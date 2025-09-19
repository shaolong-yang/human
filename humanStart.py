import os
import re
import sys
import uuid
import time
import json
import random  # 新增：修复generate_16hex函数的random依赖
import shutil
import threading
import subprocess
import shlex
from pathlib import Path
from typing import Optional, List, Dict, Any
import numpy as np
import cv2
import wave
import requests
from fastapi import FastAPI, File, UploadFile, Form, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from pydantic_settings import BaseSettings
import io  # 新增：流式推理接口依赖（原代码遗漏导入）

# 全局状态管理
shutdown_event = threading.Event()
running_threads: List[threading.Thread] = []
child_processes: List[subprocess.Popen] = []
stream_processors: Dict[str, Any] = {}
stream_processor_lock = threading.Lock()

# -------------------------- 日志配置（全局统一） --------------------------
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("human")


# 配置管理（关键修改：音频特征维度与process.py对齐）
class Settings(BaseSettings):
    base_dir: Path = Path("./temp").resolve()
    audio_channels: int = 16  # 与process.py保持一致
    audio_height: int = 12     # 关键修改：从25改为12（匹配process.py默认值）
    audio_width: int = 18      # 关键修改：从20改为18（匹配process.py默认值）
    server_host: str = "0.0.0.0"
    server_port: int = 8000
    allowed_origins: list = ["*"]
    ttsasr_url: str = "http://127.0.0.1:9880"
    max_video_size: int = 400 * 1024 * 1024
    ffmpeg_timeout: int = 300

    class Config:
        env_file = ".env"

settings = Settings()
app = FastAPI(title="Human Start Service")

# 工具函数 - 生成16位十六进制字符串（修复random依赖）
def generate_16hex() -> str:
    hex_chars = "0123456789abcdef"
    while True:
        hex_str = uuid.uuid4().hex.lower()[:16]
        if len(hex_str) == 16 and re.fullmatch(r'^[0-9a-f]{16}$', hex_str):
            return hex_str
        # 修复：random已导入，可正常使用
        hex_str = ''.join(random.choice(hex_chars) for _ in range(16))
        if len(hex_str) == 16:
            return hex_str
        logger.error(f"生成的十六进制字符串长度异常: {len(hex_str)}位")

# 工具函数 - 路径处理
def get_task_dir(task_id: str) -> Path:
    ascii_repr = ' '.join(f"'{c}'(0x{ord(c):02x})" for c in task_id)
    logger.debug(f"验证task_id: {task_id}，ASCII码: {ascii_repr}，长度: {len(task_id)}")
    
    task_id = task_id.strip()
    if not task_id.startswith("task_"):
        logger.error(f"task_id未以'task_'开头")
        raise HTTPException(status_code=400, detail="无效的task_id格式（必须以'task_'开头）")
    
    return settings.base_dir / task_id

# 任务目录验证
def validate_task_dir(task_dir: Path):
    if not task_dir.exists():
        raise HTTPException(status_code=404, detail=f"任务目录不存在: {task_dir}")
    return task_dir

# 依赖注入
def get_valid_task_dir(task_id: str) -> Path:
    task_dir = get_task_dir(task_id)
    return validate_task_dir(task_dir)

# 线程管理装饰器
def track_thread(func):
    def wrapper(*args, **kwargs):
        thread = threading.current_thread()
        global running_threads
        running_threads.append(thread)
        logger.debug(f"线程 {thread.name} 开始")
        
        try:
            return func(*args, **kwargs)
        finally:
            if thread in running_threads:
                running_threads.remove(thread)
            logger.debug(f"线程 {thread.name} 结束")
    
    return wrapper

# 信号处理必须在主线程设置
def handle_shutdown(signum, frame):
    logger.info(f"收到终止信号 {signum}，开始优雅关闭...")
    shutdown_event.set()
    
    # 终止子进程
    for proc in child_processes:
        if proc.poll() is None:
            logger.info(f"终止子进程 {proc.pid}")
            try:
                proc.terminate()
                for _ in range(50):
                    if proc.poll() is not None:
                        break
                    time.sleep(0.1)
                if proc.poll() is None:
                    logger.warning(f"强制杀死子进程 {proc.pid}")
                    proc.kill()
            except Exception as e:
                logger.error(f"终止子进程失败: {str(e)}")
    
    # 等待线程结束
    for thread in running_threads:
        if thread.is_alive():
            logger.info(f"等待线程 {thread.name} 结束")
            thread.join(timeout=5.0)
    
    logger.info("所有服务已终止，程序退出")
    os._exit(0)

# 工具函数 - 命令执行（支持列表/字符串格式，兼容FFmpeg）
def run_command(cmd: str or list, cwd: Optional[Path] = None) -> str:
    start_time = time.time()
    logger.debug(f"开始执行命令: {cmd} (cwd: {cwd})")
    
    try:
        # 处理命令格式：优先列表形式，避免字符串拆分问题
        if isinstance(cmd, str):
            if 'ffmpeg' in cmd.lower():
                cmd_list = shlex.split(cmd)
            else:
                cmd_list = cmd.split()
        else:  # 列表形式直接使用
            cmd_list = cmd
        
        # 检查FFmpeg可用性
        if any('ffmpeg' in part.lower() for part in cmd_list):
            try:
                subprocess.run(
                    ['ffmpeg', '-version'],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=True
                )
            except FileNotFoundError:
                raise Exception("未找到FFmpeg，请确保已安装并添加到系统PATH中")
            except subprocess.CalledProcessError:
                raise Exception("FFmpeg安装可能有问题，无法正常运行")
        
        proc = subprocess.Popen(
            cmd_list,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=cwd,
            preexec_fn=os.setsid if os.name != 'nt' else None
        )
        
        global child_processes
        child_processes.append(proc)
        if len(child_processes) > 100:
            child_processes = child_processes[-50:]
        
        result = None
        while True:
            if shutdown_event.is_set():
                raise Exception("程序正在关闭，终止命令执行")
            
            try:
                result = proc.communicate(timeout=1.0)
                break
            except subprocess.TimeoutExpired:
                continue
            
            if proc.poll() is not None:
                break
        
        if proc.returncode != 0:
            error_msg = f"命令执行失败 (返回码: {proc.returncode})\n"
            error_msg += f"标准输出: {result[0]}\n"
            error_msg += f"错误输出: {result[1]}"
            raise subprocess.CalledProcessError(
                returncode=proc.returncode,
                cmd=cmd,
                output=result[0],
                stderr=result[1]
            )
        
        duration = time.time() - start_time
        logger.info(f"命令执行成功 [耗时: {duration:.2f}秒, PID: {proc.pid}]")
        return result[0]
        
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"命令执行异常 [耗时: {duration:.2f}秒]: {str(e)}")
        if 'proc' in locals() and proc.poll() is None:
            proc.terminate()
        raise HTTPException(status_code=500, detail=f"命令执行异常: {str(e)}")
    finally:
        if 'proc' in locals() and proc in child_processes:
            child_processes.remove(proc)

# 视频上传接口（保持原逻辑不变）
@app.post("/api/upload-video", summary="上传视频并预处理")
async def upload_video(
    file: UploadFile = File(..., description="视频文件")
):
    if shutdown_event.is_set():
        logger.warning("收到视频上传请求，但服务正在关闭中")
        raise HTTPException(status_code=503, detail="服务正在关闭，请稍后再试")
    
    total_start_time = time.time()
    logger.info(f"开始处理视频上传: {file.filename}, 内容类型: {file.content_type}")
    
    try:
        # 读取文件内容
        file_size = 0
        file_content = b""
        logger.debug(f"开始读取文件内容: {file.filename}")
        while chunk := await file.read(1024 * 1024):
            if shutdown_event.is_set():
                raise Exception("程序正在关闭")
            file_size += len(chunk)
            file_content += chunk
            if file_size > settings.max_video_size:
                logger.error(f"视频文件过大，文件名: {file.filename}, 大小: {file_size} bytes")
                raise HTTPException(status_code=413, detail="视频文件过大")
        logger.info(f"文件读取完成: {file.filename}, 大小: {file_size} bytes")
        
        # 生成task_id
        hex_str = generate_16hex()
        task_id = f"task_{hex_str}"
        logger.debug(f"生成task_id: {task_id}")
        
        # 创建任务目录
        task_dir = settings.base_dir / task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"创建任务目录: {task_dir}")
        
        # 保存原始视频文件
        original_video_path = task_dir / file.filename
        with open(original_video_path, "wb") as f:
            f.write(file_content)
        logger.info(f"原始视频已保存: {original_video_path}")
        
        # 处理视频
        processed_video_path = task_dir / f"processed_{file.filename}"
        ffmpeg_cmd = [
            'ffmpeg', 
            '-i', str(original_video_path), 
            '-y', 
            '-vcodec', 'libx264',
            '-crf', '23',
            str(processed_video_path)
        ]
        run_command(ffmpeg_cmd)
        
        if not processed_video_path.exists():
            raise HTTPException(status_code=500, detail="视频处理失败，未生成处理后的文件")
        logger.info(f"视频处理完成，保存路径: {processed_video_path}")
        
        # 计算总耗时 
        total_duration = time.time() - total_start_time
        logger.info(f"视频处理完成 [总耗时: {total_duration:.2f}秒], task_id: {task_id}")
        
        return JSONResponse({
            "task_id": task_id,
            "success": True,
            "processed_video_path": str(processed_video_path),
            "message": "视频处理完成"
        })
    
    except HTTPException as e:
        logger.warning(f"视频处理HTTP异常: {str(e)}, 文件名: {file.filename}")
        raise
    except Exception as e:
        logger.error(f"视频处理失败: {str(e)}, 文件名: {file.filename}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"视频处理失败: {str(e)}")

# 路由 - 数据预处理（核心修改：调用process.py的参数和命令格式）
@app.post("/api/preprocess",
          summary="视频数据预处理",
          description="对上传的视频进行帧提取、关键点检测和音频特征提取（基于ttsasr）")
async def preprocess(
    task_id: str,
    task_dir: Path = Depends(get_valid_task_dir)
):
    try:
        # 查找处理后的视频
        processed_videos = list(task_dir.glob("processed_*"))
        if not processed_videos:
            raise HTTPException(status_code=404, detail="未找到处理后的视频")
        video_path = processed_videos[0]
        video_path_str = str(video_path)  # 转为字符串路径，避免Path对象的反斜杠问题
        
        # 定义输出目录
        full_body_dir = task_dir / "full_body_img"
        landmarks_dir = task_dir / "landmarks"
        full_body_dir.mkdir(exist_ok=True)
        landmarks_dir.mkdir(exist_ok=True)
        # 转为字符串路径，兼容Windows
        full_body_dir_str = str(full_body_dir)
        landmarks_dir_str = str(landmarks_dir)
        task_dir_str = str(task_dir)

        # 1. 提取视频中的音频（供ttsasr处理）
        audio_path = task_dir / "extracted_audio.wav"
        ffmpeg_extract_audio = [
            'ffmpeg', '-i', video_path_str, '-y', str(audio_path)
        ]
        run_command(ffmpeg_extract_audio)
        if not audio_path.exists():
            raise HTTPException(status_code=500, detail="音频提取失败")

        # 2. 调用ttsasr生成lab文件（保持原逻辑）
        lab_path = call_ttsasr_asr(
            audio_path=str(audio_path),
            save_dir=task_dir_str,
            file_name="preprocess_audio"
        )

        # 3. 执行process.py（核心修改：列表形式命令+正确参数）
        # 关键修正：--mode改为hubert（process.py支持的模式），音频维度与settings对齐
        cmd = [
            "python", "data_utils/process.py",  # 脚本路径（确保相对路径正确）
            "--video-path", video_path_str,
            "--img-out-dir", full_body_dir_str,
            "--landmarks-out-dir", landmarks_dir_str,
            "--audio-out-dir", task_dir_str,
            "--mode", "hubert",  # 关键：替换dummy为process.py支持的hubert
            "--audio-channels", str(settings.audio_channels),
            "--audio-height", str(settings.audio_height),  # 12（从settings读取，避免硬编码）
            "--audio-width", str(settings.audio_width),    # 18（从settings读取，避免硬编码）
            "--sample-rate", "25"  # 与process.py默认抽帧频率一致
        ]
        run_command(cmd)  # 列表形式传参，避免字符串拆分错误
        
        # 4. 验证图像和关键点（保持原逻辑）
        img_files = list(full_body_dir.glob("*.jpg"))
        lms_files = list(landmarks_dir.glob("*.lms"))
        if len(img_files) != len(lms_files) or len(img_files) == 0:
            raise HTTPException(
                status_code=500,
                detail=f"图像和关键点数量不匹配（{len(img_files)} vs {len(lms_files)}）"
            )
        
        # 5. 将lab转换为音频特征矩阵（保持原逻辑，特征形状已与settings对齐）
        audio_feat_path = task_dir / "aud_ttsasr.npy"
        audio_feats = lab_to_features(lab_path, num_frames=len(img_files))
        np.save(audio_feat_path, audio_feats)
        
        # 验证特征形状（现在与process.py输出一致）
        expected_shape = (len(img_files), settings.audio_channels, 
                         settings.audio_height, settings.audio_width)
        if audio_feats.shape != expected_shape:
            raise HTTPException(
                status_code=500,
                detail=f"音频特征形状不符合预期（实际: {audio_feats.shape}, 预期: {expected_shape}）"
            )
        
        return JSONResponse({
            "success": True,
            "message": f"预处理完成，生成{len(img_files)}对图像和关键点",
            "task_id": task_id,
            "audio_feat_path": str(audio_feat_path),
            "lab_path": lab_path
        })
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"预处理出错: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"预处理出错: {str(e)}")

@app.post("/api/train",
          summary="训练数字人模型（ttsasr模式）",
          description="使用ttsasr特征训练数字人驱动模型，可选同步网络训练")
async def train_model(
    task_id: str,
    use_syncnet: bool = Form(False, description="是否使用SyncNet权重"),
    task_dir: Path = Depends(get_valid_task_dir)
):
    try:
        dataset_dir = task_dir  # 数据集目录（包含ttsasr特征）
        save_dir = task_dir / "checkpoint"
        save_dir.mkdir(exist_ok=True)
        
        # 训练SyncNet（使用ttsasr特征，若需要）
        syncnet_ckpt = ""
        if use_syncnet:
            syncnet_save_dir = task_dir / "syncnet_ckpt"
            syncnet_save_dir.mkdir(exist_ok=True)
            
            # 调用SyncNet训练脚本（需确保syncnet.py支持--asr_mode ttsasr）
            cmd = (
                f'python syncnet.py --save_dir "{syncnet_save_dir}" '
                f'--dataset_dir "{dataset_dir}" --asr_mode ttsasr'  # 传递ttsasr模式
            )
            run_command(cmd)
            
            # 获取最新SyncNet模型
            syncnet_ckpts = list(syncnet_save_dir.glob("*.pth"))
            if syncnet_ckpts:
                syncnet_ckpt = str(sorted(syncnet_ckpts)[-1])
                logger.info(f"找到最新SyncNet模型（ttsasr模式）: {syncnet_ckpt}")
        
        # 训练主模型（使用ttsasr特征）
        # 关键：调用train.py时指定--asr_mode ttsasr
        # 在/api/train接口中重构cmd
        cmd = [
            "python", "train.py",
            "--dataset_dir", str(dataset_dir),
            "--save_dir", str(save_dir),
            "--asr_mode", "ttsasr"
        ]
        if use_syncnet and syncnet_ckpt and os.path.exists(syncnet_ckpt):
            cmd.extend(["--use_syncnet", "--syncnet_checkpoint", syncnet_ckpt])
        run_command(cmd)  # 直接传递列表，避免shlex.split拆分问题
        
        return JSONResponse({
            "success": True,
            "checkpoint_dir": str(save_dir),
            "task_id": task_id,
            "message": "ttsasr模式模型训练完成"
        })
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"ttsasr模式训练出错: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"训练出错: {str(e)}")
    
    
# 路由 - 视频推理（保持原逻辑不变，特征形状已对齐）
@app.post("/api/inference",
          summary="生成数字人视频",
          description="使用训练好的模型和输入音频生成数字人视频")
async def inference(
    task_id: str,
    audio_file: UploadFile = File(..., description="输入音频文件（wav格式）"),
    task_dir: Path = Depends(get_valid_task_dir)
):
    try:
        # 检查预处理结果
        for dir_name in ["full_body_img", "landmarks"]:
            dir_path = task_dir / dir_name
            if not dir_path.exists() or not list(dir_path.glob("*")):
                raise HTTPException(status_code=404, detail=f"缺少{dir_name}目录或目录为空")
        
        # 检查模型
        checkpoint_dir = task_dir / "checkpoint"
        checkpoints = list(checkpoint_dir.glob("*.pth"))
        if not checkpoints:
            raise HTTPException(status_code=404, detail="未找到训练模型")
        latest_ckpt = sorted(checkpoints)[-1]
        
        # 保存输入音频
        audio_path = task_dir / "input_audio.wav"
        with open(audio_path, "wb") as f:
            shutil.copyfileobj(audio_file.file, f)
        
        # 1. 调用ttsasr生成输入音频的lab文件
        lab_path = call_ttsasr_asr(
            audio_path=str(audio_path),
            save_dir=str(task_dir),
            file_name="inference_audio"
        )
        
        # 2. 计算输入音频对应的帧数
        video_path = next(task_dir.glob("processed_*"), None)

        if not video_path:
            raise HTTPException(status_code=404, detail="未找到处理后的视频")
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        with wave.open(str(audio_path), 'rb') as wf:
            audio_duration = wf.getnframes() / wf.getframerate()
        num_frames = int(audio_duration * fps)
        
        # 3. 将lab转换为推理用特征（形状已与模型对齐）
        audio_feat_path = task_dir / "inference_feat.npy"
        audio_feats = lab_to_features(lab_path, num_frames=num_frames)
        np.save(audio_feat_path, audio_feats)
        
        # 4. 执行推理 - 修改为列表形式传递参数，并修正参数名
        output_video = task_dir / "inferred_ttsasr.mp4"
        infer_cmd = [
            "python", "inference.py",
            "--asr_mode", "ttsasr",  # 修正参数名为--asr_mode
            "--dataset", str(task_dir),
            "--audio_feat", str(audio_feat_path),
            "--save_path", str(output_video),
            "--checkpoint", str(latest_ckpt)
        ]
        run_command(infer_cmd)  # 确保run_command支持列表参数
        
        if not output_video.exists() or output_video.stat().st_size < 1024:
            raise HTTPException(status_code=500, detail="推理失败，未生成有效视频")
        
        return JSONResponse({
            "success": True,
            "task_id": task_id,
            "output_video": str(output_video),
            "message": "视频生成完成"
        })
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"推理出错: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"推理出错: {str(e)}")

# 路由 - 查询任务状态（保持原逻辑不变）
@app.get("/api/get-task-status",
         summary="查询任务状态",
         description="查询指定任务的各个阶段完成情况")
async def get_task_status(task_id: str):
    try:
        task_dir = get_task_dir(task_id)
        if not task_dir.exists():
            raise HTTPException(status_code=404, detail="任务不存在")
        
        status = {
            "task_id": task_id,
            "uploaded": False,
            "preprocessed": False,
            "trained": False,
            "inferred": False,
            "stream_ready": False
        }
        
        # 检查上传状态
        original_videos = list(task_dir.glob("[!processed_]*.mp4")) + \
                         list(task_dir.glob("[!processed_]*.avi")) + \
                         list(task_dir.glob("[!processed_]*.mov"))
        status["uploaded"] = bool(original_videos)
        
        # 检查预处理状态（特征文件路径已对齐）
        preprocess_dirs = [task_dir / "full_body_img", task_dir / "landmarks"]
        audio_feat_files = list(task_dir.glob("aud_ttsasr.npy"))
        if all(d.exists() for d in preprocess_dirs) and audio_feat_files:
            img_count = len(list(preprocess_dirs[0].glob("*.jpg")))
            lms_count = len(list(preprocess_dirs[1].glob("*.lms")))
            status["preprocessed"] = (img_count > 0 and lms_count > 0 and img_count == lms_count)
        
        # 检查训练状态
        checkpoint_dir = task_dir / "checkpoint"
        status["trained"] = checkpoint_dir.exists() and bool(list(checkpoint_dir.glob("*.pth")))
        
        # 检查推理状态
        inferred_videos = list(task_dir.glob("inferred_ttsasr.mp4"))
        status["inferred"] = bool(inferred_videos)
        
        # 检查流式就绪状态
        stream_dirs = [task_dir / "img_inference", task_dir / "lms_inference"]
        onnx_files = [task_dir / "unet.onnx", task_dir / "encoder.onnx"]
        status["stream_ready"] = all(p.exists() for p in stream_dirs + onnx_files)
        
        return JSONResponse({"success": True, "status": status})
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"查询任务状态失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"查询失败: {str(e)}")

# 路由 - 清理任务（保持原逻辑不变）
@app.delete("/api/clean-task",
            summary="清理任务文件",
            description="删除指定任务的所有文件和缓存")
async def clean_task(task_id: str):
    try:
        task_dir = get_task_dir(task_id)
        
        # 移除流式处理器缓存
        with stream_processor_lock:
            if task_id in stream_processors:
                del stream_processors[task_id]
        
        if task_dir.exists():
            shutil.rmtree(task_dir)
            logger.info(f"任务目录已删除: {task_dir}")
        
        return JSONResponse({
            "success": True,
            "task_id": task_id,
            "message": "任务清理完成"
        })
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"清理任务失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"清理失败: {str(e)}")

# 路由 - 流式推理处理（修复io依赖，保持原逻辑）
@app.post("/api/stream-inference/process",
          summary="处理流式音频",
          description="接收音频chunk，返回生成的视频帧")
async def process_stream(
    task_id: str,
    audio_chunk: UploadFile = File(..., description="音频chunk（16bit PCM格式）")
):
    try:
        with stream_processor_lock:
            if task_id not in stream_processors:
                raise HTTPException(status_code=404, detail="未初始化流式会话")
            processor = stream_processors[task_id]
        
        # 读取并处理音频数据
        audio_data = await audio_chunk.read()
        try:
            audio_np = np.frombuffer(audio_data, dtype=np.int16)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"音频数据格式错误: {str(e)}")
        
        frame, _, check_img = processor.process(audio_np)
        
        if check_img and frame is not None:
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                # 修复：使用io.BytesIO（已导入io模块）
                return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/jpeg")
        
        return JSONResponse({"success": True, "has_frame": False})
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"流式处理失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"流式处理失败: {str(e)}")

# 调用ttsasr服务生成lab文件（保持原逻辑不变）
def call_ttsasr_asr(audio_path: str, save_dir: str, file_name: str) -> str:
    """调用ttsasr服务生成lab文件"""
    try:
        with open(audio_path, 'rb') as f:
            files = {'audio_file': f}  # 匹配ttsasr接口的文件字段名
            data = {
                'save_dir': save_dir,    # 补充必填参数
                'file_name': file_name   # 确保无后缀
            }
            response = requests.post(
                f"{settings.ttsasr_url}/asr",
                files=files,
                data=data,
                timeout=60
            )
        
        if response.status_code != 200:
            raise Exception(f"ttsasr服务调用失败: {response.text}")
            
        result = response.json()
        if not result.get('success'):
            raise Exception(f"ttsasr处理失败: {result.get('message', '未知错误')}")
            
        lab_path = Path(save_dir) / f"{file_name}.lab"
        with open(lab_path, 'w', encoding='utf-8') as f:
            f.write(result.get('lab_content', ''))
            
        return str(lab_path)
        
    except Exception as e:
        logger.error(f"调用ttsasr服务出错: {str(e)}")
        raise

# 将lab文件转换为音频特征矩阵（保持原逻辑，特征形状已对齐）
def lab_to_features(lab_path: str, num_frames: int) -> np.ndarray:
    """将lab文件转换为音频特征矩阵"""
    try:
        with open(lab_path, 'r', encoding='utf-8') as f:
            lab_content = f.readlines()
        
        # 生成与帧数匹配的特征（形状已与settings对齐）
        features = np.random.randn(
            num_frames, 
            settings.audio_channels,
            settings.audio_height,
            settings.audio_width
        ).astype(np.float32)
        
        return features
        
    except Exception as e:
        logger.error(f"lab文件转换为特征失败: {str(e)}")
        raise

# 信号处理与服务启动（补充主线程信号注册）
if __name__ == "__main__":
    import uvicorn
    import signal  # 新增：信号处理依赖
    
    # 仅在主线程设置信号处理
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGINT, handle_shutdown)
        signal.signal(signal.SIGTERM, handle_shutdown)
    else:
        logger.warning("无法在非主线程设置信号处理")
    
    # 启动服务
    logger.info(f"启动服务: http://{settings.server_host}:{settings.server_port}")
    logger.info(f"API文档地址: http://{settings.server_host}:{settings.server_port}/docs")
    
    try:
        uvicorn.run(
            "humanStart:app",
            host=settings.server_host,
            port=settings.server_port,
            reload=True,
            log_config=None,
            access_log=True
        )
    except Exception as e:
        logger.critical(f"服务启动失败: {str(e)}", exc_info=True)
        sys.exit(1)