import os
import re
import sys
import uuid
import time
import struct
import shutil
import base64
import wave
import io
import yaml
import requests
import threading
import hashlib
import platform
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, File, UploadFile, Form, Depends, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.fernet import Fernet
import numpy as np
import soundfile as sf
from pypinyin import pinyin, Style
from pydub import AudioSegment  # 处理音频格式转换
from faster_whisper import WhisperModel
from faster_whisper.vad import VadOptions  # ASR语音活性检测配置

# -------------------------- 跨平台依赖处理 --------------------------
try:
    import wmi  # Windows硬件信息获取（可选，无则用fallback方案）
except ImportError:
    wmi = None

# -------------------------- 模块路径配置（关键：适配GPT_SoVITS） --------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
# 导入GPT_SoVITS的TTS模块（需确保GPT_SoVITS目录在当前路径下）
gpt_sovits_path = os.path.join(current_dir, "GPT_SoVITS")
if not os.path.exists(gpt_sovits_path):
    print(f"错误：未找到GPT_SoVITS目录，预期路径：{gpt_sovits_path}")
    print("请将GPT_SoVITS文件夹放在当前脚本同级目录下")
    sys.exit(1)
sys.path.append(gpt_sovits_path)
from TTS_infer_pack.TTS import TTS, TTS_Config  # 导入GPT_SoVITS的TTS类

# -------------------------- 日志配置（全局统一） --------------------------
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("tts_asr")

# -------------------------- 全局配置（用户可根据实际情况修改） --------------------------
class Config:
    # 服务配置
    TTS_BIND_ADDR = "0.0.0.0"  # 0.0.0.0允许外部访问，127.0.0.1仅本地
    TTS_PORT = 9880
    # 安全验证（生产环境需替换为真实地址和密钥）
    PASSWORD_CHECK_URL = "https://gpt.wenyou.vip/check"  # 密码验证接口
    DEFAULT_ENCRYPTION_PASSWORD = "M)ZT*Z356n|_t3!#>KMl;!c8=_/ @Aw!"  # 示例密钥
    # TTS解密配置（对应GPT_SoVITS的加密文件）
    ENCRYPTED_ROOT_DIR = os.path.join(gpt_sovits_path)  # 加密文件目录
    SALT_FILE = ".salt"  # 盐值文件（在ENCRYPTED_ROOT_DIR下）
    TEMP_DECRYPT_DIR = os.path.join(gpt_sovits_path, "temp")  # 解密临时目录
    ENCRYPTED_FILES = [  # GPT_SoVITS需解密的文件（根据实际文件调整）
        "lgyy-e15.ckpt",
        "lgyy_e8_s416.pth",
        "tts_infer.yaml"
    ]
    DECRYPTED_TTS_CONFIG_PATH = os.path.join(TEMP_DECRYPT_DIR, "tts_infer.yaml")  # 解密后的配置文件
    # ASR配置（faster-whisper模型）
    ASR_MODEL_PATH = os.path.join(gpt_sovits_path, "small")  # 模型路径（可替换为base/large）
    PCM_CONFIG = {  # ASR要求的PCM格式（固定16kHz/16bit/单声道）
        "sample_rate": 16000,
        "bits_per_sample": 16,
        "channels": 1
    }

# 实例化配置
CFG = Config()

# -------------------------- 全局状态（避免重复初始化） --------------------------
ASR_MODEL = None  # faster-whisper模型实例
ASR_LOCK = threading.Lock()  # ASR线程锁（确保单线程调用）
TTS_PIPELINE = None  # GPT_SoVITS TTS实例
TTS_CONFIG = None  # TTS配置实例
MACHINE_ID = ""  # 设备唯一ID
ENCRYPTION_PASSWORD = ""  # 最终使用的加密密码

# -------------------------- 基础工具函数（通用功能） --------------------------
def normalize_path(path: str) -> str:
    """标准化路径（兼容Windows/Linux）"""
    normalized = os.path.normpath(path)
    logger.debug(f"[路径处理] 原始: {path} -> 标准化: {normalized}")
    return normalized

def get_machine_unique_id() -> str:
    """跨平台获取设备唯一ID（优先硬件信息，失败则用系统信息生成）"""
    logger.info("[设备标识] 生成设备唯一ID")
    try:
        system = platform.system()
        # Windows：读取主板序列号（需wmi库）
        if system == "Windows" and wmi:
            try:
                c = wmi.WMI()
                for board in c.Win32_BaseBoard():
                    if board.SerialNumber and board.SerialNumber.strip():
                        board_sn = board.SerialNumber.strip().lower()
                        logger.info(f"[设备标识] Windows主板序列号: {board_sn[:8]}...")
                        return hashlib.sha256(board_sn.encode()).hexdigest()
            except Exception as e:
                logger.warning(f"[设备标识] Windows获取主板序列号失败: {str(e)}")
        # Linux：读取CPU序列号
        elif system == "Linux":
            try:
                with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
                    for line in f:
                        if line.startswith("serial"):
                            cpu_sn = line.split(":", 1)[1].strip().lower()
                            if cpu_sn and cpu_sn != "00000000":
                                logger.info(f"[设备标识] Linux CPU序列号: {cpu_sn[:8]}...")
                                return hashlib.sha256(cpu_sn.encode()).hexdigest()
            except Exception as e:
                logger.warning(f"[设备标识] Linux获取CPU序列号失败: {str(e)}")
        # macOS：读取硬件UUID
        elif system == "Darwin":
            try:
                import subprocess
                output = subprocess.check_output(
                    ["ioreg", "-d2", "-c", "IOPlatformExpertDevice"],
                    encoding="utf-8"
                )
                for line in output.splitlines():
                    if "IOPlatformUUID" in line:
                        uuid_str = line.split('"')[3].strip().lower()
                        logger.info(f"[设备标识] macOS硬件UUID: {uuid_str[:8]}...")
                        return hashlib.sha256(uuid_str.encode()).hexdigest()
            except Exception as e:
                logger.warning(f"[设备标识] macOS获取硬件UUID失败: {str(e)}")
        # 所有方案失败：用系统信息生成稳定UUID
        fallback_info = f"{platform.node()}-{platform.processor()}-{system}-{platform.machine()}"
        fallback_id = hashlib.sha256(fallback_info.encode()).hexdigest()
        logger.warning(f"[设备标识] 硬件信息获取失败，使用系统信息生成ID: {fallback_id[:8]}...")
        return fallback_id
    except Exception as e:
        logger.error(f"[设备标识] 生成ID失败: {str(e)}", exc_info=True)
        sys.exit(1)

def get_encryption_password(machine_id: str) -> str:
    """获取加密密码（示例：优先从验证接口获取，失败用默认密钥）"""
    logger.info(f"[密码获取] 设备ID: {machine_id[:8]}...")
    try:
        # 实际生产环境：从验证接口获取密钥（这里用示例逻辑）
        resp = requests.post(
            CFG.PASSWORD_CHECK_URL,
            json={"userId": machine_id},
            timeout=10
        )
        resp.raise_for_status()
        result = resp.json()
        if result.get("key"):
            logger.info("[密码获取] 从接口成功获取密钥")
            return result["key"]
        logger.warning("[密码获取] 接口未返回有效密钥，使用默认密钥")
        return CFG.DEFAULT_ENCRYPTION_PASSWORD
    except Exception as e:
        logger.error(f"[密码获取] 接口调用失败: {str(e)}，使用默认密钥")
        return CFG.DEFAULT_ENCRYPTION_PASSWORD

def validate_password(machine_id: str, password: str) -> bool:
    """密码验证（对接外部验证接口）"""
    logger.info(f"[密码验证] 设备ID: {machine_id[:8]}...")
    try:
        resp = requests.post(
            CFG.PASSWORD_CHECK_URL,
            json={"userId": machine_id, "key": password},
            timeout=10
        )
        resp.raise_for_status()
        result = resp.json()
        is_valid = result.get("key") == password
        logger.info(f"[密码验证] 结果: {'有效' if is_valid else '无效'}")
        return is_valid
    except requests.exceptions.RequestException as e:
        logger.error(f"[密码验证] 接口调用失败: {str(e)}")
        # 开发环境可临时跳过验证（生产环境需删除此逻辑）
        logger.warning("[密码验证] 开发环境临时允许跳过验证")
        return True
    except Exception as e:
        logger.error(f"[密码验证] 未知错误: {str(e)}", exc_info=True)
        return False

# -------------------------- TTS核心功能（解密+初始化） --------------------------
def load_encryption_key(password: str, salt: bytes) -> bytes:
    """生成TTS解密密钥（基于PBKDF2HMAC算法）"""
    try:
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=480000,  # 加密强度（不建议修改）
        )
        return base64.urlsafe_b64encode(kdf.derive(password.encode("utf-8")))
    except Exception as e:
        logger.error(f"[TTS密钥] 生成失败: {str(e)}", exc_info=True)
        raise

def decrypt_single_file(encrypted_path: str, decrypted_path: str, cipher: Fernet) -> bool:
    """解密单个TTS文件（权重/配置）"""
    encrypted_path = normalize_path(encrypted_path)
    decrypted_path = normalize_path(decrypted_path)
    try:
        if not os.path.exists(encrypted_path):
            logger.error(f"[TTS解密] 文件不存在: {encrypted_path}")
            return False
        # 读取加密数据
        with open(encrypted_path, "rb") as f:
            encrypted_data = f.read()
        # 解密
        decrypted_data = cipher.decrypt(encrypted_data)
        # 保存解密文件
        os.makedirs(os.path.dirname(decrypted_path), exist_ok=True)
        with open(decrypted_path, "wb") as f:
            f.write(decrypted_data)
        logger.info(f"[TTS解密] 成功: {os.path.basename(decrypted_path)}")
        return True
    except Exception as e:
        logger.error(f"[TTS解密] 文件 {os.path.basename(encrypted_path)} 失败: {str(e)}", exc_info=True)
        return False

def decrypt_tts_resources(password: str) -> bool:
    """批量解密TTS资源文件（权重+配置）"""
    logger.info("[TTS解密] 开始批量解密资源文件")
    # 清理旧临时目录
    if os.path.exists(CFG.TEMP_DECRYPT_DIR):
        try:
            shutil.rmtree(CFG.TEMP_DECRYPT_DIR, ignore_errors=True)
            logger.info(f"[TTS解密] 清理旧临时目录: {CFG.TEMP_DECRYPT_DIR}")
        except Exception as e:
            logger.error(f"[TTS解密] 清理旧目录失败: {str(e)}", exc_info=True)
            return False
    # 检查加密根目录和盐值文件
    encrypted_root = normalize_path(CFG.ENCRYPTED_ROOT_DIR)
    salt_path = normalize_path(os.path.join(encrypted_root, CFG.SALT_FILE))
    if not os.path.exists(encrypted_root):
        logger.error(f"[TTS解密] 加密根目录不存在: {encrypted_root}")
        return False
    if not os.path.exists(salt_path):
        logger.error(f"[TTS解密] 盐值文件不存在: {salt_path}")
        return False
    # 读取盐值
    with open(salt_path, "rb") as f:
        salt = f.read()
    # 生成解密器
    try:
        key = load_encryption_key(password, salt)
        cipher = Fernet(key)
    except Exception as e:
        logger.error("[TTS解密] 解密器初始化失败", exc_info=True)
        return False
    # 批量解密
    all_success = True
    for rel_path in CFG.ENCRYPTED_FILES:
        encrypted_path = os.path.join(encrypted_root, rel_path)
        decrypted_path = os.path.join(CFG.TEMP_DECRYPT_DIR, rel_path)
        if not decrypt_single_file(encrypted_path, decrypted_path, cipher):
            all_success = False
    if all_success:
        logger.info("[TTS解密] 所有文件解密成功")
    else:
        logger.error("[TTS解密] 部分文件解密失败")
    return all_success

def init_tts_pipeline(password: str) -> tuple[Optional[TTS_Config], Optional[TTS]]:
    """初始化TTS pipeline（GPT_SoVITS）"""
    logger.info("[TTS初始化] 开始加载TTS模型")
    # 检查解密后的配置文件
    decrypted_config_path = normalize_path(CFG.DECRYPTED_TTS_CONFIG_PATH)
    if not os.path.exists(decrypted_config_path):
        logger.error(f"[TTS初始化] 配置文件不存在: {decrypted_config_path}")
        return None, None
    # 加载TTS配置
    try:
        with open(decrypted_config_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)
        tts_config = TTS_Config(decrypted_config_path)
        logger.info(f"[TTS初始化] 配置加载成功（设备: {tts_config.device}）")
    except Exception as e:
        logger.error("[TTS初始化] 配置加载失败", exc_info=True)
        return None, None
    # 初始化TTS模型
    try:
        tts_pipeline = TTS(tts_config)
        logger.info("[TTS初始化] 模型加载成功")
        return tts_config, tts_pipeline
    except Exception as e:
        logger.error("[TTS初始化] 模型加载失败", exc_info=True)
        return tts_config, None

# -------------------------- ASR核心功能（模型加载+音频处理） --------------------------
def load_asr_model() -> Optional[WhisperModel]:
    """加载faster-whisper模型（优先GPU，失败则CPU）"""
    logger.info(f"[ASR初始化] 开始加载模型: {CFG.ASR_MODEL_PATH}")
    try:
        # 优先GPU（float16精度，速度快）
        model = WhisperModel(
            CFG.ASR_MODEL_PATH,
            device="cuda",
            compute_type="float16"
        )
        logger.info("[ASR初始化] GPU模式加载成功")
        return model
    except Exception as e:
        logger.warning(f"[ASR初始化] GPU加载失败: {str(e)}，尝试CPU模式")
        # CPU模式（int8精度，节省内存）
        try:
            model = WhisperModel(
                CFG.ASR_MODEL_PATH,
                device="cpu",
                compute_type="int8"
            )
            logger.info("[ASR初始化] CPU模式加载成功")
            return model
        except Exception as e2:
            logger.error(f"[ASR初始化] CPU加载失败: {str(e2)}", exc_info=True)
            return None

def audio_to_pcm(audio_path: str) -> Optional[bytes]:
    """将WAV音频转为ASR要求的PCM格式（16kHz/16bit/单声道）"""
    audio_path = normalize_path(audio_path)
    logger.debug(f"[ASR预处理] 转换音频: {os.path.basename(audio_path)}")
    try:
        # 读取音频文件（支持WAV格式）
        audio = AudioSegment.from_wav(audio_path)
        # 调整采样率
        if audio.frame_rate != CFG.PCM_CONFIG["sample_rate"]:
            audio = audio.set_frame_rate(CFG.PCM_CONFIG["sample_rate"])
            logger.debug(f"[ASR预处理] 采样率调整: {audio.frame_rate} -> {CFG.PCM_CONFIG['sample_rate']}")
        # 调整声道数（单声道）
        if audio.channels != CFG.PCM_CONFIG["channels"]:
            audio = audio.set_channels(CFG.PCM_CONFIG["channels"])
            logger.debug(f"[ASR预处理] 声道数调整: {audio.channels} -> {CFG.PCM_CONFIG['channels']}")
        # 调整位深（16bit）
        target_sample_width = CFG.PCM_CONFIG["bits_per_sample"] // 8
        if audio.sample_width != target_sample_width:
            audio = audio.set_sample_width(target_sample_width)
            logger.debug(f"[ASR预处理] 位深调整: {audio.sample_width*8}bit -> {CFG.PCM_CONFIG['bits_per_sample']}bit")
        # 返回PCM原始数据
        return audio.raw_data
    except Exception as e:
        logger.error(f"[ASR预处理] 音频转换失败: {str(e)}", exc_info=True)
        return None

def generate_lab_from_audio(audio_path: str) -> Optional[str]:
    """ASR处理音频生成LAB文件（包含时间戳和拼音）"""
    global ASR_MODEL
    if not ASR_MODEL:
        logger.error("[ASR处理] 模型未初始化")
        return None
    # 1. 音频转PCM
    pcm_data = audio_to_pcm(audio_path)
    if not pcm_data:
        logger.error("[ASR处理] PCM转换失败")
        return None
    # 2. PCM转WAV（供faster-whisper读取）
    def pcm_to_wav(pcm: bytes) -> bytes:
        sr = CFG.PCM_CONFIG["sample_rate"]
        bits = CFG.PCM_CONFIG["bits_per_sample"]
        ch = CFG.PCM_CONFIG["channels"]
        byte_rate = sr * ch * bits // 8
        # WAV头部（44字节）
        wav_header = b""
        wav_header += b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVE"
        wav_header += b"fmt " + struct.pack("<I", 16) + struct.pack("<H", 1)  # 16bit PCM
        wav_header += struct.pack("<H", ch) + struct.pack("<I", sr)
        wav_header += struct.pack("<I", byte_rate) + struct.pack("<H", ch * bits // 8)
        wav_header += struct.pack("<H", bits)
        wav_header += b"data" + struct.pack("<I", len(pcm)) + pcm
        return wav_header
    wav_data = pcm_to_wav(pcm_data)
    wav_io = io.BytesIO(wav_data)
    wav_io.seek(0)
    # 3. ASR识别（加锁确保线程安全）
    try:
        ASR_LOCK.acquire()
        # 配置VAD（语音活性检测，过滤静音）
        vad_options = VadOptions(
            min_silence_duration_ms=100,  # 最小静音时长（毫秒）
            min_speech_duration_ms=100    # 最小语音时长（毫秒）
        )
        # 开始识别（中文，带词级时间戳）
        segments, _ = ASR_MODEL.transcribe(
            wav_io,
            language="zh",
            initial_prompt="转录为标准普通话，保留正常停顿，不添加额外标点",
            word_timestamps=True,  # 开启词级时间戳
            vad_filter=True,
            vad_parameters=vad_options,
            beam_size=1  # 速度优先（beam_size越小越快）
        )
    except Exception as e:
        logger.error(f"[ASR处理] 识别失败: {str(e)}", exc_info=True)
        return None
    finally:
        ASR_LOCK.release()
    # 4. 生成LAB数据（时间戳+拼音）
    lab_segments = []
    for seg in segments:
        for word in seg.words:
            # 清理文本（移除特殊字符）
            clean_word = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9\s]", "", word.word.strip())
            if not clean_word:
                continue
            # 数字转中文（如"123"-> "一二三"）
            clean_word = re.sub(r"\d+", lambda m: "".join([
                {"0":"零","1":"一","2":"二","3":"三","4":"四","5":"五","6":"六","7":"七","8":"八","9":"九"}[d] 
                for d in m.group()
            ]), clean_word)
            # 中文转拼音（无声调）
            pinyin_list = pinyin(clean_word, style=Style.NORMAL)
            pinyin_str = " ".join([p[0] for p in pinyin_list])  # 拼音用空格分隔
            if not pinyin_str:
                continue
            # 时间戳转为毫秒
            begin_ms = int(word.start * 1000)
            end_ms = int(word.end * 1000)
            lab_segments.append({
                "begin": begin_ms,
                "end": end_ms,
                "pinyin": pinyin_str,
                "word": clean_word
            })
    # 5. 保存LAB文件（与音频同路径，同名称）
    lab_path = os.path.splitext(audio_path)[0] + ".lab"
    try:
        with open(lab_path, "w", encoding="utf-8") as f:
            f.write("# begin_time(ms) end_time(ms) pinyins\n")  # 表头
            for seg in lab_segments:
                f.write(f"{seg['begin']} {seg['end']} {seg['pinyin']}\n")
        logger.info(f"[ASR处理] LAB文件生成成功: {os.path.basename(lab_path)}")
        return lab_path
    except Exception as e:
        logger.error(f"[ASR处理] LAB文件保存失败: {str(e)}", exc_info=True)
        return None

# -------------------------- TTS音频处理工具 --------------------------
def pack_audio(audio_np: np.ndarray, sample_rate: int, media_type: str = "wav") -> Optional[io.BytesIO]:
    """将TTS生成的numpy音频数组打包为指定格式（WAV/RAW）"""
    try:
        audio_io = io.BytesIO()
        if media_type == "wav":
            # 保存为WAV格式
            sf.write(audio_io, audio_np, sample_rate, format="wav")
        elif media_type == "raw":
            # 保存为RAW格式（PCM）
            audio_io.write(audio_np.tobytes())
        else:
            logger.error(f"[TTS打包] 不支持的媒体类型: {media_type}")
            return None
        audio_io.seek(0)
        logger.debug(f"[TTS打包] 成功（格式: {media_type}，长度: {audio_io.getbuffer().nbytes}字节）")
        return audio_io
    except Exception as e:
        logger.error(f"[TTS打包] 失败: {str(e)}", exc_info=True)
        return None

def save_tts_audio(audio_io: io.BytesIO, save_dir: str, file_name: str) -> Optional[str]:
    """保存TTS音频到指定目录（自动补全WAV后缀）"""
    # 处理文件名
    if not file_name.lower().endswith(".wav"):
        file_name += ".wav"
        logger.debug(f"[TTS保存] 自动补全后缀: {file_name}")
    # 处理保存目录
    save_dir = normalize_path(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    audio_path = normalize_path(os.path.join(save_dir, file_name))
    # 保存文件
    try:
        with open(audio_path, "wb") as f:
            f.write(audio_io.getvalue())
        logger.info(f"[TTS保存] 音频文件生成: {audio_path}")
        return audio_path
    except Exception as e:
        logger.error(f"[TTS保存] 失败: {str(e)}", exc_info=True)
        return None

# -------------------------- 服务初始化（整合所有流程） --------------------------
def initialize_service() -> bool:
    """服务全局初始化（设备ID+密码验证+TTS解密+模型加载）"""
    global MACHINE_ID, ENCRYPTION_PASSWORD, TTS_CONFIG, TTS_PIPELINE, ASR_MODEL
    logger.info("="*60)
    logger.info("         TTS+ASR整合服务 - 初始化开始")
    logger.info("="*60)
    
    # 1. 生成设备唯一ID
    MACHINE_ID = get_machine_unique_id()
    # 2. 获取并验证加密密码
    ENCRYPTION_PASSWORD = get_encryption_password(MACHINE_ID)
    if not validate_password(MACHINE_ID, ENCRYPTION_PASSWORD):
        logger.error("[初始化] 密码验证失败，服务无法启动")
        return False
    # 3. 解密TTS资源
    if not decrypt_tts_resources(ENCRYPTION_PASSWORD):
        logger.error("[初始化] TTS资源解密失败，服务无法启动")
        return False
    # 4. 初始化TTS模型
    TTS_CONFIG, TTS_PIPELINE = init_tts_pipeline(ENCRYPTION_PASSWORD)
    if not TTS_PIPELINE:
        logger.error("[初始化] TTS模型加载失败，服务无法启动")
        return False
    # 5. 初始化ASR模型
    ASR_MODEL = load_asr_model()
    if not ASR_MODEL:
        logger.error("[初始化] ASR模型加载失败，服务无法启动")
        return False
    
    logger.info("="*60)
    logger.info("         TTS+ASR整合服务 - 初始化完成")
    logger.info(f"         服务地址: http://{CFG.TTS_BIND_ADDR}:{CFG.TTS_PORT}")
    logger.info("="*60)
    return True

# -------------------------- FastAPI接口定义（API服务） --------------------------
app = FastAPI(
    title="TTS+ASR整合服务",
    description="基于GPT_SoVITS的TTS合成 + faster-whisper的ASR语音转写（生成LAB文件）",
    version="1.0.0",
    docs_url="/docs"  # 启用Swagger文档（访问http://ip:port/docs调试）
)

# 请求模型（Pydantic验证）
class TTSASRRequest(BaseModel):
    """TTS+ASR联合接口请求参数"""
    text: str  # 待合成的文本（必填）
    ref_audio_path: str  # 参考音频路径（必填，用于音色克隆）
    ref_audio_text: str  # 参考音频对应的文本（必填）
    save_dir: str  # 音频/LAB保存目录（必填）
    file_name: str  # 文件名（无后缀，必填）
    # 可选参数
    text_lang: str = "zh"  # 文本语言（默认中文）
    prompt_lang: str = "zh"  # 提示语言（默认中文）
    speed_factor: float = 1.0  # 语速（1.0为正常）
    media_type: str = "wav"  # 输出音频格式（仅支持wav/raw）

class ASRRequest(BaseModel):
    """单独ASR接口请求参数"""
    audio_path: str  # 待处理音频路径（必填，WAV格式）
    save_dir: str  # LAB保存目录（必填）
    file_name: str  # LAB文件名（无后缀，必填）

# 安全验证依赖（所有接口需通过验证）
def verify_access() -> bool:
    """接口访问验证（基于设备ID和密码）"""
    global MACHINE_ID, ENCRYPTION_PASSWORD
    if not validate_password(MACHINE_ID, ENCRYPTION_PASSWORD):
        raise HTTPException(
            status_code=401,
            detail="密码验证失败，拒绝访问",
            headers={"WWW-Authenticate": "Basic"}
        )
    return True

# -------------------------- API接口实现 --------------------------
@app.get("/health", summary="服务健康检查")
async def health_check(auth: bool = Depends(verify_access)):
    """检查服务是否正常运行（返回设备和模型状态）"""
    return {
        "status": "healthy",
        "machine_id": MACHINE_ID[:8] + "..." if MACHINE_ID else "unknown",
        "tts_device": TTS_CONFIG.device if TTS_CONFIG else "unknown",
        "asr_model": CFG.ASR_MODEL_PATH if ASR_MODEL else "unknown",
        "note": "调用接口时file_name无需带后缀，自动补全"
    }

@app.post("/tts_asr", summary="TTS+ASR联合接口")
async def tts_asr_handler(req: TTSASRRequest, auth: bool = Depends(verify_access)):
    """一步完成：文本合成音频 + 音频生成LAB文件"""
    request_id = str(uuid.uuid4())[:8]  # 生成请求ID（便于追踪）
    logger.info(f"[TTS+ASR请求-{request_id}] 收到请求: 文件名={req.file_name}")
    
    # 1. 参数校验
    if not req.text.strip():
        return JSONResponse(status_code=400, content={"message": "text（合成文本）不能为空"})
    if not os.path.exists(req.ref_audio_path):
        return JSONResponse(status_code=400, content={"message": f"参考音频不存在: {req.ref_audio_path}"})
    if not req.ref_audio_text.strip():
        return JSONResponse(status_code=400, content={"message": "ref_audio_text（参考文本）不能为空"})
    if not req.save_dir:
        return JSONResponse(status_code=400, content={"message": "save_dir（保存目录）不能为空"})
    if not req.file_name.strip():
        return JSONResponse(status_code=400, content={"message": "file_name（文件名）不能为空"})
    if req.media_type not in ["wav", "raw"]:
        return JSONResponse(status_code=400, content={"message": "media_type仅支持wav/raw"})
    
    try:
        # 2. TTS合成音频
        tts_start = time.time()
        logger.debug(f"[TTS+ASR请求-{request_id}] 开始TTS合成")
        # 构造TTS请求参数
        tts_params = {
            "text": req.text,
            "ref_audio_path": req.ref_audio_path,
            "ref_audio_text": req.ref_audio_text,
            "text_lang": req.text_lang,
            "prompt_lang": req.prompt_lang,
            "speed_factor": req.speed_factor
        }
        # 调用TTS生成音频（GPT_SoVITS返回生成器）
        tts_generator = TTS_PIPELINE.run(tts_params)
        sample_rate, audio_np = next(tts_generator)  # 获取音频数据
        # 打包音频
        audio_io = pack_audio(audio_np, sample_rate, req.media_type)
        if not audio_io:
            return JSONResponse(status_code=500, content={"message": "TTS音频打包失败"})
        # 保存音频
        audio_path = save_tts_audio(audio_io, req.save_dir, req.file_name)
        if not audio_path:
            return JSONResponse(status_code=500, content={"message": "TTS音频保存失败"})
        tts_duration = time.time() - tts_start
        logger.info(f"[TTS+ASR请求-{request_id}] TTS完成，耗时: {tts_duration:.2f}秒")
        
        # 3. ASR生成LAB文件
        asr_start = time.time()
        logger.debug(f"[TTS+ASR请求-{request_id}] 开始ASR处理")
        lab_path = generate_lab_from_audio(audio_path)
        if not lab_path:
            return JSONResponse(status_code=500, content={"message": "ASR生成LAB文件失败"})
        asr_duration = time.time() - asr_start
        logger.info(f"[TTS+ASR请求-{request_id}] ASR完成，耗时: {asr_duration:.2f}秒")
        
        # 4. 返回结果
        total_duration = tts_duration + asr_duration
        logger.info(f"[TTS+ASR请求-{request_id}] 处理完成，总耗时: {total_duration:.2f}秒")
        return {
            "status": "success",
            "request_id": request_id,
            "message": f"音频和LAB文件生成成功（总耗时: {total_duration:.2f}秒）",
            "audio_path": audio_path,
            "lab_path": lab_path,
            "text": req.text,
            "tts_duration": f"{tts_duration:.2f}秒",
            "asr_duration": f"{asr_duration:.2f}秒"
        }
    except Exception as e:
        error_msg = str(e)
        logger.error(f"[TTS+ASR请求-{request_id}] 处理失败: {error_msg}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"status": "error", "request_id": request_id, "message": f"处理失败: {error_msg}"}
        )

@app.post("/asr", summary="单独ASR接口")
async def asr_handler(
    audio_file: UploadFile = File(..., description="待处理音频文件（WAV格式）"),
    save_dir: str = Form(..., description="LAB保存目录"),
    file_name: str = Form(..., description="LAB文件名（无后缀）"),
    auth: bool = Depends(verify_access)
):
    """仅处理上传的音频文件，生成对应的LAB文件（用于humanStart预处理）"""
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[ASR单独请求-{request_id}] 收到请求: 音频文件={audio_file.filename}")
    
    # 1. 参数校验
    if not save_dir:
        return JSONResponse(status_code=400, content={"message": "save_dir（保存目录）不能为空"})
    if not file_name.strip():
        return JSONResponse(status_code=400, content={"message": "file_name（文件名）不能为空"})
    if not audio_file.filename.lower().endswith(".wav"):
        return JSONResponse(status_code=400, content={"message": "仅支持WAV格式的音频文件"})
    
    try:
        # 2. 保存上传的音频到临时目录
        temp_audio_dir = normalize_path(os.path.join(save_dir, "temp_audio"))
        os.makedirs(temp_audio_dir, exist_ok=True)
        temp_audio_path = normalize_path(os.path.join(temp_audio_dir, f"temp_{request_id}.wav"))
        with open(temp_audio_path, "wb") as f:
            f.write(await audio_file.read())
        logger.debug(f"[ASR单独请求-{request_id}] 临时音频保存: {temp_audio_path}")
        
        # 3. ASR处理生成LAB
        start_time = time.time()
        generated_lab_path = generate_lab_from_audio(temp_audio_path)
        if not generated_lab_path:
            return JSONResponse(status_code=500, content={"message": "ASR处理失败，未生成LAB文件"})
        
        # 4. 移动LAB文件到目标目录
        target_lab_path = normalize_path(os.path.join(save_dir, f"{file_name}.lab"))
        shutil.move(generated_lab_path, target_lab_path)
        duration = time.time() - start_time
        
        # 5. 清理临时音频
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
            logger.debug(f"[ASR单独请求-{request_id}] 清理临时音频: {temp_audio_path}")
        
        # 6. 读取LAB内容返回（供humanStart直接使用）
        with open(target_lab_path, "r", encoding="utf-8") as f:
            lab_content = f.read()
        
        logger.info(f"[ASR单独请求-{request_id}] 处理完成，耗时: {duration:.2f}秒")
        return {
            "success": True,
            "request_id": request_id,
            "lab_path": target_lab_path,
            "lab_content": lab_content,
            "duration": f"{duration:.2f}秒"
        }
    except Exception as e:
        error_msg = str(e)
        logger.error(f"[ASR单独请求-{request_id}] 处理失败: {error_msg}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"success": False, "request_id": request_id, "message": f"处理失败: {error_msg}"}
        )

# -------------------------- 全局异常处理（统一错误响应） --------------------------
@app.exception_handler(RequestValidationError)
async def handle_validation_error(request: Request, exc: RequestValidationError):
    """处理请求参数验证错误（422）"""
    errors = []
    for err in exc.errors():
        errors.append({
            "field": err["loc"],
            "message": err["msg"],
            "type": err["type"]
        })
    return JSONResponse(
        status_code=422,
        content={"status": "error", "code": 422, "message": "参数验证失败", "errors": errors}
    )

@app.exception_handler(HTTPException)
async def handle_http_error(request: Request, exc: HTTPException):
    """处理HTTP异常（如401/404等）"""
    return JSONResponse(
        status_code=exc.status_code,
        content={"status": "error", "code": exc.status_code, "message": exc.detail}
    )

@app.exception_handler(Exception)
async def handle_unknown_error(request: Request, exc: Exception):
    """处理未知异常（500）"""
    error_msg = str(exc)
    logger.error(f"[未知异常] {error_msg}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"status": "error", "code": 500, "message": f"服务器内部错误: {error_msg}"}
    )

# -------------------------- 程序退出清理（释放资源） --------------------------
def cleanup_on_exit():
    """程序退出时清理临时文件和资源"""
    logger.info("[退出清理] 开始清理资源")
    # 清理TTS临时解密目录
    if os.path.exists(CFG.TEMP_DECRYPT_DIR):
        try:
            shutil.rmtree(CFG.TEMP_DECRYPT_DIR, ignore_errors=True)
            logger.info(f"[退出清理] 清理TTS临时目录: {CFG.TEMP_DECRYPT_DIR}")
        except Exception as e:
            logger.error(f"[退出清理] 清理TTS目录失败: {str(e)}", exc_info=True)
    # 其他资源清理（如模型释放，Python会自动回收）
    logger.info("[退出清理] 清理完成")

# -------------------------- 主程序入口（启动服务） --------------------------
if __name__ == "__main__":
    import uvicorn
    # 1. 初始化服务（失败则退出）
    if not initialize_service():
        logger.critical("服务初始化失败，程序退出")
        sys.exit(1)
    # 2. 启动FastAPI服务（单worker确保线程安全）
    try:
        logger.info(f"[服务启动] 开始监听: http://{CFG.TTS_BIND_ADDR}:{CFG.TTS_PORT}")
        logger.info(f"[服务启动] API文档: http://{CFG.TTS_BIND_ADDR}:{CFG.TTS_PORT}/docs")
        uvicorn.run(
            app=app,
            host=CFG.TTS_BIND_ADDR,
            port=CFG.TTS_PORT,
            workers=1,  # 单进程避免模型重复加载
            log_config=None  # 使用自定义日志配置
        )
    except KeyboardInterrupt:
        logger.info("[服务停止] 收到键盘中断信号")
    except Exception as e:
        logger.critical(f"[服务崩溃] 未知错误: {str(e)}", exc_info=True)
    finally:
        # 3. 退出清理
        cleanup_on_exit()
        sys.exit(0)