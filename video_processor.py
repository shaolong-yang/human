# video_processor.py
import cv2
import ffmpeg
import librosa
import numpy as np
from typing import Tuple, Dict
# video_processor.py 新增内容
import requests

def combine_audio_video(video_path: str, audio_path: str, output_path: str):
    """调用FFmpeg合并音频和视频"""
    import subprocess
    cmd = f'ffmpeg -i "{video_path}" -i "{audio_path}" -c:v copy -c:a aac -strict experimental "{output_path}"'
    subprocess.run(cmd, shell=True, check=True)

def get_tts_audio(text: str, ref_audio: str, save_dir: str, file_name: str) -> tuple:
    """调用本地TTS+ASR服务生成音频和lab文件"""
    response = requests.post(
        "http://127.0.0.1:9880/tts_asr",
        json={
            "text": text,
            "ref_audio_path": ref_audio,
            "ref_audio_text": "参考音频对应的文本",
            "save_dir": save_dir,
            "file_name": file_name
        }
    )
    result = response.json()
    return result["audio_path"], result["lab_path"]
    
class VideoProcessor:
    def __init__(self):
        # 加载人脸检测器（可替换为更轻量的模型）
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    def convert_fps(self, input_path: str, output_path: str, target_fps: int) -> bool:
        """转换视频帧率"""
        try:
            ffmpeg.input(input_path).output(
                output_path,
                r=target_fps,  # 目标帧率
                vcodec="libx264",
                acodec="aac",
                strict="experimental"
            ).run(overwrite_output=True, quiet=True)
            return True
        except Exception as e:
            print(f"帧率转换失败: {e}")
            return False

    def validate_video(self, video_path: str, asr_mode: str) -> Dict[str, str]:
        """校验视频合法性"""
        result = {
            "valid": True,
            "errors": []
        }
        target_fps = 20 if asr_mode == "wenet" else 25

        # 1. 校验帧率
        cap = cv2.VideoCapture(video_path)
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        if not np.isclose(actual_fps, target_fps, atol=0.5):
            result["valid"] = False
            result["errors"].append(f"帧率错误（需{target_fps}fps，实际{actual_fps:.1f}fps）")

        # 2. 校验每帧人脸完整性（随机抽样100帧）
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_indices = np.random.choice(total_frames, min(100, total_frames), replace=False)
        for idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                result["valid"] = False
                result["errors"].append(f"帧{idx}读取失败")
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            if len(faces) == 0:
                result["valid"] = False
                result["errors"].append(f"帧{idx}未检测到人脸")
        cap.release()

        # 3. 校验音频（采样率、杂音）
        try:
            y, sr = librosa.load(video_path, sr=16000)
            # 检查采样率（强制转换后应为16000）
            if sr != 16000:
                result["valid"] = False
                result["errors"].append(f"音频采样率错误（需16000Hz，实际{sr}Hz）")
            # 简单杂音检测（能量阈值）
            rms = librosa.feature.rms(y=y).mean()
            if rms < 0.001:  # 能量过低可能为静音或杂音
                result["valid"] = False
                result["errors"].append("音频能量过低，可能存在杂音或静音")
        except Exception as e:
            result["valid"] = False
            result["errors"].append(f"音频校验失败: {str(e)}")

        return result

    def process_and_validate(self, input_path: str, asr_mode: str, output_dir: str) -> Tuple[bool, str, Dict]:
        """一站式处理：转换帧率 + 校验"""
        target_fps = 20 if asr_mode == "wenet" else 25
        output_path = f"{output_dir}/processed_{target_fps}fps.mp4"
        
        # 转换帧率
        if not self.convert_fps(input_path, output_path, target_fps):
            return False, "", {"valid": False, "errors": ["帧率转换失败"]}
        
        # 校验转换后的视频
        validation = self.validate_video(output_path, asr_mode)
        if validation["valid"]:
            return True, output_path, validation
        else:
            return False, output_path, validation