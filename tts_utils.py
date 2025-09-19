# human/tts_utils.py
import subprocess
import os
import logging

# --- 配置 ---
# GPT-SoVITS 项目路径和脚本
GPT_SOVITS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'GPT_SoVITS')
# 假设 GPT-SoVITS 有一个可以直接调用的推理脚本
TTS_SCRIPT_PATH = os.path.join(GPT_SOVITS_DIR, 'your_actual_tts_inference_script.py') # 需要替换成真实脚本
# --- 配置结束 ---


def synthesize_speech(text, output_path, reference_audio_path=None, reference_text=None):
    """
    调用 GPT-SoVITS 合成语音。
    注意：此函数需要根据 GPT-SoVITS 的实际 API 或 CLI 进行调整。
    这是一个通用的 subprocess 调用示例。

    :param text: 要合成的文本。
    :param output_path: 输出音频文件路径 (.wav)。
    :param reference_audio_path: (可选) 参考音频路径，用于音色克隆。
    :param reference_text: (可选) 参考音频对应的文本。
    :return: (success: bool, message: str)
    """
    try:
        if not os.path.exists(TTS_SCRIPT_PATH):
            raise FileNotFoundError(f"TTS script not found: {TTS_SCRIPT_PATH}")

        # 构建命令行参数
        # 参数需要根据 GPT-SoVITS 的实际脚本调整
        cmd = [
            'python', TTS_SCRIPT_PATH,
            '--text', text,
            '--output', output_path
        ]
        if reference_audio_path:
            cmd.extend(['--reference_audio', reference_audio_path])
        if reference_text:
            cmd.extend(['--reference_text', reference_text])

        # 执行命令
        logging.info(f"Running TTS command: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=GPT_SOVITS_DIR, capture_output=True, text=True)

        if result.returncode == 0:
            logging.info("TTS synthesis successful.")
            return True, "TTS synthesis successful."
        else:
            error_msg = f"TTS failed with return code {result.returncode}: {result.stderr}"
            logging.error(error_msg)
            return False, error_msg

    except Exception as e:
        error_msg = f"Exception during TTS synthesis: {str(e)}"
        logging.error(error_msg)
        return False, error_msg

# --- 如果 GPT-SoVITS 提供了 Python API，可以直接调用 ---
# from GPT_SoVITS.some_module import tts_function
# def synthesize_speech(text, output_path, ...):
#     try:
#         # 调用 API
#         # audio_data = tts_function(text, ...)
#         # 保存 audio_data 到 output_path
#         # ...
#         pass
#     except Exception as e:
#         ...
# --- API 调用方式结束 ---
