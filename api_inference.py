# human/api_inference.py
import os
import uuid
import subprocess
import logging
from flask import Flask, request, jsonify, send_file
# 假设项目结构中有这些模块
# from human.tts_utils import synthesize_speech # 需要根据 GPT-SoVITS 实际情况实现
# from human.inference_core import generate_video # 需要实现核心推理逻辑
import human.tts_utils as tts_utils
import human.inference_core as inference_core

# --- 配置 ---
# 基础路径，根据你的项目结构调整
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # human/
TEMP_DIR = os.path.join(BASE_DIR, 'temp')
# --- 配置结束 ---


def run_inference_api():
    """
    执行推理的核心逻辑。
    1. 接收文本和任务ID。
    2. 调用TTS生成语音。
    3. 提取语音特征。
    4. 使用训练模型生成视频。
    5. 返回视频路径。
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid JSON"}), 400

    text = data.get('text')
    task_id = data.get('task_id') # 用于定位预处理数据和模型

    if not text:
        return jsonify({"error": "Missing 'text' in request"}), 400
    if not task_id:
         # 如果没有提供 task_id，可以生成一个临时的，或者要求必须提供
         # 这里假设必须提供，因为需要知道用哪个模型和数据
         return jsonify({"error": "Missing 'task_id' in request"}), 400

    # --- 确定路径 ---
    task_dir = os.path.join(TEMP_DIR, task_id)
    if not os.path.exists(task_dir):
        return jsonify({"error": f"Task directory not found: {task_dir}"}), 404

    model_path = os.path.join(task_dir, 'checkpoint', 'unet_ttsasr_epoch_100.pth')
    if not os.path.exists(model_path):
        return jsonify({"error": f"Model file not found: {model_path}"}), 404

    # 预处理数据路径 (假设 preprocess 已经生成了这些)
    # 这些路径需要与 preprocess ctic 的输出一致
    dataset_dir = os.path.join(task_dir, 'preprocessed_data') # 示例路径
    # 需要确认 preprocess 生成的确切文件名和结构
    # 例如，可能需要 reference_img_path, coeff_initial_path 等

    # --- 创建本次推理的临时输出目录 ---
    inference_id = str(uuid.uuid4())
    inference_output_dir = os.path.join(task_dir, 'inference_outputs', inference_id)
    os.makedirs(inference_output_dir, exist_ok=True)
    audio_output_path = os.path.join(inference_output_dir, "generated_audio.wav")
    audio_feat_path = os.path.join(inference_output_dir, "generated_audio.npy") # 特征文件名
    video_output_path = os.path.join(inference_output_dir, "generated_video.mp4")

    try:
        # --- 1. 调用 TTS 生成语音 ---
        logging.info(f"Starting TTS for text: {text}")
        # 这里调用实际的 TTS 函数
        # tts_utils.synthesize_speech(text, audio_output_path) # 需要实现
        # 示例：假设 GPT-SoVITS 有一个命令行接口
        # 注意：参数需要根据 GPT-SoVITS 的实际 CLI 或 API 调整
        # 这只是一个示例命令，你需要替换为实际的调用方式
        # gpt_sovits_script = os.path.join(BASE_DIR, 'GPT_SoVITS', 'your_tts_script.py')
        # cmd = ['python', gpt_sovits_script, '--text', text, '--output', audio_output_path]
        # result = subprocess.run(cmd, capture_output=True, text=True)
        # if result.returncode != 0:
        #     logging.error(f"TTS failed: {result.stderr}")
        #     return jsonify({"error": f"TTS failed: {result.stderr}"}), 500
        #
        # 为了演示，我们创建一个静音的 wav 文件 (需要安装 soundfile: pip install soundfile)
        import soundfile as sf
        import numpy as np
        # 生成 3 秒 16kHz 的静音 (实际应调用 TTS)
        silence = np.zeros(int(3 * 16000))
        sf.write(audio_output_path, silence, 16000)
        logging.info(f"TTS completed, audio saved to: {audio_output_path}")


        # --- 2. 提取音频特征 ---
        # 这一步非常关键，必须使用与训练时完全相同的特征提取方法和参数
        logging.info(f"Extracting audio features for: {audio_output_path}")
        # 示例：假设使用 HuBERT，调用 data_utils/hubert.py
        # 实际调用需要根据你的项目结构调整
        # hubert_script = os.path.join(BASE_DIR, 'data_utils', 'hubert.py') # 或 wenet_infer.py
        # cmd = ['python', hubert_script, '--wav', audio_output_path]
        # result = subprocess.run(cmd, capture_output=True, text=True, cwd=BASE_DIR)
        # if result.returncode != 0:
        #     logging.error(f"Feature extraction failed: {result.stderr}")
        #     return jsonify({"error": f"Feature extraction failed: {result.stderr}"}), 500
        #
        # 为了演示，我们创建一个空的 npy 文件 (实际应调用特征提取)
        np.save(audio_feat_path, np.array([])) # 实际内容应由特征提取脚本生成
        logging.info(f"Audio features extracted, saved to: {audio_feat_path}")

        # --- 3. 调用视频生成 ---
        logging.info(f"Starting video generation...")
        # 调用核心推理函数
        # 需要传递模型路径、音频特征路径、输出视频路径以及其他必要参数
        # 这些参数取决于 inference_core.py 的具体实现
        # 示例参数 (需要根据实际情况调整):
        args_for_inference = {
            'asr': 'hubert', # 或 'wenet'，与训练时一致
            'dataset_dir': dataset_dir, # 预处理数据目录
            'audio_feat': audio_feat_path,
            'save_path': video_output_path,
            'checkpoint': model_path,
            # 可能还需要其他参数，如参考图像路径等
            # 'reference_img': os.path.join(dataset_dir, 'xxx.jpg'),
            # 'coeff_initial': os.path.join(dataset_dir, 'xxx.npy'),
        }
        success, message = inference_core.generate_video(args_for_inference)
        if not success:
            logging.error(f"Video generation failed: {message}")
            return jsonify({"error": f"Video generation failed: {message}"}), 500

        logging.info(f"Video generated successfully: {video_output_path}")

        # --- 4. 返回结果 ---
        # 可以返回视频文件路径，或者直接返回视频文件流
        # 返回路径示例:
        # relative_video_path = os.path.relpath(video_output_path, BASE_DIR)
        # return jsonify({"video_url": f"/{relative_video_path}", "message": "Video generated successfully"}), 200
        # 直接返回文件流示例:
        return send_file(video_output_path, as_attachment=True, download_name="generated_video.mp4")


    except Exception as e:
        logging.error(f"An error occurred during inference: {e}")
        return jsonify({"error": f"Internal server error during inference: {str(e)}"}), 500

    finally:
        # 可选：清理临时文件 (或保留一段时间供下载)
        # 注意：不要删除模型文件或预处理数据
        # import shutil
        # shutil.rmtree(inference_output_dir, ignore_errors=True) # 谨慎使用
