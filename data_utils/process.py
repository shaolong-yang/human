import argparse
import os
import cv2
import numpy as np
from typing import Tuple
from get_landmark import LandmarkDetector  # 导入实际的关键点检测器

def parse_args():
    parser = argparse.ArgumentParser(description="音频-视频同步预处理（确保时间维度对齐）")
    parser.add_argument("--video-path", required=True, help="输入视频路径")
    parser.add_argument("--img-out-dir", required=True, help="图像输出目录（full_body_img）")
    parser.add_argument("--landmarks-out-dir", required=True, help="关键点输出目录")
    parser.add_argument("--audio-out-dir", required=True, help="音频特征输出目录")
    parser.add_argument("--mode", choices=["hubert", "wenet", "ttsasr"], required=True, help="ASR模式（支持hubert/wenet/ttsasr）")
    parser.add_argument("--sample-rate", type=int, default=25, help="视频抽帧频率（每秒帧数）")
    parser.add_argument("--audio-channels", type=int, default=16, help="音频特征通道数（需与模型匹配）")
    parser.add_argument("--audio-height", type=int, default=12, help="音频特征高度（默认12）")
    parser.add_argument("--audio-width", type=int, default=18, help="音频特征宽度（默认18）")
    parser.add_argument("--landmark-model", required=True, help="关键点检测模型路径（ONNX）")
    return parser.parse_args()

def detect_landmarks(frame: np.ndarray, detector: LandmarkDetector) -> Tuple[bool, str]:
    """使用实际模型检测关键点（替换随机生成逻辑）"""
    try:
        bboxes, scores, kps = detector.detect(frame)
        if len(kps) == 0:
            return False, ""
        
        # 取第一个人脸的68个关键点（根据实际模型输出调整）
        landmarks = kps[0].reshape(-1, 2)[:68]  # 假设模型输出超过68点，取前68个
        landmarks_str = "\n".join([f"{x:.2f} {y:.2f}" for x, y in landmarks])
        return True, landmarks_str
    except Exception as e:
        print(f"关键点检测失败: {e}")
        return False, ""

def extract_synced_audio_features(video_path, audio_out_dir, asr_mode, target_num_frames, target_shape):
    """提取与视频帧同步的音频特征"""
    os.makedirs(audio_out_dir, exist_ok=True)
    # 实际应用中需替换为真实的音频特征提取逻辑
    if asr_mode == "ttsasr":
        # TTSASR模式特征提取（示例）
        audio_feat = np.random.rand(*target_shape).astype(np.float32)
    else:
        # Hubert/Wenet模式特征提取
        audio_feat = np.random.rand(*target_shape).astype(np.float32)
    
    save_path = os.path.join(audio_out_dir, "inference_feat.npy")
    np.save(save_path, audio_feat)
    return save_path

def process_video(args):
    os.makedirs(args.img_out_dir, exist_ok=True)
    os.makedirs(args.landmarks_out_dir, exist_ok=True)
    os.makedirs(args.audio_out_dir, exist_ok=True)

    # 初始化关键点检测器
    landmark_detector = LandmarkDetector(model_path=args.landmark_model)

    # 读取视频信息
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        raise IOError(f"无法打开视频文件: {args.video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / fps
    frame_interval = max(1, int(fps / args.sample_rate))
    print(
        f"视频信息: 时长={video_duration:.1f}秒, FPS={fps:.1f}, "
        f"总帧数={total_frames}, 抽帧间隔={frame_interval}（目标{args.sample_rate}fps）"
    )

    # 步骤1：提取视频帧和关键点
    frame_count = 0
    save_count = 0
    frame_timestamps = []

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                # 保存图像
                img_path = os.path.join(args.img_out_dir, f"{save_count}.jpg")
                if not cv2.imwrite(img_path, frame):
                    raise IOError(f"图像保存失败: {img_path}")
                
                # 检测关键点
                success, landmarks_str = detect_landmarks(frame, landmark_detector)
                if not success:
                    raise RuntimeError(f"第{save_count}帧关键点检测失败")

                lms_path = os.path.join(args.landmarks_out_dir, f"{save_count}.lms")
                with open(lms_path, "w", encoding="utf-8") as f:
                    f.write(landmarks_str)
                
                frame_timestamps.append(frame_count / fps)
                save_count += 1
                if save_count % 100 == 0:
                    print(f"已处理 {save_count} 帧")

            frame_count += 1
    finally:
        cap.release()

    print(f"实际保存视频帧数量: {save_count}（时间戳范围: {frame_timestamps[0]:.1f}~{frame_timestamps[-1]:.1f}秒）")
    if save_count == 0:
        raise RuntimeError("视频抽帧失败，未提取到任何帧")

    # 步骤2：提取音频特征（目标形状包含帧数）
    target_audio_shape = (save_count, args.audio_channels, args.audio_height, args.audio_width)
    audio_feat_path = extract_synced_audio_features(
        video_path=args.video_path,
        audio_out_dir=args.audio_out_dir,
        asr_mode=args.mode,
        target_num_frames=save_count,
        target_shape=target_audio_shape
    )

    # 验证音频特征形状
    audio_feats = np.load(audio_feat_path)
    if audio_feats.shape != target_audio_shape:
        raise RuntimeError(
            f"音频特征形状不符合要求（实际: {audio_feats.shape}, 预期: {target_audio_shape}）"
        )

    print(f"预处理完成：视频帧={save_count}，音频特征形状={audio_feats.shape}")

if __name__ == "__main__":
    args = parse_args()
    process_video(args)