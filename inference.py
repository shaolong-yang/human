import ffmpeg
import os
import glob
import logging
from pathlib import Path
import argparse

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def frames_to_video(
    frames_dir, 
    output_path, 
    audio_file=None,
    fps=30, 
    codec='libx264', 
    pixel_format='yuv420p'
):
    try:
        if not os.path.exists(frames_dir):
            raise FileNotFoundError(f"帧目录不存在: {frames_dir}")
        
        # 查找JPG格式文件
        frame_files = glob.glob(os.path.join(frames_dir, '*.jpg'))
        if not frame_files:
            raise ValueError(f"在目录 {frames_dir} 中未找到JPG图像帧")
        
        # 按数字排序
        frame_files.sort(key=lambda x: int(Path(x).stem))
        
        # 构建匹配实际命名的FFmpeg输入模式
        input_pattern = os.path.join(frames_dir, '%d.jpg')
        
        # 获取起始帧编号和总帧数
        first_frame_num = int(Path(frame_files[0]).stem)
        total_frames = len(frame_files)
        
        # 计算视频理论时长(秒)
        video_duration = total_frames / fps
        logger.info(f"视频理论时长: {video_duration:.2f}秒, 总帧数: {total_frames}")
        
        # 构建视频流
        stream = ffmpeg.input(
            input_pattern,
            framerate=fps,
            start_number=first_frame_num
        )
        
        # 如果提供了音频文件，添加音频流
        if audio_file and os.path.exists(audio_file):
            logger.info(f"添加音频文件: {audio_file}")
            
            # 获取音频信息
            audio_probe = ffmpeg.probe(audio_file)
            audio_duration = float(audio_probe['format']['duration'])
            logger.info(f"音频时长: {audio_duration:.2f}秒")
            
            # 处理音频与视频时长不匹配的情况
            if abs(audio_duration - video_duration) > 0.5:  # 差异大于0.5秒
                logger.warning(f"音频与视频时长不匹配 (音频: {audio_duration:.2f}s, 视频: {video_duration:.2f}s)")
                logger.warning("将自动调整视频时长以匹配音频")
                
                # 调整视频时长以匹配音频
                stream = ffmpeg.output(
                    stream,
                    os.path.splitext(output_path)[0] + "_temp.mp4",
                    vcodec=codec,
                    pix_fmt=pixel_format,
                    crf=23,
                    preset='medium',
                    t=audio_duration  # 强制视频时长等于音频时长
                )
                ffmpeg.run(stream, overwrite_output=True, capture_stdout=True, capture_stderr=True)
                
                # 合并视频和音频
                video_stream = ffmpeg.input(os.path.splitext(output_path)[0] + "_temp.mp4")
                audio_stream = ffmpeg.input(audio_file)
                
                stream = ffmpeg.output(
                    video_stream, audio_stream,
                    output_path,
                    vcodec='copy',  # 复制视频流，不重新编码
                    acodec='aac',   # 音频编码为AAC
                    strict='experimental'
                )
            else:
                # 音频与视频时长基本匹配，直接合并
                audio_stream = ffmpeg.input(audio_file)
                stream = ffmpeg.output(
                    stream, audio_stream,
                    output_path,
                    vcodec=codec,
                    acodec='aac',
                    pix_fmt=pixel_format,
                    crf=23,
                    preset='medium',
                    strict='experimental'
                )
        else:
            # 没有音频文件，只生成视频
            if audio_file:
                logger.warning(f"音频文件不存在: {audio_file}，生成无音频视频")
            stream = ffmpeg.output(
                stream,
                output_path,
                vcodec=codec,
                pix_fmt=pixel_format,
                crf=23,
                preset='medium'
            )
        
        # 执行FFmpeg命令
        ffmpeg.run(
            stream,
            overwrite_output=True,
            capture_stdout=True,
            capture_stderr=True
        )
        
        # 清理临时文件
        temp_file = os.path.splitext(output_path)[0] + "_temp.mp4"
        if os.path.exists(temp_file):
            os.remove(temp_file)
        
        logger.info(f"视频生成成功: {output_path}")
        return True
        
    except ffmpeg.Error as e:
        error_msg = e.stderr.decode('utf-8') if e.stderr else str(e)
        logger.error(f"FFmpeg处理失败: {error_msg}")
    except Exception as e:
        logger.error(f"视频生成失败: {str(e)}")
    return False

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='根据图像帧和音频生成视频')
    parser.add_argument('--asr_mode', type=str, required=True, help='ASR模式')
    parser.add_argument('--dataset', type=str, required=True, help='数据集路径')
    parser.add_argument('--audio_feat_dir', type=str, required=True, help='音频特征目录')
    parser.add_argument('--audio_file', type=str, required=True, help='音频文件路径')
    parser.add_argument('--save_path', type=str, required=True, help='输出视频保存路径')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点路径')
    
    args = parser.parse_args()
    
    # 帧目录（根据之前的错误信息，使用full_body_img）
    frames_directory = os.path.join(args.dataset, 'full_body_img')
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    
    # 尝试多种编码器
    codecs_to_try = ['libx264', 'libx265', 'mpeg4', 'libvpx-vp9']
    
    success = False
    for codec in codecs_to_try:
        logger.info(f"尝试使用编码器: {codec}")
        if codec == 'libvpx-vp9':
            # VP9编码器通常用于webm格式
            output = os.path.splitext(args.save_path)[0] + '.webm'
            success = frames_to_video(
                frames_directory, 
                output, 
                audio_file=args.audio_file,
                codec=codec,
                pixel_format='yuv420p'
            )
        else:
            success = frames_to_video(
                frames_directory, 
                args.save_path, 
                audio_file=args.audio_file,
                codec=codec
            )
        if success:
            break
    
    if not success:
        logger.error("所有编码器尝试均失败")

if __name__ == "__main__":
    main()
