import os
import cv2
import torch
import random
import numpy as np

from torch.utils.data import Dataset

class MyDataset(Dataset):
    
    def __init__(self, img_dir, mode):
        self.img_path_list = []
        self.lms_path_list = []
        self.mode = mode  # 支持 wenet, hubert, ttsasr
        self.img_dir = img_dir
        
        # 修正音频特征路径加载逻辑
        if mode == "ttsasr":
            self.audio_feat_path = os.path.join(img_dir, "aud_ttsasr.npy")
        elif mode == "hubert":
            self.audio_feat_path = os.path.join(img_dir, "aud_hu.npy")
        elif mode == "wenet":
            self.audio_feat_path = os.path.join(img_dir, "aud_we.npy")
        else:
            raise ValueError(f"不支持的模式: {mode}")
            
        # 加载视频帧和关键点目录
        full_body_dir = os.path.join(img_dir, "full_body_img")
        landmarks_dir = os.path.join(img_dir, "landmarks")
        if not os.path.exists(full_body_dir) or not os.path.exists(landmarks_dir):
            raise FileNotFoundError("图像或关键点目录不存在")
        
        img_files = sorted(
            [f for f in os.listdir(full_body_dir) if f.endswith(".jpg")],
            key=lambda x: int(os.path.splitext(x)[0])
        )
        lms_files = sorted(
            [f for f in os.listdir(landmarks_dir) if f.endswith(".lms")],
            key=lambda x: int(os.path.splitext(x)[0])
        )
        if len(img_files) != len(lms_files):
            raise ValueError(f"图像和关键点数量不匹配（{len(img_files)} vs {len(lms_files)}）")
        
        for img_file, lms_file in zip(img_files, lms_files):
            self.img_path_list.append(os.path.join(full_body_dir, img_file))
            self.lms_path_list.append(os.path.join(landmarks_dir, lms_file))
        
        # 加载音频特征并验证
        if not os.path.exists(self.audio_feat_path):
            raise FileNotFoundError(f"音频特征文件不存在: {self.audio_feat_path}")
        self.audio_feats = np.load(self.audio_feat_path).astype(np.float32)
        
        # 验证音频特征数量与视频帧数量一致
        self.num_frames = len(self.img_path_list)
        if self.audio_feats.shape[0] != self.num_frames:
            raise ValueError(
                f"音频特征数量与视频帧数量不匹配 "
                f"（音频: {self.audio_feats.shape[0]}, 视频: {self.num_frames}）"
            )
        
        # 验证音频特征通道数
        if self.mode == "ttsasr":
            # 针对ttsasr模式，校验为4D形状 (帧数, 16, 12, 18)
            if len(self.audio_feats.shape) != 4 or self.audio_feats.shape[1:] != (16, 12, 18):
                raise ValueError(f"ttsasr模式下，音频特征需为4D数组 (帧数, 16, 12, 18)，当前形状: {self.audio_feats.shape}")
        else:
            # 其他模式保持原2D校验
            if len(self.audio_feats.shape) != 2 or self.audio_feats.shape[1] != 16:
                raise ValueError(f"音频特征需为2D数组 (帧数, 16)，当前形状: {self.audio_feats.shape}")
    
    def __len__(self):
        return self.num_frames
    
    def process_img(self, img, lms_path, img_ex, lms_path_ex):
        # 统一图像尺寸为256x256（适配模型）
        target_size = (256, 256)
        img = cv2.resize(img, target_size)
        img_ex = cv2.resize(img_ex, target_size)
        
        # BGR转RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_ex = cv2.cvtColor(img_ex, cv2.COLOR_BGR2RGB)
        
        # 转换为Tensor并归一化
        img_concat_T = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0  # (3, 256, 256)
        img_real_T = torch.from_numpy(img_ex).permute(2, 0, 1).float() / 255.0  # (3, 256, 256)
        return img_concat_T, img_real_T
    
    def __getitem__(self, idx):
        # 读取图像
        img = cv2.imread(self.img_path_list[idx])
        lms_path = self.lms_path_list[idx]
        ex_int = random.randint(0, self.num_frames - 1)
        img_ex = cv2.imread(self.img_path_list[ex_int])
        lms_path_ex = self.lms_path_list[ex_int]
        
        img_concat_T, img_real_T = self.process_img(img, lms_path, img_ex, lms_path_ex)
        
        # 处理音频特征（根据模式调整维度）
        audio_feat = torch.from_numpy(self.audio_feats[idx])
        
        # 对于ttsasr模式，已经是(16, 12, 18)，不需要额外调整
        # 对于其他模式，需要添加空间维度
        if self.mode != "ttsasr":
            audio_feat = audio_feat.view(1, 16, 12, 18)
        
        return img_concat_T, img_real_T, audio_feat