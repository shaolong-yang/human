import numpy as np
import torch

class Feature_Pipeline:
    def __init__(self, engine_config, asr_mode="hubert"):
        self.configs = engine_config
        self.asr_mode = asr_mode
        self.remained_wav_ = b''
        self._waveform = b''
        self.exist_endpoint = False
        
        # 初始化特征参数
        if asr_mode == "ttsasr":
            self.num_mel_bins = self.configs.get('ttsasr_conf', {}).get('num_mel_bins', 80)
        else:
            self.num_mel_bins = self.configs['data_conf']['fbank_conf']['num_mel_bins']

    def AcceptWaveform(self, audio):
        """接收音频数据并检测端点"""
        self._waveform += audio
        # 简单端点检测：超过1秒（16000采样率）视为有效
        self.exist_endpoint = len(self._waveform) > 16000  
        return self.exist_endpoint

    def get_waveform_len(self):
        """获取当前缓存的音频长度"""
        return len(self._waveform)

    def Reset(self):
        """重置音频缓存"""
        self.remained_wav_ = b''
        self._waveform = b''

    def extract_features(self):
        """提取音频特征（根据模式适配）"""
        if self.asr_mode == "ttsasr":
            # TTSASR模式特征提取（示例）
            feat = np.random.randn(1, self.num_mel_bins, 100)  # 替换为实际逻辑
        else:
            # Hubert/Wenet模式特征提取
            feat = np.random.randn(1, self.num_mel_bins, 100)  # 替换为实际逻辑
        return torch.from_numpy(feat).float()