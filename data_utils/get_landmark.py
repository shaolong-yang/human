import cv2
import numpy as np
import logging
import onnxruntime as ort

class LandmarkDetector:
    def __init__(self, model_path, input_shape=(1, 3, 640, 640), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.input_shape = input_shape
        self.mean = np.array(mean, dtype=np.float32).reshape(1, 3, 1, 1)
        self.std = np.array(std, dtype=np.float32).reshape(1, 3, 1, 1)
        self.num_scales = 3  # 根据模型输出调整
        
        # 加载ONNX模型
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]

    def preprocess(self, img):
        """图像预处理：resize、归一化"""
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.input_shape[3], self.input_shape[2]))  # (640, 640)
        img = img.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = img.transpose(2, 0, 1)[None, ...]  # 转为[1, 3, H, W]
        return img

    def postprocess(self, outputs, h, w, threshold):
        """后处理：合并多尺度输出并解析结果"""
        all_bboxes = []
        all_scores = []
        all_kps = []
        
        # 遍历所有尺度，合并结果（每个尺度有3个输出：bbox, score, kps）
        for i in range(self.num_scales):
            bboxes = outputs[i*3][0]       # (N, 4)
            scores = outputs[i*3 + 1][0]   # (N, 1)
            kps = outputs[i*3 + 2][0]      # (N, 5, 2)
            
            all_bboxes.append(bboxes)
            all_scores.append(scores)
            all_kps.append(kps)
        
        # 合并所有尺度的结果
        bboxes = np.concatenate(all_bboxes, axis=0)  # (总N, 4)
        scores = np.concatenate(all_scores, axis=0).flatten()  # (总N,)
        kps = np.concatenate(all_kps, axis=0)  # (总N, 5, 2)
        
        # 打印维度信息用于调试
        logging.debug(f"合并后 - bboxes: {bboxes.shape}, scores: {scores.shape}")
        
        # 过滤低置信度结果
        valid_mask = scores >= threshold
        logging.debug(f"valid_mask形状: {valid_mask.shape}")
        
        # 检查维度是否匹配
        if valid_mask.shape[0] != bboxes.shape[0]:
            raise ValueError(f"维度不匹配：bboxes={bboxes.shape[0]}, valid_mask={valid_mask.shape[0]}")
        
        # 应用过滤
        bboxes = bboxes[valid_mask]
        scores = scores[valid_mask]
        kps = kps[valid_mask]
        
        # 按置信度降序排序
        sorted_idx = np.argsort(scores)[::-1]
        bboxes = bboxes[sorted_idx]
        scores = scores[sorted_idx]
        kps = kps[sorted_idx]
        
        # 坐标转换为原图尺寸
        bboxes = bboxes * np.array([w, h, w, h])  # 边界框
        kps = kps * np.array([w, h])  # 关键点
        
        return bboxes.astype(np.int32), scores, kps.astype(np.int32)

    def detect(self, img, threshold=0.5):
        """检测人脸并返回边界框和关键点"""
        img = img.copy()
        h, w = img.shape[:2]
        img = self.preprocess(img)
        
        # 推理
        outputs = self.session.run(self.output_names, {self.input_name: img})
        
        # 解析结果
        bboxes, scores, kps = self.postprocess(outputs, h, w, threshold)
        return bboxes, scores, kps

    def detect_single(self, img_path):
        """检测单张图像的人脸关键点（取置信度最高的）"""
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"无法读取图像：{img_path}")
        
        bboxes, scores, kps = self.detect(img)
        if len(bboxes) == 0:
            raise RuntimeError(f"未检测到人脸：{img_path}")
        
        # 取置信度最高的人脸
        bbox = bboxes[0]
        kp = kps[0]
        
        # 计算裁剪区域（扩大边界）
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        x1 = max(0, x1 - w // 4)
        y1 = max(0, y1 - h // 4)
        x2 = min(img.shape[1], x2 + w // 4)
        y2 = min(img.shape[0], y2 + h // 4)
        
        return kp, x1, y1, x2, y2