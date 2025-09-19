import cv2
import numpy as np
import time

class FaceDetector:
    def __init__(self, model_path, config_path, inpWidth=640, inpHeight=640, confThreshold=0.5, nmsThreshold=0.3):
        self.inpWidth = inpWidth
        self.inpHeight = inpHeight
        self.confThreshold = confThreshold
        self.nmsThreshold = nmsThreshold
        self.fmc = 3  # 特征图数量
        self._feat_stride_fpn = [8, 16, 32]  # 特征步长
        self._num_anchors = 2  # 每个位置的锚点数

        # 加载模型
        self.net = cv2.dnn.readNet(model_path, config_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def resize_image(self, srcimg):
        """调整图像尺寸并填充"""
        h, w = srcimg.shape[:2]
        scale = min(self.inpHeight / h, self.inpWidth / w)
        newh, neww = int(h * scale), int(w * scale)
        padh = (self.inpHeight - newh) // 2
        padw = (self.inpWidth - neww) // 2
        img = cv2.resize(srcimg, (neww, newh))
        img = cv2.copyMakeBorder(img, padh, self.inpHeight - newh - padh,
                                 padw, self.inpWidth - neww - padw,
                                 cv2.BORDER_CONSTANT, value=(0, 0, 0))
        return img, newh, neww, padh, padw

    def distance2bbox(self, points, distance, max_shape=None):
        """将中心点和偏移量转换为边界框"""
        x1 = points[:, 0] - distance[:, 0]
        y1 = points[:, 1] - distance[:, 1]
        x2 = points[:, 0] + distance[:, 2]
        y2 = points[:, 1] + distance[:, 3]
        if max_shape is not None:
            x1 = np.clip(x1, 0, max_shape[1])
            y1 = np.clip(y1, 0, max_shape[0])
            x2 = np.clip(x2, 0, max_shape[1])
            y2 = np.clip(y2, 0, max_shape[0])
        return np.stack([x1, y1, x2, y2], axis=-1)

    def distance2kps(self, points, distance, max_shape=None):
        """将中心点和偏移量转换为关键点"""
        kps = []
        for i in range(0, distance.shape[1], 2):
            px = points[:, 0] + distance[:, i]
            py = points[:, 1] + distance[:, i + 1]
            if max_shape is not None:
                px = np.clip(px, 0, max_shape[1])
                py = np.clip(py, 0, max_shape[0])
            kps.append(px)
            kps.append(py)
        return np.stack(kps, axis=-1)

    def detect(self, srcimg):
        t1 = time.time()
        img, newh, neww, padh, padw = self.resize_image(srcimg)
        blob = cv2.dnn.blobFromImage(img, 1.0 / 128, (self.inpWidth, self.inpHeight), 
                                    (127.5, 127.5, 127.5), swapRB=True)
        self.net.setInput(blob)

        # 前向传播获取输出
        outs = self.net.forward(self.net.getUnconnectedOutLayersNames())
        scores_list, bboxes_list, kpss_list = [], [], []
        
        for idx, stride in enumerate(self._feat_stride_fpn):
            scores = outs[idx][0]
            bbox_preds = outs[idx + self.fmc * 1][0] * stride
            kps_preds = outs[idx + self.fmc * 2][0] * stride
            
            height = blob.shape[2] // stride
            width = blob.shape[3] // stride
            anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
            anchor_centers = (anchor_centers * stride).reshape((-1, 2))
            
            if self._num_anchors > 1:
                anchor_centers = np.stack([anchor_centers] * self._num_anchors, axis=1).reshape((-1, 2))

            # 过滤低置信度结果
            pos_inds = np.where(scores >= self.confThreshold)[0]
            bboxes = self.distance2bbox(anchor_centers, bbox_preds)
            pos_scores = scores[pos_inds]
            pos_bboxes = bboxes[pos_inds]
            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)

            # 处理关键点
            kpss = self.distance2kps(anchor_centers, kps_preds)
            kpss = kpss.reshape((kpss.shape[0], -1, 2))
            pos_kpss = kpss[pos_inds]
            kpss_list.append(pos_kpss)

        # 合并所有尺度结果
        scores = np.vstack(scores_list).ravel()
        bboxes = np.vstack(bboxes_list)
        kpss = np.vstack(kpss_list)
        bboxes[:, 2:4] = bboxes[:, 2:4] - bboxes[:, 0:2]  # 转换为宽高格式

        # 映射回原图尺寸
        ratioh, ratiow = srcimg.shape[0] / newh, srcimg.shape[1] / neww
        bboxes[:, 0] = (bboxes[:, 0] - padw) * ratiow
        bboxes[:, 1] = (bboxes[:, 1] - padh) * ratioh
        bboxes[:, 2] = bboxes[:, 2] * ratiow
        bboxes[:, 3] = bboxes[:, 3] * ratioh
        kpss[:, :, 0] = (kpss[:, :, 0] - padw) * ratiow
        kpss[:, :, 1] = (kpss[:, :, 1] - padh) * ratioh

        # NMS过滤
        indices = cv2.dnn.NMSBoxes(bboxes.tolist(), scores.tolist(), self.confThreshold, self.nmsThreshold)
        return bboxes, indices, kpss