import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx
from .base_module import (Module, MobileOneBlock, GhostOneBottleneck, 
                         Conv_Block, AvgPool2d)

class AuxiliaryNet(Module):
    def __init__(self, width_factor=1):
        super(AuxiliaryNet, self).__init__()
        self.conv1 = Conv_Block(int(64 * width_factor), int(64 * width_factor), 1, 1, 0)
        self.conv2 = Conv_Block(int(80 * width_factor), int(64 * width_factor), 1, 1, 0)
        self.conv3 = Conv_Block(int(96 * width_factor), int(64 * width_factor), 1, 1, 0)
        self.conv4 = Conv_Block(int(144 * width_factor), int(64 * width_factor), 1, 1, 0)

        self.merge1 = Conv_Block(int(64 * width_factor), int(64 * width_factor), 3, 1, 1)
        self.merge2 = Conv_Block(int(64 * width_factor), int(64 * width_factor), 3, 1, 1)
        self.merge3 = Conv_Block(int(64 * width_factor), int(64 * width_factor), 3, 1, 1)

        self.conv_out = Conv_Block(int(64 * width_factor), 1, 1, 1, 0)

    def forward(self, out1, out2, out3, out4):
        output1 = self.conv1(out1)
        output2 = self.conv2(out2)
        output3 = self.conv3(out3)
        output4 = self.conv4(out4)

        up4 = F.interpolate(output4, size=[output3.size(2), output3.size(3)], mode="nearest")
        output3 = output3 + up4
        output3 = self.merge3(output3)

        up3 = F.interpolate(output3, size=[output2.size(2), output2.size(3)], mode="nearest")
        output2 = output2 + up3
        output2 = self.merge2(output2)

        up2 = F.interpolate(output2, size=[output1.size(2), output1.size(3)], mode="nearest")
        output1 = output1 + up2
        output1 = self.merge1(output1)

        output1 = self.conv_out(output1)
        return output1

class PFLD_GhostOne(Module):
    def __init__(self, width_factor=0.5, input_size=192, landmark_number=110, inference_mode=False):
        super(PFLD_GhostOne, self).__init__()

        self.inference_mode = inference_mode
        self.num_conv_branches = 6

        self.conv1 = MobileOneBlock(
            in_channels=3,
            out_channels=int(64 * width_factor),
            kernel_size=3,
            stride=2,
            padding=1,
            groups=1,
            inference_mode=self.inference_mode,
            use_se=False,
            num_conv_branches=self.num_conv_branches,
            is_linear=False
        )
        self.conv2 = MobileOneBlock(
            in_channels=int(64 * width_factor),
            out_channels=int(64 * width_factor),
            kernel_size=3,
            stride=1,
            padding=1,
            groups=int(64 * width_factor),
            inference_mode=self.inference_mode,
            use_se=False,
            num_conv_branches=self.num_conv_branches,
            is_linear=False
        )

        # 主干网络
        self.conv3_1 = GhostOneBottleneck(
            int(64 * width_factor), int(96 * width_factor), int(80 * width_factor), 
            stride=2, inference_mode=self.inference_mode, num_conv_branches=self.num_conv_branches
        )
        self.conv3_2 = GhostOneBottleneck(
            int(80 * width_factor), int(120 * width_factor), int(80 * width_factor), 
            stride=1, inference_mode=self.inference_mode, num_conv_branches=self.num_conv_branches
        )
        self.conv3_3 = GhostOneBottleneck(
            int(80 * width_factor), int(120 * width_factor), int(80 * width_factor), 
            stride=1, inference_mode=self.inference_mode, num_conv_branches=self.num_conv_branches
        )

        self.conv4_1 = GhostOneBottleneck(
            int(80 * width_factor), int(200 * width_factor), int(96 * width_factor), 
            stride=2, inference_mode=self.inference_mode, num_conv_branches=self.num_conv_branches
        )
        self.conv4_2 = GhostOneBottleneck(
            int(96 * width_factor), int(240 * width_factor), int(96 * width_factor), 
            stride=1, inference_mode=self.inference_mode, num_conv_branches=self.num_conv_branches
        )
        self.conv4_3 = GhostOneBottleneck(
            int(96 * width_factor), int(240 * width_factor), int(96 * width_factor), 
            stride=1, inference_mode=self.inference_mode, num_conv_branches=self.num_conv_branches
        )

        self.conv5_1 = GhostOneBottleneck(
            int(96 * width_factor), int(336 * width_factor), int(144 * width_factor), 
            stride=2, inference_mode=self.inference_mode, num_conv_branches=self.num_conv_branches
        )
        self.conv5_2 = GhostOneBottleneck(
            int(144 * width_factor), int(504 * width_factor), int(144 * width_factor), 
            stride=1, inference_mode=self.inference_mode, num_conv_branches=self.num_conv_branches
        )
        self.conv5_3 = GhostOneBottleneck(
            int(144 * width_factor), int(504 * width_factor), int(144 * width_factor), 
            stride=1, inference_mode=self.inference_mode, num_conv_branches=self.num_conv_branches
        )
        self.conv5_4 = GhostOneBottleneck(
            int(144 * width_factor), int(504 * width_factor), int(144 * width_factor), 
            stride=1, inference_mode=self.inference_mode, num_conv_branches=self.num_conv_branches
        )

        self.conv6 = GhostOneBottleneck(
            int(144 * width_factor), int(216 * width_factor), int(16 * width_factor), 
            stride=1, inference_mode=self.inference_mode, num_conv_branches=self.num_conv_branches
        )
        self.conv7 = MobileOneBlock(
            in_channels=int(16 * width_factor),
            out_channels=int(32 * width_factor),
            kernel_size=3,
            stride=1,
            padding=1,
            groups=1,
            inference_mode=self.inference_mode,
            use_se=False,
            num_conv_branches=self.num_conv_branches,
            is_linear=False
        )
        self.conv8 = Conv_Block(
            int(32 * width_factor), int(128 * width_factor), 
            kernel_size=input_size // 16, stride=1, padding=0, has_bn=False
        )

        # 多尺度特征融合
        self.avg_pool1 = AvgPool2d(input_size // 2)
        self.avg_pool2 = AvgPool2d(input_size // 4)
        self.avg_pool3 = AvgPool2d(input_size // 8)
        self.avg_pool4 = AvgPool2d(input_size // 16)
        self.conv_out = nn.Conv2d(int(512 * width_factor), landmark_number * 2, 1, 1, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x1 = self.avg_pool1(x)

        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x2 = self.avg_pool2(x)
        
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        x3 = self.avg_pool3(x)

        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)
        x = self.conv5_4(x)
        x4 = self.avg_pool4(x)

        x = self.conv6(x)
        x = self.conv7(x)
        x5 = self.conv8(x)

        multi_scale = torch.cat([x1, x2, x3, x4, x5], 1)
        landmarks = self.conv_out(multi_scale)
        landmarks = landmarks.view(landmarks.size(0), -1)
        return landmarks

class PFLD_GhostOne_WithSTN(Module):
    def __init__(self, width_factor=0.5, input_size=112, landmark_number=110, inference_mode=False):
        super(PFLD_GhostOne_WithSTN, self).__init__()  # 修正继承错误

        self.inference_mode = inference_mode
        self.num_conv_branches = 6

        # STN定位网络
        self.localization = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 24 * 24, 32),
            nn.ReLU(True),
            nn.Linear(32, 6)  # 输出仿射变换矩阵参数
        )
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

        # 主干网络（同PFLD_GhostOne）
        self.conv1 = MobileOneBlock(
            in_channels=3,
            out_channels=int(64 * width_factor),
            kernel_size=3,
            stride=2,
            padding=1,
            groups=1,
            inference_mode=self.inference_mode,
            use_se=False,
            num_conv_branches=self.num_conv_branches,
            is_linear=False
        )
        self.conv2 = MobileOneBlock(
            in_channels=int(64 * width_factor),
            out_channels=int(64 * width_factor),
            kernel_size=3,
            stride=1,
            padding=1,
            groups=int(64 * width_factor),
            inference_mode=self.inference_mode,
            use_se=False,
            num_conv_branches=self.num_conv_branches,
            is_linear=False
        )

        self.conv3_1 = GhostOneBottleneck(
            int(64 * width_factor), int(96 * width_factor), int(80 * width_factor), 
            stride=2, inference_mode=self.inference_mode, num_conv_branches=self.num_conv_branches
        )
        self.conv3_2 = GhostOneBottleneck(
            int(80 * width_factor), int(120 * width_factor), int(80 * width_factor), 
            stride=1, inference_mode=self.inference_mode, num_conv_branches=self.num_conv_branches
        )
        self.conv3_3 = GhostOneBottleneck(
            int(80 * width_factor), int(120 * width_factor), int(80 * width_factor), 
            stride=1, inference_mode=self.inference_mode, num_conv_branches=self.num_conv_branches
        )

        self.conv4_1 = GhostOneBottleneck(
            int(80 * width_factor), int(200 * width_factor), int(96 * width_factor), 
            stride=2, inference_mode=self.inference_mode, num_conv_branches=self.num_conv_branches
        )
        self.conv4_2 = GhostOneBottleneck(
            int(96 * width_factor), int(240 * width_factor), int(96 * width_factor), 
            stride=1, inference_mode=self.inference_mode, num_conv_branches=self.num_conv_branches
        )
        self.conv4_3 = GhostOneBottleneck(
            int(96 * width_factor), int(240 * width_factor), int(96 * width_factor), 
            stride=1, inference_mode=self.inference_mode, num_conv_branches=self.num_conv_branches
        )

        self.conv5_1 = GhostOneBottleneck(
            int(96 * width_factor), int(336 * width_factor), int(144 * width_factor), 
            stride=2, inference_mode=self.inference_mode, num_conv_branches=self.num_conv_branches
        )
        self.conv5_2 = GhostOneBottleneck(
            int(144 * width_factor), int(504 * width_factor), int(144 * width_factor), 
            stride=1, inference_mode=self.inference_mode, num_conv_branches=self.num_conv_branches
        )
        self.conv5_3 = GhostOneBottleneck(
            int(144 * width_factor), int(504 * width_factor), int(144 * width_factor), 
            stride=1, inference_mode=self.inference_mode, num_conv_branches=self.num_conv_branches
        )
        self.conv5_4 = GhostOneBottleneck(
            int(144 * width_factor), int(504 * width_factor), int(144 * width_factor), 
            stride=1, inference_mode=self.inference_mode, num_conv_branches=self.num_conv_branches
        )

        self.conv6 = GhostOneBottleneck(
            int(144 * width_factor), int(216 * width_factor), int(16 * width_factor), 
            stride=1, inference_mode=self.inference_mode, num_conv_branches=self.num_conv_branches
        )
        self.conv7 = MobileOneBlock(
            in_channels=int(16 * width_factor),
            out_channels=int(32 * width_factor),
            kernel_size=3,
            stride=1,
            padding=1,
            groups=1,
            inference_mode=self.inference_mode,
            use_se=False,
            num_conv_branches=self.num_conv_branches,
            is_linear=False
        )
        self.conv8 = Conv_Block(
            int(32 * width_factor), int(128 * width_factor), 
            kernel_size=input_size // 16, stride=1, padding=0, has_bn=False
        )

        self.avg_pool1 = AvgPool2d(input_size // 2)
        self.avg_pool2 = AvgPool2d(input_size // 4)
        self.avg_pool3 = AvgPool2d(input_size // 8)
        self.avg_pool4 = AvgPool2d(input_size // 16)
        self.conv_out = nn.Conv2d(int(512 * width_factor), landmark_number * 2, 1, 1, 0)

    def stn(self, x):
        """空间变换网络"""
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 24 * 24)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x

    def forward(self, x):
        x = self.stn(x)  # 应用STN变换
        x = self.conv1(x)
        x = self.conv2(x)
        x1 = self.avg_pool1(x)

        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x2 = self.avg_pool2(x)
        
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        x3 = self.avg_pool3(x)

        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)
        x = self.conv5_4(x)
        x4 = self.avg_pool4(x)

        x = self.conv6(x)
        x = self.conv7(x)
        x5 = self.conv8(x)

        multi_scale = torch.cat([x1, x2, x3, x4, x5], 1)
        landmarks = self.conv_out(multi_scale)
        landmarks = landmarks.view(landmarks.size(0), -1)
        return landmarks

def check_onnx(torch_out, torch_in, onnx_path):
    """验证ONNX模型导出正确性"""
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    import onnxruntime
    ort_session = onnxruntime.InferenceSession(onnx_path)
    ort_inputs = {ort_session.get_inputs()[0].name: torch_in.cpu().numpy()}
    ort_outs = ort_session.run(None, ort_inputs)
    np.testing.assert_allclose(torch_out[0].cpu().numpy(), ort_outs[0][0], rtol=1e-03, atol=1e-05)
    print("Exported model has been tested with ONNXRuntime, and the result looks good!")