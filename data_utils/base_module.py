import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class Module(nn.Module):
    """基础模块类（兼容原有代码）"""
    def __init__(self):
        super().__init__()

class Conv_Block(Module):
    """卷积+BN+激活单元"""
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding, 
                 group=1, has_bn=True, is_linear=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, groups=group, bias=False)
        self.bn = nn.BatchNorm2d(out_channel) if has_bn else nn.Identity()
        self.activation = nn.ReLU() if not is_linear else nn.Identity()

    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))

class InvertedResidual(Module):
    def __init__(self, in_channel, out_channel, stride, use_res_connect, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        exp_channel = in_channel * expand_ratio
        self.use_res_connect = use_res_connect
        self.inv_res = nn.Sequential(
            Conv_Block(in_channel=in_channel, out_channel=exp_channel, kernel_size=1, stride=1, padding=0),
            Conv_Block(in_channel=exp_channel, out_channel=exp_channel, kernel_size=3, stride=stride, padding=1,
                       group=exp_channel),
            Conv_Block(in_channel=exp_channel, out_channel=out_channel, kernel_size=1, stride=1, padding=0,
                       is_linear=True)
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.inv_res(x)
        else:
            return self.inv_res(x)

class GhostModule(Module):
    def __init__(self, in_channel, out_channel, is_linear=False):
        super(GhostModule, self).__init__()
        self.out_channel = out_channel
        init_channel = math.ceil(out_channel / 2)
        new_channel = init_channel

        self.primary_conv = Conv_Block(in_channel, init_channel, 1, 1, 0, is_linear=is_linear)
        self.cheap_operation = Conv_Block(init_channel, new_channel, 3, 1, 1, group=init_channel, is_linear=is_linear)

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.out_channel, :, :]

class GhostBottleneck(Module):
    def __init__(self, in_channel, hidden_channel, out_channel, stride):
        super(GhostBottleneck, self).__init__()
        assert stride in [1, 2]

        self.ghost_conv = nn.Sequential(
            # GhostModule
            GhostModule(in_channel, hidden_channel, is_linear=False),
            # DepthwiseConv-linear
            Conv_Block(hidden_channel, hidden_channel, 3, stride, 1, group=hidden_channel,
                       is_linear=True) if stride == 2 else nn.Sequential(),
            # GhostModule-linear
            GhostModule(hidden_channel, out_channel, is_linear=True)
        )

        if stride == 1 and in_channel == out_channel:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                Conv_Block(in_channel, in_channel, 3, stride, 1, group=in_channel, is_linear=True),
                Conv_Block(in_channel, out_channel, 1, 1, 0, is_linear=True)
            )

    def forward(self, x):
        return self.ghost_conv(x) + self.shortcut(x)

class SEBlock(nn.Module):
    """Squeeze and Excite模块"""
    def __init__(self, in_channels: int, rd_ratio: float = 0.0625) -> None:
        super(SEBlock, self).__init__()
        self.reduce = nn.Conv2d(in_channels=in_channels,
                                out_channels=int(in_channels * rd_ratio),
                                kernel_size=1,
                                stride=1,
                                bias=True)
        self.expand = nn.Conv2d(in_channels=int(in_channels * rd_ratio),
                                out_channels=in_channels,
                                kernel_size=1,
                                stride=1,
                                bias=True)

    def forward(self, inputs: Tensor) -> Tensor:
        b, c, h, w = inputs.size()
        x = F.avg_pool2d(inputs, kernel_size=[h, w])
        x = self.reduce(x)
        x = F.relu(x)
        x = self.expand(x)
        x = torch.sigmoid(x)
        x = x.view(-1, c, 1, 1)
        return inputs * x

class MobileOneBlock(nn.Module):
    """MobileOne基础块"""
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 groups: int = 1,
                 inference_mode: bool = False,
                 use_se: bool = False,
                 num_conv_branches: int = 1,
                 is_linear: bool = False) -> None:
        super(MobileOneBlock, self).__init__()
        self.inference_mode = inference_mode
        self.groups = groups
        self.stride = stride
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_conv_branches = num_conv_branches

        self.se = SEBlock(out_channels) if use_se else nn.Identity()
        self.activation = nn.Identity() if is_linear else nn.ReLU()

        if inference_mode:
            self.reparam_conv = nn.Conv2d(in_channels=in_channels,
                                          out_channels=out_channels,
                                          kernel_size=kernel_size,
                                          stride=stride,
                                          padding=padding,
                                          dilation=dilation,
                                          groups=groups,
                                          bias=True)
        else:
            # 可重参数化的跳跃连接
            self.rbr_skip = nn.BatchNorm2d(num_features=in_channels) \
                if out_channels == in_channels and stride == 1 else None

            # 可重参数化的卷积分支
            self.rbr_conv = nn.ModuleList([
                self._conv_bn(kernel_size=kernel_size, padding=padding)
                for _ in range(num_conv_branches)
            ])

            # 可重参数化的尺度分支
            self.rbr_scale = self._conv_bn(kernel_size=1, padding=0) if kernel_size > 1 else None

    def _conv_bn(self, kernel_size: int, padding: int) -> nn.Sequential:
        """创建卷积+BN层"""
        return nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels,
                      out_channels=self.out_channels,
                      kernel_size=kernel_size,
                      stride=self.stride,
                      padding=padding,
                      groups=self.groups,
                      bias=False),
            nn.BatchNorm2d(num_features=self.out_channels)
        )

    def _fuse_bn_tensor(self, branch) -> Tuple[Tensor, Tensor]:
        """融合BN和卷积层"""
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            bn = branch.bn
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = torch.zeros((self.in_channels, input_dim, 
                                           self.kernel_size, self.kernel_size),
                                           dtype=branch.weight.dtype,
                                           device=branch.weight.device)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 
                                 self.kernel_size//2, self.kernel_size//2] = 1
                self.id_tensor = kernel_value
            kernel = self.id_tensor
            bn = branch

        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def _get_kernel_bias(self) -> Tuple[Tensor, Tensor]:
        """获取重参数化后的 kernel 和 bias"""
        kernel_scale, bias_scale = 0, 0
        if self.rbr_scale is not None:
            kernel_scale, bias_scale = self._fuse_bn_tensor(self.rbr_scale)
            pad = self.kernel_size // 2
            kernel_scale = F.pad(kernel_scale, [pad, pad, pad, pad])

        kernel_identity, bias_identity = 0, 0
        if self.rbr_skip is not None:
            kernel_identity, bias_identity = self._fuse_bn_tensor(self.rbr_skip)

        kernel_conv, bias_conv = 0, 0
        for branch in self.rbr_conv:
            _kernel, _bias = self._fuse_bn_tensor(branch)
            kernel_conv += _kernel
            bias_conv += _bias

        return kernel_conv + kernel_scale + kernel_identity, bias_conv + bias_scale + bias_identity

    def reparameterize(self):
        """重参数化（训练转推理）"""
        if self.inference_mode:
            return
        kernel, bias = self._get_kernel_bias()
        self.reparam_conv = nn.Conv2d(
            in_channels=self.rbr_conv[0].conv.in_channels,
            out_channels=self.rbr_conv[0].conv.out_channels,
            kernel_size=self.rbr_conv[0].conv.kernel_size,
            stride=self.rbr_conv[0].conv.stride,
            padding=self.rbr_conv[0].conv.padding,
            dilation=self.rbr_conv[0].conv.dilation,
            groups=self.rbr_conv[0].conv.groups,
            bias=True
        )
        self.reparam_conv.weight.data = kernel
        self.reparam_conv.bias.data = bias

        # 删除无用分支
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_conv')
        self.__delattr__('rbr_scale')
        if hasattr(self, 'rbr_skip'):
            self.__delattr__('rbr_skip')
        self.inference_mode = True

    def forward(self, x: Tensor) -> Tensor:
        if self.inference_mode:
            return self.activation(self.se(self.reparam_conv(x)))

        # 训练时多分支前向
        identity_out = self.rbr_skip(x) if self.rbr_skip is not None else 0
        scale_out = self.rbr_scale(x) if self.rbr_scale is not None else 0
        out = scale_out + identity_out

        for branch in self.rbr_conv:
            out += branch(x)

        return self.activation(self.se(out))

class GhostOneModule(Module):
    def __init__(self, in_channel, out_channel, is_linear=False, inference_mode=False, num_conv_branches=1):
        super(GhostOneModule, self).__init__()
        self.out_channel = out_channel
        half_outchannel = math.ceil(out_channel / 2)

        self.inference_mode = inference_mode
        self.num_conv_branches = num_conv_branches

        self.primary_conv = MobileOneBlock(
            in_channels=in_channel,
            out_channels=half_outchannel,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            inference_mode=self.inference_mode,
            use_se=False,
            num_conv_branches=self.num_conv_branches,
            is_linear=is_linear
        )
        self.cheap_operation = MobileOneBlock(
            in_channels=half_outchannel,
            out_channels=half_outchannel,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=half_outchannel,
            inference_mode=self.inference_mode,
            use_se=False,
            num_conv_branches=self.num_conv_branches,
            is_linear=is_linear
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out

class GhostOneBottleneck(Module):
    def __init__(self, in_channel, hidden_channel, out_channel, stride, inference_mode=False, num_conv_branches=1):
        super(GhostOneBottleneck, self).__init__()
        assert stride in [1, 2]

        self.inference_mode = inference_mode
        self.num_conv_branches = num_conv_branches

        self.ghost_conv = nn.Sequential(
            # GhostModule
            GhostOneModule(in_channel, hidden_channel, is_linear=False, 
                          inference_mode=self.inference_mode, num_conv_branches=self.num_conv_branches),
            # DepthwiseConv-linear
            MobileOneBlock(
                in_channels=hidden_channel,
                out_channels=hidden_channel,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=hidden_channel,
                inference_mode=self.inference_mode,
                use_se=False,
                num_conv_branches=self.num_conv_branches,
                is_linear=True
            ) if stride == 2 else nn.Sequential(),
            # GhostModule-linear
            GhostOneModule(hidden_channel, out_channel, is_linear=True, 
                          inference_mode=self.inference_mode, num_conv_branches=self.num_conv_branches)
        )

    def forward(self, x):
        return self.ghost_conv(x)