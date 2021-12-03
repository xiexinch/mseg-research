import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule
from torch.nn.modules.batchnorm import _BatchNorm
from mmcv.cnn import ConvModule, kaiming_init, constant_init, normal_init, build_upsample_layer
from ..builder import BACKBONES
from .mobilenet_v2 import MobileNetV2


class ChannelAttention(BaseModule):

    def __init__(self, channels, reduction=16, init_cfg=None):
        super(ChannelAttention, self).__init__(init_cfg)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1,
                      bias=False), nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_max_pool = self.max_pool(x)
        x_avg_pool = self.avg_pool(x)
        x_max = self.se(x_max_pool)
        x_avg = self.se(x_avg_pool)
        out = self.sigmoid(x_max + x_avg)
        return out


class SpatialAttention(BaseModule):

    def __init__(self, kernel_size=7, init_cfg=None):
        super(SpatialAttention, self).__init__(init_cfg)
        self.conv = ConvModule(
            in_channels=2,
            out_channels=1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            act_cfg=None)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_max, _ = torch.max(x, dim=1, keepdim=True)
        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_cat = torch.cat([x_max, x_avg], dim=1)

        out = self.conv(x_cat)
        out = self.sigmoid(out)
        return out


class BilateralAttn(BaseModule):

    def __init__(self, channels, reduction, kernel_size, init_cfg=None):
        super(BilateralAttn, self).__init__(init_cfg)
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(kernel_size)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m.weight, mode='fan_out')
                if m.bias is not None:
                    constant_init(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m.weight, 1)
                constant_init(m.bias, 0)
            elif isinstance(m, nn.Linear):
                normal_init(m.weight, std=0.001)
                if m.bias is not None:
                    constant_init(m.bias, 0)

    def forward(self, x):
        residual = x.clone()
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x + residual


@BACKBONES.register_module()
class AttnMobileNetV2(MobileNetV2):
    def __init__(self, **kwargs):
        super(AttnMobileNetV2, self).__init__(**kwargs)
        # 第二层加attn
        self.cbam = BilateralAttn(self.arch_settings[2][1], 16, 7)
        self.cbam.init_weights()

    def forward(self, x):
        x = self.conv1(x)

        outs = []
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            x = layer(x)
            if i == 2:
                x = self.cbam(x)
            if i in self.out_indices:
                outs.append(x)

        if len(outs) == 1:
            return outs[0]
        else:
            return tuple(outs)
