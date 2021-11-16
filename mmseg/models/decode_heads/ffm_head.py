import torch
import torch.nn as nn
from mmcv.cnn import DepthwiseSeparableConvModule
from mmcv.ops import CARAFE

from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.ops import resize


@HEADS.register_module()
class FFMHead(BaseDecodeHead):

    def __init__(
            self,
            in_channels=(64, 256),
            channels=128,
            **kwargs):

        super(FFMHead, self).__init__(in_channels, channels, **kwargs)

        # 滤波器
        self.up_conv = DepthwiseSeparableConvModule(
            in_channels[-1], in_channels[0], 3, padding=1)

        # spatial attn
        self.avg_pool = nn.AdaptiveMaxPool2d(1)

        # semantic channels attn
        self.softmax = nn.Softmax(dim=1)

        # A*context + B*spatial + bias
        self.fuse_conv = DepthwiseSeparableConvModule(
            in_channels=in_channels * 2,
            out_channels=channels,
            kernel_size=3, padding=1)

    def forward(self, inputs):
        x = self._transform_inputs(inputs)
        x_8, x_32 = x[0], x[1]

        # 先模糊再上采样
        x_32 = self.up_conv(x_32)
        x_32 = resize(x_32, scale_factor=4, mode='bilinear')

        # 空间注意力
        x_8 = self.avg_pool(x) * x_8
        x = torch.cat([x_8, x_32], dim=1)
        x = self.fuse_conv(x)
        out = self.conv_cfg(x)
        return out
