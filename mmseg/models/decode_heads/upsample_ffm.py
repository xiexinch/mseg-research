import torch
import torch.nn as nn
from mmcv.cnn import DepthwiseSeparableConvModule
from mmcv.runner import BaseModule
from mmcv.cnn.bricks.activation import build_activation_layer

from mmseg.models.builder import HEADS
from mmseg.ops import resize
from mmseg.models.decode_heads.decode_head import BaseDecodeHead


class AttnModule(BaseModule):

    def __init__(
            self,
            in_channels,
            out_channels,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='Sigmoid'),
            init_cfg=None):
        super(AttnModule, self).__init__(init_cfg)
        self.conv = DepthwiseSeparableConvModule(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            dw_norm_cfg=norm_cfg,
            dw_act_cfg=None,
            pw_norm_cfg=None,
            pw_act_cfg=None)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.activate = build_activation_layer(act_cfg)

    def forward(self, inputs):
        x = self.conv(inputs)
        x_attn = self.activate(self.pool(x))
        x = x * x_attn
        return x


@HEADS.register_module()
class UpsampleFFMHead(BaseDecodeHead):
    def __init__(
            self,
            in_channels=(128, 256, 512),
            norm_cfg=dict(type='BN'),
            **kwargs):

        super(UpsampleFFMHead, self).__init__(in_channels, **kwargs)

        self.conv_8 = AttnModule(
            in_channels=in_channels[0],
            out_channels=in_channels[2],
            norm_cfg=norm_cfg)

        self.conv_16 = AttnModule(
            in_channels=in_channels[1],
            out_channels=in_channels[2],
            norm_cfg=norm_cfg)

        self.activate = torch.nn.Sigmoid()

    def forward(self, inputs):
        x = self._transform_inputs(inputs)
        x_8, x_16, x_32 = x[0], x[1], x[2]

        x_16_ = resize(x_32, scale_factor=2,
                       mode='bilinear', align_corners=True)
        x_16 = self.conv_16(x_16)
        x_16 = x_16 + x_16_

        x_8_ = resize(x_16, scale_factor=2,
                      mode='bilinear', align_corners=True)
        x_8 = self.conv_8(x_8)
        x_8 = x_8 + x_8_

        return self.conv_seg(x_8)
