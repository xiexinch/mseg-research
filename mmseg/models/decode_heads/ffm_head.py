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
            in_channels=(32, 320),
            channels=128,
            kernel_size=3,
            num_convs=2,
            concat_input=True,
            dilation=1,
            dw_cfg=None,
            **kwargs):

        super(FFMHead, self).__init__(in_channels, channels, **kwargs)

        if dw_cfg is None:
            dw_cfg = dict(
                dw_norm_cfg='default',
                dw_act_cfg='default',
                pw_norm_cfg='default',
                pw_act_cfg='default',
            )

        # 滤波器
        self.up_conv = DepthwiseSeparableConvModule(
            in_channels[-1], in_channels[-1], 3, padding=1, **dw_cfg)

        # spatial attn
        self.avg_pool = nn.AdaptiveMaxPool2d(1)

        # semantic channels attn
        self.softmax = nn.Softmax(dim=1)

        # A*context + B*spatial + bias
        conv_padding = (kernel_size // 2) * dilation
        convs = []
        for i in range(num_convs):
            convs.append(
                DepthwiseSeparableConvModule(
                    in_channels=sum(in_channels) if i == 0 else channels,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    padding=conv_padding,
                    dilation=dilation,
                    **dw_cfg)
            )
        self.fuse_conv = nn.Sequential(*convs)

        self.concat_input = concat_input
        if self.concat_input:
            self.conv_cat = DepthwiseSeparableConvModule(
                sum(in_channels) + channels,
                self.channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                norm_cfg=self.norm_cfg,
                **dw_cfg)

    def forward(self, inputs):
        x = self._transform_inputs(inputs)
        x_8, x_32 = x[0], x[1]

        # 先模糊再上采样
        x_32 = self.up_conv(x_32)
        x_32 = resize(x_32, scale_factor=4,
                      mode='bilinear', align_corners=True)

        # 空间注意力
        x_8 = self.avg_pool(x_8) * x_8
        x_cat = torch.cat([x_8, x_32], dim=1)
        x = self.fuse_conv(x_cat)

        if self.concat_input:
            x = self.conv_cat(torch.cat([x_cat, x], dim=1))
        out = self.conv_seg(x)
        return out
