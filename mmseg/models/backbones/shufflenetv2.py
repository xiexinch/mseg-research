import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import ConvModule, constant_init, normal_init
from mmcv.runner import BaseModule
from torch.nn.modules.batchnorm import _BatchNorm
from mmseg.models.builder import BACKBONES


def channel_shuffle(x, groups):
    """Channel Shuffle operation.

    This function enables cross-group information flow for multiple groups
    convolution layers.

    Args:
        x (Tensor): The input tensor.
        groups (int): The number of groups to divide the input tensor
            in the channel dimension.

    Returns:
        Tensor: The output tensor after channel shuffle operation.
    """

    batch_size, num_channels, height, width = x.size()
    assert (num_channels % groups == 0), ('num_channels should be '
                                          'divisible by groups')
    channels_per_group = num_channels // groups

    x = x.view(batch_size, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batch_size, -1, height, width)

    return x


class InvertedResidual(BaseModule):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            reduce_channel=True,
            stride=1,
            conv_cfg=None,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU'),
            with_cp=False,
            init_cfg=None):
        super(InvertedResidual, self).__init__(init_cfg)
        self.stride = stride
        self.with_cp = with_cp

        branch_features = out_channels // 2 if reduce_channel else out_channels

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                ConvModule(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=self.stride,
                    padding=1,
                    groups=in_channels,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=None
                ),
                ConvModule(
                    in_channels,
                    branch_features,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg
                )
            )

        self.branch2 = nn.Sequential(
            ConvModule(
                in_channels if (self.stride > 1) else branch_features,
                branch_features,
                kernel_size=1,
                stride=1,
                padding=0,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg
            ),
            ConvModule(
                branch_features,
                branch_features,
                kernel_size=3,
                stride=self.stride,
                padding=1,
                groups=branch_features,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=None
            ),
            ConvModule(
                branch_features,
                branch_features,
                kernel_size=1,
                stride=1,
                padding=0,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg
            )
        )

    def forward(self, x: torch.Tensor):
        def _inner_forward(x: torch.Tensor):

            if self.stride > 1:
                out = torch.cat([self.branch1(x), self.branch2(x)], dim=1)
            else:
                x1, x2 = x.chunk(2, dim=1)
                print(x1.shape, x2.shape)
                out = torch.cat([x1, self.branch2(x2)], dim=1)

            out = channel_shuffle(out, 2)
            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        return out


class StemBlock(BaseModule):
    def __init__(
            self,
            in_channels=3,
            out_channels=64,
            conv_cfg=None,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU'),
            init_cfg=None):
        super(StemBlock, self).__init__(init_cfg)
        self.conv1 = ConvModule(
            in_channels, out_channels, 3, stride=2, padding=1,
            conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

        self.conv_branch = nn.Sequential(
            ConvModule(
                out_channels,
                out_channels // 2,
                kernel_size=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg),
            ConvModule(
                out_channels // 2,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg))
        self.fuse_conv = ConvModule(
            in_channels=out_channels * 2,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def forward(self, x):
        x = self.conv1(x)
        x_conv = self.conv_branch(x)
        x_pool = nn.AdaptiveMaxPool2d(x_conv.shape[2:])(x)
        out = self.fuse_conv(torch.cat([x_conv, x_pool], dim=1))
        return out


class ResidualLayers(BaseModule):
    def __init__(
            self,
            in_channels,
            out_channels,
            norm_cfg,
            act_cfg,
            with_cp,
            init_cfg=None):
        super(ResidualLayers, self).__init__(init_cfg)
        self.blocks = []
        for i in range(4):
            out_channels = out_channels // 2 if i <= 2 else out_channels
            block = InvertedResidual(
                in_channels=in_channels,
                out_channels=out_channels // 2,
                reduce_channel=False if 0 < i < 3 else True,
                stride=2 if i == 0 else 1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg, with_cp=with_cp)
            in_channels = out_channels
            self.blocks.append(block)

        self.fuse_conv = ConvModule(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def forward(self, x):
        outs = []
        for block in self.blocks:
            x = block(x)
            outs.append(x)
        out = self.fuse_conv(torch.cat(outs, dim=1))
        return out


@BACKBONES.register_module()
class LiteShuffleNetV2(BaseModule):
    def __init__(
            self,
            in_channels=3,
            channels=[64, 256, 512, 1024],
            out_indices=(0, 1, 2, 3),
            frozen_stages=-1,
            conv_cfg=None,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU'),
            norm_eval=False,
            with_cp=False,
            init_cfg=None):
        super(LiteShuffleNetV2, self).__init__(init_cfg)
        self.stage_blocks = [4, 8, 4]
        for index in out_indices:
            if index not in range(0, 4):
                raise ValueError('the item in out_indices must in '
                                 f'range(0, 4). But received {index}')
        if frozen_stages not in range(-1, 4):
            raise ValueError('frozen_stages must be in range(-1, 4). '
                             f'But received {frozen_stages}')
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.norm_eval = norm_eval
        self.with_cp = with_cp

        self.stem = StemBlock(
            in_channels=in_channels, out_channels=channels[0],
            norm_cfg=norm_cfg, act_cfg=act_cfg, conv_cfg=conv_cfg)

        self.stages = nn.ModuleList()
        for i in range(1, len(channels)):
            self.stages.append(
                ResidualLayers(
                    channels[i - 1],
                    channels[i],
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    with_cp=with_cp))

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            for param in self.stem.parameters():
                param.requires_grad = False

        for i in range(self.frozen_stages):
            m = self.stages[i]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def init_weights(self):
        super(LiteShuffleNetV2, self).init_weights()

        if (isinstance(self.init_cfg, dict) and self.init_cfg['type'] == 'Pretrained'):
            return

        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, mean=0, std=1.0 / m.weight.shape[1])
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                constant_init(m.weight, val=1, bias=0.0001)
                if isinstance(m, _BatchNorm):
                    if m.running_mean is not None:
                        nn.init.constant_(m.running_mean, 0)

    def forward(self, x):
        outs = []
        x = self.stem(x)
        print(x.shape)
        outs.append(x)

        for stage in self.stages:
            x = stage(x)
            outs.append(x)
        outs = [outs[i] for i in self.out_indices]
        return tuple(outs)

    def train(self, mode=True):
        super(LiteShuffleNetV2, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
