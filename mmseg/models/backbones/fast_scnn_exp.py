import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmcv.runner import BaseModule, Sequential

from .fast_scnn import FeatureFusionModule, LearningToDownsample, GlobalFeatureExtractor
from mmseg.ops import resize
from ..builder import BACKBONES
from ..utils import InvertedResidual


class ASPPGlobalFeatureExtractor(BaseModule):

    def __init__(
            self,
            in_channels=64,
            block_channels=(64, 96, 128),
            expand_ratio=6,
            num_blocks=(3, 3, 3),
            strides=(2, 2, 1),
            dilations=(1, 6, 12),
            conv_cfg=None,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU'),
            init_cfg=None):
        super(ASPPGlobalFeatureExtractor, self).__init__(init_cfg)
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        assert len(block_channels) == len(num_blocks) == 3

        bottlenecks = []
        for i in range(3):
            bottlenecks.append(
                self._make_layer(
                    in_channels=in_channels if i == 0 else block_channels[i - 1],
                    out_channels=block_channels[i],
                    blocks=num_blocks[i],
                    dilation=dilations[i],
                    stride=strides[i],
                    expand_ratio=expand_ratio))
        self.bottlenecks = Sequential(**bottlenecks)

    def _make_layer(
            self,
            in_channels,
            out_channels,
            blocks,
            dilation=1,
            stride=1,
            expand_ratio=6):
        layers = []
        for i in range(blocks):
            layers.append(
                InvertedResidual(
                    in_channels=in_channels if i == 0 else out_channels,
                    out_channels=out_channels,
                    stride=stride if i == 0 else 1,
                    dilation=dilation,
                    expand_ratio=expand_ratio,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))
        return Sequential(**layers)

    def forward(self, x):
        return self.bottlenecks(x)


@BACKBONES.register_module()
class FastSCNNEXP(BaseModule):

    def __init__(
            self,
            in_channels=3,
            downsample_dw_channels=(32, 48),
            global_in_channels=64,
            global_block_channels=(64, 96, 128),
            expand_ratio=6,
            global_block_strides=(2, 2, 1),
            aspp_dilations=(1, 6, 12),
            global_out_channels=128,
            fusion_out_channels=128,
            out_indices=(0, 1, 2),
            aspp_global_extractor=True,
            conv_cfg=None,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU'),
            align_corners=False,
            dw_act_cfg=None,
            init_cfg=None):
        super(FastSCNNEXP, self).__init__(init_cfg)
        self.out_indices = out_indices

        self.learning_to_downsample = LearningToDownsample(
            in_channels,
            downsample_dw_channels,
            global_in_channels,
            conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            dw_act_cfg=dw_act_cfg)

        if aspp_global_extractor:
            self.global_feature_extractor = ASPPGlobalFeatureExtractor(
                global_in_channels,
                global_block_channels,
                expand_ratio=expand_ratio,
                strides=global_block_strides,
                dilations=aspp_dilations,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
        else:
            self.global_feature_extractor = GlobalFeatureExtractor(
                global_in_channels,
                global_block_channels,
                global_out_channels,
                strides=global_block_strides,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                align_corners=align_corners)

        self.feature_fusion = FeatureFusionModule(
            higher_in_channels=global_in_channels,
            lower_in_channels=global_out_channels,
            fusion_out_channels=fusion_out_channels,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            dwconv_act_cfg=act_cfg,
            align_corners=align_corners)

    def forward(self, x):
        higher_res_features = self.learning_to_downsample(x)
        lower_res_features = self.global_feature_extractor(higher_res_features)
        fusion_output = self.feature_fusion(
            higher_res_features, lower_res_features)
        outs = [higher_res_features, lower_res_features, fusion_output]
        outs = [outs[i] for i in self.out_indices]
        return tuple(outs)
