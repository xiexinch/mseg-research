from mmcv.cnn.bricks.wrappers import Linear
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmcv.runner import BaseModule, Sequential
from mmcv.cnn.bricks.transformer import FFN

from mmseg.models.utils import self_attention_block

from .fast_scnn import FeatureFusionModule, LearningToDownsample, GlobalFeatureExtractor
from mmseg.ops import resize
from ..builder import BACKBONES
from ..utils import InvertedResidual, nchw_to_nlc, nlc_to_nchw, PatchEmbed


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
        self.bottlenecks = Sequential(*bottlenecks)

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
        return Sequential(*layers)

    def forward(self, x):
        return self.bottlenecks(x)


class SelfAttention(BaseModule):

    def __init__(
            self,
            in_channels,
            embed_dims,
            patch_size=4,
            num_heads=8,
            qkv_bias=False,
            attn_drop=0.,
            proj_drop=0.,
            mlp_ratio=4,
            fc_drop=0.,
            init_cfg=None):
        super(SelfAttention, self).__init__(init_cfg)
        self.patch_embed = PatchEmbed(
            in_channels,
            embed_dims,
            patch_size
        )

        # attn
        self.num_heads = num_heads
        self.scale = (embed_dims // num_heads) ** -0.5
        self.qkv = Linear(embed_dims, embed_dims*3, qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop)

        # mlp
        hidden_dims = embed_dims * mlp_ratio
        self.mlp = Sequential([
            Linear(embed_dims, hidden_dims),
            nn.ReLU(),
            nn.Dropout(fc_drop),
            Linear(hidden_dims, embed_dims),
            nn.Dropout(fc_drop)
        ])

        self.conv_out = DepthwiseSeparableConvModule(
            embed_dims,
            in_channels,
            kernel_size=1)

    def forward(self, x):
        out_hw_shape = x.shape[2:]
        # downsample 4x
        x, hw_shape = self.patch_embed(x)
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        x = self.mlp(x)

        x = nlc_to_nchw(x, hw_shape)
        x = resize(x, size=out_hw_shape)
        x = self.conv_out(x)
        return x


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
            with_self_attn=False,
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

        self.with_self_attn = with_self_attn
        if self.with_self_attn:
            self.attn = SelfAttention(
                global_in_channels,
                embed_dims=128)

        self.feature_fusion = FeatureFusionModule(
            higher_in_channels=global_in_channels,
            lower_in_channels=global_out_channels,
            out_channels=fusion_out_channels,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            dwconv_act_cfg=act_cfg,
            align_corners=align_corners)

    def forward(self, x):
        higher_res_features = self.learning_to_downsample(x)
        lower_res_features = self.global_feature_extractor(higher_res_features)
        if self.with_self_attn:
            higher_res_features_attn = self.attn(higher_res_features)
            fusion_output = self.feature_fusion(
                higher_res_features_attn, lower_res_features)
        else:
            fusion_output = self.feature_fusion(
                higher_res_features, lower_res_features)

        outs = [higher_res_features, lower_res_features, fusion_output]
        outs = [outs[i] for i in self.out_indices]
        return tuple(outs)
