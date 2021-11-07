# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.runner import BaseModule, ModuleList

import torch

from ..builder import BACKBONES
from ..utils import (build_transformer_layer, ResLayer, PatchEmbed,
                     PatchMerging, nlc_to_nchw)
from .resnet import BasicBlock
from .bisenetv2 import StemBlock
from .swin import SwinBlockSequence


@BACKBONES.register_module()
class HybridTransformer(BaseModule):

    def __init__(self,
                 in_channels,
                 out_indices=(0, 1, 2, 3),
                 base_channels=64,
                 embed_dims=128,
                 patch_embed_cfg=dict(),
                 num_res_stages=2,
                 num_res_layers=[2, 2],
                 num_transformer_layers=[2, 2],
                 num_heads=[8, 8],
                 mlp_ratio=3,
                 res_norm_cfg=dict(type='BN'),
                 patch_merging_cfg=dict(),
                 transformer_cfg=None,
                 init_cfg=None):
        super(HybridTransformer, self).__init__(init_cfg)

        if transformer_cfg == None:
            transformer_cfg = dict(
                drop_rate=0.,
                attn_drop_rate=0.,
                drop_path_rate=0.,
                num_fcs=2,
                qkv_bias=True,
                act_cfg=dict(type='GELU'),
                norm_cfg=dict(type='LN'),
                batch_first=True)

        self.stem = StemBlock(
            in_channels=in_channels, out_channels=base_channels)

        self.high_level_stages = ModuleList()
        for i in range(num_res_stages):
            out_channels = base_channels * 2**i
            self.high_level_stages.append(
                ResLayer(
                    BasicBlock,
                    base_channels,
                    out_channels,
                    num_res_layers[i],
                    stride=1 if i == 0 else 2,
                    norm_cfg=res_norm_cfg))
            base_channels = out_channels

        self.patch_embed = PatchEmbed(out_channels, embed_dims,
                                      **patch_embed_cfg)

        self.x16_stage = ModuleList()

        for i in range(num_transformer_layers[0]):
            self.x16_stage.append(
                build_transformer_layer(
                    dict(
                        type='ViTTransformerLayer',
                        embed_dims=embed_dims,
                        num_heads=num_heads[0],
                        feedforward_channels=mlp_ratio * embed_dims,
                        **transformer_cfg)))

        self.patch_merging = PatchMerging(embed_dims, embed_dims * 2,
                                          **patch_merging_cfg)

        embed_dims = embed_dims * 2
        self.x32_stage = ModuleList()
        for i in range(num_transformer_layers[1]):
            self.x32_stage.append(
                build_transformer_layer(
                    dict(
                        type='ViTTransformerLayer',
                        embed_dims=embed_dims,
                        num_heads=num_heads[1],
                        feedforward_channels=mlp_ratio * embed_dims,
                        **transformer_cfg)))

        self.out_indices = out_indices

    def forward(self, x):
        outs = []
        # x2
        x = self.stem(x)
        # x8
        for stage in self.high_level_stages:
            x = stage(x)
            outs.append(x)

        # x16
        x_16, hw_shape = self.patch_embed(x)
        for layer in self.x16_stage:
            x_16 = layer(x_16)
        outs.append(nlc_to_nchw(x_16, hw_shape))

        # x32
        x_32, hw_shape = self.patch_merging(x_16, hw_shape)
        for layer in self.x32_stage:
            x_32 = layer(x_32)
        outs.append(nlc_to_nchw(x_32, hw_shape))
        outs = [outs[i] for i in self.out_indices]
        return tuple(outs)


@BACKBONES.register_module()
class HybridSwin(HybridTransformer):

    def __init__(self,
                 embed_dims=128,
                 num_transformer_layers=[2, 2],
                 num_heads=[12, 24],
                 mlp_ratio=4,
                 **kwargs):
        super(HybridSwin, self).__init__(kwargs)

        self.x16_stage = ModuleList()
        for i in range(num_transformer_layers[0]):
            shift = False if i % 2 == 0 else True
            self.x16_stage.append(
                build_transformer_layer(
                    dict(
                        type='SwinTransformerLayer',
                        embed_dims=embed_dims,
                        num_heads=num_heads[0],
                        feedforward_channels=mlp_ratio * embed_dims,
                        window_size=7,
                        shift=shift,
                        shift_size=7 // 2 if shift else 0)))

        embed_dims = embed_dims * 2
        self.x32_stage = ModuleList()
        for i in range(num_transformer_layers[1]):
            shift = False if i % 2 == 0 else True
            self.x32_stage.append(
                build_transformer_layer(
                    dict(
                        type='SwinTransformerLayer',
                        embed_dims=embed_dims,
                        num_heads=num_heads[1],
                        feedforward_channels=mlp_ratio * embed_dims,
                        window_size=7,
                        shift=shift,
                        shift_size=7 // 2 if shift else 0)))


@BACKBONES.register_module()
class HybridSwinResNet(BaseModule):

    def __init__(self,
                 in_channels,
                 embed_dims,
                 embed_cfg=None,
                 out_indices=(0, 1, 2, 3),
                 window_size=7,
                 num_heads=[3, 6],
                 mlp_ratio=4,
                 depths=[2, 2],
                 patch_norm=True,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 res_norm_cfg=dict(type='BN'),
                 num_res_layers=[2, 2],
                 swin_act_cfg=dict(type='GELU'),
                 swin_norm_cfg=dict(type='LN'),
                 with_cp=False,
                 init_cfg=None):
        super(HybridSwinResNet, self).__init__(init_cfg)
        self.out_indices = out_indices
        if embed_cfg is None:
            embed_cfg = dict(kernel_size=4, stride=4, padding='corner')
        # downsample 4x
        self.patch_embed = PatchEmbed(
            in_channels=in_channels,
            embed_dims=embed_dims,
            conv_type='Conv2d',
            norm_cfg=swin_norm_cfg if patch_norm else None,
            **embed_cfg)

        # set stochastic depth decay rule
        total_depth = sum(depths)
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, total_depth)
        ]

        for i in range(2):
            stage = SwinBlockSequence(
                embed_dims=embed_dims,
                num_heads=num_heads[i],
                feedforward_channels=mlp_ratio * embed_dims,
                window_size=window_size,
                depth=depths[i],
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                downsample=PatchMerging(
                    in_channels=embed_dims,
                    out_channels=embed_dims * 2,
                    stride=2,
                    norm_cfg=swin_norm_cfg if patch_norm else None)
                if i == 0 else None,
                act_cfg=swin_act_cfg,
                norm_cfg=swin_norm_cfg,
                with_cp=with_cp)
            self.add_module(f'stage_{2**(i + 2)}x', stage)
            embed_dims *= 2

        base_channels = embed_dims // 2
        for i in range(2, 4):
            out_channels = base_channels * 2
            stage = ResLayer(
                BasicBlock,
                base_channels,
                out_channels,
                num_res_layers[i - 2],
                stride=2,
                norm_cfg=res_norm_cfg)
            base_channels = out_channels
            self.add_module(f'stage_{2**(i + 2)}x', stage)

    def forward(self, x):
        x, hw_shape = self.patch_embed(x)
        outs = []
        for i in range(2):
            stage = self.__getattr__(f'stage_{2**(i + 2)}x')
            x, hw_shape, _, _ = stage(x, hw_shape)
            outs.append(nlc_to_nchw(x, hw_shape))

        x = nlc_to_nchw(x, hw_shape)
        for i in range(2, 4):
            stage = self.__getattr__(f'stage_{2**(i + 2)}x')
            x = stage(x)
            outs.append(x)
        outs = [outs[i] for i in self.out_indices]
        return outs
