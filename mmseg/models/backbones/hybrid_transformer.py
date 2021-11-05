# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.runner import BaseModule, ModuleList


from ..builder import BACKBONES
from ..utils import (build_transformer_layer, ResLayer,
                     PatchEmbed, PatchMerging, nlc_to_nchw)
from .resnet import BasicBlock
from .bisenetv2 import StemBlock


@BACKBONES.register_module()
class HybridTransformer(BaseModule):

    def __init__(
        self,
        in_channels,
        transformer_cfg,
        out_indices=(0, 1, 2, 3),
        base_channels=64,
        embed_dims=128,
        patch_embed_cfg=dict(),
        num_res_stages=2,
        num_reslayers=[2, 2],
        num_transformer_layers=[2, 2],
        num_heads=[12, 24],
        stride=[2, 2],
        depth=[6, 2],
        mlp_ratio=3,
        res_norm_cfg=dict(type='BN'),
        patch_merging_cfg=dict(),
        init_cfg=None
    ):
        super(HybridTransformer, self).__init__(init_cfg)

        self.stem = StemBlock(
            in_channels=in_channels,
            out_channels=base_channels
        )

        self.high_level_stages = ModuleList()
        for i in range(num_res_stages):
            out_channels = base_channels * 2
            self.high_level_stages.append(
                ResLayer(BasicBlock, base_channels, out_channels,
                         num_reslayers[i], norm_cfg=res_norm_cfg))
            base_channels = out_channels

        self.patch_embed = PatchEmbed(
            out_channels, embed_dims, **patch_embed_cfg)

        self.x16_stage = ModuleList()
        transformer_cfg['embed_dims'] = embed_dims
        for i in range(num_transformer_layers[0]):
            shift = False if i % 2 == 0 else True
            self.x16_stage.append(build_transformer_layer(
                dict(
                    type='SwinTransformerLayer',
                    embed_dims=embed_dims,
                    feedforward_channels=mlp_ratio * embed_dims,
                    window_size=7,
                    shift=shift,
                    shift_size=7 // 2 if shift else 0
                )))

        self.patch_merging = PatchMerging(
            embed_dims, embed_dims * 2, **patch_merging_cfg)

        transformer_cfg['embed_dims'] = embed_dims * 2
        self.x32_stage = ModuleList()
        for i in range(num_transformer_layers[1]):
            self.x32_stage.append(build_transformer_layer(transformer_cfg))

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
        x, hw_shape = self.patch_embed(x)
        x_16 = self.x16_stage(x)
        outs.append(nlc_to_nchw(x_16, hw_shape))

        # x32
        x, hw_shape = self.patch_merging(x_16, hw_shape)
        x_32 = self.x32_stage(x)
        outs.append(nlc_to_nchw(x_32, hw_shape))
        outs = [outs[i] for i in range(self.out_indices)]
        return tuple(outs)


@BACKBONES.register_module()
class HybridResNet(BaseModule):

    def __init__(
        self,
        in_channels,
        base_channels,
        out_indices=(0, 1, 2, 3),
        init_cfg=None
    ):
        super(HybridTransformer, self).__init__(init_cfg)
