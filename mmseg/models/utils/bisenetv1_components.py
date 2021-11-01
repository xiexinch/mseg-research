import torch
import torch.nn as nn
from mmcv.cnn.bricks.norm import build_norm_layer
from mmcv.cnn.bricks.transformer import build_transformer_layer
from mmcv.runner import BaseModule, ModuleList
from mmcv.cnn import ConvModule

from mmseg.models.builder import MODELS
from mmseg.ops.wrappers import resize
from ..utils.embed import PatchMerging
from ..utils import PatchEmbed
from ..backbones.vit import TransformerEncoderLayer
from ..backbones.swin import SwinBlock

SPATIAL_PATH = MODELS
FFM = MODELS


def build_spatial_path(cfg):
    return SPATIAL_PATH.build(cfg)


def build_ffm(cfg):
    return FFM.build(cfg)


@SPATIAL_PATH.register_module()
class ShiftWindowTransformerSpatialPath(BaseModule):
    """Custom ShiftWindow transformer sp"""

    def __init__(self,
                 patch_embed_cfg=dict(
                     in_channels=3, embed_dims=96, kernel_size=4),
                 num_heads=3,
                 mlp_ratio=4,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_cfg=dict(type='LN'),
                 act_cfg=dict(type='GELU'),
                 final_downsample=False,
                 init_cfg=None):
        super(ShiftWindowTransformerSpatialPath,
              self).__init__(init_cfg=init_cfg)
        self.patch_embed = PatchEmbed(**patch_embed_cfg)
        self.num_heads = num_heads
        self.embed_dims = patch_embed_cfg['embed_dims']
        self.mlp_ratio = mlp_ratio
        feedforward_channels = self.embed_dims * self.mlp_ratio
        self.blocks = ModuleList()
        for i in range(2):
            block = SwinBlock(
                embed_dims=self.embed_dims,
                num_heads=self.num_heads,
                feedforward_channels=feedforward_channels,
                shift=False if i % 2 == 0 else True,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=drop_path_rate,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
            self.blocks.append(block)
        if final_downsample:
            self.downsample = PatchMerging(
                in_channels=self.embed_dims,
                out_channels=self.embed_dims * 2,
                norm_cfg=norm_cfg)
        else:
            self.downsample = None
        self.norm = build_norm_layer(
            norm_cfg,
            self.embed_dims * 2 if final_downsample else self.embed_dims)[1]

    def forward(self, x):
        x, hw_shape = self.patch_embed(x)
        for block in self.blocks:
            x = block(x, hw_shape)
        if self.downsample:
            x, hw_shape = self.downsample(x, hw_shape)

        x = self.norm(x)
        B, _, C = x.shape
        x = x.reshape(B, hw_shape[0], hw_shape[1], C).permute(0, 3, 1,
                                                              2).contiguous()
        return x


@SPATIAL_PATH.register_module()
class TransformerSpatialPath(BaseModule):
    """Custom Transformer based SpatialPath
    """

    def __init__(self,
                 in_channels=3,
                 embed_dims=128,
                 patch_embed_kernel=16,
                 num_layers=3,
                 num_heads=8,
                 feedforward_channels=256,
                 norm_cfg=dict(type='LN'),
                 act_cfg=dict(type='GELU'),
                 init_cfg=None):
        super(TransformerSpatialPath, self).__init__(init_cfg)

        self.patch_embed = PatchEmbed(
            in_channels, embed_dims, kernel_size=patch_embed_kernel)
        self.layers = ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                TransformerEncoderLayer(
                    embed_dims,
                    num_heads,
                    feedforward_channels,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
        self.norm = build_norm_layer(norm_cfg, embed_dims)[1]

    def forward(self, img):
        x, hw_shape = self.patch_embed(img)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        B, _, C = x.shape
        x = x.reshape(B, hw_shape[0], hw_shape[1], C).permute(0, 3, 1,
                                                              2).contiguous()
        return x


@SPATIAL_PATH.register_module()
class SpatialPath(BaseModule):
    """Spatial Path to preserve the spatial size of the original input image
    and encode affluent spatial information.

    Args:
        in_channels(int): The number of channels of input
            image. Default: 3.
        num_channels (Tuple[int]): The number of channels of
            each layers in Spatial Path.
            Default: (64, 64, 64, 128).
    Returns:
        x (torch.Tensor): Feature map for Feature Fusion Module.
    """

    def __init__(self,
                 in_channels=3,
                 num_channels=(64, 64, 64, 128),
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 init_cfg=None):
        super(SpatialPath, self).__init__(init_cfg=init_cfg)
        assert len(num_channels) == 4, 'Length of input channels \
                                        of Spatial Path must be 4!'

        self.layers = []
        for i in range(len(num_channels)):
            layer_name = f'layer{i + 1}'
            self.layers.append(layer_name)
            if i == 0:
                self.add_module(
                    layer_name,
                    ConvModule(
                        in_channels=in_channels,
                        out_channels=num_channels[i],
                        kernel_size=7,
                        stride=2,
                        padding=3,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg))
            elif i == len(num_channels) - 1:
                self.add_module(
                    layer_name,
                    ConvModule(
                        in_channels=num_channels[i - 1],
                        out_channels=num_channels[i],
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg))
            else:
                self.add_module(
                    layer_name,
                    ConvModule(
                        in_channels=num_channels[i - 1],
                        out_channels=num_channels[i],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg))

    def forward(self, x):
        for i, layer_name in enumerate(self.layers):
            layer_stage = getattr(self, layer_name)
            x = layer_stage(x)
        return x


@SPATIAL_PATH.register_module()
class DetailBranch(BaseModule):
    """Detail Branch with wide channels and shallow layers to capture low-level
    details and generate high-resolution feature representation.

    Args:
        detail_channels (Tuple[int]): Size of channel numbers of each stage
            in Detail Branch, in paper it has 3 stages.
            Default: (64, 64, 128).
        in_channels (int): Number of channels of input image. Default: 3.
        conv_cfg (dict | None): Config of conv layers.
            Default: None.
        norm_cfg (dict | None): Config of norm layers.
            Default: dict(type='BN').
        act_cfg (dict): Config of activation layers.
            Default: dict(type='ReLU').
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    Returns:
        x (torch.Tensor): Feature map of Detail Branch.
    """

    def __init__(self,
                 detail_channels=(64, 64, 128),
                 in_channels=3,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 init_cfg=None):
        super(DetailBranch, self).__init__(init_cfg=init_cfg)
        detail_branch = []
        for i in range(len(detail_channels)):
            if i == 0:
                detail_branch.append(
                    nn.Sequential(
                        ConvModule(
                            in_channels=in_channels,
                            out_channels=detail_channels[i],
                            kernel_size=3,
                            stride=2,
                            padding=1,
                            conv_cfg=conv_cfg,
                            norm_cfg=norm_cfg,
                            act_cfg=act_cfg),
                        ConvModule(
                            in_channels=detail_channels[i],
                            out_channels=detail_channels[i],
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            conv_cfg=conv_cfg,
                            norm_cfg=norm_cfg,
                            act_cfg=act_cfg)))
            else:
                detail_branch.append(
                    nn.Sequential(
                        ConvModule(
                            in_channels=detail_channels[i - 1],
                            out_channels=detail_channels[i],
                            kernel_size=3,
                            stride=2,
                            padding=1,
                            conv_cfg=conv_cfg,
                            norm_cfg=norm_cfg,
                            act_cfg=act_cfg),
                        ConvModule(
                            in_channels=detail_channels[i],
                            out_channels=detail_channels[i],
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            conv_cfg=conv_cfg,
                            norm_cfg=norm_cfg,
                            act_cfg=act_cfg),
                        ConvModule(
                            in_channels=detail_channels[i],
                            out_channels=detail_channels[i],
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            conv_cfg=conv_cfg,
                            norm_cfg=norm_cfg,
                            act_cfg=act_cfg)))
        self.detail_branch = nn.ModuleList(detail_branch)

    def forward(self, x):
        for stage in self.detail_branch:
            x = stage(x)
        return x


@FFM.register_module()
class TransformerDecoderFeatureFusionLayer(BaseModule):
    """Feature Fusion Module based on Transformer Decoder
    """

    def __init__(self,
                 transformer_decoder_cfg,
                 in_channels=128,
                 embed_dims=256,
                 num_layers=2,
                 patch_embed_kernel=2,
                 norm_cfg=dict(type='LN'),
                 init_cfg=None):
        if init_cfg is None:
            init_cfg = [
                dict(type='Kaiming', layer='Conv2d'),
                dict(
                    type='Constant',
                    val=1,
                    layer=['_BatchNorm', 'GroupNorm', 'LayerNorm'])
            ]
        super().__init__(init_cfg)
        self.patch_embed_spatial = PatchEmbed(
            in_channels, embed_dims, kernel_size=patch_embed_kernel)
        self.patch_embed_context = PatchEmbed(
            in_channels, embed_dims, kernel_size=patch_embed_kernel)
        self.layers = ModuleList()
        for _ in range(num_layers):
            layer = build_transformer_layer(transformer_decoder_cfg)
            self.layers.append(layer)
        self.norm = build_norm_layer(norm_cfg, embed_dims)[1]

        self.conv = ConvModule(
            embed_dims, embed_dims, kernel_size=3, padding=1, stride=1)

    def forward(self, spatial_path, context_path):
        x_spatial, _ = self.patch_embed_spatial(spatial_path)
        x_context, hw_shape = self.patch_embed_context(context_path)
        for i, layer in enumerate(self.layers):
            if i == 0:
                x_query = layer(x_context, x_spatial, x_spatial)
            else:
                x_query = layer(x_query)
        x = self.norm(x_query)
        B, _, C = x.shape
        out = x.reshape(B, hw_shape[0], hw_shape[1],
                        C).permute(0, 3, 1, 2).contiguous()
        out = self.conv(out)
        out = resize(out, scale_factor=2, mode='bilinear', align_corners=False)
        return out


@FFM.register_module()
class FeatureFusionModule(BaseModule):
    """Feature Fusion Module to fuse low level output feature of Spatial Path
    and high level output feature of Context Path.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
    Returns:
        x_out (torch.Tensor): Feature map of Feature Fusion Module.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 init_cfg=None):
        super(FeatureFusionModule, self).__init__(init_cfg=init_cfg)
        self.conv1 = ConvModule(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.conv_atten = nn.Sequential(
            ConvModule(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg), nn.Sigmoid())

    def forward(self, x_sp, x_cp):
        x_concat = torch.cat([x_sp, x_cp], dim=1)
        x_fuse = self.conv1(x_concat)
        x_atten = self.gap(x_fuse)
        # Note: No BN and more 1x1 conv in paper.
        x_atten = self.conv_atten(x_atten)
        x_atten = x_fuse * x_atten
        x_out = x_atten + x_fuse
        return x_out
