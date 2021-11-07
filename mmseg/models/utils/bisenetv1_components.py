from mmcv.cnn.bricks import conv
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn.bricks.norm import build_norm_layer
from mmcv.cnn.bricks.transformer import build_transformer_layer
from mmcv.runner import BaseModule, ModuleList
from mmcv.cnn import ConvModule, Linear

from mmseg.models.builder import MODELS, build_backbone
from mmseg.models.utils.shape_convert import nchw_to_nlc
from mmseg.ops.wrappers import resize
from ..utils.embed import PatchMerging
from ..utils import PatchEmbed, nlc_to_nchw
from ..backbones.vit import TransformerEncoderLayer
from ..backbones.swin import SwinBlock
from ..backbones.mit import TransformerEncoderLayer as MiTTransformerLayer

CONTEXT_PATH = MODELS
SPATIAL_PATH = MODELS
FFM = MODELS


def build_context_path(cfg):
    return CONTEXT_PATH.build(cfg)


def build_spatial_path(cfg):
    return SPATIAL_PATH.build(cfg)


def build_ffm(cfg):
    return FFM.build(cfg)


@SPATIAL_PATH.register_module()
class MiTSpatialPath(BaseModule):
    """First stage of MixVisionTransformer
    """

    def __init__(self,
                 embed_dims=64,
                 out_channels=128,
                 num_layers=3,
                 num_heads=1,
                 patch_size=7,
                 stride=4,
                 sr_ratio=8,
                 mlp_ratio=4,
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN', eps=1e-6),
                 final_dowsample=True,
                 final_attn=False,
                 init_cfg=None):
        super(MiTSpatialPath, self).__init__(init_cfg)
        self.patch_embed = PatchEmbed(
            in_channels=3,
            embed_dims=embed_dims,
            kernel_size=patch_size,
            stride=stride,
            padding=patch_size // 2,
            norm_cfg=norm_cfg)

        dpr = torch.linspace(0, drop_path_rate, num_layers)
        self.layers = ModuleList()
        for i in range(num_layers):
            layer = MiTTransformerLayer(
                embed_dims=embed_dims,
                num_heads=num_heads,
                feedforward_channels=mlp_ratio * embed_dims,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=dpr[i],
                qkv_bias=qkv_bias,
                act_cfg=act_cfg,
                norm_cfg=norm_cfg,
                sr_ratio=sr_ratio)
            self.layers.append(layer)

        self.final_downsample = final_dowsample
        self.final_attn = final_attn
        if self.final_downsample:
            # dowsample 2x
            self.downsample = PatchMerging(embed_dims, out_channels)
            if final_attn:
                self.final_attn = MiTTransformerLayer(
                    embed_dims=out_channels,
                    num_heads=2,
                    feedforward_channels=mlp_ratio * out_channels,
                    drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate,
                    drop_path_rate=drop_path_rate,
                    qkv_bias=qkv_bias,
                    act_cfg=act_cfg,
                    norm_cfg=norm_cfg,
                    sr_ratio=sr_ratio // 2)
        else:
            self.project = Linear(embed_dims, out_channels)

    def forward(self, x):
        x, hw_shape = self.patch_embed(x)
        for layer in self.layers:
            x = layer(x, hw_shape)
        if self.final_downsample:
            x, hw_shape = self.downsample(x, hw_shape)
            if self.final_attn:
                x = self.final_attn(x, hw_shape)
        else:
            x = self.project(x)
        return x


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
        # TODO segformer
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
class CPVecSPMapFFM(BaseModule):

    def __init__(
        self,
        transformer_decoder_cfg,
        in_channels=128,
        embed_dims=256,
        num_layers=1,
        patch_size=1,
        stride=None,
        padding='corner',
        norm_cfg=dict(type='BN'),
        final_upsample=True,
        final_fuse=False,
        init_cfg=None
    ):
        super().__init__(init_cfg)
        self.patch_embed = PatchEmbed(
            in_channels,
            embed_dims,
            kernel_size=patch_size,
            stride=stride,
            padding=padding)
        self.layers = ModuleList()
        for _ in range(num_layers):
            layer = build_transformer_layer(transformer_decoder_cfg)
            self.layers.append(layer)

        self.final_upsample = final_upsample
        if self.final_upsample:
            self.up_conv = ConvModule(
                embed_dims, embed_dims, 3, padding=1, norm_cfg=norm_cfg)
        # self.final_fuse = final_fuse

    def forward(self, spatial_path, context_path):
        x_spatial, hw_shape = self.patch_embed(spatial_path)
        x_context = nchw_to_nlc(context_path)
        for i, layer in enumerate(self.layers):
            if i == 0:
                x = layer(x_spatial, x_context, x_context)
            else:
                x = layer(x)
        x = nlc_to_nchw(x, hw_shape)
        if self.final_upsample:
            x_16 = self.up_conv(x)
            x = resize(x_16, scale_factor=2, mode='bilinear')
            # if self.final_fuse:
            #     x = spatial_path * F.sigmoid(x)

        return x


@FFM.register_module()
class CPVecSPMapFFMReverse(BaseModule):

    def __init__(
        self,
        transformer_decoder_cfg,
        in_channels=128,
        embed_dims=256,
        num_layers=1,
        patch_size=1,
        stride=None,
        padding='corner',
        norm_cfg=dict(type='BN'),
        final_upsample=True,
        final_fuse=False,
        cp_up_rate=2,
        init_cfg=None
    ):
        super().__init__(init_cfg)
        self.patch_embed = PatchEmbed(
            in_channels,
            embed_dims,
            kernel_size=patch_size,
            stride=stride,
            padding=padding)
        self.layers = ModuleList()
        for _ in range(num_layers):
            layer = build_transformer_layer(transformer_decoder_cfg)
            self.layers.append(layer)

        self.final_upsample = final_upsample
        if self.final_upsample:
            self.up_conv = ConvModule(
                embed_dims, embed_dims, 3, padding=1, norm_cfg=norm_cfg)
        # self.final_fuse = final_fuse
        self.cp_up_rate = cp_up_rate

    def forward(self, spatial_path, context_path):
        x_spatial, hw_shape = self.patch_embed(spatial_path)
        x_context = nchw_to_nlc(context_path)
        for i, layer in enumerate(self.layers):
            if i == 0:
                x = layer(x_context, x_spatial, x_spatial)
            else:
                x = layer(x)
        x = nlc_to_nchw(x, [l // self.cp_up_rate for l in hw_shape])
        if self.final_upsample:
            x_16 = self.up_conv(x)
            x = resize(x_16, scale_factor=2, mode='bilinear')
            # if self.final_fuse:
            #     x = spatial_path * F.sigmoid(x)

        return x


@FFM.register_module()
class CPMapSPVecFFM(BaseModule):

    def __init__(self,
                 transformer_decoder_cfg,
                 in_channels=128,
                 embed_dims=256,
                 num_layers=1,
                 patch_size=1,
                 stride=None,
                 padding='corner',
                 cp_up_rate=4,
                 init_cfg=None):
        super().__init__(init_cfg)
        # patch embed for context path feature
        self.patch_embed = PatchEmbed(
            in_channels,
            embed_dims,
            kernel_size=patch_size,
            stride=stride,
            padding=padding)

        self.layers = ModuleList()
        for _ in range(num_layers):
            layer = build_transformer_layer(transformer_decoder_cfg)
            self.layers.append(layer)

        self.cp_up_rate = cp_up_rate

    def forward(self, spatial_path, context_path):
        x_context, hw_shape = self.patch_embed(context_path)
        for i, layer in enumerate(self.layers):
            if i == 0:
                x = layer(spatial_path, x_context, x_context)
            else:
                x = layer(x)
        return nlc_to_nchw(x, [i * self.cp_up_rate for i in hw_shape])


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
class TransformerDecoderFFM(BaseModule):
    """Feature Fusion Module based on Transformer Decoder
    """

    def __init__(self,
                 transformer_decoder_cfg,
                 num_layers=2,
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

        self.layers = ModuleList()
        for _ in range(num_layers):
            layer = build_transformer_layer(transformer_decoder_cfg)
            self.layers.append(layer)

    def forward(self, spatial_path, context_path):
        hw_shape = spatial_path.shape[2:]
        x_spatial = nchw_to_nlc(spatial_path)
        x_context = nchw_to_nlc(context_path)
        for i, layer in enumerate(self.layers):
            if i == 0:
                x = layer(x_context, x_spatial, x_spatial)
            else:
                x = layer(x)

        B, _, C = x.shape
        out = x.reshape(B, hw_shape[0], hw_shape[1],
                        C).permute(0, 3, 1, 2).contiguous()

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


@FFM.register_module()
class BaseTransformerDecoder(BaseModule):

    def __init__(self,
                 transformer_decoder_cfg,
                 context_in_channels,
                 spatial_in_channels,
                 embed_dims,
                 spatial_dw_rate,
                 context_dw_rate,
                 num_layers=1,
                 init_cfg=None):
        super(BaseTransformerDecoder, self).__init__(init_cfg)
        # query
        self.spatial_patch_embed = PatchEmbed(
            spatial_in_channels, embed_dims, kernel_size=spatial_dw_rate)
        # key, value
        self.context_patch_embed = PatchEmbed(
            context_in_channels, embed_dims, kernel_size=context_dw_rate)

        # TransformerDecoderLayer
        self.layers = ModuleList()
        for _ in range(num_layers):
            layer = build_transformer_layer(transformer_decoder_cfg)
            self.layers.append(layer)

    def forward(self, spatial_path, context_path):
        x_spatial, hw_shape = self.spatial_patch_embed(spatial_path)
        x_context, _ = self.context_patch_embed(context_path)
        for i, layer in enumerate(self.layers):
            if i == 0:
                x_query = layer(x_spatial, x_context, x_context)
            else:
                x_query = layer(x_query)

        # reshape
        B, _, C = x_query.shape
        out = x_query.reshape(B, hw_shape[0], hw_shape[1],
                              C).permute(0, 3, 1, 2).contiguous()
        return out


@CONTEXT_PATH.register_module()
class SimpleContextPath(BaseModule):

    def __init__(self, backbone_cfg, init_cfg=None):
        super(SimpleContextPath, self).__init__(init_cfg)
        self.backbone = build_backbone(backbone_cfg)

    def forward(self, x):
        return self.backbone(x)


@CONTEXT_PATH.register_module()
class ContextPath(BaseModule):
    """Context Path to provide sufficient receptive field.

    Args:
        backbone_cfg:(dict): Config of backbone of
            Context Path.
        context_channels (Tuple[int]): The number of channel numbers
            of various modules in Context Path.
            Default: (128, 256, 512).
        align_corners (bool, optional): The align_corners argument of
            resize operation. Default: False.
    Returns:
        x_16_up, x_32_up (torch.Tensor, torch.Tensor): Two feature maps
            undergoing upsampling from 1/16 and 1/32 downsampling
            feature maps. These two feature maps are used for Feature
            Fusion Module and Auxiliary Head.
    """

    def __init__(self,
                 backbone_cfg,
                 context_channels=(128, 256, 512),
                 align_corners=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 init_cfg=None):
        super(ContextPath, self).__init__(init_cfg=init_cfg)
        assert len(context_channels) == 3, 'Length of input channels \
                                           of Context Path must be 3!'

        self.backbone = build_backbone(backbone_cfg)
        if hasattr(self.backbone, 'train'):
            self.backbone.train()

        self.align_corners = align_corners
        self.arm16 = AttentionRefinementModule(context_channels[1],
                                               context_channels[0])
        self.arm32 = AttentionRefinementModule(context_channels[2],
                                               context_channels[0])
        self.conv_head32 = ConvModule(
            in_channels=context_channels[0],
            out_channels=context_channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.conv_head16 = ConvModule(
            in_channels=context_channels[0],
            out_channels=context_channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.gap_conv = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            ConvModule(
                in_channels=context_channels[2],
                out_channels=context_channels[0],
                kernel_size=1,
                stride=1,
                padding=0,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg))

    def forward(self, x):
        x_4, x_8, x_16, x_32 = self.backbone(x)
        x_gap = self.gap_conv(x_32)

        x_32_arm = self.arm32(x_32)
        x_32_sum = x_32_arm + x_gap
        x_32_up = resize(input=x_32_sum, size=x_16.shape[2:], mode='nearest')
        x_32_up = self.conv_head32(x_32_up)

        x_16_arm = self.arm16(x_16)
        x_16_sum = x_16_arm + x_32_up
        x_16_up = resize(input=x_16_sum, size=x_8.shape[2:], mode='nearest')
        x_16_up = self.conv_head16(x_16_up)

        return x_8, x_16_up, x_32_up


class AttentionRefinementModule(BaseModule):
    """Attention Refinement Module (ARM) to refine the features of each stage.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
    Returns:
        x_out (torch.Tensor): Feature map of Attention Refinement Module.
    """

    def __init__(self,
                 in_channels,
                 out_channel,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 init_cfg=None):
        super(AttentionRefinementModule, self).__init__(init_cfg=init_cfg)
        self.conv_layer = ConvModule(
            in_channels=in_channels,
            out_channels=out_channel,
            kernel_size=3,
            stride=1,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.atten_conv_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            ConvModule(
                in_channels=out_channel,
                out_channels=out_channel,
                kernel_size=1,
                bias=False,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=None), nn.Sigmoid())

    def forward(self, x):
        x = self.conv_layer(x)
        x_atten = self.atten_conv_layer(x)
        x_out = x * x_atten
        return x_out
