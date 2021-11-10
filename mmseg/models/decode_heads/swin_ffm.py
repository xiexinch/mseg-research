import torch
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_norm_layer
from mmcv.runner import BaseModule, ModuleList
from mmcv.cnn.bricks.transformer import build_dropout, FFN
from mmcv.utils import to_2tuple


from mmseg.models.utils import PatchEmbed, nchw_to_nlc, nlc_to_nchw

from mmseg.models.builder import HEADS
from mmseg.models.backbones.swin import WindowMSA, ShiftWindowMSA
from mmseg.ops import resize

from mmseg.models.decode_heads.decode_head import BaseDecodeHead


@HEADS.register_module()
class Swin_FFM(BaseDecodeHead):

    def __init__(
            self,
            in_channels=(64, 256),
            embed_dims=256,
            num_heads=4,
            mlp_ratial=4,
            window_size=7,
            qkv_bias=True,
            qk_scale=0.,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.,
            act_cfg=dict(type='GELU'),
            norm_cfg=dict(type='LN'),
            conv_norm_cfg=dict(type='BN'),
            **kwargs):

        super(Swin_FFM, self).__init__(in_channels, **kwargs)

        self.fuse_layers = ModuleList()
        for i in range(3):
            layer = SwinDecoderBlock(
                embed_dims=embed_dims,
                num_heads=num_heads,
                feedforward_channels=mlp_ratial * embed_dims,
                window_size=window_size,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=drop_path_rate,
                act_cfg=act_cfg,
                norm_cfg=norm_cfg,
                init_cfg=None
            )
            self.fuse_layers.append(layer)

        # overlap patch embed
        _downsample_cfg = [
            (7, 4, 2),
            (3, 2, 1),
            (1, 1, 'corner')
        ]
        self.spatial_downsamples = ModuleList()
        for i in range(3):
            downsample = PatchEmbed(
                in_channels=in_channels[0],
                embed_dims=in_channels[-1],
                kernel_size=_downsample_cfg[i][0],
                stride=_downsample_cfg[i][1],
                padding=_downsample_cfg[i][2]
            )
            self.spatial_downsamples.append(downsample)

        self.upsample_convs = ModuleList()
        for i in range(3):
            self.upsample_convs.append(
                ConvModule(
                    in_channels=embed_dims,
                    out_channels=embed_dims,
                    kernel_size=1,
                    norm_cfg=conv_norm_cfg)
            )

    def forward(self, inputs):
        x = self._transform_inputs(inputs)
        assert len(x) == 2
        x_spatial, x = x[0], x[1]
        for i in range(3):
            x_spatial_, hw_shape = self.spatial_downsamples[i](x_spatial)
            x = nchw_to_nlc(x)
            x = self.fuse_layers[i](
                x, x_spatial_, x_spatial_, hw_shape)
            x = nlc_to_nchw(x, hw_shape)
            x = self.upsample_convs[i](x)
            x = resize(x, scale_factor=2, mode='bilinear')
        x = self.conv_seg(x)
        return x


class WindowCrossMSA(WindowMSA):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.qkv = None

    def forward(self, q, k, v, mask=None):
        B, N, C = q.shape

        qkv = torch.cat([q, k, v], dim=2).reshape(B, N, 3, self.num_heads,
                                                  C // self.num_heads).permute(
                                                      2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1],
                self.window_size[0] * self.window_size[1],
                -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B // nW, nW, self.num_heads, N,
                             N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class ShiftWindowCrossMSA(ShiftWindowMSA):

    def __init__(self,
                 embed_dims,
                 num_heads,
                 window_size,
                 shift_size=0,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop_rate=0,
                 proj_drop_rate=0,
                 dropout_layer=dict(type='DropPath', drop_prob=0.),
                 init_cfg=None):
        super().__init__(
            embed_dims=embed_dims,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=0,
            qkv_bias=True,
            qk_scale=None,
            attn_drop_rate=0,
            proj_drop_rate=0,
            dropout_layer=dict(type='DropPath', drop_prob=0.),
            init_cfg=init_cfg)

        self.window_size = window_size
        self.shift_size = shift_size
        assert 0 <= self.shift_size < self.window_size

        self.w_msa = WindowCrossMSA(
            embed_dims=embed_dims,
            num_heads=num_heads,
            window_size=to_2tuple(window_size),
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=proj_drop_rate,
            init_cfg=None)

        self.drop = build_dropout(dropout_layer)

    def forward(self, query, key, value, hw_shape, key_shape=None):
        B, qL, C = query.shape
        _, kL, _ = key.shape
        H, W = hw_shape
        if key_shape is not None:
            kH, kW = key_shape
        else:
            kH, kW = H, W
        assert qL == H * W, 'input feature has wrong size'
        query = query.view(B, H, W, C)
        key = key.view(B, kH, kW, C)
        value = value.view(B, kH, kW, C)

        # pad feature maps to multiples of window size
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        pad_kr = (self.window_size - kW % self.window_size) % self.window_size
        pad_kb = (self.window_size - kH % self.window_size) % self.window_size

        query = F.pad(query, (0, 0, 0, pad_r, 0, pad_b))
        key = F.pad(key, (0, 0, 0, pad_kr, 0, pad_kb))
        value = F.pad(value, (0, 0, 0, pad_kr, 0, pad_kb))

        H_pad, W_pad = query.shape[1], query.shape[2]

        attn_mask = None

        # nW*B, window_size, window_size, C
        query_windows = self.window_partition(query)
        key_windows = self.window_partition(key)
        value_windows = self.window_partition(value)
        # nW*B, window_size*window_size, C
        query_windows = query_windows.view(-1, self.window_size**2, C)
        key_windows = key_windows.view(-1, self.window_size**2, C)
        value_windows = value_windows.view(-1, self.window_size**2, C)

        # W-MSA/SW-MSA (nW*B, window_size*window_size, C)
        attn_windows = self.w_msa(
            query_windows, key_windows, value_windows, mask=attn_mask)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size,
                                         self.window_size, C)

        # B H' W' C
        x = self.window_reverse(attn_windows, H_pad, W_pad)

        if pad_r > 0 or pad_b:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        x = self.drop(x)
        return x


class SwinDecoderBlock(BaseModule):

    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 window_size=7,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 with_cp=False,
                 init_cfg=None):
        super(SwinDecoderBlock, self).__init__(init_cfg)

        self.with_cp = with_cp

        self.attn = ShiftWindowMSA(
            embed_dims=embed_dims,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=window_size // 2,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            init_cfg=None)
        self.norm1 = build_norm_layer(norm_cfg, embed_dims)[1]

        self.cross_attn = ShiftWindowCrossMSA(
            embed_dims=embed_dims,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=0,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            init_cfg=None)
        self.norm2 = build_norm_layer(norm_cfg, embed_dims)[1]

        self.ffn = FFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            num_fcs=2,
            ffn_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            act_cfg=act_cfg,
            add_identity=True,
            init_cfg=None)
        self.norm3 = build_norm_layer(norm_cfg, embed_dims)[1]

    def forward(self, x, k, v, hw_shape, key_shape=None):
        identity = x
        x = self.norm1(x)
        x = self.attn(x, hw_shape)
        x = x + identity
        identity = x
        x = self.norm2(x)
        x = self.cross_attn(x, k, v, hw_shape, key_shape)
        x = x + identity
        identity = x
        x = self.norm3(x)
        x = self.ffn(x, identity=identity)
        return x
