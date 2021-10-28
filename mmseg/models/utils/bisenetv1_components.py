from mmcv.runner import BaseModule, ModuleList
from ..utils import PatchEmbed
from ..backbones.vit import TransformerEncoderLayer


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

    def forward(self, img):
        x, hw_shape = self.patch_embed(img)
        for layer in self.layers:
            x = layer(x)
        B, _, C = x.shape
        x = x.reshape(B, hw_shape[0], hw_shape[1], C).permute(0, 3, 1,
                                                              2).contiguous()
        return x
