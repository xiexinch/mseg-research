from ..builder import MODELS
from ..backbones.vit import TransformerEncoderLayer as ViTTransformerLayer
from ..backbones.mit import TransformerEncoderLayer as MiTTransformerLayer
from ..backbones.swin import SwinBlock as SwinTransformerLayer

TRANSFORMER_LAYER = MODELS

TRANSFORMER_LAYER.register_module(
    'ViTTransformerLayer', module=ViTTransformerLayer)
TRANSFORMER_LAYER.register_module(
    'MiTTransformerLayer', module=MiTTransformerLayer)
TRANSFORMER_LAYER.register_module(
    'SwinTransformerLayer', module=SwinTransformerLayer)


def build_transformer_layer(cfg):
    return TRANSFORMER_LAYER.build(cfg)
