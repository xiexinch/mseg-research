_base_ = [
    '../_base_/models/bisenetv1_r18-d32.py', './cityscapes_0125.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]

model = dict(
    backbone=dict(
        type='BiSeNetV1EXPCFG',
        spatial_path_cfg=dict(type='SpatialPath', norm_cfg=dict(type='BN')),
        ffm_cfg=dict(
            type='TransformerDecoderFeatureFusionLayer',
            transformer_decoder_cfg=dict(
                type='BaseTransformerLayer',
                attn_cfgs=dict(
                    type='MultiheadAttention',
                    embed_dims=256,
                    num_heads=8,
                    attn_drop=0.1),
                feedforward_channels=512,  # need ablation study
                ffn_dropout=0.1,
                #operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm') detr
                operation_order=('self_attn', 'norm', 'ffn', 'norm')),
            in_channels=128,
            embed_dims=256)))

lr_config = dict(warmup='linear', warmup_iters=1000)
optimizer = dict(lr=0.025)
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
)
