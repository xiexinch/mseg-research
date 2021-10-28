_base_ = [
    '../_base_/models/bisenetv1_r18-d32.py',
    './cityscapes_0125.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_20k.py'
]

model = dict(backbone=dict(
    type='BiSeNetV1EXPCFG',
    spatial_path_cfg=dict(
        type='ShiftWindowTransformerSpatialPath',
        patch_embed_cfg=dict(
            in_channels=3,
            embed_dims=64,
            kernel_size=4,
        ),
        num_heads=4,
        mlp_ratio=4,
        attn_drop_rate=0.,
        drop_path_rate=0.,
        norm_cfg=dict(type='LN'),
        act_cfg=dict(type='GELU'),
        final_downsample=True,
        init_cfg=None)))

lr_config = dict(warmup='linear', warmup_iters=1000)
optimizer = dict(lr=0.025)
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
)
