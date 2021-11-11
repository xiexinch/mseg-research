_base_ = [
    '../_base_/models/swinffm_hybridswin-d32.py', './cityscapes_0125.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_20k.py'
]

model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='SimpleBiSeNet',
        backbone_cfg=dict(
            type='ResNet',
            in_channels=3,
            depth=18,
            stem_channels=64,
            base_channels=64,
            num_stages=4,
            out_indices=[3],
            dilations=(1, 1, 1, 1),
            strides=(1, 2, 2, 2),
            norm_eval=False,
            style='pytorch',
            contract_dilation=True),
        spatial_path_cfg=dict(
            type='ShiftWindowTransformerSpatialPath',
            patch_embed_cfg=dict(
                in_channels=3,
                embed_dims=256,
                kernel_size=4),
            num_heads=4,
            mlp_ratio=4,
            attn_drop_rate=0.,
            drop_path_rate=0.,
            norm_cfg=dict(type='LN'),
            act_cfg=dict(type='GELU'),
            final_downsample=True)
    ),
    decode_head=dict(
        type='Swin_FFM',
        in_channels=(512, 512),
        embed_dims=512,
        channels=512,
        in_index=(0, 1)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=512,
        channels=64,
        num_convs=2,
        num_classes=19,
        in_index=0)
)

lr_config = dict(warmup='linear', warmup_iters=1000)
optimizer = dict(lr=0.025)
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
)
