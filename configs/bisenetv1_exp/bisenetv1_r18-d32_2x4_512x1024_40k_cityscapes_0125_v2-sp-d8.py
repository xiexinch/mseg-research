_base_ = [
    '../_base_/models/bisenetv1_r18-d32.py',
    './cityscapes_0125.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]

model = dict(backbone=dict(
    type='BiSeNetV1EXPCFG',
    spatial_path_cfg=dict(
        type='DetailBranch'),
    ffm_cfg=dict(
        type='FeatureFusionModule',
        in_channels=256,
        out_channels=256
    )))

lr_config = dict(warmup='linear', warmup_iters=1000)
optimizer = dict(lr=0.025)
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
)
