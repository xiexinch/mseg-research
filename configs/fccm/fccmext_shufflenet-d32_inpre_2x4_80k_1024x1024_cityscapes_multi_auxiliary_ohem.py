_base_ = [
    '../_base_/datasets/cityscapes_1024x1024.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='ShuffleNetV2',
        out_indices=(0, 1, 2, 3),
        init_cfg=dict(type='Pretrained',
                      checkpoint='open-mmlab://shufflenet_v2')
    ),
    decode_head=dict(
        type='FCCMHead_EXT',
        in_channels=(232, 1024),
        channels=1024,
        with_fuse_attn=True,
        in_index=(1, 3),
        input_transform='multiple_select',
        num_classes=19,
        dropout_ratio=0.1,
        norm_cfg=norm_cfg,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=[
        dict(
            type='FCNHead',
            in_channels=116,
            channels=128,
            num_convs=2,
            num_classes=19,
            in_index=0,
            norm_cfg=norm_cfg,
            concat_input=False,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
        dict(
            type='FCNHead',
            in_channels=232,
            channels=256,
            num_convs=2,
            num_classes=19,
            in_index=1,
            norm_cfg=norm_cfg,
            concat_input=False,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
        dict(
            type='FCNHead',
            in_channels=464,
            channels=512,
            num_convs=2,
            num_classes=19,
            in_index=2,
            norm_cfg=norm_cfg,
            concat_input=False,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
        dict(
            type='FCNHead',
            in_channels=1024,
            channels=1024,
            num_convs=2,
            num_classes=19,
            in_index=3,
            norm_cfg=norm_cfg,
            concat_input=False,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    ]
)

sampler = dict(type='OHEMPixelSampler', thresh=0.7, min_kept=10000)
lr_config = dict(warmup='linear', warmup_iters=1000)
optimizer = dict(lr=0.05)
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
)
