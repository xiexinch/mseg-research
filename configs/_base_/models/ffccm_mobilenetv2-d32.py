# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='MobileNetV2',
        widen_factor=1.,
        strides=(1, 2, 2, 2, 1, 2, 1),
        dilations=(1, 1, 1, 1, 1, 1, 1),
        out_indices=(1, 2, 4, 6),
        init_cfg=dict(type='Pretrained', checkpoint='mmcls://mobilenet_v2')),
    decode_head=dict(
        type='FFCCMHead',
        in_channels=(32, 320),
        channels=352,
        with_fuse_attn=True,
        in_index=(1, 3),
        input_transform='multiple_select',
        num_classes=19,
        dropout_ratio=0.1,
        norm_cfg=norm_cfg,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=96,
        in_index=2,
        channels=64,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
