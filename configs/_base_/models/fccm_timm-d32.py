# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='TIMMBackbone',
        pretrained=True,
        features_only=True,
        model_name='resnet50_gn',
        out_indices=(1, 2, 3, 4)
    ),
    decode_head=dict(
        type='FCCMHead',
        in_channels=(512, 2048),
        channels=2560,
        with_fuse_attn=False,
        in_index=(1, 3),
        input_transform='multiple_select',
        num_classes=19,
        dropout_ratio=0.1,
        norm_cfg=norm_cfg,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=1024,
        in_index=2,
        channels=512,
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
