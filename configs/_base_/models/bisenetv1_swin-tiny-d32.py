# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
backbone_norm_cfg = dict(type='LN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='BiSeNetV1',
        in_channels=3,
        context_channels=(192, 384, 768),
        spatial_channels=(64, 64, 64, 192),
        out_indices=(0, 1, 2),
        out_channels=256,
        backbone_cfg=dict(
            type='SwinTransformer',
            pretrain_img_size=224,
            embed_dims=96,
            patch_size=4,
            window_size=7,
            mlp_ratio=4,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            strides=(4, 2, 2, 2),
            out_indices=(0, 1, 2, 3),
            qkv_bias=True,
            qk_scale=None,
            patch_norm=True,
            drop_rate=0.,
            attn_drop_rate=0.3,
            use_abs_pos_embed=False,
            act_cfg=dict(type='GELU'),
            norm_cfg=backbone_norm_cfg,
            frozen_stages=2),
        norm_cfg=norm_cfg,
        align_corners=False,
        init_cfg=None),
    decode_head=dict(
        type='FCNHead',
        in_channels=256,
        in_index=0,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=[
        dict(
            type='FCNHead',
            in_channels=192,
            channels=64,
            num_convs=1,
            num_classes=19,
            in_index=1,
            norm_cfg=norm_cfg,
            concat_input=False,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
        dict(
            type='FCNHead',
            in_channels=192,
            channels=64,
            num_convs=1,
            num_classes=19,
            in_index=2,
            norm_cfg=norm_cfg,
            concat_input=False,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    ],
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
