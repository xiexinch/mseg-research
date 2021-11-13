# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='HybridSwinResNet',
        in_channels=3,
        embed_dims=64,
        embed_cfg=None,
        out_indices=(0, 1, 2, 3),
        window_size=7,
        num_heads=[4, 8],
        mlp_ratio=4,
        depths=[2, 2],
        patch_norm=True,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.1,
        res_norm_cfg=norm_cfg,
        num_res_layers=[2, 2],
        swin_act_cfg=dict(type='GELU'),
        swin_norm_cfg=dict(type='LN'),
        with_cp=False,
        init_cfg=None
    ),
    decode_head=dict(
        type='UpsampleFFMHead',
        in_channels=(128, 256, 512),
        channels=512,
        in_index=(1, 2, 3),
        input_transform='multiple_select',
        num_classes=19,
        norm_cfg=norm_cfg,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=256,
        channels=64,
        num_convs=2,
        num_classes=19,
        in_index=2,
        norm_cfg=norm_cfg,
        concat_input=False,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
