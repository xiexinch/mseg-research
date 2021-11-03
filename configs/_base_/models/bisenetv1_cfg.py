# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='BiSeNetV1EXPCFG',
        context_path_cfg=dict(
            type='SimpleContextPath',
            backbone_cfg=dict(
                type='ResNet',
                in_channels=3,
                depth=18,
                num_stages=4,
                out_indices=[2,  3],
                dilations=(1, 1, 1, 1),
                strides=(1, 2, 2, 2),
                norm_cfg=norm_cfg,
                norm_eval=False,
                style='pytorch',
                contract_dilation=True)),
        spatial_path_cfg=dict(
            type='MiTSpatialPath',
            embed_dims=64,
            out_channels=128,
            num_layers=2,
            num_heads=8,
            patch_size=7,
            stride=4,
            sr_ratio=8,
            mlp_ratio=4,
            qkv_bias=True,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.,
            act_cfg=dict(type='GELU'),
            norm_cfg=dict(type='LN', eps=1e-6),
            final_dowsample=True,
            final_attn=False),
        ffm_cfg=dict(
            type='CPMapSPVecFFM',
            transformer_decoder_cfg=dict(
                type='BaseTransformerLayer',
                attn_cfgs=dict(
                    type='MultiheadAttention',
                    embed_dims=128,
                    num_heads=8,
                    attn_drop=0.1),
                ffn_cfgs=dict(
                    type='FFN',
                    embed_dims=128,
                    feedforward_channels=512,
                    num_fcs=2,
                    ffn_drop=0.,
                    act_cfg=dict(type='ReLU', inplace=True)),
                operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                 'ffn', 'norm'),
                batch_first=True),
            in_channels=512,
            embed_dims=128,
            num_layers=1,
            patch_size=1,
            stride=None,
            padding='corner',
            cp_up_rate=4
        ),
        out_indices=(0, 2)),
    decode_head=dict(
        type='FCNHead',
        in_channels=128,
        in_index=0,
        channels=128,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
#     auxiliary_head=[
#     auxiliary_head=dict(
#             type='FCNHead',
#             in_channels=128,
#             channels=64,
#             num_convs=1,
#             num_classes=19,
#             in_index=1,
#             norm_cfg=norm_cfg,
#             concat_input=False,
#             align_corners=False,
#             loss_decode=dict(
#                 type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
        auxiliary_head=dict(
            type='FCNHead',
            in_channels=256,
            channels=64,
            num_convs=1,
            num_classes=19,
            in_index=1,
            norm_cfg=norm_cfg,
            concat_input=False,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
#     ],
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)