# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='BiSeNetSWCFG',
        backbone_cfg=dict(
            type='HybridTransformer',
            in_channels=3,
            out_indices=(1, 3),
            base_channels=64,
            embed_dims=128,
            patch_embed_cfg=dict(
                conv_type='Conv2d',
                kernel_size=2,
                stride=2,
                padding='corner',
                dilation=1,
                bias=True,
                norm_cfg=None
            ),
            num_res_stages=2,
            num_res_layers=[2, 2],
            num_transformer_layers=[2, 2],
            num_heads=[8, 8],
            mlp_ratio=4,
            res_norm_cfg=norm_cfg,
            patch_merging_cfg=dict(
                kernel_size=2,
                stride=None,
                padding='corner',
                dilation=1,
                bias=False,
                norm_cfg=dict(type='LN')
            ),
            transformer_cfg=dict(
                drop_rate=0.,
                attn_drop_rate=0.1,
                drop_path_rate=0.1,
                num_fcs=2,
                qkv_bias=True,
                act_cfg=dict(type='GELU'),
                norm_cfg=dict(type='LN'),
                batch_first=True
            )
        ),
        ffm_cfg=dict(
            type='CPVecSPMapFFM',
            transformer_decoder_cfg=dict(
                type='BaseTransformerLayer',
                attn_cfgs=dict(
                    type='MultiheadAttention',
                    embed_dims=256,
                    num_heads=8,
                    attn_drop=0.1),
                ffn_cfgs=dict(
                    type='FFN',
                    embed_dims=256,
                    feedforward_channels=1024,
                    num_fcs=2,
                    ffn_drop=0.,
                    act_cfg=dict(type='ReLU', inplace=True)),
                operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                 'ffn', 'norm'),
                batch_first=True),
            in_channels=128,
            embed_dims=256,
            num_layers=2,
            patch_size=3,
            stride=2,
            padding='corner',
            norm_cfg=norm_cfg,
            final_upsample=True)
    ),
    decode_head=dict(
        type='FCNHead',
        in_channels=256,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
    ),
    auxiliary_head=[
        dict(
            type='FCNHead',
            in_channels=256,
            channels=64,
            num_convs=1,
            num_classes=19,
            in_channels=1,
            norm_cfg=norm_cfg,
            concat_input=False,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
        dict(
            type='FCNHead',
            in_channels=128,
            channels=64,
            num_convs=1,
            num_classes=19,
            in_index=0,
            norm_cfg=norm_cfg,
            concat_input=False,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4))
    ],
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)
