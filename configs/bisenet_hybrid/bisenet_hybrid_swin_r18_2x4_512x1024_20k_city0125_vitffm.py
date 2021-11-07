_base_ = [
    '../_base_/models/bisenet_hybrid.py', './cityscapes_0125.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_20k.py'
]
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    backbone=dict(
        type='BiSeNetSWContext',
        context_path_cfg=dict(
            type='ContextPath',
            backbone_cfg=dict(
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
                res_norm_cfg=dict(type='BN'),
                num_res_layers=[2, 2],
                swin_act_cfg=dict(type='GELU'),
                swin_norm_cfg=dict(type='LN'),
                with_cp=False,
                init_cfg=None
            ),
            norm_cfg=norm_cfg,
        ),
        ffm_cfg=dict(
            type='TransformerDecoderFFM',
            transformer_decoder_cfg=dict(
                type='BaseTransformerLayer',
                attn_cfgs=dict(
                    type='MultiheadAttention',
                    embed_dims=128,
                    num_heads=4,
                    attn_drop=0.1),
                ffn_cfgs=dict(
                    type='FFN',
                    embed_dims=128,
                    feedforward_channels=512,
                    num_fcs=2,
                    ffn_drop=0.,
                    act_cfg=dict(type='ReLU', inplace=True)),
                operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                 'ffn', 'norm'))
        )
    ),
    decode_head=dict(
        type='FCNHead',
        in_channels=128,
        in_index=2,
        channels=128,
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
            in_channels=128,
            channels=64,
            num_convs=1,
            num_classes=19,
            in_index=1,
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
    ]
)

lr_config = dict(warmup='linear', warmup_iters=1000)
optimizer = dict(lr=0.025)
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
)
