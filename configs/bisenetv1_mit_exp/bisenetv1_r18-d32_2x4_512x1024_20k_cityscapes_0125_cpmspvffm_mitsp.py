_base_ = [
    '../_base_/models/bisenetv1_r18-d32.py', './cityscapes_0125.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_20k.py'
]

model = dict(
    backbone=dict(
        type='BiSeNetV1EXPCFG',
        context_path_cfg=dict(
            type='SimpleContextPath',
            backbone_cfg=dict(
                type='ResNet',
                in_channels=3,
                depth=18,
                num_stages=4,
                out_indices=[1, 2, 3],
                dilations=(1, 1, 1, 1),
                strides=(1, 2, 2, 2),
                norm_cfg=dict(type='BN', requires_grad=True),
                norm_eval=False,
                style='pytorch',
                contract_dilation=True)),
        spatial_path_cfg=dict(
            type='MiTSpatialPath',
            embed_dims=64,
            out_channels=512,
            num_layers=3,
            num_heads=1,
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
                    embed_dims=512,
                    num_heads=8,
                    attn_drop=0.1),
                ffn_cfgs=dict(
                    type='FFN',
                    embed_dims=512,
                    feedforward_channels=1024,
                    num_fcs=2,
                    ffn_drop=0.,
                    act_cfg=dict(type='ReLU', inplace=True)),
                operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                 'ffn', 'norm'),
                batch_first=True),
            in_channels=512,
            embed_dims=512,
            num_layers=1,
            patch_size=1,
            stride=None,
            padding='corner',
            cp_up_rate=4
        )))

lr_config = dict(warmup='linear', warmup_iters=1000)
optimizer = dict(lr=0.025)
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
)
