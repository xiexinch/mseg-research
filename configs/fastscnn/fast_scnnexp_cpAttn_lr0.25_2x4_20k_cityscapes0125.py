_base_ = [
    '../_base_/models/fast_scnn.py', '../_base_/datasets/cityscapes.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_20k.py'
]

model = dict(
    backbone=dict(
        type='FastSCNNEXP',
        aspp_dilations=(1, 1, 1),
        spatial_self_attn=False,
        context_self_attn=True,
    )
)

# Re-config the data sampler.
data = dict(samples_per_gpu=4, workers_per_gpu=4)

# Re-config the optimizer.
optimizer = dict(type='SGD', lr=0.25, momentum=0.9, weight_decay=4e-5)
