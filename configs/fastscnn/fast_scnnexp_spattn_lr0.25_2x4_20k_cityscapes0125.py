_base_ = [
    '../_base_/models/fast_scnn.py', '../_base_/datasets/cityscapes.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_20k.py'
]

model = dict(
    backbone=dict(
        type='FastSCNNEXP',
        spatial_self_attn=True,
        context_self_attn=False,
    )
)

# Re-config the data sampler.
data = dict(samples_per_gpu=4, workers_per_gpu=4)

# Re-config the optimizer.
optimizer = dict(type='SGD', lr=0.25, momentum=0.9, weight_decay=4e-5)
