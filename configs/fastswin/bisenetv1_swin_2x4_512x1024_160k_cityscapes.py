_base_ = [
    '../_base_/models/bisenetv1_swin-tiny-d32.py',
    '../_base_/datasets/cityscapes.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]


lr_config = dict(warmup='linear', warmup_iters=1000)
optimizer = dict(lr=0.025)
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
)

find_unused_parameters = True
