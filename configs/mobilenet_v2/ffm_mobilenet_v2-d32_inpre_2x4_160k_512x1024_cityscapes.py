_base_ = [
    '../_base_/models/ffm_mobilenetv2-d32.py',
    '../_base_/datasets/cityscapes.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py'
]

lr_config = dict(warmup='linear', warmup_iters=1000)

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
)
