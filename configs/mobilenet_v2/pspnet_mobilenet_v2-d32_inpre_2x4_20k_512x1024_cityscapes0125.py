_base_ = [
    '../_base_/models/pspnet_mobilenet_v2-d32.py', './cityscapes_0125.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_20k.py'
]

lr_config = dict(warmup='linear', warmup_iters=1000)
# optimizer = dict(lr=0.025)

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
)
