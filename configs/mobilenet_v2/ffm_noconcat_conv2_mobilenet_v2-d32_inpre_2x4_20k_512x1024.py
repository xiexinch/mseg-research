_base_ = [
    '../_base_/models/ffm_mobilenetv2-d32.py', './cityscapes_0125.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_20k.py'
]

model = dict(decode_head=dict(
    num_convs=2,
    concat_input=False
))

lr_config = dict(warmup='linear', warmup_iters=1000)
# optimizer = dict(lr=0.025)

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
)
