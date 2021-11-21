_base_ = [
    '../_base_/models/fccm_mobilenetv2-d32.py', '../_base_/datasets/cityscapes.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]

model = dict(decode_head=dict(type='FCCMHead_EXT'))

lr_config = dict(warmup='linear', warmup_iters=1000)
# optimizer = dict(lr=0.025)

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
)
