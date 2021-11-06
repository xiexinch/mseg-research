_base_ = [
    '../_base_/models/bisenet_hybrid.py', './cityscapes_0125.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_20k.py'
]

model = dict(
    backbone=dict(
        ffm_cfg=dict(
            type='CPVecSPMapFFMReverse',
        )
    )
)

lr_config = dict(warmup='linear', warmup_iters=1000)
optimizer = dict(lr=0.025)
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
)
