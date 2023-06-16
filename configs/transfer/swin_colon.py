# Only for evaluation
_base_ = [
    'mmpretrain::_base_/models/swin_transformer/base_384.py',
    '../datasets/colon.py',
    '../swin_schedule.py',
    '../custom_imports.py', 
    'mmpretrain::_base_/default_runtime.py',
]


model = dict(
    backbone=dict(
        init_cfg=dict(
            type='Pretrained',
            checkpoint=
            'https://download.openmmlab.com/mmclassification/v0/swin-transformer/convert/swin_base_patch4_window12_384_22kto1k-d59b0d1d.pth',
            prefix='backbone',
        )),
    head=dict(num_classes=2),
)

default_hooks = dict(
    checkpoint = dict(type='CheckpointHook', interval=1, max_keep_ckpts=1),
    logger=dict(interval=50),
)
