# Only for evaluation
_base_ = [
    '../swin_schedule.py',
    '../datasets/endoscopy.py',
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
    head=dict(_delete_=True, type='MultiLabelLinearClsHead', num_classes=4, in_channels=1024, 
               loss=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),),
)

default_hooks = dict(
    checkpoint = dict(type='CheckpointHook', interval=1, max_keep_ckpts=1),
    logger=dict(interval=50),
)