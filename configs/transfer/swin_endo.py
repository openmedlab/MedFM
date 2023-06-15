# Only for evaluation
_base_ = [
    'mmpretrain::_base_/models/swin_transformer/base_384.py',
    '../datasets/endoscopy.py',
    'mmpretrain::_base_/schedules/imagenet_bs1024_adamw_swin.py',
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
    head=dict(type='MultiLabelLinearClsHead', num_classes=4, loss=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),),
)
