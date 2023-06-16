_base_ = [
    '../datasets/colon.py',
    'mmpretrain::_base_/models/efficientnet_b4.py',
    '../custom_imports.py', 
    'mmpretrain::_base_/default_runtime.py',
    'mmpretrain::_base_/schedules/imagenet_bs256_coslr.py',
]


model = dict(
    backbone=dict(
        init_cfg=dict(
            type='Pretrained',
            checkpoint=
            'https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b4_3rdparty_8xb32-aa_in1k_20220119-45b8bd2b.pth',
            prefix='backbone',
        )),
    head=dict(num_classes=2),
)

default_hooks = dict(
    checkpoint = dict(type='CheckpointHook', interval=1, max_keep_ckpts=1),
    logger=dict(interval=50),
)
