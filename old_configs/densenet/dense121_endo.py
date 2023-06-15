_base_ = [
    '../_base_/models/densenet/densenet121_multilabel.py',
    '../_base_/datasets/endoscopy.py', '../_base_/schedules/imagenet_dense.py',
    '../_base_/default_runtime.py', '../_base_/custom_imports.py'
]

model = dict(
    backbone=dict(
        init_cfg=dict(
            type='Pretrained',
            checkpoint='pretrain/densenet121_4xb256_in1k_20220426-07450f99.pth',
            prefix='backbone',
        )),
    head=dict(num_classes=4),
)
