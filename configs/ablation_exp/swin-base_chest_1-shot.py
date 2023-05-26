# Only for evaluation
_base_ = [
    '../_base_/models/swin_transformer/base_384_multilabel.py',
    '../_base_/datasets/chest.py',
    '../_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '../_base_/default_runtime.py', '../_base_/custom_imports.py'
]

model = dict(
    backbone=dict(
        init_cfg=dict(
            type='Pretrained',
            checkpoint=
            'pretrain/swin_base_patch4_window12_384_22kto1k-d59b0d1d.pth',
            prefix='backbone',
        )),
    head=dict(num_classes=19),
)
dataset = 'chest'
nshot = 1
data = dict(
    samples_per_gpu=4,  # use 2 gpus, total 128
    train=dict(
        ann_file=f'data/MedFMC/{dataset}/{dataset}_{nshot}-shot_train.txt'),
    val=dict(ann_file=f'data/MedFMC/{dataset}/{dataset}_{nshot}-shot_val.txt'),
    test=dict(ann_file=f'data/MedFMC/{dataset}/test_WithLabel.txt'))
