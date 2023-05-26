_base_ = [
    '../_base_/models/efficientnet_b4.py', '../_base_/datasets/colon.py',
    '../_base_/schedules/imagenet_bs256_coslr.py',
    '../_base_/default_runtime.py', '../_base_/custom_imports.py'
]

model = dict(
    backbone=dict(
        init_cfg=dict(
            type='Pretrained',
            checkpoint=
            'pretrain/efficientnet-b4_3rdparty_8xb32-aa_in1k_20220119-45b8bd2b.pth',
            prefix='backbone',
        )),
    head=dict(num_classes=2),
)
dataset = 'colon'
nshot = 10
data = dict(
    samples_per_gpu=4,  # use 2 gpus, total 128
    train=dict(
        ann_file=f'data/MedFMC/{dataset}/{dataset}_{nshot}-shot_train.txt'),
    val=dict(ann_file=f'data/MedFMC/{dataset}/{dataset}_{nshot}-shot_val.txt'),
    test=dict(ann_file=f'data/MedFMC/{dataset}/test_WithLabel.txt'))
