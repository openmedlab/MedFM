_base_ = [
    '../_base_/custom_imports.py',
    '../_base_/datasets/endoscopy.py',
    '../_base_/schedules/imagenet_bs4096_AdamW.py',
    '../_base_/default_runtime.py',
]

lr = 1e-4
exp_num = 1
dataset = 'endo'
nshot = 5
run_name = f'in21k-vitb16_bs4_lr{lr}_{nshot}-shot_{dataset}'
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='VisionTransformer',
        arch='b',
        img_size=224,
        patch_size=16,
        drop_rate=0.1,
    ),
    head=dict(
        type='MultiLabelLinearClsHead',
        num_classes=4,
        in_channels=768,
        loss=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=1.0),
    ))
data = dict(
    samples_per_gpu=4,  # use 2 gpus, total 128
    train=dict(
        ann_file=f'data/MedFMC/{dataset}/{dataset}_{nshot}-shot_train.txt'),
    val=dict(ann_file=f'data/MedFMC/{dataset}/{dataset}_{nshot}-shot_val.txt'),
    test=dict(ann_file=f'data/MedFMC/{dataset}/test_WithLabel.txt'))
optimizer = dict(lr=lr)

log_config = dict(
    interval=10, hooks=[
        dict(type='TextLoggerHook'),
    ])

load_from = 'work_dirs/vit-base-p16_3rdparty_pt-64xb64_in1k-224_20210928-02284250.pth'
work_dir = f'work_dirs/exp{exp_num}/{run_name}'

runner = dict(type='EpochBasedRunner', max_epochs=20)

# yapf:disable
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
    ])
