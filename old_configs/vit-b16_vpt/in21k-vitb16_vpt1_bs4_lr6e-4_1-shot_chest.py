_base_ = [
    '../_base_/custom_imports.py',
    '../_base_/datasets/chest.py',
    '../_base_/schedules/imagenet_dense.py',
    '../_base_/default_runtime.py',
]

lr = 6e-4
n = 1
vpl = 1
dataset = 'chest'
nshot = 1
run_name = f'in21k-vitb16_vpt-{vpl}_bs4_lr{lr}_{nshot}-shot_{dataset}'

model = dict(
    type='ImageClassifier',
    backbone=dict(type='PromptedVisionTransformer', prompt_length=1),
    head=dict(
        type='MultiLabelLinearClsHead',
        num_classes=19,
        in_channels=768,
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
work_dir = f'work_dirs/vpt/{run_name}'

runner = dict(type='EpochBasedRunner', max_epochs=20)

# yapf:disable
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
    ])
