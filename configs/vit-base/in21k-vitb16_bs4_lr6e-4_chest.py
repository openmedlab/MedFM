_base_ = [
    '../_base_/custom_imports.py',
    '../_base_/datasets/chest.py',
    '../_base_/schedules/imagenet_bs4096_AdamW.py',
    '../_base_/default_runtime.py',
]

lr = 1e-4
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='MedFMC_VisionTransformer',
        arch='b',
        img_size=224,
        patch_size=16,
        drop_rate=0.1),
    head=dict(
        type='MultiLabelLinearClsHead',
        num_classes=19,
        in_channels=768,
        loss=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=1.0),
    ))

optimizer = dict(lr=lr)

log_config = dict(
    interval=10, hooks=[
        dict(type='TextLoggerHook'),
    ])

load_from = 'work_dirs/vit-base-p16_3rdparty_pt-64xb64_in1k-224_20210928-02284250.pth'

runner = dict(type='EpochBasedRunner', max_epochs=20)

# yapf:disable
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
    ])
