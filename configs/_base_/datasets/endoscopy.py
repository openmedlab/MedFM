# dataset settings
dataset_type = 'Endoscopy'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='RandomResizedCrop',
        size=384,
        backend='pillow',
        interpolation='bicubic'),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=384, backend='pillow', interpolation='bicubic'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_prefix='data/MedFMC/endo/images',
        ann_file='data/MedFMC/endo/train_20.txt',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_prefix='data/MedFMC/endo/images',
        ann_file='data/MedFMC/endo/val_20.txt',
        pipeline=test_pipeline),
    test=dict(
        # replace `data/val` with `data/test` for standard test
        type=dataset_type,
        data_prefix='data/MedFMC/endo/images',
        ann_file='data/MedFMC/endo/test_WithLabel.txt',
        pipeline=test_pipeline))
# evaluation = dict(interval=1, metric='mAP', save_best='auto')
evaluation = dict(interval=1, metric='AUC_multilabel', save_best='auto')
