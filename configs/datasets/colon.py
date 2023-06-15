# dataset settings
dataset_type = 'Colon'
data_preprocessor = dict(
    num_classes=2,
    # RGB format normalization parameters
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    # convert image from BGR to RGB
    to_rgb=True,
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='RandomResizedCrop',
        scale=384,
        backend='pillow',
        interpolation='bicubic'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=384, backend='pillow', interpolation='bicubic'),
    dict(type='PackInputs'),
]

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_prefix='data/MedFMC_train/colon/images',
        ann_file='data_backup/MedFMC/colon/train_20.txt',
        pipeline=train_pipeline,),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=4,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_prefix='data/MedFMC_train/colon/images',
        ann_file='data_backup/MedFMC/colon/val_20.txt',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)


test_dataloader = dict(
    batch_size=4,
    num_workers=2,
    dataset=dict(
        # replace `data/val` with `data/test` for standard test
        type=dataset_type,
        data_prefix='data/MedFMC_train/colon/images',
        ann_file='data_backup/MedFMC/colon/test_WithLabel.txt',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)

val_evaluator = [dict(type='Accuracy', topk=(1, )), dict(type='AUC')]
test_evaluator = val_evaluator
