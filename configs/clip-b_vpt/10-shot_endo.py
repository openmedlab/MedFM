_base_ = [
    '../datasets/endoscopy.py',
    '../swin_schedule.py',
    'mmpretrain::_base_/default_runtime.py',
    '../custom_imports.py',
]


lr = 1e-3
vpl = 1
dataset = 'endo'
exp_num = 1
nshot = 10
run_name = f'clip-b_{nshot}-shot_ptokens-{vpl}_{dataset}'

data_preprocessor = dict(
    num_classes=4,
    # RGB format normalization parameters
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    # convert image from BGR to RGB
    to_rgb=True,
)

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='PromptedViT',
        prompt_length=vpl,
        patch_size=16,
        out_type='cls_token',
        arch='b',
        pre_norm=True,
        img_size=384,
        init_cfg=dict(
            type='Pretrained',
            checkpoint=
            'https://download.openmmlab.com/mmclassification/v0/clip/clip-vit-base-p16_laion2b-in12k-pre_3rdparty_in1k-384px_20221220-84ed0cc0.pth',
            prefix='backbone',
        ),
    ),
    neck=None,
    head=dict(
        type='MultiLabelLinearClsHead',
        num_classes=4,
        in_channels=768,
    ))

# data settings
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
    dict(
        type='ResizeEdge',
        scale=384,
        edge='short',
        backend='pillow',
        interpolation='bicubic'),
    dict(type='CenterCrop', crop_size=384),
    dict(type='PackInputs'),
]


train_dataloader = dict(
    batch_size=4, 
    dataset=dict(
        ann_file=f'data_anns/MedFMC/{dataset}/{dataset}_{nshot}-shot_train_exp{exp_num}.txt',
        pipeline=train_pipeline),
)

val_dataloader = dict(
    batch_size=8,  
    dataset=dict(
        ann_file=f'data_anns/MedFMC/{dataset}/{dataset}_{nshot}-shot_val_exp{exp_num}.txt',
        pipeline=test_pipeline),
)

test_dataloader = dict(
    batch_size=8,  
    dataset=dict(
        ann_file=f'data_anns/MedFMC/{dataset}/test_WithLabel.txt',
        pipeline=test_pipeline),
)

optim_wrapper = dict(optimizer=dict(lr=lr))

default_hooks = dict(
    checkpoint = dict(type='CheckpointHook', interval=1, max_keep_ckpts=1, save_best="auto"),
    logger=dict(interval=50),
)

work_dir = f'work_dirs/clip-b/exp{exp_num}/{run_name}'

