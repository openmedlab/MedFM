_base_ = [
    '../datasets/endoscopy.py',
    '../swin_schedule.py',
    'mmpretrain::_base_/default_runtime.py',
    '../custom_imports.py',
]


lr = 5e-3
vpl = 1
dataset = 'endo'
exp_num = 1
nshot = 10
run_name = f'eva02-b_{nshot}-shot_ptokens-{vpl}_{dataset}'

data_preprocessor = dict(
    # RGB format normalization parameters
    mean=[0.48145466 * 255, 0.4578275 * 255, 0.40821073 * 255],
    std=[0.26862954 * 255, 0.26130258 * 255, 0.27577711 * 255],
    # convert image from BGR to RGB
    to_rgb=True,
)

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='PromptedViTEVA02',
        prompt_length=vpl,
        patch_size=14,
        sub_ln=True,
        final_norm=False,
        out_type='avg_featmap',
        arch='b',
        img_size=448,
        init_cfg=dict(
            type='Pretrained',
            checkpoint=
            'https://download.openmmlab.com/mmpretrain/v1.0/eva02/eva02-base-p14_in21k-pre_in21k-medft_3rdparty_in1k-448px_20230505-5cd4d87f.pth',
            prefix='backbone',),
        ),
    neck=None,
    head=dict(
        type='MultiLabelLinearClsHead',
        num_classes=4,
        in_channels=768,
    ))

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='RandomResizedCrop',
        scale=448,
        backend='pillow',
        interpolation='bicubic'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='ResizeEdge',
        scale=448,
        edge='short',
        backend='pillow',
        interpolation='bicubic'),
    dict(type='CenterCrop', crop_size=448),
    dict(type='PackInputs'),
]

train_dataloader = dict(
    batch_size=1, 
    dataset=dict(ann_file=f'data_backup/MedFMC/{dataset}/{dataset}_{nshot}-shot_train_exp{exp_num}.txt',
    pipeline=train_pipeline),
)

val_dataloader = dict(
    batch_size=2,  
    dataset=dict(ann_file=f'data_backup/MedFMC/{dataset}/{dataset}_{nshot}-shot_val_exp{exp_num}.txt',
    pipeline=test_pipeline),
)

test_dataloader = dict(
    batch_size=2,  
    dataset=dict(ann_file=f'data_backup/MedFMC/{dataset}/test_WithLabel.txt',
    pipeline=test_pipeline),
)

optim_wrapper = dict(optimizer=dict(lr=lr))

default_hooks = dict(
    checkpoint = dict(type='CheckpointHook', interval=1, max_keep_ckpts=1, save_best="auto"),
    logger=dict(interval=50),
)

work_dir = f'work_dirs/eva02-b/exp{exp_num}/{run_name}'

