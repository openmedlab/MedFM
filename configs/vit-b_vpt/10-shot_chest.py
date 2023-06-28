_base_ = [
    '../datasets/chest.py',
    '../swin_schedule.py',
    'mmpretrain::_base_/default_runtime.py',
    '../custom_imports.py',
]


lr = 5e-3
vpl = 1  
dataset = 'chest'
exp_num = 1
nshot = 10
run_name = f'vit-b_{nshot}-shot_ptokens-{vpl}_{dataset}'

# dataset setting
data_preprocessor = dict(
    mean=[127.5, 127.5, 127.5],
    std=[127.5, 127.5, 127.5],
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
        img_size=384,
        init_cfg=dict(
            type='Pretrained',
            checkpoint=
            'https://download.openmmlab.com/mmclassification/v0/vit/finetune/vit-base-p16_in21k-pre-3rdparty_ft-64xb64_in1k-384_20210928-98e8652b.pth',
            prefix='backbone',
        ),
        ),
    neck=None,
    head=dict(
        type='MultiLabelLinearClsHead',
        num_classes=19,
        in_channels=768,
    ))

train_dataloader = dict(
    batch_size=4, 
    dataset=dict(ann_file=f'data_anns/MedFMC/{dataset}/{dataset}_{nshot}-shot_train_exp{exp_num}.txt'),
)

val_dataloader = dict(
    batch_size=8,  
    dataset=dict(ann_file=f'data_anns/MedFMC/{dataset}/{dataset}_{nshot}-shot_val_exp{exp_num}.txt'),
)

test_dataloader = dict(
    batch_size=4,  
    dataset=dict(ann_file=f'data_anns/MedFMC/{dataset}/test_WithLabel.txt'),
)

optim_wrapper = dict(optimizer=dict(lr=lr))

default_hooks = dict(
    checkpoint = dict(type='CheckpointHook', interval=1, max_keep_ckpts=1, save_best="auto"),
    logger=dict(interval=50),
)

work_dir = f'work_dirs/vit-b/exp{exp_num}/{run_name}'
