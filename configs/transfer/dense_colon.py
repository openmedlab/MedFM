# Only for evaluation
_base_ = [
    '../datasets/colon.py',
    'mmpretrain::_base_/models/densenet/densenet121.py',
    '../custom_imports.py', 
    'mmpretrain::_base_/default_runtime.py',
]


# optimizer
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001))

# learning policy
param_scheduler = dict(type='CosineAnnealingLR', by_epoch=True, eta_min=1e-5,)

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=20, val_interval=1)
val_cfg = dict()
test_cfg = dict()

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=256)


model = dict(
    backbone=dict(
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmclassification/v0/densenet/densenet121_4xb256_in1k_20220426-07450f99.pth',
            prefix='backbone',
        )),
    head=dict(num_classes=2),
)

train_dataloader = dict(batch_size=6)

default_hooks = dict(
    checkpoint = dict(type='CheckpointHook', interval=1, max_keep_ckpts=1, save_best="auto"),
    logger=dict(interval=50),
)
