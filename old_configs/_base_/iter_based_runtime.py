# yapf:disable
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
    ])
# yapf:enable
checkpoint_config = dict(interval=100, max_keep_ckpts=1)
# evaluation = dict(by_epoch=False, metric=['accuracy', 'class_accuracy', 'bag_accuracy', 'bag_class_accuracy'], interval=1000)
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
