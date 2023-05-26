# runner = dict(type='IterBasedRunner', max_iters=1000)
optimizer = dict(type='SGD', lr=0.001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0.,
    by_epoch=False,
    warmup='constant',
    warmup_by_epoch=False,
    warmup_iters=20,
    warmup_ratio=0.005)
