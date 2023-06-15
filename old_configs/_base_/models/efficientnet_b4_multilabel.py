# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(type='EfficientNet', arch='b4'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='MultiLabelLinearClsHead',
        num_classes=1000,
        in_channels=1792,
    ))
# custom_imports = dict(
#     imports=[
#         'medfmc.datasets.medical_datasets',
#     ], allow_failed_imports=False)
