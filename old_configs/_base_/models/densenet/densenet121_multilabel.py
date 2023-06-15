# Model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(type='DenseNet', arch='121'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='MultiLabelLinearClsHead',
        num_classes=1000,
        in_channels=1024,
    ))

# custom_imports = dict(
#     imports=[
#         'medfmc.datasets.medical_datasets',
#     ], allow_failed_imports=False)
