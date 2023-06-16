# Copyright (c) OpenMMLab. All rights reserved.

import numpy as np
from mmpretrain.datasets import CustomDataset
from mmengine.fileio import join_path

from mmpretrain.registry import DATASETS


@DATASETS.register_module()
class Chest19(CustomDataset):

    CLASSES = [
        'pleural_effusion', 'nodule', 'pneumonia', 'cardiomegaly',
        'hilar_enlargement', 'fracture_old', 'fibrosis',
        'aortic_calcification', 'tortuous_aorta', 'thickened_pleura', 'TB',
        'pneumothorax', 'emphysema', 'atelectasis', 'calcification',
        'pulmonary_edema', 'increased_lung_markings', 'elevated_diaphragm',
        'consolidation'
    ]

    def __init__(self, **kwargs):
        super(Chest19, self).__init__(**kwargs)

    def load_data_list(self):
        data_list = []
        with open(self.ann_file) as f:
            samples = [x.strip() for x in f.readlines()]
            for item in samples:
                filename, imglabel = item.split(' ')
                gt_label = np.asarray(
                    list(map(int, imglabel.split(','))), dtype=np.int8)

                gt_label = np.where(gt_label == 1)[0].tolist()
                img_path = join_path(self.img_prefix, filename)
                info = {
                    'img_path': img_path, 
                    'gt_label': gt_label
                }

                data_list.append(info)

        return data_list


@DATASETS.register_module()
class Endoscopy(CustomDataset):

    CLASSES = ['ulcer', 'erosion', 'polyp', 'tumor']

    def __init__(self, **kwargs):
        super(Endoscopy, self).__init__(**kwargs)

    def load_data_list(self):
        data_list = []
        with open(self.ann_file) as f:
            samples = [x.strip() for x in f.readlines()]
            for item in samples:
                filename = item[:-8]
                imglabel = item[-7:]
                gt_label = np.asarray(
                    list(map(int, imglabel.split(' '))), dtype=np.int8)
                gt_label = np.where(gt_label == 1)[0].tolist()
                img_path = join_path(self.img_prefix, filename)
                info = {
                    'img_path': img_path, 
                    'gt_label': gt_label
                }


                data_list.append(info)

        return data_list


@DATASETS.register_module()
class Colon(CustomDataset):

    CLASSES = ['negtive', 'positive']

    def __init__(self, **kwargs):
        super(Colon, self).__init__(**kwargs)

    def load_data_list(self):
        data_list = []
        with open(self.ann_file) as f:
            samples = [x.strip() for x in f.readlines()]
            for item in samples:
                filename = item[:-2]
                gt_label = int(item[-1:])

                img_path = join_path(self.img_prefix, filename)
                info = {
                    'img_path': img_path, 
                    'gt_label': gt_label
                }

                data_list.append(info)

        return data_list
