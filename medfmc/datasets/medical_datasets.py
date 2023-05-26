# Copyright (c) OpenMMLab. All rights reserved.

import numpy as np
from mmcls.core import average_performance, mAP
from mmcls.core.evaluation import precision_recall_f1, support
from mmcls.datasets.base_dataset import BaseDataset as OLD_BaseDataset
from mmcls.datasets.builder import DATASETS
from mmcls.datasets.multi_label import \
    MultiLabelDataset as OLD_MultiLabelDataset
from mmcls.models.losses import accuracy

from medfmc.core.evaluation import AUC_multiclass, AUC_multilabel


class BaseDataset(OLD_BaseDataset):

    def evaluate(self,
                 results,
                 metric='accuracy',
                 metric_options=None,
                 indices=None,
                 logger=None):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
                Default value is `accuracy`.
            metric_options (dict, optional): Options for calculating metrics.
                Allowed keys are 'topk', 'thrs' and 'average_mode'.
                Defaults to None.
            indices (list, optional): The indices of samples corresponding to
                the results. Defaults to None.
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Defaults to None.
        Returns:
            dict: evaluation results
        """
        if metric_options is None:
            metric_options = {'topk': (1, 5)}
        if isinstance(metric, str):
            metrics = [metric]
        else:
            metrics = metric
        allowed_metrics = [
            'accuracy', 'precision', 'recall', 'f1_score', 'support',
            'AUC_multiclass'
        ]
        eval_results = {}
        results = np.vstack(results)
        gt_labels = self.get_gt_labels()
        if indices is not None:
            gt_labels = gt_labels[indices]
        num_imgs = len(results)
        assert len(gt_labels) == num_imgs, 'dataset testing results should '\
            'be of the same length as gt_labels.'

        invalid_metrics = set(metrics) - set(allowed_metrics)
        if len(invalid_metrics) != 0:
            raise ValueError(f'metric {invalid_metrics} is not supported.')

        topk = metric_options.get('topk', (1, 5))
        thrs = metric_options.get('thrs')
        average_mode = metric_options.get('average_mode', 'macro')

        if 'AUC_multiclass' in metrics:
            AUC_value = AUC_multiclass(results, gt_labels)
            eval_results['AUC_multiclass'] = AUC_value

        if 'accuracy' in metrics:
            if thrs is not None:
                acc = accuracy(results, gt_labels, topk=topk, thrs=thrs)
            else:
                acc = accuracy(results, gt_labels, topk=topk)
            if isinstance(topk, tuple):
                eval_results_ = {
                    f'accuracy_top-{k}': a
                    for k, a in zip(topk, acc)
                }
            else:
                eval_results_ = {'accuracy': acc}
            if isinstance(thrs, tuple):
                for key, values in eval_results_.items():
                    eval_results.update({
                        f'{key}_thr_{thr:.2f}': value.item()
                        for thr, value in zip(thrs, values)
                    })
            else:
                eval_results.update(
                    {k: v.item()
                     for k, v in eval_results_.items()})

        if 'support' in metrics:
            support_value = support(
                results, gt_labels, average_mode=average_mode)
            eval_results['support'] = support_value

        precision_recall_f1_keys = ['precision', 'recall', 'f1_score']
        if len(set(metrics) & set(precision_recall_f1_keys)) != 0:
            if thrs is not None:
                precision_recall_f1_values = precision_recall_f1(
                    results, gt_labels, average_mode=average_mode, thrs=thrs)
            else:
                precision_recall_f1_values = precision_recall_f1(
                    results, gt_labels, average_mode=average_mode)
            for key, values in zip(precision_recall_f1_keys,
                                   precision_recall_f1_values):
                if key in metrics:
                    if isinstance(thrs, tuple):
                        eval_results.update({
                            f'{key}_thr_{thr:.2f}': value
                            for thr, value in zip(thrs, values)
                        })
                    else:
                        eval_results[key] = values

        return eval_results


class MultiLabelDataset(OLD_MultiLabelDataset):

    def evaluate(self,
                 results,
                 metric='AUC_multilabel',
                 metric_options=None,
                 indices=None,
                 logger=None):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
                Default value is 'mAP'. Options are 'mAP', 'CP', 'CR', 'CF1',
                'OP', 'OR' and 'OF1', 'AUC_multilabel.
            metric_options (dict, optional): Options for calculating metrics.
                Allowed keys are 'k' and 'thr'. Defaults to None
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Defaults to None.

        Returns:
            dict: evaluation results
        """
        if metric_options is None or metric_options == {}:
            metric_options = {'thr': 0.5}

        if isinstance(metric, str):
            metrics = [metric]
        else:
            metrics = metric
        allowed_metrics = [
            'mAP', 'CP', 'CR', 'CF1', 'OP', 'OR', 'OF1', 'AUC_multilabel'
        ]
        eval_results = {}
        results = np.vstack(results)
        gt_labels = self.get_gt_labels()
        if indices is not None:
            gt_labels = gt_labels[indices]
        num_imgs = len(results)
        assert len(gt_labels) == num_imgs, 'dataset testing results should '\
            'be of the same length as gt_labels.'

        invalid_metrics = set(metrics) - set(allowed_metrics)
        if len(invalid_metrics) != 0:
            raise ValueError(f'metric {invalid_metrics} is not supported.')

        if 'mAP' in metrics:
            mAP_value = mAP(results, gt_labels)
            eval_results['mAP'] = mAP_value
        if 'AUC_multilabel' in metrics:
            AUC_value = AUC_multilabel(results, gt_labels)
            eval_results['AUC_multilabel'] = AUC_value
        if len(set(metrics) - {'mAP'}) != 0:
            performance_keys = ['CP', 'CR', 'CF1', 'OP', 'OR', 'OF1']
            performance_values = average_performance(results, gt_labels,
                                                     **metric_options)
            for k, v in zip(performance_keys, performance_values):
                if k in metrics:
                    eval_results[k] = v

        return eval_results


@DATASETS.register_module()
class Chest19(MultiLabelDataset):

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

    def load_annotations(self):

        data_infos = []
        with open(self.ann_file) as f:
            samples = [x.strip() for x in f.readlines()]
            for item in samples:
                filename, imglabel = item.split(' ')
                gt_label = np.asarray(
                    list(map(int, imglabel.split(','))), dtype=np.int8)

                info = {'img_prefix': self.data_prefix}
                info['img_info'] = {'filename': filename}
                info['gt_label'] = gt_label

                data_infos.append(info)

        return data_infos


@DATASETS.register_module()
class Endoscopy(MultiLabelDataset):

    CLASSES = ['ulcer', 'erosion', 'polyp', 'tumor']

    def __init__(self, **kwargs):
        super(Endoscopy, self).__init__(**kwargs)

    def load_annotations(self):

        data_infos = []
        with open(self.ann_file) as f:
            samples = [x.strip() for x in f.readlines()]
            for item in samples:
                filename = item[:-8]
                imglabel = item[-7:]
                gt_label = np.asarray(
                    list(map(int, imglabel.split(' '))), dtype=np.int8)

                info = {'img_prefix': self.data_prefix}
                info['img_info'] = {'filename': filename}
                info['gt_label'] = gt_label

                data_infos.append(info)

        return data_infos


@DATASETS.register_module()
class Colon(BaseDataset):

    CLASSES = ['negtive', 'positive']

    def __init__(self, **kwargs):
        super(Colon, self).__init__(**kwargs)

    def load_annotations(self):

        data_infos = []
        with open(self.ann_file) as f:
            samples = [x.strip() for x in f.readlines()]
            for item in samples:
                filename = item[:-2]
                imglabel = int(item[-1:])
                gt_label = np.array(imglabel, dtype=np.int64)

                info = {'img_prefix': self.data_prefix}
                info['img_info'] = {'filename': filename}
                info['gt_label'] = gt_label

                data_infos.append(info)

        return data_infos
