import math
import torch
from typing import List, Optional, Sequence, Union
import numpy as np
from mmpretrain.structures import label_to_onehot
from sklearn import metrics
from mmengine.evaluator import BaseMetric
from mmpretrain.registry import METRICS


def compute_auc(cls_scores, cls_labels):
    cls_aucs = []
    for i in range(cls_scores.shape[1]):
        scores_per_class = cls_scores[:, i]
        labels_per_class = cls_labels[:, i]
        try:
            auc_per_class = metrics.roc_auc_score(labels_per_class,
                                                  scores_per_class)
            # print('class {} auc = {:.2f}'.format(i + 1, auc_per_class * 100))
        except ValueError:
            pass
        cls_aucs.append(auc_per_class * 100)

    return cls_aucs


def cal_metrics_multilabel(target, cosine_scores):
    """Calculate mean AUC with given dataset information and cosine scores."""

    sample_num = target.shape[0]
    cls_num = cosine_scores.shape[1]

    gt_labels = np.zeros((sample_num, cls_num))
    for k in range(target.shape[0]):
        label = target[k]
        gt_labels[k, :] = label

    cls_scores = np.zeros((sample_num, cls_num))
    for k in range(target.shape[0]):
        cos_score = cosine_scores[k]
        norm_scores = [1 / (1 + math.exp(-1 * v)) for v in cos_score]
        cls_scores[k, :] = np.array(norm_scores)

    cls_aucs = compute_auc(cls_scores, gt_labels)
    mean_auc = np.mean(cls_aucs)

    return mean_auc


def cal_metrics_multiclass(target, cosine_scores):

    sample_num = target.shape[0]
    cls_num = cosine_scores.shape[1]

    gt_labels = np.zeros((sample_num, cls_num))
    for k in range(target.shape[0]):
        label = target[k]
        one_hot_label = np.array([int(i == label) for i in range(cls_num)])
        gt_labels[k, :] = one_hot_label

    cls_scores = np.zeros((sample_num, cls_num))
    for k in range(target.shape[0]):
        cos_score = cosine_scores[k]

        norm_scores = [math.exp(v) for v in cos_score]
        norm_scores /= np.sum(norm_scores)

        cls_scores[k, :] = np.array(norm_scores)

    cls_aucs = compute_auc(cls_scores, gt_labels)
    mean_auc = np.mean(cls_aucs)

    return mean_auc


@METRICS.register_module()
class AUC(BaseMetric):
    r"""AUC.

    Args:
        multilabel: 
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.

    """
    default_prefix: Optional[str] = 'AUC'

    def __init__(self,
                 multilabel: bool = False,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.multilabel = multilabel


    def process(self, data_batch, data_samples: Sequence[dict]):
        """Process one batch of data samples.

        The processed results should be stored in ``self.results``, which will
        be used to computed the metrics when all batches have been processed.

        Args:
            data_batch: A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        for data_sample in data_samples:
            result = dict()

            result['pred_score'] = data_sample['pred_score'].clone()
            num_classes = result['pred_score'].size()[-1]

            if 'gt_score' in data_sample:
                result['gt_score'] = data_sample['gt_score'].clone()
            else:
                result['gt_score'] = label_to_onehot(data_sample['gt_label'],
                                                     num_classes)

            # Save the result to `self.results`.
            self.results.append(result)

    def compute_metrics(self, results: List):
        """Compute the metrics from processed results.

        Args:
            results (dict): The processed results of each batch.

        Returns:
            Dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        # NOTICE: don't access `self.results` from the method.
        target = torch.stack([res['gt_score'] for res in results])
        pred = torch.stack([res['pred_score'] for res in results])

        if self.multilabel:
            res = cal_metrics_multilabel(target, pred)
        else:
            res = cal_metrics_multilabel(target, pred)

        result_metrics = dict()
        if self.multilabel:
            result_metrics['AUC_multilabe'] = float(res)
        else:
            result_metrics['AUC_multiclass'] = float(res)
        return result_metrics



def AUC_multiclass(pred, target):
    """Calculate the AUC with respect of classes. This metric is used for Colon
    Dataset.

    Args:
        pred (torch.Tensor | np.ndarray): The model prediction with shape
            (N, C), where C is the number of classes.
        target (torch.Tensor | np.ndarray): The target of each prediction with
            shape (N, C), where C is the number of classes. 1 stands for
            positive examples, 0 stands for negative examples and -1 stands for
            difficult examples.

    Returns:
        float: A single float as mAP value.
    """
    auc = cal_metrics_multiclass(target, pred)

    return auc


def AUC_multilabel(pred, target):
    """Calculate the AUC with respect of classes. This metric is used for
    Endoscopy and Chest Dataset.

    Args:
        pred (torch.Tensor | np.ndarray): The model prediction with shape
            (N, C), where C is the number of classes.
        target (torch.Tensor | np.ndarray): The target of each prediction with
            shape (N, C), where C is the number of classes. 1 stands for
            positive examples, 0 stands for negative examples and -1 stands for
            difficult examples.

    Returns:
        float: A single float as mAP value.
    """
    # auc = cal_metrics_multilabel(target, pred)
    cls_aucs = compute_auc(pred, target)
    mean_auc = np.mean(cls_aucs)
    return mean_auc




