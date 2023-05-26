import math
import numpy as np
from sklearn import metrics


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
