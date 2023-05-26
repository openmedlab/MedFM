import argparse
import math
import mmcv
import numpy as np
import os
import random
import time
import torch
import yaml
from mmcls.apis import init_model
from mmcls.datasets.pipelines import Compose
from mmcv.parallel import collate, scatter
from sklearn import metrics
from sklearn.linear_model import LogisticRegression


def inference_model(model, img):
    """Inference image(s) with the classifier. Noted that it is modified from
    MMClasification repository. It only returns feature map of certain model
    backbone rather than prediction label or scores.

    Args:
        model (nn.Module): The loaded classifier.
        img (str/ndarray): The image filename or loaded image.

    Returns:
        result (Tensor): The feature map of certain model backbone.
    """

    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    if isinstance(img, str):
        if cfg.data.test.pipeline[0]['type'] != 'LoadImageFromFile':
            cfg.data.test.pipeline.insert(0, dict(type='LoadImageFromFile'))
        data = dict(img_info=dict(filename=img), img_prefix=None)
    else:
        if cfg.data.test.pipeline[0]['type'] == 'LoadImageFromFile':
            cfg.data.test.pipeline.pop(0)
        data = dict(img=img)
    test_pipeline = Compose(cfg.data.test.pipeline)
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]

    # forward the model
    with torch.no_grad():
        result = model.extract_feat(data['img'], stage='pre_logits')

    return result


def extract_model_fea(model, img_file):
    """Extract feature map of given image file."""
    img_array = mmcv.imread(img_file)
    extract_feat = inference_model(model, img_array)
    return extract_feat


def load_annotations(ann_file):
    """Load annotation information from the file."""
    data_infos = []
    with open(ann_file) as f:
        samples = [x.strip() for x in f.readlines()]
        for item in samples:
            filename = item[:-2]
            imglabel = int(item[-1:])
            info = {}
            info['filename'] = filename
            info['gt_label'] = imglabel
            data_infos.append(info)

    return data_infos


def gen_support_set_twoclass(img_list, K_shot, data_type):
    """Generate Colon dataset list for few-shot learning task.

    Args:
        img_list (List[dict]): List of input dataset file information,
            whose elements contains filename and annotation labels
            of each image.
        K_shot (int): The number of certain shot learning, such as 1-shot,
            5-shot and 10-shot.
        data_type (str): The name of dataset.
    """

    # sort study_id
    pos_study_ids = []
    neg_study_ids = []
    for item in img_list:
        img_name = item['filename']
        study_id = None
        if data_type == 'colon':
            study_id = img_name[:-9]
        else:
            raise ValueError(f'Invalid dataset type {dataset_type}.')
        label = item['gt_label']

        if label == 1 and study_id not in pos_study_ids:
            pos_study_ids.append(study_id)
        if label == 0 and study_id not in neg_study_ids:
            neg_study_ids.append(study_id)

    print(len(pos_study_ids), len(neg_study_ids))
    random.shuffle(pos_study_ids)
    random.shuffle(neg_study_ids)

    pick_pos_study_ids = pos_study_ids[:K_shot]
    pick_neg_study_ids = neg_study_ids[:K_shot]

    support_pos_set = []
    support_neg_set = []
    for item in img_list:
        img_name = item['filename']
        study_id = None
        if data_type == 'colon':
            study_id = img_name[:-9]
        label = item['gt_label']

        if study_id in pick_pos_study_ids and label == 1:
            support_pos_set.append(img_name)
        if study_id in pick_neg_study_ids and label == 0:
            support_neg_set.append(img_name)

    print(len(support_pos_set), support_pos_set[0])
    print(len(support_neg_set), support_neg_set[0])

    support_set = []
    support_set.append(support_neg_set)
    support_set.append(support_pos_set)

    return support_set


def compute_cls_centers(support_set, images_dir, model, fea_dim):
    """Compute class center of each class in Meta-Baseline method.

    Args:
        support_set (List[list]): List of few shot samples in given dataset.
        images_dir (str): The path of given dataset images.
        model (ImageClassifier): The image classifier model used in Meta-Baseline method.
        fea_dim (str): The dimension of feature map.
    Returns:
        cls_centers (list): The list of class centers.
    """

    cls_num = len(support_set)
    cls_centers = []
    for cls_idx in range(cls_num):
        imgs_per_cls = support_set[cls_idx]
        cls_center = torch.zeros([1, fea_dim])
        for item in imgs_per_cls:
            img_file = os.path.join(images_dir, item)
            img_fea = extract_model_fea(model, img_file)
            img_fea = img_fea.cpu()
            img_fea = img_fea / img_fea.norm(dim=1, keepdim=True)
            cls_center = cls_center + img_fea
        cls_center = cls_center / len(imgs_per_cls)
        cls_center = cls_center / cls_center.norm(dim=1, keepdim=True)

        cls_centers.append(cls_center)

    return cls_centers


def fewshot_test(query_set, cls_centers, images_dir, model):
    """Few-Shot test in Meta Baseline method.

    Args:
        query_set (List[list]): List of test samples in given dataset for few shot task.
        cls_centers (list): The list of class centers.
        images_dir (str): The path of given dataset images.
        model (ImageClassifier): The image classifier model used in Meta-Baseline method.
    Returns:
        pred_accuracy (float): The prediction accuracy.
        cosine_scores_query_set (List[List]): The cosine distance for each class in
            test dataset. It has N elements corresponding to N samples in test set,
            each element has C scores corresponding to C classes in ground truth.
    """

    cls_num = len(cls_centers)
    correct_num = 0
    cosine_scores_query_set = []
    for item in query_set:
        img_name = item['filename']
        img_label = item['gt_label']

        img_file = os.path.join(images_dir, img_name)
        img_fea = extract_model_fea(model, img_file)
        img_fea = img_fea.cpu()
        img_fea = img_fea / img_fea.norm(dim=1, keepdim=True)

        max_cosine = 0
        pred_cls = -1
        cosine_scores_per_img = []
        for cls_idx in range(cls_num):
            cls_center = cls_centers[cls_idx]
            norm_inner = torch.inner(img_fea, cls_center)
            cosine_simlarity = norm_inner.item()
            cosine_scores_per_img.append(cosine_simlarity)
            if max_cosine < cosine_simlarity:
                max_cosine = cosine_simlarity
                pred_cls = cls_idx

        if img_label == pred_cls:
            correct_num += 1

        cosine_scores_query_set.append(cosine_scores_per_img)

    pred_accuracy = correct_num / len(query_set)
    print('correct_num = ', correct_num, 'pred_accuracy = ', pred_accuracy)
    print(len(cosine_scores_query_set), cosine_scores_query_set[0])

    return pred_accuracy, cosine_scores_query_set


def compute_auc(cls_scores, cls_labels):
    """Compute AUC of given prediction scores and ground truth."""

    cls_aucs = []
    for i in range(cls_scores.shape[1]):
        scores_per_class = cls_scores[:, i]
        labels_per_class = cls_labels[:, i]
        auc_per_class = metrics.roc_auc_score(labels_per_class,
                                              scores_per_class)
        print('class {} auc = {:.2f}'.format(i + 1, auc_per_class * 100))

        cls_aucs.append(auc_per_class * 100)

    return cls_aucs


def cal_metrics(img_infos, cosine_scores):
    """Calculate mean AUC with given dataset information and cosine scores.

    This method is only implemented in Meta Baseline method.
    """
    print('run cal_metrics ...')

    sample_num = len(img_infos)
    cls_num = len(cosine_scores[0])

    print('sample_num = ', sample_num, ', cls_num = ', cls_num)

    gt_labels = np.zeros((sample_num, cls_num))
    for k in range(len(img_infos)):
        sample = img_infos[k]
        label = sample['gt_label']
        one_hot_label = np.array([int(i == label) for i in range(cls_num)])
        gt_labels[k, :] = one_hot_label

    cls_scores = np.zeros((sample_num, cls_num))
    for k in range(len(img_infos)):
        cos_score = cosine_scores[k]

        norm_scores = [math.exp(v) for v in cos_score]
        norm_scores /= np.sum(norm_scores)

        cls_scores[k, :] = np.array(norm_scores)

    cls_aucs = compute_auc(cls_scores, gt_labels)
    print(np.mean(cls_aucs), cls_aucs)
    mean_auc = np.mean(cls_aucs)

    return mean_auc


def baseline_cls_model(support_set, images_dir, model, fea_dim):
    """Compute class center of each class in Baseline method.

    Args:
        support_set (List[list]): List of few shot samples in given dataset.
        images_dir (str): The path of given dataset images.
        model (ImageClassifier): The image classifier model used in Meta-Baseline method.
        fea_dim (str): The dimension of feature map.
    Returns:
        clf (LogisticRegression): The Logistic Regression function which includes
            training datasets. It is executed from sklearn package.
    """

    sample_num = 0
    for cls_set in support_set:
        sample_num += len(cls_set)

    X_train = np.zeros((sample_num, fea_dim))
    cls_num = len(support_set)
    sample_idx = 0
    Y_train = []
    for cls_idx in range(cls_num):
        imgs_per_cls = support_set[cls_idx]
        for item in imgs_per_cls:
            img_file = os.path.join(images_dir, item)
            img_fea = extract_model_fea(model, img_file)
            img_fea = img_fea.cpu()
            img_fea = img_fea / img_fea.norm(dim=1, keepdim=True)
            X_train[sample_idx, :] = img_fea
            sample_idx += 1
            Y_train.append(cls_idx)

    clf = LogisticRegression(random_state=0).fit(X_train, Y_train)

    return clf


def baseline_test(query_set, images_dir, model, clf, cls_num, fea_dim):
    """Few-Shot test in Baseline method.

    Args:
        query_set (List[list]): List of test samples in given dataset for few shot task.
        images_dir (str): The path of given dataset images.
        model (ImageClassifier): The image classifier model used in Meta-Baseline method.
        clf (LogisticRegression): The Logistic Regression function which includes
            training datasets. It is executed from sklearn package.
        cls_num (int): The number of classes.
        fea_dim (str): The dimension of feature map.
    Returns:
        acc (float): The value of Acc(accuracy).
        auc (float): The value of AUC.
    """

    sample_num = len(query_set)
    print('sample_num = ', sample_num)

    X_Test = np.zeros((sample_num, fea_dim))
    Y_test = []
    sample_idx = 0
    for item in query_set:
        img_name = item['filename']
        img_label = item['gt_label']
        img_file = os.path.join(images_dir, img_name)
        img_fea = extract_model_fea(model, img_file)
        img_fea = img_fea.cpu()
        img_fea = img_fea / img_fea.norm(dim=1, keepdim=True)

        X_Test[sample_idx, :] = img_fea
        Y_test.append(img_label)

        sample_idx += 1

    prob_Test = clf.predict_proba(X_Test)

    pred_labels = []
    for i in range(sample_num):
        sample_prob = prob_Test[i]
        label = np.argmax(sample_prob)
        pred_labels.append(label)

    acc = metrics.accuracy_score(Y_test, pred_labels)
    print('acc = ', acc)

    auc = cal_metrics(query_set, prob_Test)
    print('auc = ', auc)

    return acc, auc


def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument(
        '--method',
        default='baseline',
        help='Method of few shot learning. "baseline" or "meta-baseline".')
    parser.add_argument(
        '--dataset', default='colon', help='dataset to be used')
    parser.add_argument(
        '--save-dir', default='fewshot', help='the dir to save the log')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    # step 1. load model
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('DEVICE = ', DEVICE)

    # load config file
    filepath = os.path.join(os.getcwd(), './configs/baseline_multiclass.yaml')
    with open(filepath, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    # placeholder for each model
    swin_model, effi_model, dens_model = None, None, None
    # whether to run certain model
    run_swin = cfg['model_run']['run_swin']
    run_effi = cfg['model_run']['run_effi']
    run_dens = cfg['model_run']['run_dens']

    # which method would be used in few shot learning task
    method_type = args.method

    # which dataset would be used
    dataset_type = args.dataset

    if run_swin:
        start_time = time.time()
        swin_model_config = cfg['model_cfg']['swin_model_config']
        swin_model_checkpoint = cfg['model_cfg']['swin_model_checkpoint']
        swin_model = init_model(
            swin_model_config, swin_model_checkpoint, device=DEVICE)
        print('load swin transformer model success ...')
        time_swin_model_load = time.time() - start_time
        print('time_swin_model_load = ', time_swin_model_load)

    if run_effi:
        start_time = time.time()
        effi_model_config = cfg['model_cfg']['effi_model_config']
        effi_model_checkpoint = cfg['model_cfg']['effi_model_checkpoint']
        effi_model = init_model(
            effi_model_config, effi_model_checkpoint, device=DEVICE)
        print('load efficientnet model success ...')
        time_effi_model_load = time.time() - start_time
        print('time_effi_model_load = ', time_effi_model_load)

    if run_dens:
        start_time = time.time()
        dens_model_config = cfg['model_cfg']['dens_model_config']
        dens_model_checkpoint = cfg['model_cfg']['dens_model_checkpoint']
        dens_model = init_model(
            dens_model_config, dens_model_checkpoint, device=DEVICE)
        print('load densenet model success ...')
        time_dens_model_load = time.time() - start_time
        print('time_dens_model_load = ', time_dens_model_load)

    # step 2. load image filenames and labels
    images_dir = cfg['data_cfg'][dataset_type]['images_dir']
    train_list_txt = cfg['data_cfg'][dataset_type]['train_list_txt']
    test_list_txt = cfg['data_cfg'][dataset_type]['test_list_txt']

    train_img_infos = load_annotations(train_list_txt)
    test_img_infos = load_annotations(test_list_txt)

    print('train num: ', len(train_img_infos), ', test num: ',
          len(test_img_infos))

    mmcv.mkdir_or_exist(args.save_dir)
    save_file_name = dataset_type + '_' + method_type + '_result.txt'
    fp = open(os.path.join(args.save_dir, save_file_name), 'w')

    # step 3. start few shot test
    # total class number of dataset
    N_way = cfg['data_cfg'][dataset_type]['N_way']
    # samples num of each class in support set
    K_shot_set = cfg['method_cfg']['K_shot_set']
    # number of repeated times of experiments
    max_iters = cfg['method_cfg']['max_iters']

    for K_shot in K_shot_set:

        mean_swin_run_time = 0
        mean_effi_run_time = 0
        mean_dens_run_time = 0

        print('\n----------------test K shot = {}----------------'.format(
            K_shot))
        fp.write('\n----------------test K shot = {}----------------\n'.format(
            K_shot))

        swin_accs = []
        swin_aucs = []
        effi_accs = []
        effi_aucs = []
        dens_accs = []
        dens_aucs = []

        for iter in range(max_iters):

            print('\n----------------test iter = {}----------------'.format(
                iter))

            # support set randomly sampled from the train set
            if dataset_type == 'colon':
                support_set = gen_support_set_twoclass(train_img_infos, K_shot,
                                                       'colon')
            else:
                raise ValueError(f'Invalid dataset type {dataset_type}.')

            # model 1: swin model
            if swin_model is not None:
                print('run swin-base model now ...')
                start_time = time.time()
                if method_type == 'baseline':
                    clf = baseline_cls_model(support_set, images_dir,
                                             swin_model, 1024)
                    time_swin = time.time() - start_time
                    print('time_swin = ', time_swin)
                    mean_swin_run_time += time_swin
                    acc, auc = baseline_test(test_img_infos, images_dir,
                                             swin_model, clf, N_way, 1024)
                elif method_type == 'meta-baseline':
                    cls_centers = compute_cls_centers(support_set, images_dir,
                                                      swin_model, 1024)
                    time_swin = time.time() - start_time
                    print('time_swin = ', time_swin)
                    mean_swin_run_time += time_swin
                    acc, cosine_scores = fewshot_test(test_img_infos,
                                                      cls_centers, images_dir,
                                                      swin_model)
                    auc = cal_metrics(test_img_infos, cosine_scores)
                else:
                    raise ValueError(f'Invalid method type {method_type}.')
                print('iter ', iter, ': auc = ', auc, ', acc = ', acc)
                swin_accs.append(acc)
                swin_aucs.append(auc)

            # model 2: efficientnet model
            if effi_model is not None:
                print('run efficient-b4 model now ...')
                start_time = time.time()
                if method_type == 'baseline':
                    clf = baseline_cls_model(support_set, images_dir,
                                             effi_model, 1792)
                    time_effi = time.time() - start_time
                    print('time_effi = ', time_effi)
                    mean_effi_run_time += time_effi
                    acc, auc = baseline_test(test_img_infos, images_dir,
                                             effi_model, clf, N_way, 1792)
                elif method_type == 'meta-baseline':
                    cls_centers = compute_cls_centers(support_set, images_dir,
                                                      effi_model, 1792)
                    time_effi = time.time() - start_time
                    print('time_effi = ', time_effi)
                    mean_effi_run_time += time_effi
                    acc, cosine_scores = fewshot_test(test_img_infos,
                                                      cls_centers, images_dir,
                                                      effi_model)
                    auc = cal_metrics(test_img_infos, cosine_scores)
                else:
                    raise ValueError(f'Invalid method type {method_type}.')
                print('iter ', iter, ': auc = ', auc, ', acc = ', acc)
                effi_accs.append(acc)
                effi_aucs.append(auc)

            # model 3: densenet model
            if dens_model is not None:
                print('run dense121 model now ...')
                start_time = time.time()
                if method_type == 'baseline':
                    clf = baseline_cls_model(support_set, images_dir,
                                             dens_model, 1024)
                    time_dens = time.time() - start_time
                    print('time_dens = ', time_dens)
                    mean_dens_run_time += time_dens
                    acc, auc = baseline_test(test_img_infos, images_dir,
                                             dens_model, clf, N_way, 1024)
                elif method_type == 'meta-baseline':
                    cls_centers = compute_cls_centers(support_set, images_dir,
                                                      dens_model, 1024)
                    time_dens = time.time() - start_time
                    print('time_dens = ', time_dens)
                    mean_dens_run_time += time_dens
                    acc, cosine_scores = fewshot_test(test_img_infos,
                                                      cls_centers, images_dir,
                                                      dens_model)
                    auc = cal_metrics(test_img_infos, cosine_scores)
                else:
                    raise ValueError(f'Invalid method type {method_type}.')
                print('iter ', iter, ': auc = ', auc, ', acc = ', acc)
                dens_accs.append(acc)
                dens_aucs.append(auc)

        mean_swin_run_time /= max_iters
        mean_effi_run_time /= max_iters
        mean_dens_run_time /= max_iters
        #
        mean_swin_run_time += time_swin_model_load
        mean_effi_run_time += time_effi_model_load
        mean_dens_run_time += time_dens_model_load
        #
        print('mean_dens_run_time = {} s, {} h'.format(
            mean_dens_run_time, mean_dens_run_time / 3600))
        print('mean_effi_run_time = {} s, {} h'.format(
            mean_effi_run_time, mean_effi_run_time / 3600))
        print('mean_swin_run_time = {} s, {} h'.format(
            mean_swin_run_time, mean_swin_run_time / 3600))
        fp.write('mean_dens_run_time = {} s, {} h\n'.format(
            mean_dens_run_time, mean_dens_run_time / 3600))
        fp.write('mean_effi_run_time = {} s, {} h\n'.format(
            mean_effi_run_time, mean_effi_run_time / 3600))
        fp.write('mean_swin_run_time = {} s, {} h\n'.format(
            mean_swin_run_time, mean_swin_run_time / 3600))

        if dens_model is not None:
            print('{} shot dens accuracy: max = {:.2f}, mean = {:.2f}'.format(
                K_shot,
                np.max(dens_accs) * 100,
                np.mean(dens_accs) * 100))
            print('{} shot dens auc: max = {:.2f}, mean = {:.2f}'.format(
                K_shot, np.max(dens_aucs), np.mean(dens_aucs)))
            fp.write(
                '{} shot dens accuracy: max = {:.2f}, mean = {:.2f}\n'.format(
                    K_shot,
                    np.max(dens_accs) * 100,
                    np.mean(dens_accs) * 100))
            fp.write(
                '{} shot dens auc: max = {:.2f}, mean = {:.2f}\n\n'.format(
                    K_shot, np.max(dens_aucs), np.mean(dens_aucs)))

        if effi_model is not None:
            print('{} shot effi accuracy: max = {:.2f}, mean = {:.2f}'.format(
                K_shot,
                np.max(effi_accs) * 100,
                np.mean(effi_accs) * 100))
            print('{} shot effi auc: max = {:.2f}, mean = {:.2f}'.format(
                K_shot, np.max(effi_aucs), np.mean(effi_aucs)))
            fp.write(
                '{} shot effi accuracy: max = {:.2f}, mean = {:.2f}\n'.format(
                    K_shot,
                    np.max(effi_accs) * 100,
                    np.mean(effi_accs) * 100))
            fp.write(
                '{} shot effi auc: max = {:.2f}, mean = {:.2f}\n\n'.format(
                    K_shot, np.max(effi_aucs), np.mean(effi_aucs)))

        if swin_model is not None:
            print('{} shot swin accuracy: max = {:.2f}, mean = {:.2f}'.format(
                K_shot,
                np.max(swin_accs) * 100,
                np.mean(swin_accs) * 100))
            print('{} shot swin auc: max = {:.2f}, mean = {:.2f}'.format(
                K_shot, np.max(swin_aucs), np.mean(swin_aucs)))
            fp.write(
                '{} shot swin accuracy: max = {:.2f}, mean = {:.2f}\n'.format(
                    K_shot,
                    np.max(swin_accs) * 100,
                    np.mean(swin_accs) * 100))
            fp.write(
                '{} shot swin auc: max = {:.2f}, mean = {:.2f}\n\n'.format(
                    K_shot, np.max(swin_aucs), np.mean(swin_aucs)))

    fp.close()
