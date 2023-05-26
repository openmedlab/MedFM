import os
import random
import yaml
from baseline_multiclass import gen_support_set_twoclass, load_annotations
from baseline_multilabel import (gen_support_set, gen_support_set_endo,
                                 load_chest_annotations, load_endo_annotations)

K_shot_lst = [1, 5, 10]
# make total validation number not larger than 3000
val_num = 3000
dataset_type_lst = ['endo', 'colon', 'chest']
exp_num_total = 1

for exp_num in range(1, 1 + exp_num_total):
    for dataset_type in dataset_type_lst:
        for K_shot in K_shot_lst:
            # load config file of Colon
            if dataset_type == 'colon':
                filepath = os.path.join(os.getcwd(),
                                        './configs/baseline_multiclass.yaml')
            elif dataset_type == 'chest' or dataset_type == 'endo':
                filepath = os.path.join(os.getcwd(),
                                        './configs/baseline_multilabel.yaml')

            with open(filepath, 'r') as f:
                cfg = yaml.load(f, Loader=yaml.FullLoader)
            train_list_txt = cfg['data_cfg'][dataset_type]['train_list_txt']

            if dataset_type == 'colon':
                train_img_infos = load_annotations(train_list_txt)
            elif dataset_type == 'endo':
                train_img_infos = load_endo_annotations(train_list_txt)
            elif dataset_type == 'chest':
                train_img_infos = load_chest_annotations(train_list_txt)
            else:
                raise ValueError(f'Invalid dataset type {dataset_type}.')

            if dataset_type == 'colon':
                support_set = gen_support_set_twoclass(train_img_infos, K_shot,
                                                       'colon')
                few_shot_lst = []
                with open(f'colon_{K_shot}-shot_train_exp{exp_num}.txt',
                          'w') as f:
                    for i, i_class in enumerate(support_set):
                        for j_id in support_set[i]:
                            f.write(j_id + ' ' + str(i) + '\n')
                            few_shot_lst.append(j_id)
                # generate validation set txt file
                few_shot_val_lst = []
                with open(f'colon_{K_shot}-shot_val_exp{exp_num}.txt',
                          'w') as f:
                    val_cur = 0
                    while val_cur < val_num:
                        num = random.randint(0, len(train_img_infos) - 1)
                        filename = train_img_infos[num]['filename']
                        gt_label = train_img_infos[num]['gt_label']
                        if (filename not in few_shot_lst) and (
                                filename not in few_shot_val_lst):
                            few_shot_val_lst.append(filename)
                            f.write(filename + ' ' + str(gt_label) + '\n')
                            val_cur += 1
                        if len(few_shot_val_lst) + len(few_shot_lst) == len(
                                train_img_infos):
                            print('The validation set are not enough...')
                            print(
                                len(train_img_infos), len(few_shot_lst),
                                len(few_shot_val_lst))
                            break

            if dataset_type == 'endo':
                # total class number of dataset
                N_way = cfg['data_cfg'][dataset_type]['N_way']
                support_set = gen_support_set_endo(train_img_infos, N_way,
                                                   K_shot)
                # used in generating .txt file
            elif dataset_type == 'chest':
                N_way = cfg['data_cfg'][dataset_type]['N_way']
                support_set = gen_support_set(train_img_infos, N_way, K_shot)
            if dataset_type == 'endo' or dataset_type == 'chest':
                few_shot_lst = []
                with open(
                        f'{dataset_type}_{K_shot}-shot_train_exp{exp_num}.txt',
                        'w') as f:
                    for i, i_class in enumerate(support_set):
                        for j_id in support_set[i]:
                            j_pid, j_label = j_id
                            k_line = j_pid
                            few_shot_lst.append(j_id)
                            for k, k_label in enumerate(j_label.tolist()):
                                if dataset_type == 'endo':
                                    sep_str = ' '
                                elif dataset_type == 'chest' and k == 0:
                                    sep_str = ' '
                                elif dataset_type == 'chest' and k != 0:
                                    sep_str = ','
                                k_line += sep_str + str(k_label)
                            f.write(k_line + '\n')

                few_shot_val_lst = []
                with open(f'{dataset_type}_{K_shot}-shot_val_exp{exp_num}.txt',
                          'w') as f:
                    val_cur = 0
                    while val_cur < val_num:
                        num = random.randint(0, len(train_img_infos) - 1)
                        filename = train_img_infos[num]['filename']
                        gt_label = train_img_infos[num]['gt_label']
                        if (filename not in few_shot_lst) and (
                                filename not in few_shot_val_lst):
                            few_shot_val_lst.append(filename)
                            k_line = filename
                            for j, j_label in enumerate(gt_label.tolist()):
                                if dataset_type == 'endo':
                                    sep_str = ' '
                                elif dataset_type == 'chest' and j == 0:
                                    sep_str = ' '
                                elif dataset_type == 'chest' and j != 0:
                                    sep_str = ','
                                k_line += sep_str + str(j_label)
                            f.write(k_line + '\n')
                            val_cur += 1
                        if len(few_shot_val_lst) + len(few_shot_lst) == len(
                                train_img_infos):
                            print('The validation set are not enough...')
                            print(
                                len(train_img_infos), len(few_shot_lst),
                                len(few_shot_val_lst))
                            break
