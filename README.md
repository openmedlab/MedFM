# NeurIPS 2023 - MedFM: Foundation Model Prompting for Medical Image Classification Challenge 2023

A naive baseline and submission demo for the [Foundation Model Prompting for Medical Image Classification Challenge 2023 (MedFM)](https://medfm2023.grand-challenge.org/medfm2023/).

## ✨ Notification

Please check out [master branch](https://github.com/openmedlab/MedFM/tree/master). Third party implementation of MedFMC baseline is supported! It is based on the [MMPreTrain](https://github.com/open-mmlab/mmpretrain), with backbone of **`ViT-cls`**, **`ViT-eva02`**, **`ViT-dinov2`**, **`Swin-cls`** and **`ViT-clip`**.
More details could be found in its [document](https://mmpretrain.readthedocs.io/en/latest/index.html). Thanks [Ezra-Yu](https://github.com/Ezra-Yu) for this excellent work.

## 🛠️ Installation

Install requirements by

```bash
$ conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.1 -c pytorch
$ pip install mmcls==0.25.0 openmim scipy scikit-learn ftfy regex tqdm
$ mim install mmcv-full==1.6.0
```

We suggest you install PyTorch successfully first, then install OpenMMLab packages and their dependencies.

Moreover, you can use other Computer Vision or other foundation models such as [EVA](https://github.com/baaivision/EVA) and [CLIP](https://github.com/openai/CLIP).

## 📊 Results

The results of ChestDR, ColonPath and Endo in MedFMC dataset and their corresponding configs on each task are shown as below.

### Few-shot Learning Results

We utilize [Visual Prompt Tuning](https://github.com/KMnP/vpt) method as the few-shot learning baseline, whose backbone is Swin Transformer.
The results are shown as below:

#### ChestDR

| N Shot | Crop Size | Epoch |  mAP  |  AUC  |                                      Config                                      |
| :----: | :-------: | :---: | :---: | :---: | :------------------------------------------------------------------------------: |
|   1    |  384x384  |  20   | 13.14 | 56.49 | [config](configs/swin-b_vpt/in21k-swin-b_vpt5_bs4_lr5e-2_1-shot_chest_adamw.py)  |
|   5    |  384x384  |  20   | 17.05 | 64.86 | [config](configs/swin-b_vpt/in21k-swin-b_vpt5_bs4_lr5e-2_5-shot_chest_adamw.py)  |
|   10   |  384x384  |  20   | 19.01 | 66.68 | [config](configs/swin-b_vpt/in21k-swin-b_vpt5_bs4_lr5e-2_10-shot_chest_adamw.py) |

#### ColonPath

| N Shot | Crop Size | Epoch |  Acc  |  AUC  |                                      Config                                      |
| :----: | :-------: | :---: | :---: | :---: | :------------------------------------------------------------------------------: |
|   1    |  384x384  |  20   | 77.60 | 84.69 | [config](configs/swin-b_vpt/in21k-swin-b_vpt5_bs4_lr5e-2_1-shot_colon_adamw.py)  |
|   5    |  384x384  |  20   | 89.29 | 96.07 | [config](configs/swin-b_vpt/in21k-swin-b_vpt5_bs4_lr5e-2_5-shot_colon_adamw.py)  |
|   10   |  384x384  |  20   | 91.21 | 97.14 | [config](configs/swin-b_vpt/in21k-swin-b_vpt5_bs4_lr5e-2_10-shot_colon_adamw.py) |

#### Endo

| N Shot | Crop Size | Epoch |  mAP  |  AUC  |                                     Config                                      |
| :----: | :-------: | :---: | :---: | :---: | :-----------------------------------------------------------------------------: |
|   1    |  384x384  |  20   | 19.70 | 62.18 | [config](configs/swin-b_vpt/in21k-swin-b_vpt5_bs4_lr5e-2_1-shot_endo_adamw.py)  |
|   5    |  384x384  |  20   | 23.88 | 67.48 | [config](configs/swin-b_vpt/in21k-swin-b_vpt5_bs4_lr5e-2_5-shot_endo_adamw.py)  |
|   10   |  384x384  |  20   | 25.62 | 71.41 | [config](configs/swin-b_vpt/in21k-swin-b_vpt5_bs4_lr5e-2_10-shot_endo_adamw.py) |

### Transfer Learning on 20% (Fully Supervised Task)

Noted that MedFMC mainly focuses on few-shot learning i.e., transfer learning task.
Thus, fully supervised learning tasks below only use 20% training data to make corresponding comparisons.

#### ChestDR

|    Backbone     | Crop Size | Epoch |  mAP  |  AUC  |                        Config                         |
| :-------------: | :-------: | :---: | :---: | :---: | :---------------------------------------------------: |
|   DenseNet121   |  384x384  |  20   | 24.48 | 75.25 |     [config](configs/densenet/dense121_chest.py)      |
| EfficientNet-B5 |  384x384  |  20   | 29.08 | 77.21 |    [config](configs/efficientnet/eff-b5_chest.py)     |
|     Swin-B      |  384x384  |  20   | 31.07 | 78.56 | [config](configs/swin_transformer/swin-base_chest.py) |

#### ColonPath

|    Backbone     | Crop Size | Epoch |  Acc  |  AUC  |                        Config                         |
| :-------------: | :-------: | :---: | :---: | :---: | :---------------------------------------------------: |
|   DenseNet121   |  384x384  |  20   | 92.73 | 98.27 |     [config](configs/densenet/dense121_colon.py)      |
| EfficientNet-B5 |  384x384  |  20   | 94.04 | 98.58 |    [config](configs/efficientnet/eff-b5_colon.py)     |
|     Swin-B      |  384x384  |  20   | 94.68 | 98.35 | [config](configs/swin_transformer/swin-base_colon.py) |

#### Endo

|    Backbone     | Crop Size | Epoch |  mAP  |  AUC  |                        Config                        |
| :-------------: | :-------: | :---: | :---: | :---: | :--------------------------------------------------: |
|   DenseNet121   |  384x384  |  20   | 41.13 | 80.19 |     [config](configs/densenet/dense121_endo.py)      |
| EfficientNet-B5 |  384x384  |  20   | 36.95 | 78.23 |    [config](configs/efficientnet/eff-b5_endo.py)     |
|     Swin-B      |  384x384  |  20   | 41.38 | 79.42 | [config](configs/swin_transformer/swin-base_endo.py) |

## 🎫 License

This project is released under the [Apache 2.0 license](LICENSE).

## 🙌 Usage

### Data preparation

Prepare data following [MMClassification](https://github.com/open-mmlab/mmclassification). The data structure looks like below:

```text
data/
├── MedFMC
│   ├── chest
│   │   ├── images
│   │   ├── chest_X-shot_train_expY.txt
│   │   ├── chest_X-shot_val_expY.txt
│   │   ├── train_20.txt
│   │   ├── val_20.txt
│   │   ├── trainval.txt
│   │   ├── test_WithLabel.txt
│   ├── colon
│   │   ├── images
│   │   ├── colon_X-shot_train_expY.txt
│   │   ├── colon_X-shot_val_expY.txt
│   │   ├── train_20.txt
│   │   ├── val_20.txt
│   │   ├── trainval.txt
│   │   ├── test_WithLabel.txt
│   ├── endo
│   │   ├── images
│   │   ├── endo_X-shot_train_expY.txt
│   │   ├── endo_X-shot_val_expY.txt
│   │   ├── train_20.txt
│   │   ├── val_20.txt
│   │   ├── trainval.txt
│   │   ├── test_WithLabel.txt
```

Noted that the `.txt` files includes data split information for fully supervised learning and few-shot learning tasks.
The public dataset is splitted to `trainval.txt` and `test_WithLabel.txt`, and `trainval.txt` is also splitted to `train_20.txt` and `val_20.txt` where `20` means the training data makes up 20% of `trainval.txt`.
And the `test_WithoutLabel.txt` of each dataset is validation set.

Corresponding `.txt` files are stored at `./data_backup/` folder, the few-shot learning data split files `{dataset}_{N_shot}-shot_train/val_exp{N_exp}.txt` could also be generated as below:

```shell
python tools/generate_few-shot_file.py
```

Where `N_shot` is 1,5 and 10, respectively, the shot is of patient(i.e., 1-shot means images of certain one patient are all counted as one), not number of images.

The `images` in each dataset folder contains its images, which could be achieved from original dataset.

### Training and evaluation using OpenMMLab codebases.

In this repository we provided many config files for fully supervised task (only uses 20% of original traning set, please check out the `.txt` files which split dataset)
and few-shot learning task.

The config files of fully supervised transfer learning task are stored at `./configs/densenet`, `./configs/efficientnet`, `./configs/vit-base` and
`./configs/swin_transformer` folders, respectively. The config files of few-shot learning task are stored at `./configs/ablation_exp` and `./configs/vit-b16_vpt` folders.

For the training and testing, you can directly use commands below to train and test the model:

```bash
# you need to export path in terminal so the `custom_imports` in config would work
export PYTHONPATH=$PWD:$PYTHONPATH
# Training
# you can choose a config file like `configs/vit-b16_vpt/in21k-vitb16_vpt1_bs4_lr6e-4_1-shot_chest.py` to train its model
python tools/train.py $CONFIG

# Evaluation
# Endo and ChestDR utilize mAP as metric
python tools/test.py $CONFIG $CHECKPOINT --metrics mAP
python tools/test.py $CONFIG $CHECKPOINT --metrics AUC_multilabel
# Colon utilizes accuracy as metric
python tools/test.py $CONFIG $CHECKPOINT --metrics accuracy --metric-options topk=1
python tools/test.py $CONFIG $CHECKPOINT --metrics AUC_multiclass

```

The repository is built upon [MMClassification/MMPretrain](https://github.com/open-mmlab/mmpretrain/tree/master). More details could be found in its [document](https://mmpretrain.readthedocs.io/en/mmcls-0.x/).

### Generating Submission results of Validation Phase

Noted:

- The order of filanames of all CSV files must follow the order of provided `colon_val.csv`, `chest_val.csv` and `endo_val.csv`! You can see files in `./data_backup/result_sample` for more details.
- The name of CSV files in `result.zip` must be the same names `xxx_N-shot_submission.csv` below.

Run

```bash
python tools/test_prediction.py $DATASETPATH/test_WithoutLabel.txt $DATASETPATH/images/ $CONFIG $CHECKPOINT --output-prediction $DATASET_N-shot_submission.csv
```

For example:

```bash
python tools/test_prediction.py data/MedFMC/endo/test_WithoutLabel.txt data/MedFMC/endo/images/ $CONFIG $CHECKPOINT --output-prediction endo_10-shot_submission.csv
```

You can generate all prediction results of `endo_N-shot_submission.csv`, `colon_N-shot_submission.csv` and `chest_N-shot_submission.csv` and zip them into `result.zip` file. Then upload it to Grand Challenge website.

```
result/
├── endo_1-shot_submission.csv
├── endo_5-shot_submission.csv
├── endo_10-shot_submission.csv
├── colon_1-shot_submission.csv
├── colon_5-shot_submission.csv
├── colon_10-shot_submission.csv
├── chest_1-shot_submission.csv
├── chest_5-shot_submission.csv
├── chest_10-shot_submission.csv
```

Then using `zip` to make them as `.zip` file(i.e., `result_sample.zip` in `./data_backup` folder) and upload it to submission site of [Grand Challenge MedFMC Validation Phase](https://medfm2023.grand-challenge.org/evaluation/challenge-validation-results-submission-only/submissions/create/).

## 🖊️ Citation

```
@article{wang2023real,
  title={A real-world dataset and benchmark for foundation model adaptation in medical image classification},
  author={Wang, Dequan and Wang, Xiaosong and Wang, Lilong and Li, Mengzhang and Da, Qian and Liu, Xiaoqiang and Gao, Xiangyu and Shen, Jun and He, Junjun and Shen, Tian and others},
  journal={Scientific Data},
  volume={10},
  number={1},
  pages={574},
  year={2023},
  publisher={Nature Publishing Group UK London}
}
```
