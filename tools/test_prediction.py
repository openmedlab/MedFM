import mmcv
import numpy as np
import os
import torch
from argparse import ArgumentParser
from mmcls.apis import inference_model, init_model
from mmcls.datasets.pipelines import Compose
from mmcv.parallel import collate, scatter


def inference_model(model, img):
    """Inference image(s) with the classifier.

    Args:
        model (nn.Module): The loaded classifier.
        img (str/ndarray): The image filename or loaded image.

    Returns:
        result (dict): The classification results that contains
            `class_name`, `pred_label` and `pred_score`.
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
        scores = model(return_loss=False, **data)
    return scores


def main():
    parser = ArgumentParser()
    parser.add_argument('img_file', help='Names of test image files')
    parser.add_argument('img_path', help='Path of test image files')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--output-prediction',
        help='where to save prediction in csv file',
        default=False)
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)
    # test a bundle of images
    if args.output_prediction:
        with open(args.output_prediction, 'w') as f_out:
            for line in open(args.img_file, 'r'):
                image_name = line.split('\n')[0]
                file = os.path.join(args.img_path, image_name)
                result = inference_model(model, file)[0]
                f_out.write(image_name)
                for j in range(len(result)):
                    f_out.write(',' + str(np.around(result[j], 8)))
                f_out.write('\n')


if __name__ == '__main__':
    main()
