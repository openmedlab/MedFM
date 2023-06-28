import numpy as np
import os
from argparse import ArgumentParser
from mmpretrain import ImageClassificationInferencer
from mmengine.fileio import list_dir_or_file, list_from_file


def main():
    parser = ArgumentParser()
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('images', help='Names of test images, could a image list file or a folder')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--out',
        help='where to save prediction in csv file',
        default="result.csv")
    parser.add_argument(
        '--batch-size',
        help='the batch-size of the inferencer',
        type=int,
        default=1)
    args = parser.parse_args()
    
    # get all the inference image list
    if os.path.isfile(args.images) and args.images.endswith(".txt"):
        images = [image for image in list_from_file(args.images)]
    elif os.path.isdir(args.images):
        images = [
            os.path.join(args.images, image)
            for image in list_dir_or_file(args.images, suffix='.png', list_dir=False)
        ]
    else:
        raise ValueError("please set `args.images` a '.txt' file or a folder.")

    # build the model from a config file and a checkpoint file
    inferencer = ImageClassificationInferencer(
                                    model=args.config, 
                                    pretrained=args.checkpoint, 
                                    device=args.device)

    results = inferencer(images, batch_size=args.batch_size)

    # test a bundle of images
    with open(args.out, 'w') as f_out:
        for image, res in zip(images, results):
            print(os.path.basename(image), res['pred_label'])
            f_out.write(os.path.basename(image))
            for j in range(len(res['pred_scores'])):
                f_out.write(' ' + str(np.around(res['pred_scores'][j], 8)))
            f_out.write('\n')


if __name__ == '__main__':
    main()
