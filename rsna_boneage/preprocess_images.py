import argparse
import os
import pathlib
from datetime import datetime
from typing import Tuple

import pandas as pd
import torch
from PIL import Image

from boneage.rsna_bone_dataloading import get_image_transforms
from util.eval import print_progress

DIMENSION_POOL = {
    'resnet_min': (224, 224),
    'inception_min': (299, 299),
    '500x500': (500, 500),
}

parser = argparse.ArgumentParser(description='Preprocessing of RSNA images (dimension scaling).')
parser.add_argument('annotation_dir', type=str,
                    help='Directory containing the annotations for the raw images.')
parser.add_argument('target_dir', type=str,
                    help='Target base dir where to save preprocessed images.')
parser.add_argument('--target-dim', type=str, choices=DIMENSION_POOL.keys(), default='resnet_min',
                    required=False, help='Dimensions to resize images to.')


def preprocess_image(annotation_tuple, target_dir: str, preprocessing_name: str,
                     target_dimensions: Tuple[int, int]):
    img_preprocessed_path = pathlib.Path(os.path.join(
        target_dir,
        f'{pathlib.Path(annotation_tuple.img_path).stem}_preprocessed_{preprocessing_name}.pt'
    )).as_posix()

    img = Image.open(annotation_tuple.img_path)
    img = get_image_transforms(target_dimensions=target_dimensions)(img)
    torch.save(img, img_preprocessed_path)

    return img_preprocessed_path


def preprocess_all(annotation_dir: str, target_dir: str, target_dimensions: Tuple[int, int]):
    preprocessing_name = f'{target_dimensions[0]}x{target_dimensions[1]}'
    print('Preprocessing <', preprocessing_name, '> started; saving images to:', target_dir)

    train_val_dir = os.path.join(target_dir, 'train_val')
    test_dir = os.path.join(target_dir, 'test')
    os.makedirs(train_val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    df_test_annotations = pd.read_csv(
        os.path.join(annotation_dir, 'test_annotations.csv'), index_col='id')
    df_train_val_annotations = pd.read_csv(
        os.path.join(annotation_dir, 'train_val_annotations.csv'), index_col='id')
    df_train_annotations = pd.read_csv(
        os.path.join(annotation_dir, 'train_annotations.csv'), index_col='id')
    df_val_annotations = pd.read_csv(
        os.path.join(annotation_dir, 'val_annotations.csv'), index_col='id')

    # Progress Logging
    position = 1
    data_length = len(df_test_annotations) + len(df_train_val_annotations)
    start = datetime.now()
    def _prog_logger(position: int):
        if position % 200 == 0:
            print_progress(position, data_length, start)
        return position + 1

    # Loop Test Dataset
    for annotation_tuple in df_test_annotations.itertuples():
        path = preprocess_image(annotation_tuple, test_dir, preprocessing_name, target_dimensions)
        df_test_annotations.at[annotation_tuple.Index, 'img_preprocessed_path'] = path
        position = _prog_logger(position)
        

    # Loop Train + Val Dataset
    for df in [df_train_annotations, df_val_annotations]:
        for annotation_tuple in df.itertuples():
            path = preprocess_image(annotation_tuple, train_val_dir, preprocessing_name,
                                    target_dimensions)
            df.at[annotation_tuple.Index, 'img_preprocessed_path'] = path
            df_train_val_annotations.at[annotation_tuple.Index, 'img_preprocessed_path'] = path
            position = _prog_logger(position)

    df_test_annotations.to_csv(
        os.path.join(annotation_dir, f'test_annotations_preprocessed_{preprocessing_name}.csv'))
    df_train_val_annotations.to_csv(
        os.path.join(annotation_dir,
                     f'train_val_annotations_preprocessed_{preprocessing_name}.csv'))
    df_train_annotations.to_csv(
        os.path.join(annotation_dir, f'train_annotations_preprocessed_{preprocessing_name}.csv'))
    df_val_annotations.to_csv(
        os.path.join(annotation_dir, f'val_annotations_preprocessed_{preprocessing_name}.csv'))


if __name__ == '__main__':
    args = parser.parse_args()
    annotation_dir = args.annotation_dir
    target_dir = args.target_dir
    target_dimensions = DIMENSION_POOL[args.target_dim]

    preprocess_all(annotation_dir, target_dir, target_dimensions)
