import argparse
import pandas as pd
import os
import shutil

parser = argparse.ArgumentParser(description='Copy images from given annotation file and update '
                                             'image paths.')
parser.add_argument('annotation_dir', metavar='ANNOTATION_DIR', type=str,
                    help='Directory, where the annotation files are located.')
parser.add_argument('img_base_dir', metavar='IMG_BASE_DIR', type=str,
                    help='Directory, where the (training) images of the RSNA Boneage challenge '
                         'are located.')
parser.add_argument('--target-dir', metavar='TARGET_DIR', type=str, default=None, required=False,
                    help='Directory, where the (split) images / sets should be saved. Defaults to '
                         'the annotation directory.')
args = parser.parse_args()

ANNOTATION_DIR = args.annotation_dir
IMG_BASE_DIR = args.img_base_dir
TARGET_DIR = args.target_dir if args.target_dir else ANNOTATION_DIR


def main():
     target_dir_train_val = os.path.join(TARGET_DIR, 'train_val')
     target_dir_test = os.path.join(TARGET_DIR, 'test')
     for dir in [target_dir_train_val, target_dir_test]:
          os.makedirs(dir, exist_ok=True)

     df_test_annotations = pd.read_csv(
          os.path.join(ANNOTATION_DIR, 'test_annotations.csv'), index_col='id')
     df_train_val_annotations = pd.read_csv(
          os.path.join(ANNOTATION_DIR, 'train_val_annotations.csv'), index_col='id')
     df_train_annotations = pd.read_csv(
          os.path.join(ANNOTATION_DIR, 'train_annotations.csv'), index_col='id')
     df_val_annotations = pd.read_csv(
          os.path.join(ANNOTATION_DIR, 'val_annotations.csv'), index_col='id')

     for annotation_tuple in df_test_annotations.itertuples():
          img_path = copy_image(annotation_tuple, target_dir_test)
          df_test_annotations.at[annotation_tuple.Index, 'img_path'] = img_path

     # Loop Train + Val Dataset
     for df in [df_train_annotations, df_val_annotations]:
          for annotation_tuple in df.itertuples():
               img_path = copy_image(annotation_tuple, target_dir_train_val)
               df.at[annotation_tuple.Index, 'img_path'] = img_path
               df_train_val_annotations.at[annotation_tuple.Index, 'img_path'] = img_path

     df_test_annotations.to_csv(os.path.join(ANNOTATION_DIR, 'test_annotations.csv'))
     df_train_val_annotations.to_csv(os.path.join(ANNOTATION_DIR, 'train_val_annotations.csv'))
     df_train_annotations.to_csv(os.path.join(ANNOTATION_DIR, 'train_annotations.csv'))
     df_val_annotations.to_csv(os.path.join(ANNOTATION_DIR, 'val_annotations.csv'))


def copy_image(annotation_tuple, target_dir):
     image_filename = f'{annotation_tuple.Index}.png'
     source_path = os.path.join(IMG_BASE_DIR, image_filename)
     target_path = os.path.join(target_dir, image_filename)
     shutil.copy2(source_path, target_path)
     return target_path


if __name__ == '__main__':
     main()
