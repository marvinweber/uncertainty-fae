import pandas
from boneage.rsna_bone_dataloading import get_image_transforms
from PIL import Image
import torch
import os
import pathlib


DIMENSION_POOL = {
    'resnet_min': (224, 224),
    'inception_min': (299, 299),
}
TARGET_DIMENSIONS = DIMENSION_POOL['inception_min']


def preprocess_image(annotation_tuple):
    preprocessed_path = pathlib.Path(os.path.join(
        os.path.dirname(annotation_tuple.img_path),
        f'{pathlib.Path(annotation_tuple.img_path).stem}_preprocessed.pt')).as_posix()

    img = Image.open(annotation_tuple.img_path)
    img = get_image_transforms(target_dimensions=TARGET_DIMENSIONS)(img)
    torch.save(img, preprocessed_path)

    return preprocessed_path


df_test_annotations = pandas.read_csv('./images/test_annotations.csv', index_col='id')
df_train_val_annotations = pandas.read_csv('./images/train_val_annotations.csv', index_col='id')
df_train_annotations = pandas.read_csv('./images/train_annotations.csv', index_col='id')
df_val_annotations = pandas.read_csv('./images/val_annotations.csv', index_col='id')

# Loop Test Dataset
for annotation_tuple in df_test_annotations.itertuples():
    preprocessed_path = preprocess_image(annotation_tuple)
    df_test_annotations.at[annotation_tuple.Index, 'img_preprocessed_path'] = preprocessed_path

# Loop Train + Val Dataset
for df in [df_train_annotations, df_val_annotations]:
    for annotation_tuple in df.itertuples():
        preprocessed_path = preprocess_image(annotation_tuple)
        df.at[annotation_tuple.Index, 'img_preprocessed_path'] = preprocessed_path
        df_train_val_annotations.at[annotation_tuple.Index, 'img_preprocessed_path'] = preprocessed_path


df_test_annotations.to_csv('./images/with_preprocessed_test_annotations.csv')
df_train_val_annotations.to_csv('./images/with_preprocessed_train_val_annotations.csv')
df_train_annotations.to_csv('./images/with_preprocessed_train_annotations.csv')
df_val_annotations.to_csv('./images/with_preprocessed_val_annotations.csv')
