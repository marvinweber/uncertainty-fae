import csv
import os
import pandas
import pathlib
import random
import shutil


raw_annotation_file = 'C:/Users/marvi/Downloads/rsna-bone-age-dataset/boneage-training-dataset.csv'
img_base_path = 'C:/Users/marvi/Downloads/rsna-bone-age-dataset/boneage-training-dataset/boneage-training-dataset'

PERCENTAGE_TEST = 0.1
TRAIN_VAL_PERCENTAGE_TRAIN = 0.8
SEPARATE_TEST_IMAGES = True

raw_annotations = pandas.read_csv(raw_annotation_file)
train_val_dir = './images/train_val'
test_dir = './images/test'

CSV_COLUMNS = ['id', 'img_path', 'img_preprocessed_path', 'boneage', 'male']

if SEPARATE_TEST_IMAGES:
    for dir in [train_val_dir, test_dir]:
        if os.path.exists(dir):
            shutil.rmtree(dir)
        os.makedirs(dir, exist_ok=True)

    test_annotations_file = open('./images/test_annotations.csv', 'w', newline='')
    csv_writer_test_annotations = csv.writer(test_annotations_file)
    csv_writer_test_annotations.writerow(CSV_COLUMNS)

    train_val_annotations_file = open('./images/train_val_annotations.csv', 'w', newline='')
    csv_writer_train_val_annotations = csv.writer(train_val_annotations_file)
    csv_writer_train_val_annotations.writerow(CSV_COLUMNS)

    for annotation in raw_annotations.itertuples():
        filename = f'{annotation.id}.png'
        source_path = os.path.join(img_base_path, filename)
        source_path = pathlib.Path(source_path).as_posix()

        r = random.random()
        is_test_image = r <= PERCENTAGE_TEST
        folder_to = test_dir if is_test_image else train_val_dir
        target_path = os.path.join(folder_to, filename)
        target_path = pathlib.Path(target_path).absolute().as_posix()

        shutil.copy2(source_path, target_path)

        (csv_writer_test_annotations if is_test_image else csv_writer_train_val_annotations).writerow([
            annotation.id,
            target_path,
            '',
            annotation.boneage,
            annotation.male,
        ])
    test_annotations_file.close()
    train_val_annotations_file.close()


train_annotations_file = open('./images/train_annotations.csv', 'w', newline='')
csv_writer_train_annotations = csv.writer(train_annotations_file)
csv_writer_train_annotations.writerow(CSV_COLUMNS)

val_annotations_file = open('./images/val_annotations.csv', 'w', newline='')
csv_writer_val_annotations = csv.writer(val_annotations_file)
csv_writer_val_annotations.writerow(CSV_COLUMNS)

train_val_annotations = pandas.read_csv('./images/train_val_annotations.csv')
for annotation in train_val_annotations.itertuples():
    r = random.random()
    is_train_image = r <= TRAIN_VAL_PERCENTAGE_TRAIN

    (csv_writer_train_annotations if is_train_image else csv_writer_val_annotations).writerow([
        annotation.id,
        annotation.img_path,
        annotation.img_preprocessed_path,
        annotation.boneage,
        annotation.male,
    ])

print('Done!')
