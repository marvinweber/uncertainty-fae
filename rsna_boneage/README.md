# RSNA Boneage Model and Uncertainty Quantification

## Execution
Execute all files from root as working directory.

## Scripts
- `training.py`: Train a Model
- `eval_single_image.py`: Evaluate performance and uncertainty for single given image and model.
- `eval_model.py`: Evaluate performance and uncertainty for given model on entire validation set.
- `preprocess_images.py`: Rescale images to given target dimension and save as torch Tensor.
- `split_datasets_and_create_annotations.py`: Split dataset into train/val and test dataset.
