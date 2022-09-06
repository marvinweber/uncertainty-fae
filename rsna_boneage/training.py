import argparse
import os
from datetime import datetime

import pandas as pd
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torchsummary import summary

from util.model_provider import get_model_and_datamodule

parser = argparse.ArgumentParser(description='Training of RSNA Boneage Models')
parser.add_argument('model_train_config', metavar='MODEL_TRAIN_CONFIG', type=str,
                    help='Model Configuration for Training (csv file).')
parser.add_argument('--max-epochs', metavar='MAX_EPOCHS', type=int, default=100, required=False,
                    help='Maximum Epochs to train.')
parser.add_argument('--early-stopping-patience', metavar='EARLY_STOPPING_PATIENCE', type=int,
                    default=10, required=False, help='Patience for EarlyStopping Callback.')
parser.add_argument('--save-dir', metavar='SAVE_DIR', type=str, default='train_logs',
                    required=False,
                    help='Directory to save training logs (checkpoints, metrics) to.')
parser.add_argument('--save-top-k-checkpoints', metavar='SAVE_TOP_K_CHECKPOINTS', type=int,
                    default=5, required=False, help='Amount of k best checkpoints to save.')


def train_all(model_train_config_filepath: str, max_epochs: int, early_stopping_patience: int,
              save_dir: str = 'train_logs', save_top_k_checkpoints: int = 5):
    version = datetime.now().strftime('%Y-%m-%d-%H-%M')
    train_configs = pd.read_csv(model_train_config_filepath, index_col='model_id')
    train_log = train_configs.copy(True)
    train_log_file = os.path.join(save_dir, f'train_log_{version}.csv')

    print(f'Training Started. Max Epochs: {max_epochs}, Early Stopping '
          f'Patience: {early_stopping_patience}.')

    for train_config in train_configs.itertuples():
        print('\n\n\n######### NEXT MODEL ###############')

        img_input_dimensions=(train_config.img_input_width, train_config.img_input_height)
        model_log_name = (f'{train_config.name}_{train_config.img_input_width}x'
                          f'{train_config.img_input_height}')
        if train_config.with_gender_input:
            model_log_name = f'{model_log_name}_with_gender_input'
        
        print('Model:', model_log_name)
        log_dir = os.path.join(save_dir, model_log_name, version)
        checkpoint_dir = os.path.join(log_dir, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)

        train_log.at[train_config.Index, 'max_epochs'] = max_epochs
        train_log.at[train_config.Index, 'early_stopping_patience'] = early_stopping_patience
        train_log.at[train_config.Index, 'log_name'] = model_log_name
        train_log.at[train_config.Index, 'log_dir'] = log_dir
        train_log.at[train_config.Index, 'checkpoint_dir'] = checkpoint_dir
        train_log.to_csv(train_log_file)

        model, rsna_boneage_datamodule = get_model_and_datamodule(
            train_config.name,
            img_input_dimension=img_input_dimensions,
            with_gender_input=train_config.with_gender_input)

        tb_logger = pl_loggers.TensorBoardLogger(
            save_dir=save_dir, name=model_log_name, version=version)
        csv_logger = pl_loggers.CSVLogger(
            save_dir=save_dir, name=model_log_name, version=version)

        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir, monitor='val_loss', save_top_k=save_top_k_checkpoints,
            save_last=True, filename='{epoch}-{val_loss:2f}')
        early_stopping_callback = EarlyStopping(
            monitor='val_loss', mode='min', patience=early_stopping_patience)
        callbacks = [checkpoint_callback, early_stopping_callback]

        trainer = Trainer(accelerator='gpu', max_epochs=max_epochs, log_every_n_steps=50,
                          logger=[csv_logger, tb_logger], callbacks=callbacks)
        trainer.fit(model, datamodule=rsna_boneage_datamodule)
        print('Training done!')


if __name__ == '__main__':
    args = parser.parse_args()
    
    model_train_config_filepath = args.model_train_config
    max_epochs = args.max_epochs
    early_stopping_patience = args.early_stopping_patience
    save_dir = args.save_dir
    save_top_k_checkpoints = args.save_top_k_checkpoints

    train_all(model_train_config_filepath, max_epochs, early_stopping_patience,
              save_dir=save_dir, save_top_k_checkpoints=save_top_k_checkpoints)
