import argparse
import logging
import os
from datetime import datetime
from typing import Dict, Tuple

import yaml

from rsna_boneage.model_provider import RSNAModelProvider
from uncertainty.model import TrainLoadMixin
from util import ModelProvider
from util.training import TrainConfig

logger = logging.getLogger('UNCERTAINTY_FAE_TRAINING')

TRAIN_CONFIG_FILENAME = 'config.yml'
TRAIN_RESULT_FILENAME = 'train_result.yml'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Training of RSNA Boneage Models')
    parser.add_argument('model_name', metavar='MODEL_NAME', type=str,
                        help='Name of the model to train (from model config file).')
    parser.add_argument('--configuration', metavar='CONFIGURATION', type=str, required=False,
                        default='./config/models.yml',
                        help='Path to the configuration file for available models.')
    parser.add_argument('--max-epochs', metavar='MAX_EPOCHS', type=int, default=100, required=False,
                        help='Maximum Epochs to train.')
    parser.add_argument('--early-stopping-patience', metavar='EARLY_STOPPING_PATIENCE', type=int,
                        default=10, required=False, help='Patience for EarlyStopping Callback.')
    parser.add_argument('--save-dir', metavar='SAVE_DIR', type=str, default='train_logs',
                        required=False,
                        help='Directory to save training logs (checkpoints, metrics) to.')
    parser.add_argument('--save-top-k-checkpoints', metavar='SAVE_TOP_K_CHECKPOINTS', type=int,
                        default=2, required=False, help='Amount of k best checkpoints to save.')
    parser.add_argument('--no-resume', action='store_true', required=False, default=False,
                        help='Flag to disable resume of existing training.')
    parser.add_argument('--version', metavar='VERSION', type=str, required=False, default=None,
                        help='Version name of the training (will be timestamp, if not defined). '
                             'If the version already exists and --no-resume is NOT set, training '
                             'will be resumed if possible.')
    parser.add_argument('--sub-version', type=str, required=False, default=None,
                        help='Subversion can be used to create multiple model versions and is '
                             'useful to train a deep ensemble, for example. Note that this is '
                             'setting is only allowed if --version is set, too (because '
                             'subversions would most likely not be located in the same parent '
                             'directory).')
    parser.add_argument('--debug', action='store_true', required=False, default=False,
                        help='Flag to enable output of DEBUG logs.')
    return parser.parse_args()


PROVIDER_MAPPING: Dict[str, ModelProvider] = {
    'rsna_boneage': RSNAModelProvider,
}


def train_model(train_model_name: str, model_configurations: dict, config_defaults: dict,
                train_config: TrainConfig) -> None:
    version = train_config.start_time if train_config.version is None else train_config.version
    logger.info('Training Started...')
    logger.info('Start Time: %s', train_config.start_time)
    logger.info('Model Name: %s', train_model_name)
    logger.info('Version: %s', version)

    try:
        if train_model_name not in model_configurations:
            raise ValueError(f'Unkown Model: "{train_model_name}"!')
        model_config: dict = model_configurations[train_model_name]
        if model_config['data'] not in PROVIDER_MAPPING.keys():
            raise ValueError(f'Unkown or unsupported Dataset: "{model_config["data"]}"!')
        model_provider_cls = PROVIDER_MAPPING[model_config['data']]
        model_provider = model_provider_cls.get_provider(**model_config['provider_config'])

        log_dir = os.path.abspath(
            os.path.join(train_config.save_dir, model_config['data'], train_model_name, version))
        if train_config.sub_version:
            log_dir = os.path.join(log_dir, train_config.sub_version)

        # Check whether training with this version has been started already before (in this case we
        # either try to resume the training, or abort - if resume is not allowed).
        version_exists = os.path.exists(log_dir)
        if version_exists and train_config.no_resume:
            raise RuntimeError('Version already exists but --no-resume is set to true! Abort!')

        # Save Train and Model Configuration for later verifications or sanity checks
        if not version_exists:
            os.makedirs(log_dir, exist_ok=False)
            logger.debug('Dumping train and model configuration...')
            with open(os.path.join(log_dir, TRAIN_CONFIG_FILENAME), 'w') as file:
                config = {
                    'train_config': train_config.__dict__,
                    'model_config': model_config,
                    'config_defaults': config_defaults,
                }
                yaml.dump(config, file)
        else:
            logger.info('Trying to RESUME the training...')

        # Log warning if training seems to be done already
        if os.path.exists(os.path.join(log_dir, TRAIN_RESULT_FILENAME)):
            logger.warning(f'Training started, even though {TRAIN_RESULT_FILENAME} already exists, '
                           'indicating that the training probably already has been finished!')

        litmodel_kwargs = (model_config['litmodel_config']
                           if 'litmodel_config' in model_config
                           else {})
        model = model_provider.get_model(litmodel_kwargs=litmodel_kwargs)

        # Annotation files and img base dirs
        train_af, train_d = _get_annotation_file_and_img_dir(model_config, config_defaults, 'train')
        val_af, val_d = _get_annotation_file_and_img_dir(model_config, config_defaults, 'val')
        test_af, test_d = _get_annotation_file_and_img_dir(model_config, config_defaults, 'test')

        datamodule = model_provider.get_lightning_data_module(
            train_annotation_file=train_af,
            val_annotation_file=val_af,
            test_annotation_file=test_af,
            img_train_base_dir=train_d,
            img_val_base_dir=val_d,
            img_test_base_dir=test_d,
            batch_size=train_config.batch_size,
        )

        if not isinstance(model, TrainLoadMixin):
            raise ValueError('Model must implement `TrainLoadMixin`!')

        model_cls = model.__class__
        train_result = model_cls.train_model(
            log_dir, datamodule, model, train_config, is_resume=version_exists)

        if not train_result.interrupted:
            logger.info('Training succesfully finished!')

            logger.debug('Dumping train result...')
            train_result.trainer = None  # we don't need this
            with open(os.path.join(log_dir, TRAIN_RESULT_FILENAME), 'w') as file:
                yaml.dump(train_result.__dict__, file)
        else:
            logger.warning('Training was interrupted! Maybe resume is possible.')
        logger.info(f'Logs are in: {log_dir}')

    except Exception:
        logger.critical('Training failed (see exception info below for reason)!', exc_info=True)

    logger.info('DONE')


def _get_annotation_file_and_img_dir(
        model_config: dict, config_defaults: dict, dataset: str) -> Tuple[str, str]:
    annotation_file = (model_config.get('datasets', {}).get('annotations', {}).get(dataset, {})
            or config_defaults[model_config['data']]['datasets']['annotations'][dataset])
    img_base_dir = (model_config.get('datasets', {}).get('img_base_dirs', {}).get(dataset, {})
            or config_defaults[model_config['data']]['datasets']['img_base_dirs'][dataset])
    return annotation_file, img_base_dir


if __name__ == '__main__':
    args = parse_args()
    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=level, format='%(name)s - %(asctime)s - %(levelname)s: %(message)s')

    train_model_name = args.model_name
    configuration_path = args.configuration
    train_config = TrainConfig(
        max_epochs=args.max_epochs, early_stopping_patience=args.early_stopping_patience,
        save_top_k_checkpoints=args.save_top_k_checkpoints, save_dir=args.save_dir,
        start_time=datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f'), no_resume=args.no_resume,
        version=args.version, sub_version=args.sub_version)

    with open(configuration_path, 'r') as f:
        configuration = yaml.safe_load(f)

    model_configurations = configuration['models']
    configuration_defaults = configuration['defaults']

    train_model(train_model_name, model_configurations, configuration_defaults, train_config)
