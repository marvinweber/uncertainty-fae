import logging
import os

import yaml

from rsna_boneage.model_provider import RSNAModelProvider
from uncertainty_fae.model import TrainLoadMixin
from uncertainty_fae.util import ModelProvider, TrainConfig
from uncertainty_fae.util.config import parse_cli_args

logger = logging.getLogger('UNCERTAINTY_FAE_TRAINING')

TRAIN_CONFIG_FILENAME = 'config.yml'
TRAIN_RESULT_FILENAME = 'train_result.yml'

PROVIDER_MAPPING: dict[str, ModelProvider] = {
    'rsna_boneage': RSNAModelProvider,
}


def train_model(train_model_name: str, train_config: TrainConfig) -> None:
    version = train_config.start_time if train_config.version is None else train_config.version
    logger.info('Training Started...')
    logger.info('Start Time: %s', train_config.start_time)
    logger.info('Model Name: %s', train_model_name)
    logger.info('Version: %s', version)
    if train_config.sub_version:
        logger.info('Subversion: %s', train_config.sub_version)

    try:
        if train_model_name not in train_config.model_configurations:
            raise ValueError(f'Unkown Model: "{train_model_name}"!')
        model_config: dict = train_config.model_configurations[train_model_name]

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
                }
                yaml.dump(config, file)
        else:
            logger.info('Trying to RESUME the training...')

        # Log warning if training seems to be done already
        if os.path.exists(os.path.join(log_dir, TRAIN_RESULT_FILENAME)):
            logger.info(f'File "{TRAIN_RESULT_FILENAME}" already exists, indicating that the '
                        'training has already been finished! SKIPPING!')
            return

        model, datamodule = train_config.get_model_and_datamodule(
            PROVIDER_MAPPING,
            train_model_name,
        )

        if not isinstance(model, TrainLoadMixin):
            raise ValueError('Model must implement `TrainLoadMixin`!')

        model_cls = model.__class__
        train_result = model_cls.train_model(
            log_dir,
            datamodule,
            model,
            train_config,
            is_resume=version_exists,
        )

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


if __name__ == '__main__':
    cli_config = parse_cli_args('training')

    level = logging.DEBUG if cli_config['debug'] else logging.INFO
    logging.basicConfig(level=level, format='%(name)s - %(asctime)s - %(levelname)s: %(message)s')

    train_config = TrainConfig(cli_config)
    train_model_name = cli_config['model_name']
    train_model(train_model_name, train_config)
    logger.info('DONE')
