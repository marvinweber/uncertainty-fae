import logging
import os

import yaml

from rsna_boneage.model_provider import RSNAModelProvider
from uncertainty_fae.model import TrainLoadMixin
from uncertainty_fae.util import ModelProvider
from uncertainty_fae.util.config import TrainConfig, parse_cli_args

logger = logging.getLogger('UNCERTAINTY_FAE_TRAINING')

TRAIN_CONFIG_FILENAME = 'config.yml'
TRAIN_RESULT_FILENAME = 'train_result.yml'

PROVIDER_MAPPING: dict[str, ModelProvider] = {
    'rsna_boneage': RSNAModelProvider,
}


def train_model(train_model_name: str, model_configurations: dict, config_defaults: dict,
                train_config: TrainConfig) -> None:
    version = train_config.start_time if train_config.version is None else train_config.version
    logger.info('Training Started...')
    logger.info('Start Time: %s', train_config.start_time)
    logger.info('Model Name: %s', train_model_name)
    logger.info('Version: %s', version)
    if train_config.sub_version:
        logger.info('Subversion: %s', train_config.sub_version)

    try:
        if train_model_name not in model_configurations:
            raise ValueError(f'Unkown Model: "{train_model_name}"!')
        model_config: dict = model_configurations[train_model_name]
        if model_config['data'] not in PROVIDER_MAPPING.keys():
            raise ValueError(f'Unkown or unsupported Dataset: "{model_config["data"]}"!')
        model_provider_cls = PROVIDER_MAPPING[model_config['data']]
        model_provider = model_provider_cls.get_provider(
            train_config, **model_config['provider_config'])

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
            logger.info(f'File "{TRAIN_RESULT_FILENAME}" already exists, indicating that the '
                        'training has already been finished! SKIPPING!')
            return

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
            num_workers=train_config.dataloader_num_workers,
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


def _get_annotation_file_and_img_dir(
        model_config: dict, config_defaults: dict, dataset: str) -> tuple[str, str]:
    annotation_file = (model_config.get('datasets', {}).get('annotations', {}).get(dataset, {})
            or config_defaults[model_config['data']]['datasets']['annotations'][dataset])
    img_base_dir = (model_config.get('datasets', {}).get('img_base_dirs', {}).get(dataset, {})
            or config_defaults[model_config['data']]['datasets']['img_base_dirs'][dataset])
    return annotation_file, img_base_dir


if __name__ == '__main__':
    cli_config = parse_cli_args('training')
    level = logging.DEBUG if cli_config['debug'] else logging.INFO
    logging.basicConfig(level=level, format='%(name)s - %(asctime)s - %(levelname)s: %(message)s')

    train_model_name = cli_config['model_name']
    configuration_path = cli_config['configuration']
    train_config = TrainConfig(cli_config)

    with open(configuration_path, 'r') as f:
        configuration = yaml.safe_load(f)

    model_configurations = configuration['models']
    configuration_defaults = configuration['defaults']

    train_model(train_model_name, model_configurations, configuration_defaults, train_config)
    logger.info('DONE')
