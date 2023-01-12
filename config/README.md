# Configurations

This directory contains (example) configuration files.

## `models.yml`
This configuration file defines available models and where data for training,
valdiation and testing can be found.

The `defaults` section defines per data type where the annotation files and base
directories for training, validation, and test data can be found. These values
can be overwritten on a per model basis.

The models in the configurations used for the MA-Thesis can be found in the
`models.ma.yml` file.

Configs ending with `*.config.yml` are ignored by Git (gitignore).
