# Configurations

This directory contains (example) configuration files.

Configs ending with `*.config.yml` are ignored by Git (gitignore).

## Available Config Files
The "Model Configuration" files define the available models and their
parameters. They are required for training and evaluation, as both need to
create model.  
The "Evaluation Configuration" files are only required for the evaluation and
define the setup of the evaluation. For example, they define what models to
evaluate and where to find respective checkpoint files. 

### Model Configuration
* `models.example.yml`: Demonstrates different available configuration options
  (refer to the comments in this example configuration file for details).
* `models.ma.yml`: Includes the model setups used for the Master Thesis.

### Evaluation Configuration
* `evaluation.ma.clavicle.yml`: Configuration for the evaluation of the Clavicle
  CT models.
* `evaluation.ma.rsna-boneage.yml`: Configuration for the evaluation of the
  RSNA Bone Age models.
