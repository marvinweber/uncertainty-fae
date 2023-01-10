# Uncertainty Quantification in Forensic Age Estimation

## Directory Structure
- `config`: (Example) config files.
- `notebooks`: Notebooks with examples and or smaller evaluations, etc.
- `results`: Contains a few results (CSV statistics). See `results/README.md`
  for details.
- `scripts`: Python scripts for training, evaluation, etc.
- `slurm_commands`: Scripts and Description for Slurm Job Usage (for training and
  evaluation) for the FAE framework.
- `src`: Main/ Source-Code for the Uncertainty Framework.
  - `src/clavicle_ct`: Code for the *Clavicle CT* dataset.
  - `src/rsna_boneage`: Code for the *RSNA Bone Age* dataset.
  - `src/swa_gaussian`: Base code for SWAG implementation and PyTorch Lightning
    integration (see `swa_gaussian/README.md` for details).
  - `src/uncertainty_fae`: Code for the generic uncertainty part/ framework
    (e.g., interfaces, evaulation code, and so on...).
  - `src/uncertainty_fae/util`: Utility methods and classes (mainly used by or
    helful for usage with the uncertainty framework).
- `util`: Utility scripts; see `util/README.md` for details.

## Docker Setup
A `Dockerfile` is provided which defines a working environemnt for this project
and allows to reproduce trainings and evaluations.

The PyTorch Version is pinned in the Dockerfile itself, all other requirements
are managed and pinned via the `requirements.txt` file.

You can build the image as follows:
```bash
# Go to project root dir
cd /project/root/uncertainty-fae

# Build image
docker build -t uncertainty-fae-dev:1.0 .
```
See below for how to start a container with proper bind mounts.

### Bind Mounts
The following Bind-Mounts are used accross scripts and commands and as default
values in this repository. If you mount directories in the same way, all
example commands should work out of the box (inside the Docker contianer!):

- Code: `/app`
- Datasets: `/data_fae_uq`
- ML-Logs: `/ml_logs`
- Eval Logs: `/ml_eval`

You may adjust/use according to your setup/needs, of course.

### Container Start Example
The following command starts the Docker Container for the Docker Image
`uncertainty-fae-dev:1.0` with the above listed bind mounts. You must adjust the
host paths (and the user id mapping) according to your setup:
```bash
docker run -d \
  --name=<uq-fae> \
  -u <proper-user-id-mapping> \
  -v <code/repo-path>:/app \
  -v <data-path>:/data_fae_uq \
  -v <ml-logs-path>:/ml_logs \
  -v <ml-eval-path>:/ml_eval \
  --gpus all \
  --shm-size=8g \
  uncertainty-fae-dev:1.0
```

## Notebook Usage
If you want to write and/or use the notebooks, you should install the
`ipykernel` and `ipywidgets` packages:

```bash
# Inside the Container
python -m pip install ipykernel ipywidgets
```

## Uncertainty Framework
The `uncertainty_fae` together with the provided `scripts` can be seen as a
lightweight "framework" to conduct uncertainty experiments in the context of
age estimation (probably also for similar, regression-based tasks).

The package defines a few interfaces and utility classes/methods, that help to
implement and test UQ integrations.

## Datasets

### RSNA Bone Age
Source:
[RSNA Pediatric Bone Age Challenge](https://pubs.rsna.org/doi/10.1148/radiol.2018180736)  
Download: [Dataset and Annotations](https://www.rsna.org/education/ai-resources-and-training/ai-image-challenge/RSNA-Pediatric-Bone-Age-Challenge-2017)  
Content: X-ray scans of hands labeled with sex and expert-assessed bone age.

Models and Data Handling regarding this dateset are located in the
`src/rsna_boneage` package.

### Clavicle CT
Unfortunately, the **Clavicle CT** dataset is an internal data set of the
University Hospital of LMU Munich, hence, it cannot be provided publicly.

The architecture (network, LitModels and UQ integration) can still be inspected
from the `src/clavicle_ct` package.

## Scripts
The `scripts` directory contains a few scripts to run trainings, evaluations, 
and so on. Training and Evaluation are described below, apart from that:

- `create_baseline_model_predictions.py`: Can be used to create predictions with
  a model not using Uncertainty Quantification (as baseline/ for comparison).
- `clavicle_ct/generate_ood_images.py`: Script to generate/ extract OoD patches
  (see thesis for more information) from the full-body CT scans.
- `clavicle_ct/train_autoencoder.py`: Script to train an autoencoder for the
  Clavicle CT Base model (the weights of the autoencoder were used as
  initialization of the AgeNet; see thesis).

### Training
The training script allows to train a specific model for a specific dataset
(RSNA Bone Age or Clavicle CT).  
The help `python scripts/training.py --help` should explain most parameters,
for more information refer to uses in the `slurm_commands` directory.

A config (example ones are provided in the `config` directory) must be provided.
It defines available models and their parameters.

This example call trains the SWAG UQ Model for the Clavicle CT datset for
a maximum of 100 epochs:
```bash
python scripts/training.py --max-epochs 100 --config config/models.ma.yml clavicle_swag
```

### Evaluation
The `scripts/uq_evaluation.py` can be used to forwared any dataset (train,
val, test) through one of the (in the config) defined models.  
It can be used to create predictions (they are stored in CSV files) and to
create evaluation plots and metrics.

A evaluation config is required and it should define models, where to find
their checkpoints (from training), and the model parameters.  
Example evaluation configs are provided in the `config` directory and the help
of the command (`python scripts/uq_evaluation.py --help`) explains available
flags and parameters.
