# Commands (for SLURM Job Creation)
This directory contains scripts to execute commands (e.g., training or
evaluation of a model) inside a singularity container. The scripts can be used
to submit jobs to a Slurm cluster.

You can find the Slurm documentation here:
https://slurm.schedmd.com/documentation.html

## Singularity
The Slurm scripts in this directory use a Singularity Container to run the
trainings and evaluations.  
Singularity Docs: https://docs.sylabs.io/guides/3.10/user-guide/

You can use the `Dockerfile` from the root directory, build it, and the convert
it to a Singularity image.  
To convert a Docker Image to a Singularity Image file use the following
commands:
```bash
# Export/ Save Docker Image
sudo docker save <image_id> -o local.tar
# Create Singularity SIF file
singularity build local_tar.sif docker-archive://local.tar
```

## Usage of the Scripts

### Environment and Settings
- `UQ_FAE_SING_IMG`: Path to the singularity image to use.
- `UQ_FAE_SING_MOUNTS`: List of mounts for the singularity container. See
  examples below for usage info.

You generally should see the scripts as a starting point / inspiration for your
own adjustments.  
At least, you have to define the before given environment variables and set
the `<slurm-account>` and `<slurm-partion>` to values corresponding to your
Slurm cluster.

### Generic SBatch
The `generic_sbatch_uncertainty.sh` script can be used to submit arbitrary
commands that should/can run within the singularity container.

Example usage:
```bash
# Export path to image and directories that should be mounted
export UQ_FAE_SING_IMG="path/to/singularity/image/uncertainty-fae-dev_0-3.sif"
export UQ_FAE_SING_MOUNTS=\"/data/rsna_dataset:/data_fae_uq,/data/models:/ml_logs,/data/results:/ml_eval,/code/uncertainty-fae:/app\"

# Test, if Torch and Cuda work
sbatch \
  --account=<slurm-account> \
  --partition=<slurm-partition> \
  --export="UQ_FAE_SING_IMG=${UQ_FAE_SING_IMG},UQ_FAE_SING_MOUNTS=${UQ_FAE_SING_MOUNTS}" \
  generic_sbatch_uncertainty.sh \
  python -c 'import torch; print(torch.__version__); print(torch.cuda.is_available())'

# Train a SWAG model
sbatch \
  --account=<slurm-account> \
  --partition=<slurm-partition> \
  --export="UQ_FAE_SING_IMG=${UQ_FAE_SING_IMG},UQ_FAE_SING_MOUNTS=${UQ_FAE_SING_MOUNTS}" \
  generic_sbatch_uncertainty.sh \
  python /app/scripts/training.py \
  --max-epochs 250 \
  --batch-size 40 \
  --dataloader-num-workers 24 \
  --save-dir /ml_logs \
  --configuration /app/config/models.yml \
  --version swag1 \
  rsna_inception_500_gender_swag
```

### Submission - Bash Helper Scripts

The `env_export.sh` Script can be adjusted (according to (Slurm) setup) and
afterwards be used to easily set all required envs with one command:
```bash
. env_export.sh
```

#### Training - Multiple Sub-Version Submission
The `submit_multiple_subversions.sh` bash script can be used to submit a bunch
of model trainings at once.

Set environment variables:
- `UQ_SLURM_ACCOUNT`: Slurm account.
- `UQ_SLURM_PARTITION`: Slurm partition.

Usage:
```bash
./submit_multiple_subversions.sh rsna_resnet50_500_gender_mcd deepensemble 1 10
```
This will submit 10 Array Jobs for Subversions 1 ... 10 and Version=deepensemble
for the given model.

#### Evaluation - Baseline Predictions
The `submit_baseline_prediction_generation.sh` job submits an sbatch job that
generates baseline predictions (i.e., a "normal" model without any UQ) for given
model, dataset type, and checkpoint.

The following command could be used for validation data for RSNA Bone Age,
for example:
```bash
./submit_baseline_prediction_generation.sh val rsna_resnet50_500_gender_mcd /ml_logs/rsna_boneage/rsna_resnet50_500_gender_mcd/mcd_150/1/checkpoints/xyz-best-checkpoint.ckpt
```

#### Evaluation - Submission of Generation of Multiple Evaluation Predictions
To submit prediction generation runs for all evaluation models/configs given
in the example file (`eval-config.example.yml`), for eval version
`rsna_final_eval_1` and `val` dataset, run the follwoing command:

```bash
./submit_evaluation_jobs.sh rsna_final_eval_1 val rsna_resnet50_500_gender_mcd_10 rsna_resnet50_500_gender_mcd_100 rsna_resnet50_500_gender_de_10 rsna_resnet50_500_gender_de_20 rsna_resnet50_500_gender_laplace rsna_resnet50_500_gender_swag rsna_resnet50_500_gender_variance_mcd_10 rsna_resnet50_500_gender_variance_mcd_100 rsna_resnet50_500_gender_variance_de_10 rsna_resnet50_500_gender_variance_de_20
```

This command only runs predictions, no plots are created yet (as the different
jobs create predictions for different evaluation models which should be combined
in comparison plots afterwards).

To **create plots**, you can use the following srun command, for example:
```
./srun_evaluation_plot_generation.sh rsna_evaluation_best_ckpt val /app/config/evaluation.ma.rsna-boneage.yml
```
