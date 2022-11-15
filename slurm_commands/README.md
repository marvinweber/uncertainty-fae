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
export UQ_FAE_SING_IMG="path/to/singularity/image/uncertainty-fae-dev_0-2.sif"
export UQ_FAE_SING_MOUNTS=\"/data/rsna_dataset:/data_fae_uncertainty,/data/uncertainty-fae:/app\"

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
  python training.py \
  --max-epochs 250 \
  --batch-size 40 \
  --dataloader-num-workers 24 \
  --save-dir /ml_logs \
  --configuration /app/config/models.yml \
  --version swag1 \
  rsna_inception_500_gender_swag
```

### Submit Multiple Subversions
The `submit_multiple_subversions.sh` bash script can be used to submit a bunch
of model trainings at once.

Usage:
```bash
./submit_multiple_subversions.sh rsna_resnet50_500_gender_mcd deepensemble 1 10
```
This will submit 10 Array Jobs for Subversions 1 ... 10 and Version=deepensemble
for the given model.
