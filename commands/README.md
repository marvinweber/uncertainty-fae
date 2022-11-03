# Commands (for SLURM Job Creation)
This directory contains scripts to execute commands (e.g., training or
evaluation of a model) inside a singularity container. The scripts can be used
to submit jobs to a Slurm cluster.

## Singularity
TBD TODO

## Usage of the Scripts
The `generic_sbatch_uncertainty.sh` script can be used to submit arbitrary
commands that should/can run within the singularity container.  

You have to define the container/image path in the env variable:
`$UQ_FAE_SINGULARITY_IMG`.

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
```
