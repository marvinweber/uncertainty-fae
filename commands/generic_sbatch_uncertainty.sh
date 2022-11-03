#!/bin/sh
#
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=24
#SBATCH --mem=32GB
#SBATCH -o logs/slurm/slurm%j.log
#SBATCH -e logs/slurm/slurm%j.err
#SBATCH -J uncertainty_fae_model_training

echo "Using Image: ${UQ_FAE_SING_IMG}"
echo "Mounts: ${UQ_FAE_SING_MOUNTS}"
echo "###############################"

singularity exec --pwd /app --nv --no-home -B "$UQ_FAE_SING_MOUNTS" "$UQ_FAE_SING_IMG" "$@"
