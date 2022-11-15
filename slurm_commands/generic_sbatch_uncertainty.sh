#!/bin/sh
#
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=32
#SBATCH --mem=32GB
#SBATCH -o logs/slurm/slurm%A_%a.log
#SBATCH -e logs/slurm/slurm%A_%a.log
#SBATCH -J uncertainty_fae_model_training
#SBATCH --array=1-3%1

echo "Using Image: ${UQ_FAE_SING_IMG}"
echo "Mounts: ${UQ_FAE_SING_MOUNTS}"
echo "###############################"
echo ""

singularity exec --pwd /app --nv --no-home --cleanenv --env TORCH_HOME=/tmp/torch -B "$UQ_FAE_SING_MOUNTS" "$UQ_FAE_SING_IMG" "$@"
