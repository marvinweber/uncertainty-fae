#!/bin/bash

# Helper Script to srun evaluation plot generation.

if ! [ $# -eq 2 ]
then
  echo "Invalid Argument Amount!"
  cat <<EOF
    Usage: ./srun_evaluation_plot_generation.sh EVAL_VERSION DATASET_TYPE

      EVAL_VERSION: Name of the evaluation version.
      DATASET_TYPE: Which dataset to use for evaluation.

    NOTE: Please adjust <slurm-account> and <slurm-partition> as required
          (via env variables)!
EOF
  exit 1
fi

eval_version_name=$1
dataset_type=$2


jobname="${eval_version_name}_${dataset_type}_evaluation"

srun \
  --account="${UQ_SLURM_ACCOUNT}" \
  --partition="${UQ_SLURM_PARTITION}" \
  --cpus-per-task=2 \
  --mem=2GB \
  --job-name="${jobname}" \
  --unbuffered \
  --export="UQ_FAE_SING_IMG=${UQ_FAE_SING_IMG},UQ_FAE_SING_MOUNTS=${UQ_FAE_SING_MOUNTS}" \
  generic_sbatch_uncertainty.sh \
  python /app/scripts/uq_evaluation.py \
  --eval-dir /ml_eval \
  --configuration /app/config/models.yml \
  --eval-configuration /app/config/eval-config.example.yml \
  --only-plotting \
  --dataset-type "${dataset_type}" \
  "${eval_version_name}"
