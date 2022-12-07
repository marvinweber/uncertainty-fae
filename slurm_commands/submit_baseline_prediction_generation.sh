#!/bin/bash

# Helper Script to submit a Job for Baseline Prediction Generation.

if ! [ $# -eq 3 ]
then
  cat <<EOF
    Invalid Argument Amount!

    Usage: ./submit_baseline_rediction_generation.sh DATASET_TYPE MODEL_NAME CHECKPOINT_PATH

      DATASET_TYPE: Which dataset to use for evaluation (train/val/test).
      MODEL_NAME: Name of the model to use (from model configuration).
      CHECKPOINT_PATH: Path of the checkpoint to use for predictions/ the model.

    Jobs are submitted in HOLD state, so they have to be released manually
    afterwards with: scontrol release <job-id>

    NOTE: Please adjust <slurm-account> and <slurm-partition> as required
          (via env variables)!
EOF
  exit 1
fi

dataset_type=$1
model_name=$2
checkpoint_path=$3

jobname="baseline_preds_${dataset_type}_${model_name}"
logname="${jobname}_%A_%a.log"
echo "Submitting Baseline Prediction Generation - Dataset-Type=${dataset_type} Model=${model_name} Checkpoint=${checkpoint_patch} "
echo "Job: Name=${jobname} Logname=${logname}"

sbatch \
  --account="${UQ_SLURM_ACCOUNT}" \
  --partition="${UQ_SLURM_PARTITION}" \
  --output="logs/slurm/${logname}" \
  --error="logs/slurm/${logname}" \
  --job-name="${jobname}" \
  --hold \
  --export="UQ_FAE_SING_IMG=${UQ_FAE_SING_IMG},UQ_FAE_SING_MOUNTS=${UQ_FAE_SING_MOUNTS}" \
  --array=1-1%1 \
  generic_sbatch_uncertainty.sh \
  python /app/scripts/create_baseline_model_predictions.py \
  --batch-size 20 \
  --dataloader-num-workers 32 \
  --eval-dir /ml_eval \
  --configuration /app/config/models.ma.yml \
  --dataset-type "${dataset_type}" \
  "${model_name}" \
  "${checkpoint_path}"
