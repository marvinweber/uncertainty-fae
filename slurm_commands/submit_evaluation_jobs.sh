#!/bin/bash

# Helper Script to submit a single job per evaluation model.

if [ $# -lt 3 ]
then
  echo "Invalid Argument Amount!"
  cat <<EOF
    Usage: ./submit_evaluation_jobs.sh EVAL_VERSION DATASET_TYPE EVAL_CFG_NAME_1 ... EVAL_CFG_NAME_2

      EVAL_VERSION: Name of the evaluation version.
      DATASET_TYPE: Which dataset to use for evaluation.
      EVAL_CFG_NAME_1 ... EVAL_CFG_NAME_2:
          One to many eval configuration names for each of which a separate job
          will be scheduled.

    Jobs are submitted in HOLD state, so they have to be released manually
    afterwards with: scontrol release <job-id-1> <job-id-2> ...

    Job is submitted as Array Job with two Jobs (currently, only OOD and Eval
    data generation can be separated; there is no "in-pred-generation" checkpoint
    or similar).

    NOTE: Please adjust <slurm-account> and <slurm-partition> as required
          (via env variables)!
EOF
  exit 1
fi

eval_version_name=$1
dataset_type=$2

for eval_cfg_name in "${@:3}"
do
  jobname="${eval_version_name}_${dataset_type}_${eval_cfg_name}"
  logname="${jobname}_%A_%a.log"
  echo "Submitting Evaluation Version=${eval_version_name} Config=${eval_cfg_name} Dataset-Type=${dataset_type} as: Jobname=${jobname} Logname=${logname}"

  sbatch \
    --account="${UQ_SLURM_ACCOUNT}" \
    --partition="${UQ_SLURM_PARTITION}" \
    --output="logs/slurm/${logname}" \
    --error="logs/slurm/${logname}" \
    --job-name="${jobname}" \
    --hold \
    --export="UQ_FAE_SING_IMG=${UQ_FAE_SING_IMG},UQ_FAE_SING_MOUNTS=${UQ_FAE_SING_MOUNTS}" \
    --array=1-2%1 \
    generic_sbatch_uncertainty.sh \
    python /app/scripts/uq_evaluation.py \
    --batch-size 40 \
    --dataloader-num-workers 32 \
    --eval-dir /ml_eval \
    --model-logs-dir /ml_logs \
    --configuration /app/config/models.yml \
    --eval-configuration /app/config/eval-config.example.yml \
    --eval-only "${eval_cfg_name}" \
    --only-predictions \
    --dataset-type "${dataset_type}" \
    "${eval_version_name}"
done
