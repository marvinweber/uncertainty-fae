#!/bin/bash

# Helper Script to Submit multiple Sub-Versions of a Model (for Training)
# Helpful for submitting a Deep ensemble training, for example

if ! [ $# -eq 4 ]
then
  echo "Invalid Argument Amount!"
  cat <<EOF
    Usage: ./script.sh MODEL VERSION_NAME SUBVERSION_START SUBVERSION_END

      MODEL: Name of the model to train.
      VERSION_NAME: Name of the version.
      SUBVERSION_START/ SUBVERSION_END:
          Training for sub-versions SUBVERSION_START ... SUBVERSION_END
          will be submitted.

    Jobs are submitted in HOLD state, so they have to be released manually
    afterwards with: scontrol release <job-id-1> <job-id-2> ...

    NOTE: Please adjust <slurm-account> and <slurm-partition> as required!
EOF
fi

model=$1
version=$2
start=$3
end=$4

for subversion in $( eval echo {$start..$end} )
do
  jobname="${model}_${version}_${subversion}"
  logname="${jobname}_%A_%a.log"
  echo "Submitting Model=${model} Version=${version} Subversion=${subversion} Jobname=${jobname} Logname=${logname}"
  sbatch \
    --account=<slurm-account> \
    --partition=<slurm-partition> \
    --output="logs/slurm/${logname}" \
    --error="logs/slurm/${logname}" \
    --job-name="${jobname}" \
    --hold \
    --export="UQ_FAE_SING_IMG=${UQ_FAE_SING_IMG},UQ_FAE_SING_MOUNTS=${UQ_FAE_SING_MOUNTS}" \
    generic_sbatch_uncertainty.sh \
    python /app/scripts/training.py \
    --max-epochs 400 \
    --batch-size 40 \
    --early-stopping-patience 400 \
    --dataloader-num-workers 32 \
    --save-dir /ml_logs \
    --configuration /app/config/models.yml \
    --version "${version}" \
    --sub-version "${subversion}" \
    "${model}"
done
