#!/bin/bash
#SBATCH --job-name=n2_sqd
#SBATCH --partition=gpu-mid
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/02_sqd_%j.out
#SBATCH --error=logs/02_sqd_%j.err
#SBATCH --container-image=$HOME/qfm-hpc-py312.sqsh
#SBATCH --container-mounts=$SLURM_SUBMIT_DIR:/workspace
#SBATCH --container-workdir=/workspace
#
# QRMI credentials are set automatically by the Slurm SPANK plugin on SCC
# when a QPU resource is allocated. For manual testing without SPANK,
# uncomment and fill in the variables below:
#
# export QRMI_JOB_QPU_RESOURCES=ibm_torino
# export QRMI_JOB_QPU_TYPES=qiskit-runtime-service
# export ibm_torino_QRMI_IBM_QRS_ENDPOINT=https://quantum.cloud.ibm.com/api/v1
# export ibm_torino_QRMI_IBM_QRS_IAM_ENDPOINT=https://iam.cloud.ibm.com
# export ibm_torino_QRMI_IBM_QRS_IAM_APIKEY=your_apikey
# export ibm_torino_QRMI_IBM_QRS_SERVICE_CRN=your_crn

mkdir -p logs

source /opt/conda/etc/profile.d/conda.sh
conda activate qfrag-env

echo "Step 2: SQD Solve"
echo "Job ID: $SLURM_JOB_ID"

python 02_run_sqd.py \
  --config config_N2_sto-3g.yaml \
  --data-dir /workspace/results \
  --results-dir /workspace/results/sqd_results \
  --wait
