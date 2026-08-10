#!/bin/bash
#SBATCH --job-name=ewf_03_solve
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/03_solve_%j.out
#SBATCH --error=logs/03_solve_%j.err

# Partition, account, container image, mount, and QRMI credential env vars
# are all passed by run_workflow.sh at sbatch submission time.
# QRMI credentials are read from environment variables — never stored here.

set -euo pipefail

export PYTHONPATH=/workspace${PYTHONPATH:+:$PYTHONPATH}

source /opt/conda/etc/profile.d/conda.sh
conda activate qfrag-env

export CUPY_CACHE_DIR=/tmp/cupy_cache
mkdir -p $CUPY_CACHE_DIR

DEMO_DIR=/workspace/examples/hpc_ewf_sqd_demo

CONFIG="${QFM_TRIAL_CONFIG:-$DEMO_DIR/config_alanine_sto-3g.yaml}"

echo "============================================================"
echo "Step 3: Fragment Solving (FCI + SQD)"
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURMD_NODENAME"
echo "Config:  $CONFIG"
echo "Started: $(date)"
echo "============================================================"

python -u "$DEMO_DIR/03_solve.py" \
    --config "$CONFIG" \
    --wait \
    --max-wait-time 14400

echo "============================================================"
echo "Step 3 complete: $(date)"
echo "============================================================"
