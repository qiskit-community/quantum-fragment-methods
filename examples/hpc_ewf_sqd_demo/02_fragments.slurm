#!/bin/bash
#SBATCH --job-name=ewf_02_fragments
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/02_fragments_%j.out
#SBATCH --error=logs/02_fragments_%j.err

# Partition, account, container image, and mount are passed by run_workflow.sh
# at sbatch submission time — they are not hardcoded here.

set -euo pipefail

export PYTHONPATH=/workspace${PYTHONPATH:+:$PYTHONPATH}

source /opt/conda/etc/profile.d/conda.sh
conda activate qfrag-env

export CUPY_CACHE_DIR=/tmp/cupy_cache
mkdir -p $CUPY_CACHE_DIR

DEMO_DIR=/workspace/examples/hpc_ewf_sqd_demo

CONFIG="${QFM_TRIAL_CONFIG:-$DEMO_DIR/config_alanine_sto-3g.yaml}"

echo "============================================================"
echo "Step 2: EWF Fragment Construction"
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURMD_NODENAME"
echo "Config:  $CONFIG"
echo "Started: $(date)"
echo "============================================================"

python -u "$DEMO_DIR/02_fragments.py" \
    --config "$CONFIG"

echo "============================================================"
echo "Step 2 complete: $(date)"
echo "============================================================"
