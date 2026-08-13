#!/bin/bash
#SBATCH --job-name=ewf_01_meanfield
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/workspace/examples/hpc_demos/alanine_ewf_sqd_demo/logs/01_meanfield_%j.out
#SBATCH --error=/workspace/examples/hpc_demos/alanine_ewf_sqd_demo/logs/01_meanfield_%j.err

# Partition, account, container image, and mount are passed by run_workflow.sh
# at sbatch submission time — they are not hardcoded here.

set -euo pipefail

# Pyxis does not forward image-baked ENV into the job environment automatically.
# Re-declare PYTHONPATH so /workspace/quantum_fragment_methods (live GPFS copy)
# takes precedence over the baked-in /opt copy.
export PYTHONPATH=/workspace${PYTHONPATH:+:$PYTHONPATH}

source /opt/conda/etc/profile.d/conda.sh
conda activate qfrag-env

export CUPY_CACHE_DIR=/tmp/cupy_cache
mkdir -p $CUPY_CACHE_DIR

DEMO_DIR=/workspace/examples/hpc_demos/alanine_ewf_sqd_demo

# Use trial config if set by trial_run.sh, otherwise use the production config
CONFIG="${QFM_TRIAL_CONFIG:-$DEMO_DIR/config_alanine_sto-3g.yaml}"

echo "============================================================"
echo "Step 1: Mean-Field Calculation (Hartree-Fock)"
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURMD_NODENAME"
echo "Config:  $CONFIG"
echo "Started: $(date)"
echo "============================================================"

python -u "$DEMO_DIR/01_meanfield.py" \
    --config "$CONFIG"

echo "============================================================"
echo "Step 1 complete: $(date)"
echo "============================================================"
