#!/bin/bash
#SBATCH --job-name=prot_01_mf
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/workspace/examples/hpc_demos/proteasome_ewf_sqd_demo/logs/01_meanfield_%j.out
#SBATCH --error=/workspace/examples/hpc_demos/proteasome_ewf_sqd_demo/logs/01_meanfield_%j.err

# Partition, account, container image, mount, and QFM_PROTEASOME_CONFIG are all
# passed by run_workflow.sh at sbatch submission time — not hardcoded here.

set -euo pipefail

export PYTHONPATH=/workspace${PYTHONPATH:+:$PYTHONPATH}

source /opt/conda/etc/profile.d/conda.sh
conda activate qfrag-env

export CUPY_CACHE_DIR=/tmp/cupy_cache
mkdir -p $CUPY_CACHE_DIR

DEMO_DIR=/workspace/examples/hpc_demos/proteasome_ewf_sqd_demo

# QFM_PROTEASOME_CONFIG is set by run_workflow.sh to one of:
#   config_ixazomib_complex_sto-3g.yaml
#   config_bortezomib_complex_sto-3g.yaml
#   config_ixazomib_ligand_sto-3g.yaml
#   config_bortezomib_ligand_sto-3g.yaml
CONFIG="${QFM_PROTEASOME_CONFIG:?QFM_PROTEASOME_CONFIG not set}"

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
