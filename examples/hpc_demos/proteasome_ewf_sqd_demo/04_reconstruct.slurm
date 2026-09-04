#!/bin/bash
#SBATCH --job-name=prot_04_recon
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/gpfs/scc6000/proj/toffoli/thaddeus/quantum-fragment-methods/examples/hpc_demos/proteasome_ewf_sqd_demo/logs/04_reconstruct_%j.out
#SBATCH --error=/gpfs/scc6000/proj/toffoli/thaddeus/quantum-fragment-methods/examples/hpc_demos/proteasome_ewf_sqd_demo/logs/04_reconstruct_%j.err

set -euo pipefail

export PYTHONPATH=/workspace${PYTHONPATH:+:$PYTHONPATH}

source /opt/conda/etc/profile.d/conda.sh
conda activate qfrag-env

export CUPY_CACHE_DIR=/tmp/cupy_cache
mkdir -p $CUPY_CACHE_DIR

DEMO_DIR=/workspace/examples/hpc_demos/proteasome_ewf_sqd_demo

CONFIG="${QFM_PROTEASOME_CONFIG:?QFM_PROTEASOME_CONFIG not set}"

echo "============================================================"
echo "Step 4: Energy Reconstruction"
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURMD_NODENAME"
echo "Config:  $CONFIG"
echo "Started: $(date)"
echo "============================================================"

python -u "$DEMO_DIR/04_reconstruct.py" \
    --config "$CONFIG"

echo "============================================================"
echo "Step 4 complete: $(date)"
echo "Results: $DEMO_DIR/results/"
echo "============================================================"
