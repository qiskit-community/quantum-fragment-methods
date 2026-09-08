#!/bin/bash
#SBATCH --job-name=prot_03a_ccsd
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=96
#SBATCH --mem=1900G
#SBATCH --time=1-00:00:00
#SBATCH --output=/gpfs/scc6000/proj/toffoli/thaddeus/quantum-fragment-methods/examples/hpc_demos/proteasome_ewf_sqd_demo/logs/03_solve_ccsd_%j.out
#SBATCH --error=/gpfs/scc6000/proj/toffoli/thaddeus/quantum-fragment-methods/examples/hpc_demos/proteasome_ewf_sqd_demo/logs/03_solve_ccsd_%j.err

# Runs CCSD on all SQD-eligible fragments (classical reference + LUCJ seed).
# QFM_PROTEASOME_CONFIG is passed by run_workflow.sh.

set -euo pipefail

export PYTHONPATH=/workspace${PYTHONPATH:+:$PYTHONPATH}

source /opt/conda/etc/profile.d/conda.sh
conda activate qfrag-env

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

export CUPY_CACHE_DIR=/tmp/cupy_cache
mkdir -p $CUPY_CACHE_DIR

DEMO_DIR=/workspace/examples/hpc_demos/proteasome_ewf_sqd_demo
CONFIG="${QFM_PROTEASOME_CONFIG:?QFM_PROTEASOME_CONFIG not set}"

echo "============================================================"
echo "Step 3a: CCSD Fragment Solving (classical reference)"
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURMD_NODENAME"
echo "Config:  $CONFIG"
echo "Started: $(date)"
echo "============================================================"

python -u "$DEMO_DIR/03_solve_ccsd.py" \
    --config "$CONFIG"

echo "============================================================"
echo "Step 3a complete: $(date)"
echo "============================================================"
