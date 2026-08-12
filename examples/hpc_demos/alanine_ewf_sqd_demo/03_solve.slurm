#!/bin/bash
#SBATCH --job-name=ewf_03_solve
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=192
#SBATCH --mem=1800G
#SBATCH --time=72:00:00
#SBATCH --output=logs/03_solve_%j.out
#SBATCH --error=logs/03_solve_%j.err

# Partition, account, container image, mount, and QRMI credential env vars
# are all passed by run_workflow.sh at sbatch submission time.
# QRMI credentials are read from environment variables — never stored here.

set -euo pipefail

export PYTHONPATH=/workspace${PYTHONPATH:+:$PYTHONPATH}

source /opt/conda/etc/profile.d/conda.sh
conda activate qfrag-env

# Use all allocated CPUs for PySCF FCI/CCSD parallelism
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

export CUPY_CACHE_DIR=/tmp/cupy_cache
mkdir -p $CUPY_CACHE_DIR

DEMO_DIR=/workspace/examples/hpc_demos/alanine_ewf_sqd_demo

CONFIG="${QFM_TRIAL_CONFIG:-$DEMO_DIR/config_alanine_sto-3g.yaml}"

echo "============================================================"
echo "Step 3: Fragment Solving (FCI + SQD)"
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURMD_NODENAME"
echo "Config:  $CONFIG"
echo "Started: $(date)"
echo "============================================================"

echo "[$(date +%T)] Probing imports..."
python -u -c "
import time, sys
t = time.time()
def ck(lbl):
    print(f'  {lbl}: {time.time()-t:.1f}s', flush=True)
import yaml; ck('yaml')
from pyscf import gto, scf; ck('pyscf')
from quantum_fragment_methods.application.embedding.base import EmbeddingResult, Fragment; ck('embedding.base')
from quantum_fragment_methods.application.solvers.classical_zoo.fci import FCI; ck('fci')
import pickle, h5py; ck('pickle+h5py')
print('imports OK', flush=True)
"
echo "[$(date +%T)] Imports done. Starting solver..."

python -u "$DEMO_DIR/03_solve.py" \
    --config "$CONFIG" \
    --wait \
    --max-wait-time 21600

echo "============================================================"
echo "Step 3 complete: $(date)"
echo "============================================================"
