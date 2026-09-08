#!/bin/bash
#SBATCH --job-name=prot_03c_sqd_col
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=0-04:00:00
#SBATCH --output=/gpfs/scc6000/proj/toffoli/thaddeus/quantum-fragment-methods/examples/hpc_demos/proteasome_ewf_sqd_demo/logs/03_solve_sqd_collect_%j.out
#SBATCH --error=/gpfs/scc6000/proj/toffoli/thaddeus/quantum-fragment-methods/examples/hpc_demos/proteasome_ewf_sqd_demo/logs/03_solve_sqd_collect_%j.err

# Retrieves QPU counts from IBM Quantum and runs SBD postprocessing.
# Submit this job MANUALLY once 03_solve_sqd_monitor.sh confirms all QPU
# jobs are in COMPLETED/DONE state.
#
# QRMI credentials and QFM_PROTEASOME_CONFIG are passed by run_workflow.sh
# or manually at sbatch time.

set -euo pipefail

export PYTHONPATH=/workspace${PYTHONPATH:+:$PYTHONPATH}

source /opt/conda/etc/profile.d/conda.sh
conda activate qfrag-env

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

DEMO_DIR=/workspace/examples/hpc_demos/proteasome_ewf_sqd_demo
CONFIG="${QFM_PROTEASOME_CONFIG:?QFM_PROTEASOME_CONFIG not set}"

echo "============================================================"
echo "Step 3c: SQD Result Collection + SBD Postprocessing"
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURMD_NODENAME"
echo "Config:  $CONFIG"
echo "Started: $(date)"
echo "============================================================"

python -u "$DEMO_DIR/03_solve_sqd_collect.py" \
    --config "$CONFIG"

echo "============================================================"
echo "Step 3c complete: $(date)"
echo "solver_results.pkl written — run 04_reconstruct.py next."
echo "============================================================"
