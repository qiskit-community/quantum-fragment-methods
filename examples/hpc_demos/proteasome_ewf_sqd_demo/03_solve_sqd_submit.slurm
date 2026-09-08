#!/bin/bash
#SBATCH --job-name=prot_03b_sqd_sub
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-01:00:00
#SBATCH --output=/gpfs/scc6000/proj/toffoli/thaddeus/quantum-fragment-methods/examples/hpc_demos/proteasome_ewf_sqd_demo/logs/03_solve_sqd_submit_%j.out
#SBATCH --error=/gpfs/scc6000/proj/toffoli/thaddeus/quantum-fragment-methods/examples/hpc_demos/proteasome_ewf_sqd_demo/logs/03_solve_sqd_submit_%j.err

# Submits all SQD QPU jobs to IBM Quantum and exits immediately.
# Needs --gres=gpu:1 only for the container; the actual work is API calls.
# Wall time is short (1h) because this job just submits and exits.
#
# QRMI credentials and QFM_PROTEASOME_CONFIG are passed by run_workflow.sh.

set -euo pipefail

export PYTHONPATH=/workspace${PYTHONPATH:+:$PYTHONPATH}

source /opt/conda/etc/profile.d/conda.sh
conda activate qfrag-env

DEMO_DIR=/workspace/examples/hpc_demos/proteasome_ewf_sqd_demo
CONFIG="${QFM_PROTEASOME_CONFIG:?QFM_PROTEASOME_CONFIG not set}"

echo "============================================================"
echo "Step 3b: SQD QPU Job Submission"
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURMD_NODENAME"
echo "Config:  $CONFIG"
echo "Started: $(date)"
echo "============================================================"

python -u "$DEMO_DIR/03_solve_sqd_submit.py" \
    --config "$CONFIG"

echo "============================================================"
echo "Step 3b complete: $(date)"
echo "QPU jobs submitted — HPC node released."
echo "Monitor with: bash $DEMO_DIR/03_solve_sqd_monitor.sh <run_name> --watch 300"
echo "============================================================"
