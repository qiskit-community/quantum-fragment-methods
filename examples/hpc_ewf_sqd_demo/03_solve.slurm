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

DEMO_DIR=/workspace/quantum-fragment-methods/examples/hpc_ewf_sqd_demo

echo "============================================================"
echo "Step 3: Fragment Solving (FCI + SQD)"
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURMD_NODENAME"
echo "Started: $(date)"
echo "============================================================"

mkdir -p "$DEMO_DIR/results"

python "$DEMO_DIR/03_solve.py" \
    --config   "$DEMO_DIR/config_alanine_sto-3g.yaml" \
    --data-dir "$DEMO_DIR/data" \
    --output-dir "$DEMO_DIR/data" \
    --results-dir "$DEMO_DIR/results" \
    --wait \
    --max-wait-time 14400

echo "============================================================"
echo "Step 3 complete: $(date)"
echo "============================================================"
