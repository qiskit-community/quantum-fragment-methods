#!/bin/bash
#SBATCH --job-name=ewf_04_reconstruct
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/04_reconstruct_%j.out
#SBATCH --error=logs/04_reconstruct_%j.err

# Partition, account, container image, and mount are passed by run_workflow.sh
# at sbatch submission time — they are not hardcoded here.
# No QRMI credentials needed for this classical reconstruction step.

set -euo pipefail

DEMO_DIR=/workspace/quantum-fragment-methods/examples/hpc_ewf_sqd_demo

echo "============================================================"
echo "Step 4: Energy Reconstruction"
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURMD_NODENAME"
echo "Started: $(date)"
echo "============================================================"

python "$DEMO_DIR/04_reconstruct.py" \
    --config      "$DEMO_DIR/config_alanine_sto-3g.yaml" \
    --data-dir    "$DEMO_DIR/data" \
    --results-dir "$DEMO_DIR/results"

echo "============================================================"
echo "Step 4 complete: $(date)"
echo "Results: $DEMO_DIR/results/summary.json"
echo "============================================================"
