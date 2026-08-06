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

DEMO_DIR=/workspace/quantum-fragment-methods/examples/hpc_ewf_sqd_demo

echo "============================================================"
echo "Step 2: EWF Fragment Construction"
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURMD_NODENAME"
echo "Started: $(date)"
echo "============================================================"

python "$DEMO_DIR/02_fragments.py" \
    --config "$DEMO_DIR/config_alanine_sto-3g.yaml" \
    --data-dir "$DEMO_DIR/data" \
    --output-dir "$DEMO_DIR/data"

echo "============================================================"
echo "Step 2 complete: $(date)"
echo "============================================================"
