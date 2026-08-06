#!/bin/bash
#SBATCH --job-name=ewf_01_meanfield
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/01_meanfield_%j.out
#SBATCH --error=logs/01_meanfield_%j.err

# Partition, account, container image, and mount are passed by run_workflow.sh
# at sbatch submission time — they are not hardcoded here.

set -euo pipefail

DEMO_DIR=/workspace/quantum-fragment-methods/examples/hpc_ewf_sqd_demo

echo "============================================================"
echo "Step 1: Mean-Field Calculation (Hartree-Fock)"
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURMD_NODENAME"
echo "Started: $(date)"
echo "============================================================"

mkdir -p "$DEMO_DIR/data"

python "$DEMO_DIR/01_meanfield.py" \
    --config "$DEMO_DIR/config_alanine_sto-3g.yaml" \
    --output-dir "$DEMO_DIR/data"

echo "============================================================"
echo "Step 1 complete: $(date)"
echo "============================================================"
