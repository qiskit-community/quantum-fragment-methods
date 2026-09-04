#!/bin/bash
#SBATCH --job-name=prot_05_dE
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/workspace/examples/hpc_demos/proteasome_ewf_sqd_demo/logs/05_binding_energy_%j.out
#SBATCH --error=/workspace/examples/hpc_demos/proteasome_ewf_sqd_demo/logs/05_binding_energy_%j.err

# This step has no --dependency — it is submitted manually once all four
# 04_reconstruct jobs (complex and ligand for each inhibitor) have completed.

set -euo pipefail

export PYTHONPATH=/workspace${PYTHONPATH:+:$PYTHONPATH}

source /opt/conda/etc/profile.d/conda.sh
conda activate qfrag-env

DEMO_DIR=/workspace/examples/hpc_demos/proteasome_ewf_sqd_demo

echo "============================================================"
echo "Step 5: Binding Energy Assembly"
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURMD_NODENAME"
echo "Started: $(date)"
echo "============================================================"

python -u "$DEMO_DIR/05_binding_energy.py" \
    --results-dir "$DEMO_DIR/results" \
    --output      "$DEMO_DIR/results/binding_energies.json"

echo "============================================================"
echo "Step 5 complete: $(date)"
echo "Results: $DEMO_DIR/results/binding_energies.json"
echo "============================================================"
