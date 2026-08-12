#!/bin/bash
#SBATCH --job-name=n2_meanfield
# --partition is passed by run_workflow.sh from .hpc_config (SLURM_PARTITION)
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/01_meanfield_%j.out
#SBATCH --error=logs/01_meanfield_%j.err
#SBATCH --container-workdir=/workspace
# NOTE: Submit via run_workflow.sh (not sbatch directly) — logs/ must exist
# before submission or Slurm cannot open the --output file. run_workflow.sh
# creates logs/ and results/ automatically before calling sbatch.

mkdir -p logs results

source /opt/conda/etc/profile.d/conda.sh
conda activate qfrag-env

# CuPy writes a kernel cache on first import. /u/ is read-only and /gpfs is
# not mounted inside the container, so use /tmp (writable tmpfs on every node).
export CUPY_CACHE_DIR=/tmp/cupy_cache
mkdir -p $CUPY_CACHE_DIR

echo "Step 1: Mean-Field Calculation"
echo "Job ID: $SLURM_JOB_ID"

python examples/hpc_demos/N2_hpc_sqd_demo/01_setup_molecule.py --output-dir /workspace/results
