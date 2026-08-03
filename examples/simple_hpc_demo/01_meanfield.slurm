#!/bin/bash
#SBATCH --job-name=n2_meanfield
#SBATCH --partition=gpu-mid
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/01_meanfield_%j.out
#SBATCH --error=logs/01_meanfield_%j.err
#SBATCH --container-image=$HOME/qfm-hpc-py312.sqsh
#SBATCH --container-mounts=$SLURM_SUBMIT_DIR:/workspace
#SBATCH --container-workdir=/workspace

mkdir -p logs results

source /opt/conda/etc/profile.d/conda.sh
conda activate qfrag-env

echo "Step 1: Mean-Field Calculation"
echo "Job ID: $SLURM_JOB_ID"

python 01_setup_molecule.py --output-dir /workspace/results
