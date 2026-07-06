#!/bin/bash
#SBATCH --job-name=01_meanfield
#SBATCH --partition=gpu-mid
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/01_meanfield_%j.out
#SBATCH --error=logs/01_meanfield_%j.err

# TODO: Add environment setup
# TODO: Run Hartree-Fock calculation
# TODO: Save mean-field results

echo "Step 1: Mean-Field Calculation"
echo "Job ID: $SLURM_JOB_ID"