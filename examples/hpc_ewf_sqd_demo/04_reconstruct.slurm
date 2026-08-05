#!/bin/bash
#SBATCH --job-name=04_reconstruct
#SBATCH --partition=gpu-mid
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/04_reconstruct_%j.out
#SBATCH --error=logs/04_reconstruct_%j.err

# TODO: Load all fragment solutions
# TODO: Reconstruct total energy
# TODO: Calculate correlation energy
# TODO: Save final results

echo "Step 4: Energy Reconstruction"
echo "Job ID: $SLURM_JOB_ID"