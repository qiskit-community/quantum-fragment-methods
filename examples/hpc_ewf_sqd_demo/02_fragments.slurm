#!/bin/bash
#SBATCH --job-name=02_fragments
#SBATCH --partition=gpu-mid
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/02_fragments_%j.out
#SBATCH --error=logs/02_fragments_%j.err

# TODO: Load mean-field results
# TODO: Create embedder
# TODO: Generate fragments
# TODO: Save fragment data

echo "Step 2: Fragment Construction"
echo "Job ID: $SLURM_JOB_ID"