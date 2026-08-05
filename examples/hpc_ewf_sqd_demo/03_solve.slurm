#!/bin/bash
#SBATCH --job-name=03_solve
#SBATCH --partition=gpu-mid
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/03_solve_%j.out
#SBATCH --error=logs/03_solve_%j.err

# TODO: Load fragment data
# TODO: Create solver
# TODO: Solve all fragments
# TODO: Save fragment solutions

echo "Step 3: Fragment Solving"
echo "Job ID: $SLURM_JOB_ID"