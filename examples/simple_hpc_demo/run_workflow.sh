#!/bin/bash
# Submit N2 SQD workflow with job dependency chain.
# Step 2 runs only after step 1 completes successfully.

set -e

mkdir -p logs

JOB1=$(sbatch --parsable 01_meanfield.slurm)
echo "Submitted step 1 (meanfield): job $JOB1"

JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 02_sqd_solve.slurm)
echo "Submitted step 2 (SQD solve): job $JOB2 (depends on $JOB1)"

echo ""
echo "Monitor with:  squeue -u \$USER"
echo "Logs:          logs/01_meanfield_${JOB1}.out"
echo "               logs/02_sqd_${JOB2}.out"
