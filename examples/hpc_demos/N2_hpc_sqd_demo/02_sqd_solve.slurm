#!/bin/bash
#SBATCH --job-name=n2_sqd
# --partition is passed by run_workflow.sh from .hpc_config (SLURM_PARTITION)
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=37:00:00
#SBATCH --output=/workspace/examples/hpc_demos/N2_hpc_sqd_demo/logs/02_sqd_%j.out
#SBATCH --error=/workspace/examples/hpc_demos/N2_hpc_sqd_demo/logs/02_sqd_%j.err
#SBATCH --container-workdir=/workspace
#
# QRMI credentials are injected as environment variables by run_workflow.sh,
# which sources .qrmi_config on the host before calling sbatch and passes
# each variable in via --container-env. /gpfs is not mounted inside the container.

mkdir -p logs

source /opt/conda/etc/profile.d/conda.sh
conda activate qfrag-env

# CuPy writes a kernel cache on first import. /u/ is read-only and /gpfs is
# not mounted inside the container, so use /tmp (writable tmpfs on every node).
export CUPY_CACHE_DIR=/tmp/cupy_cache
mkdir -p $CUPY_CACHE_DIR

echo "Step 2: SQD Solve"
echo "Job ID: $SLURM_JOB_ID"

python examples/hpc_demos/N2_hpc_sqd_demo/02_run_sqd.py \
  --config examples/hpc_demos/N2_hpc_sqd_demo/config_N2_sto-3g.yaml \
  --data-dir /workspace/results \
  --results-dir /workspace/results/sqd_results \
  --wait \
  --max-wait-time 129600
