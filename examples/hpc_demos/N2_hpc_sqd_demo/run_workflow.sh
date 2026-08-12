#!/bin/bash
# Submit N2 SQD workflow with job dependency chain.
# Step 2 runs only after step 1 completes successfully.
#
# NOTE: Always submit via this script rather than calling sbatch directly.
# Slurm evaluates --output paths at submission time; logs/ must exist first.
# run_workflow.sh creates logs/ and results/ before calling sbatch.
#
# CONFIGURATION: copy <repo-root>/.hpc_config.template to .hpc_config and
# fill in your cluster values before running this script.

set -e

# ---------------------------------------------------------------------------
# Load cluster configuration from .hpc_config (repo root, two levels up)
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
HPC_CONFIG="$REPO_ROOT/.hpc_config"

if [ -f "$HPC_CONFIG" ]; then
    set -a; source "$HPC_CONFIG"; set +a
else
    echo "ERROR: $HPC_CONFIG not found."
    echo "Copy .hpc_config.template to .hpc_config and fill in your cluster values."
    exit 1
fi

# ---------------------------------------------------------------------------
# Validate required variables
# ---------------------------------------------------------------------------
: "${CONTAINER_IMAGE:?CONTAINER_IMAGE not set in .hpc_config}"
: "${SLURM_PARTITION:?SLURM_PARTITION not set in .hpc_config}"
: "${QRMI_CREDS:?QRMI_CREDS not set in .hpc_config}"

mkdir -p logs results

# ---------------------------------------------------------------------------
# Container mount and env flags
# ---------------------------------------------------------------------------
# $SLURM_SUBMIT_DIR and $HOME are not expanded in #SBATCH directives by Pyxis.
# Pass container options as CLI arguments so the shell expands them first.
MOUNT="$REPO_ROOT:/workspace"

# NVIDIA_DRIVER_CAPABILITIES tells Pyxis/enroot to inject the host NVIDIA
# driver libraries so CuPy/GPU4PySCF can see the GPU.
ENV_FLAGS="--container-env=NVIDIA_DRIVER_CAPABILITIES=compute,utility"

# ---------------------------------------------------------------------------
# Inject QRMI credentials
# Source from the shared filesystem here (host shell) and pass each variable
# into the container via --container-env. The shared filesystem is not
# mounted inside the container, so the job script body cannot source it.
# ---------------------------------------------------------------------------
if [ -f "$QRMI_CREDS" ]; then
    set -a; source "$QRMI_CREDS"; set +a
    for var in QRMI_JOB_QPU_RESOURCES QRMI_JOB_QPU_TYPES $(env | grep _QRMI_ | cut -d= -f1); do
        ENV_FLAGS="$ENV_FLAGS --container-env=$var"
    done
else
    echo "WARNING: QRMI credentials not found at $QRMI_CREDS"
    echo "QPU job submission will fail. See README.md § 'QRMI credentials'."
fi

# ---------------------------------------------------------------------------
# Optional: pass --account if set
# ---------------------------------------------------------------------------
ACCOUNT_FLAG=""
if [ -n "$SLURM_ACCOUNT" ]; then
    ACCOUNT_FLAG="--account=$SLURM_ACCOUNT"
fi

# ---------------------------------------------------------------------------
# Submit
# ---------------------------------------------------------------------------
JOB1=$(sbatch --parsable \
    --partition="$SLURM_PARTITION" \
    $ACCOUNT_FLAG \
    --container-image="$CONTAINER_IMAGE" \
    --container-mounts="$MOUNT" \
    $ENV_FLAGS \
    01_meanfield.slurm)
echo "Submitted step 1 (meanfield): job $JOB1"

JOB2=$(sbatch --parsable \
    --dependency=afterok:$JOB1 \
    --partition="$SLURM_PARTITION" \
    $ACCOUNT_FLAG \
    --container-image="$CONTAINER_IMAGE" \
    --container-mounts="$MOUNT" \
    $ENV_FLAGS \
    02_sqd_solve.slurm)
echo "Submitted step 2 (SQD solve): job $JOB2 (depends on $JOB1)"

echo ""
echo "Monitor with:  squeue -u \$USER"
echo "Logs:          logs/01_meanfield_${JOB1}.out"
echo "               logs/02_sqd_${JOB2}.out"
