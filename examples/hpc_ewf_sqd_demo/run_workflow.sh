#!/bin/bash
# Submit the EWF+SQD alanine workflow as a 4-job Slurm dependency chain.
# Steps 2, 3, and 4 run only after their predecessor completes successfully.
#
# CONFIGURATION: copy <repo-root>/.hpc_config.template to .hpc_config and
# fill in your cluster values before running this script.
#
# NOTE: QRMI credentials are injected only for job 3 (the QPU solve step).
# Jobs 1, 2, and 4 are classical and do not need QPU access.

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

# ---------------------------------------------------------------------------
# Create output directories before submission
# (Slurm evaluates --output paths at submission time; dirs must exist first)
# ---------------------------------------------------------------------------
mkdir -p "$SCRIPT_DIR/logs" "$SCRIPT_DIR/data" "$SCRIPT_DIR/results"

# ---------------------------------------------------------------------------
# Container mount and base env flags
# ---------------------------------------------------------------------------
MOUNT="$REPO_ROOT:/workspace"

BASE_ENV="--container-env=NVIDIA_DRIVER_CAPABILITIES=compute,utility"

# ---------------------------------------------------------------------------
# Optional: --account flag
# ---------------------------------------------------------------------------
ACCOUNT_FLAG=""
if [ -n "${SLURM_ACCOUNT:-}" ]; then
    ACCOUNT_FLAG="--account=$SLURM_ACCOUNT"
fi

# ---------------------------------------------------------------------------
# QRMI credential flags (only injected into job 3)
# Source credentials here (host shell) and pass each variable into the
# container via --container-env. The shared filesystem is not mounted inside
# the container, so job scripts cannot source it themselves.
# ---------------------------------------------------------------------------
QRMI_ENV_FLAGS=""
if [ -f "$QRMI_CREDS" ]; then
    set -a; source "$QRMI_CREDS"; set +a
    for var in QRMI_JOB_QPU_RESOURCES QRMI_JOB_QPU_TYPES $(env | grep _QRMI_ | cut -d= -f1); do
        QRMI_ENV_FLAGS="$QRMI_ENV_FLAGS --container-env=$var"
    done
else
    echo "WARNING: QRMI credentials not found at $QRMI_CREDS"
    echo "QPU job submission (step 3) will fail. See README.md for setup instructions."
fi

# ---------------------------------------------------------------------------
# Submit job 1 — Mean-field (no QRMI needed)
# ---------------------------------------------------------------------------
JOB1=$(sbatch --parsable \
    --partition="$SLURM_PARTITION" \
    $ACCOUNT_FLAG \
    --container-image="$CONTAINER_IMAGE" \
    --container-mounts="$MOUNT" \
    --container-workdir=/workspace \
    $BASE_ENV \
    "$SCRIPT_DIR/01_meanfield.slurm")
echo "Submitted step 1 (meanfield):    job $JOB1"

# ---------------------------------------------------------------------------
# Submit job 2 — Fragment construction (no QRMI needed)
# ---------------------------------------------------------------------------
JOB2=$(sbatch --parsable \
    --dependency=afterok:$JOB1 \
    --partition="$SLURM_PARTITION" \
    $ACCOUNT_FLAG \
    --container-image="$CONTAINER_IMAGE" \
    --container-mounts="$MOUNT" \
    --container-workdir=/workspace \
    $BASE_ENV \
    "$SCRIPT_DIR/02_fragments.slurm")
echo "Submitted step 2 (fragments):    job $JOB2 (depends on $JOB1)"

# ---------------------------------------------------------------------------
# Submit job 3 — Solve fragments (QRMI credentials injected)
# ---------------------------------------------------------------------------
JOB3=$(sbatch --parsable \
    --dependency=afterok:$JOB2 \
    --partition="$SLURM_PARTITION" \
    $ACCOUNT_FLAG \
    --container-image="$CONTAINER_IMAGE" \
    --container-mounts="$MOUNT" \
    --container-workdir=/workspace \
    $BASE_ENV \
    $QRMI_ENV_FLAGS \
    "$SCRIPT_DIR/03_solve.slurm")
echo "Submitted step 3 (solve/QPU):    job $JOB3 (depends on $JOB2)"

# ---------------------------------------------------------------------------
# Submit job 4 — Energy reconstruction (no QRMI needed)
# ---------------------------------------------------------------------------
JOB4=$(sbatch --parsable \
    --dependency=afterok:$JOB3 \
    --partition="$SLURM_PARTITION" \
    $ACCOUNT_FLAG \
    --container-image="$CONTAINER_IMAGE" \
    --container-mounts="$MOUNT" \
    --container-workdir=/workspace \
    $BASE_ENV \
    "$SCRIPT_DIR/04_reconstruct.slurm")
echo "Submitted step 4 (reconstruct):  job $JOB4 (depends on $JOB3)"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "Workflow submitted: $JOB1 → $JOB2 → $JOB3 → $JOB4"
echo ""
echo "Monitor with:  squeue -u \$USER"
echo ""
echo "Logs:"
echo "  logs/01_meanfield_${JOB1}.out"
echo "  logs/02_fragments_${JOB2}.out"
echo "  logs/03_solve_${JOB3}.out"
echo "  logs/04_reconstruct_${JOB4}.out"
echo ""
echo "Results: examples/hpc_ewf_sqd_demo/results/summary.json"
