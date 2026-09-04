#!/bin/bash
# Submit the proteasome EWF+SQD workflow as a dependency-chained set of Slurm
# jobs.  The workflow runs steps 1–4 independently for all four systems:
#
#   ixazomib_complex   (533 atoms, charge -1)
#   ixazomib_ligand    ( 42 atoms, charge  0)
#   bortezomib_complex (544 atoms, charge -1)
#   bortezomib_ligand  ( 53 atoms, charge  0)
#
# Each system runs its own 1→2→3→4 chain.  Step 5 (binding energy) is
# submitted with --dependency=afterok on ALL four step-4 jobs.
#
# CONFIGURATION: copy <repo-root>/.hpc_config.template to .hpc_config and
# fill in your cluster values before running this script.

set -e

# ---------------------------------------------------------------------------
# Load cluster configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
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
# Create output directories
# ---------------------------------------------------------------------------
mkdir -p "$SCRIPT_DIR/logs" "$SCRIPT_DIR/data" "$SCRIPT_DIR/results"

# ---------------------------------------------------------------------------
# Container flags
# ---------------------------------------------------------------------------
MOUNT="$REPO_ROOT:/workspace"
BASE_ENV="--container-env=NVIDIA_DRIVER_CAPABILITIES=compute,utility"

ACCOUNT_FLAG=""
if [ -n "${SLURM_ACCOUNT:-}" ]; then
    ACCOUNT_FLAG="--account=$SLURM_ACCOUNT"
fi

# ---------------------------------------------------------------------------
# QRMI credential flags (injected only into step 3 jobs)
# ---------------------------------------------------------------------------
QRMI_ENV_FLAGS=""
if [ -f "$QRMI_CREDS" ]; then
    set -a; source "$QRMI_CREDS"; set +a
    for var in QRMI_JOB_QPU_RESOURCES QRMI_JOB_QPU_TYPES $(env | grep _QRMI_ | cut -d= -f1); do
        QRMI_ENV_FLAGS="$QRMI_ENV_FLAGS --container-env=$var"
    done
else
    echo "WARNING: QRMI credentials not found at $QRMI_CREDS"
    echo "QPU solve steps will use CCSD fallback (trial mode)."
fi

# ---------------------------------------------------------------------------
# Helper: submit a 4-job chain for one system and return the step-4 job ID
# ---------------------------------------------------------------------------
submit_system() {
    local SYSTEM_NAME="$1"     # e.g. ixazomib_complex
    local CONFIG_FILE="$SCRIPT_DIR/config_${SYSTEM_NAME}_sto-3g.yaml"

    echo ""
    echo "─── Submitting system: $SYSTEM_NAME ───"

    # Step 1 — Mean-field
    JOB1=$(sbatch --parsable \
        --partition="$SLURM_PARTITION" \
        $ACCOUNT_FLAG \
        --container-image="$CONTAINER_IMAGE" \
        --container-mounts="$MOUNT" \
        --container-workdir=/workspace \
        $BASE_ENV \
        --container-env=QFM_PROTEASOME_CONFIG="$CONFIG_FILE" \
        "$SCRIPT_DIR/01_meanfield.slurm")
    echo "  Step 1 (meanfield):   job $JOB1"

    # Step 2 — Fragmentation
    JOB2=$(sbatch --parsable \
        --dependency=afterok:$JOB1 \
        --partition="$SLURM_PARTITION" \
        $ACCOUNT_FLAG \
        --container-image="$CONTAINER_IMAGE" \
        --container-mounts="$MOUNT" \
        --container-workdir=/workspace \
        $BASE_ENV \
        --container-env=QFM_PROTEASOME_CONFIG="$CONFIG_FILE" \
        "$SCRIPT_DIR/02_fragments.slurm")
    echo "  Step 2 (fragments):   job $JOB2  (depends on $JOB1)"

    # Step 3 — Solve fragments (QRMI credentials injected)
    JOB3=$(sbatch --parsable \
        --dependency=afterok:$JOB2 \
        --partition="$SLURM_PARTITION" \
        $ACCOUNT_FLAG \
        --container-image="$CONTAINER_IMAGE" \
        --container-mounts="$MOUNT" \
        --container-workdir=/workspace \
        $BASE_ENV \
        $QRMI_ENV_FLAGS \
        --container-env=QFM_PROTEASOME_CONFIG="$CONFIG_FILE" \
        "$SCRIPT_DIR/03_solve.slurm")
    echo "  Step 3 (solve/QPU):   job $JOB3  (depends on $JOB2)"

    # Step 4 — Energy reconstruction
    JOB4=$(sbatch --parsable \
        --dependency=afterok:$JOB3 \
        --partition="$SLURM_PARTITION" \
        $ACCOUNT_FLAG \
        --container-image="$CONTAINER_IMAGE" \
        --container-mounts="$MOUNT" \
        --container-workdir=/workspace \
        $BASE_ENV \
        --container-env=QFM_PROTEASOME_CONFIG="$CONFIG_FILE" \
        "$SCRIPT_DIR/04_reconstruct.slurm")
    echo "  Step 4 (reconstruct): job $JOB4  (depends on $JOB3)"

    # Return step-4 job ID to caller via a named variable
    eval "${SYSTEM_NAME//[^a-zA-Z0-9]/_}_JOB4=$JOB4"
}

# ---------------------------------------------------------------------------
# Submit all four systems
# ---------------------------------------------------------------------------
submit_system "ixazomib_complex"
submit_system "ixazomib_ligand"
submit_system "bortezomib_complex"
submit_system "bortezomib_ligand"

# Collect the four step-4 job IDs
ALL_RECON_DEPS="afterok:${ixazomib_complex_JOB4}:${ixazomib_ligand_JOB4}:${bortezomib_complex_JOB4}:${bortezomib_ligand_JOB4}"

# ---------------------------------------------------------------------------
# Step 5 — Binding energy (runs only after all four reconstructions succeed)
# ---------------------------------------------------------------------------
JOB5=$(sbatch --parsable \
    --dependency="$ALL_RECON_DEPS" \
    --partition="$SLURM_PARTITION" \
    $ACCOUNT_FLAG \
    --container-image="$CONTAINER_IMAGE" \
    --container-mounts="$MOUNT" \
    --container-workdir=/workspace \
    $BASE_ENV \
    "$SCRIPT_DIR/05_binding_energy.slurm")
echo ""
echo "  Step 5 (binding ΔE):  job $JOB5  (depends on all step-4 jobs)"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "Proteasome EWF+SQD Workflow Submitted"
echo "Submitted at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "Execution graph (all four systems run in parallel):"
echo ""
echo "  ixazomib_complex:   1→2→3→4 ─┐"
echo "  ixazomib_ligand:    1→2→3→4 ─┤"
echo "                                 ├─→ 5 (binding ΔE)"
echo "  bortezomib_complex: 1→2→3→4 ─┤"
echo "  bortezomib_ligand:  1→2→3→4 ─┘"
echo ""
echo "Monitor with:  squeue -u \$USER"
echo "Check logs:    tail -f logs/03_solve_<JOBID>.out"
echo ""
echo "Final result:  results/binding_energies.json"
