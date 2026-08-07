#!/bin/bash
# Trial run: submit all 4 stages using FCI for every fragment (no QPU).
#
# This validates the full pipeline end-to-end before committing a QPU job.
# Uses config_alanine_sto-3g_trial.yaml which sets orbital_threshold: 999
# so all fragments route to FCI.
#
# After this run succeeds:
#   1. Check data/embedding_data.pkl fragment sizes to tune orbital_threshold
#      in config_alanine_sto-3g.yaml for the production run.
#   2. Run bash run_workflow.sh for the full EWF+SQD production workflow.
#
# CONFIGURATION: .hpc_config must be filled in at the repo root.

set -e

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

: "${CONTAINER_IMAGE:?CONTAINER_IMAGE not set in .hpc_config}"
: "${SLURM_PARTITION:?SLURM_PARTITION not set in .hpc_config}"

mkdir -p "$SCRIPT_DIR/logs" "$SCRIPT_DIR/data" "$SCRIPT_DIR/results"

MOUNT="$REPO_ROOT:/workspace"

# Config path must be the container-internal path (/workspace/...), not the
# GPFS path, because GPFS is not mounted inside the container — only /workspace is.
# Replace the REPO_ROOT prefix with /workspace to get the in-container path.
TRIAL_CONFIG="/workspace/${SCRIPT_DIR#$REPO_ROOT/}/config_alanine_sto-3g_trial.yaml"

BASE_ENV="--container-env=NVIDIA_DRIVER_CAPABILITIES=compute,utility"

ACCOUNT_FLAG=""
if [ -n "${SLURM_ACCOUNT:-}" ]; then
    ACCOUNT_FLAG="--account=$SLURM_ACCOUNT"
fi

# ---------------------------------------------------------------------------
# Helper: submit a slurm script with the trial config injected via env var
# Each slurm script reads TRIAL_CONFIG from the environment if set.
# ---------------------------------------------------------------------------
submit() {
    local script="$1"
    local dep_flag="${2:-}"

    sbatch --parsable \
        ${dep_flag:+--dependency=afterok:$dep_flag} \
        --partition="$SLURM_PARTITION" \
        $ACCOUNT_FLAG \
        --container-image="$CONTAINER_IMAGE" \
        --container-mounts="$MOUNT" \
        --container-workdir=/workspace \
        $BASE_ENV \
        --container-env=QFM_TRIAL_CONFIG \
        "$script"
}

# Export so the container can see it
export QFM_TRIAL_CONFIG="$TRIAL_CONFIG"

echo "Trial run using: $TRIAL_CONFIG"
echo ""

JOB1=$(submit "$SCRIPT_DIR/01_meanfield.slurm")
echo "Submitted step 1 (meanfield):    job $JOB1"

JOB2=$(submit "$SCRIPT_DIR/02_fragments.slurm" "$JOB1")
echo "Submitted step 2 (fragments):    job $JOB2 (depends on $JOB1)"

JOB3=$(submit "$SCRIPT_DIR/03_solve.slurm" "$JOB2")
echo "Submitted step 3 (solve/FCI):    job $JOB3 (depends on $JOB2)"

JOB4=$(submit "$SCRIPT_DIR/04_reconstruct.slurm" "$JOB3")
echo "Submitted step 4 (reconstruct):  job $JOB4 (depends on $JOB3)"

echo ""
echo "Trial workflow submitted: $JOB1 → $JOB2 → $JOB3 → $JOB4"
echo ""
echo "Monitor with:  squeue -u \$USER"
echo ""
echo "Logs:"
echo "  logs/01_meanfield_${JOB1}.out"
echo "  logs/02_fragments_${JOB2}.out"
echo "  logs/03_solve_${JOB3}.out"
echo "  logs/04_reconstruct_${JOB4}.out"
echo ""
echo "After completion:"
echo "  python -c \"import pickle; d=pickle.load(open('data/alanine_sto-3g_trial/embedding_data.pkl','rb')); [print(f'  frag {k}: n_orb={v[\\\"n_orbitals\\\"]}') for k,v in d['fragment_meta'].items()]\""
echo "  cat results/alanine_sto-3g_trial/summary.json"
