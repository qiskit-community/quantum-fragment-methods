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
# MODES
# -----
# Default (no flag): classical CCSD reference.
#   Chain: 1 → 2 → 3 (03_solve.slurm, CCSD fallback) → 4 → 5
#   No QPU resource required.
#
# --qpu: fire-and-forget QPU workflow.
#   Chain: 1 → 2 → 3a (03_solve_ccsd.slurm) → 4
#   Step 3b (QPU submit) and 3c (collect + merge) are printed as manual
#   commands to run after 3a completes — they require QRMI credentials and
#   should be submitted manually once the CCSD reference run is confirmed OK.
#   3c → 4 → 5 can be queued once QPU jobs are confirmed COMPLETED via
#   03_solve_sqd_monitor.sh.
#
# CONFIGURATION: copy <repo-root>/.hpc_config.template to .hpc_config and
# fill in your cluster values before running this script.

set -e

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
QPU_MODE=0
for arg in "$@"; do
    case "$arg" in
        --qpu) QPU_MODE=1 ;;
        --help|-h)
            echo "Usage: $0 [--qpu]"
            echo ""
            echo "  (default)  Submit classical CCSD reference workflow (03_solve.slurm)"
            echo "  --qpu      Submit QPU split workflow (03_solve_ccsd.slurm step 3a;"
            echo "             prints manual commands for 3b QPU submit + 3c collect)"
            exit 0
            ;;
        *) echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

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
# QRMI credential flags (injected into QPU-facing jobs: 3, 3b, 3c)
# ---------------------------------------------------------------------------
QRMI_ENV_FLAGS=""
if [ -f "$QRMI_CREDS" ]; then
    set -a; source "$QRMI_CREDS"; set +a
    for var in QRMI_JOB_QPU_RESOURCES QRMI_JOB_QPU_TYPES $(env | grep _QRMI_ | cut -d= -f1); do
        QRMI_ENV_FLAGS="$QRMI_ENV_FLAGS --container-env=$var"
    done
else
    echo "WARNING: QRMI credentials not found at $QRMI_CREDS"
    echo "QPU solve steps will use CCSD fallback."
fi

# ---------------------------------------------------------------------------
# Helper: submit a chain for one system and return the step-4 job ID
#
# Classical mode:  1 → 2 → 3 (monolithic FCI/CCSD) → 4
# QPU mode:        1 → 2 → 3a (CCSD ref, blocks step 4 too) → 4
#                  3b/3c are printed for manual submission after QPU completes
# ---------------------------------------------------------------------------
submit_system() {
    local SYSTEM_NAME="$1"

    # Host-side config path (verified to exist before submitting)
    local HOST_CONFIG="$SCRIPT_DIR/config_${SYSTEM_NAME}_sto-3g.yaml"
    if [ ! -f "$HOST_CONFIG" ]; then
        echo "ERROR: config not found: $HOST_CONFIG"
        exit 1
    fi

    # Container-relative path (repo root mounted at /workspace)
    local CONFIG_FILE="/workspace${HOST_CONFIG#$REPO_ROOT}"

    echo ""
    echo "─── Submitting system: $SYSTEM_NAME ───"

    # Export config path into shell env so Pyxis can forward with
    # --container-env=VAR (no =value in the flag — avoids doubling bug).
    export QFM_PROTEASOME_CONFIG="$CONFIG_FILE"

    # Step 1 — Mean-field
    JOB1=$(sbatch --parsable \
        --partition="$SLURM_PARTITION" \
        $ACCOUNT_FLAG \
        --container-image="$CONTAINER_IMAGE" \
        --container-mounts="$MOUNT" \
        --container-workdir=/workspace \
        $BASE_ENV \
        --container-env=QFM_PROTEASOME_CONFIG \
        "$SCRIPT_DIR/01_meanfield.slurm")
    echo "  Step 1 (meanfield):   job $JOB1"

    # Step 2 — EWF Fragmentation
    JOB2=$(sbatch --parsable \
        --dependency=afterok:$JOB1 \
        --partition="$SLURM_PARTITION" \
        $ACCOUNT_FLAG \
        --container-image="$CONTAINER_IMAGE" \
        --container-mounts="$MOUNT" \
        --container-workdir=/workspace \
        $BASE_ENV \
        --container-env=QFM_PROTEASOME_CONFIG \
        "$SCRIPT_DIR/02_fragments.slurm")
    echo "  Step 2 (fragments):   job $JOB2  (depends on $JOB1)"

    if [ "$QPU_MODE" -eq 0 ]; then
        # ── Classical mode: monolithic FCI/CCSD solver ────────────────────────
        JOB3=$(sbatch --parsable \
            --dependency=afterok:$JOB2 \
            --partition="$SLURM_PARTITION" \
            $ACCOUNT_FLAG \
            --container-image="$CONTAINER_IMAGE" \
            --container-mounts="$MOUNT" \
            --container-workdir=/workspace \
            $BASE_ENV \
            $QRMI_ENV_FLAGS \
            --container-env=QFM_PROTEASOME_CONFIG \
            "$SCRIPT_DIR/03_solve.slurm")
        echo "  Step 3 (solve/CCSD):  job $JOB3  (depends on $JOB2)"

        # Step 4 — Energy reconstruction
        JOB4=$(sbatch --parsable \
            --dependency=afterok:$JOB3 \
            --partition="$SLURM_PARTITION" \
            $ACCOUNT_FLAG \
            --container-image="$CONTAINER_IMAGE" \
            --container-mounts="$MOUNT" \
            --container-workdir=/workspace \
            $BASE_ENV \
            --container-env=QFM_PROTEASOME_CONFIG \
            "$SCRIPT_DIR/04_reconstruct.slurm")
        echo "  Step 4 (reconstruct): job $JOB4  (depends on $JOB3)"

    else
        # ── QPU mode: split 3a (CCSD ref) → manual 3b (QPU submit) → 3c (collect) → 4 ──
        #
        # 3a: runs CCSD on all SQD-eligible fragments (classical reference +
        #     blocks step 4 until classical results are in hand).
        JOB3A=$(sbatch --parsable \
            --dependency=afterok:$JOB2 \
            --partition="$SLURM_PARTITION" \
            $ACCOUNT_FLAG \
            --container-image="$CONTAINER_IMAGE" \
            --container-mounts="$MOUNT" \
            --container-workdir=/workspace \
            $BASE_ENV \
            --container-env=QFM_PROTEASOME_CONFIG \
            "$SCRIPT_DIR/03_solve_ccsd.slurm")
        echo "  Step 3a (CCSD ref):   job $JOB3A  (depends on $JOB2)"

        # Steps 3b and 3c are fire-and-forget / manual:
        #   3b submits QPU jobs and exits immediately (cheap: ~1h wall time)
        #   3c retrieves counts and runs SBD once all QPU jobs are DONE
        # They are printed below in the QPU commands section — not chained
        # automatically because QPU queue time is unpredictable.
        #
        # Step 4 chains off 3a for now (uses ccsd_results.pkl as solver_results.pkl
        # proxy).  Re-run step 4 after 3c completes if you want QPU-corrected RDMs.
        JOB4=$(sbatch --parsable \
            --dependency=afterok:$JOB3A \
            --partition="$SLURM_PARTITION" \
            $ACCOUNT_FLAG \
            --container-image="$CONTAINER_IMAGE" \
            --container-mounts="$MOUNT" \
            --container-workdir=/workspace \
            $BASE_ENV \
            --container-env=QFM_PROTEASOME_CONFIG \
            "$SCRIPT_DIR/04_reconstruct.slurm")
        echo "  Step 4 (reconstruct): job $JOB4  (depends on $JOB3A)"

        # Stash info for QPU manual commands printed after all systems
        QPU_MANUAL_SYSTEMS="$QPU_MANUAL_SYSTEMS $SYSTEM_NAME:$JOB3A"
        export "QPU_JOB3A_${SYSTEM_NAME//[^a-zA-Z0-9]/_}=$JOB3A"
    fi

    # Return step-4 job ID to caller via a named variable
    eval "${SYSTEM_NAME//[^a-zA-Z0-9]/_}_JOB4=$JOB4"
}

# ---------------------------------------------------------------------------
# Submit all four systems
# ---------------------------------------------------------------------------
QPU_MANUAL_SYSTEMS=""
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
if [ "$QPU_MODE" -eq 0 ]; then
    echo "Proteasome EWF+CCSD Classical Reference Workflow Submitted"
else
    echo "Proteasome EWF+SQD QPU Workflow — Step 3a Submitted"
fi
echo "Submitted at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "Execution graph (all four systems run in parallel):"
echo ""
if [ "$QPU_MODE" -eq 0 ]; then
    echo "  ixazomib_complex:   1→2→3→4 ─┐"
    echo "  ixazomib_ligand:    1→2→3→4 ─┤"
    echo "                                 ├─→ 5 (binding ΔE)"
    echo "  bortezomib_complex: 1→2→3→4 ─┤"
    echo "  bortezomib_ligand:  1→2→3→4 ─┘"
else
    echo "  ixazomib_complex:   1→2→3a→4 ─┐"
    echo "  ixazomib_ligand:    1→2→3a→4 ─┤"
    echo "                                  ├─→ 5 (binding ΔE, CCSD energies)"
    echo "  bortezomib_complex: 1→2→3a→4 ─┤"
    echo "  bortezomib_ligand:  1→2→3a→4 ─┘"
    echo ""
    echo "  QPU steps 3b + 3c run AFTER 3a completes — see manual commands below."
fi
echo ""
echo "Monitor:      squeue -u \$USER"
echo "Check logs:   tail -f $SCRIPT_DIR/logs/03_solve_<JOBID>.out"
echo ""
echo "Final result: $SCRIPT_DIR/results/binding_energies.json"

# ---------------------------------------------------------------------------
# QPU mode: print manual commands for 3b (submit) and 3c (collect)
# ---------------------------------------------------------------------------
if [ "$QPU_MODE" -eq 1 ]; then
    DEMO="$SCRIPT_DIR"
    IMG="$CONTAINER_IMAGE"
    REPO="$REPO_ROOT"

    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "QPU Steps — Run MANUALLY after 3a jobs complete"
    echo "═══════════════════════════════════════════════════════════════"
    echo ""
    echo "For each system, after its 3a job is COMPLETED:"
    echo ""
    echo "  1. Monitor 3a:   sacct -j <JOB3A_ID> --format=State -X"
    echo ""
    echo "  2. Submit QPU jobs (3b) — one per system, costs only ~1h node:"
    echo ""

    for SYSTEM_NAME in ixazomib_complex ixazomib_ligand bortezomib_complex bortezomib_ligand; do
        VARNAME="QPU_JOB3A_${SYSTEM_NAME//[^a-zA-Z0-9]/_}"
        J3A="${!VARNAME:-<JOB3A_ID>}"
        HOST_CONFIG="$SCRIPT_DIR/config_${SYSTEM_NAME}_sto-3g.yaml"
        CONFIG_FILE="/workspace${HOST_CONFIG#$REPO_ROOT}"
        echo "    # $SYSTEM_NAME (after 3a job $J3A completes)"
        echo "    export QFM_PROTEASOME_CONFIG=\"$CONFIG_FILE\""
        echo "    sbatch --dependency=afterok:$J3A \\"
        echo "        --partition=$SLURM_PARTITION \\"
        echo "        --container-image=$IMG \\"
        echo "        --container-mounts=\"$REPO:/workspace\" \\"
        echo "        --container-workdir=/workspace \\"
        echo "        $BASE_ENV \\"
        echo "        $QRMI_ENV_FLAGS \\"
        echo "        --container-env=QFM_PROTEASOME_CONFIG \\"
        echo "        \"$DEMO/03_solve_sqd_submit.slurm\""
        echo ""
    done

    echo ""
    echo "  3. Monitor QPU jobs (login node, no allocation needed):"
    for SYSTEM_NAME in ixazomib_complex ixazomib_ligand bortezomib_complex bortezomib_ligand; do
        echo "    bash $DEMO/03_solve_sqd_monitor.sh ${SYSTEM_NAME}_sto-3g --watch 300"
    done

    echo ""
    echo "  4. Once all QPU jobs DONE, run 3c (collect + SBD + merge):"
    echo "     Submit one job per system with --dependency=afterok:<3b_JOB_ID>"
    echo ""
    for SYSTEM_NAME in ixazomib_complex ixazomib_ligand bortezomib_complex bortezomib_ligand; do
        HOST_CONFIG="$SCRIPT_DIR/config_${SYSTEM_NAME}_sto-3g.yaml"
        CONFIG_FILE="/workspace${HOST_CONFIG#$REPO_ROOT}"
        echo "    # $SYSTEM_NAME"
        echo "    export QFM_PROTEASOME_CONFIG=\"$CONFIG_FILE\""
        echo "    sbatch --dependency=afterok:<3b_JOB_ID> \\"
        echo "        --partition=$SLURM_PARTITION \\"
        echo "        --container-image=$IMG \\"
        echo "        --container-mounts=\"$REPO:/workspace\" \\"
        echo "        --container-workdir=/workspace \\"
        echo "        $BASE_ENV \\"
        echo "        $QRMI_ENV_FLAGS \\"
        echo "        --container-env=QFM_PROTEASOME_CONFIG \\"
        echo "        \"$DEMO/03_solve_sqd_collect.slurm\""
        echo ""
    done

    echo ""
    echo "  5. After all 3c jobs complete, re-run step 4 + 5 with QPU-corrected data:"
    echo "     sbatch --dependency=afterok:<3c_ixa_cx>:<3c_ixa_lig>:<3c_bo2_cx>:<3c_bo2_lig> \\"
    echo "         ... \"$DEMO/05_binding_energy.slurm\""
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
fi
