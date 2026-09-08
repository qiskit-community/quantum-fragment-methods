#!/usr/bin/env bash
# 03_solve_sqd_monitor.sh
#
# Cheap IBM Quantum job status poller.  Reads job_manifest.json and calls the
# QRMI status API for each fragment.  Runs entirely from the login node — no
# GPU or HPC allocation needed.
#
# Usage:
#   bash 03_solve_sqd_monitor.sh <run_name> [--watch <seconds>]
#
#   run_name  e.g. ixazomib_complex_sto-3g
#   --watch N re-poll every N seconds until all jobs are terminal (default: one-shot)
#
# Examples:
#   bash 03_solve_sqd_monitor.sh ixazomib_complex_sto-3g
#   bash 03_solve_sqd_monitor.sh ixazomib_complex_sto-3g --watch 300
#
# Prerequisites:
#   - QRMI env vars set (source .qrmi_config or equivalent)
#   - qrmi[ibm] installed in the active Python env

set -euo pipefail

DEMO_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$DEMO_DIR/../../.." && pwd)"
HPC_CONFIG="$REPO_ROOT/.hpc_config"

# Load cluster config for QRMI creds path
if [ -f "$HPC_CONFIG" ]; then
    set -a; source "$HPC_CONFIG"; set +a
fi

# Load QRMI credentials if available
if [ -n "${QRMI_CREDS:-}" ] && [ -f "$QRMI_CREDS" ]; then
    set -a; source "$QRMI_CREDS"; set +a
fi

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
RUN_NAME="${1:-}"
WATCH_INTERVAL=0

if [ -z "$RUN_NAME" ]; then
    echo "Usage: $0 <run_name> [--watch <seconds>]"
    echo "  run_name: e.g. ixazomib_complex_sto-3g"
    exit 1
fi

shift
while [[ $# -gt 0 ]]; do
    case "$1" in
        --watch) WATCH_INTERVAL="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

MANIFEST="$DEMO_DIR/data/${RUN_NAME}/sqd_jobs/job_manifest.json"

if [ ! -f "$MANIFEST" ]; then
    echo "ERROR: manifest not found: $MANIFEST"
    echo "Has 03_solve_sqd_submit.py been run for $RUN_NAME?"
    exit 1
fi

# ---------------------------------------------------------------------------
# Python status checker — lightweight, no GPU, no pyscf needed
# ---------------------------------------------------------------------------
poll_once() {
    python3 - "$MANIFEST" << 'PYEOF'
import json, sys, os
from pathlib import Path

manifest_path = Path(sys.argv[1])
manifest = json.loads(manifest_path.read_text())

# Import QRMI — credentials must already be in env
try:
    from quantum_fragment_methods.qpu.qrmi import QRMIBackend
except ImportError:
    print("ERROR: quantum_fragment_methods not importable. Activate the correct Python env.")
    sys.exit(1)

# Determine backend name from any entry in the manifest
backend_name = None
for entry in manifest.values():
    backend_name = entry.get("backend_name") or backend_name
    break

# Minimal config — only need backend_name for QRMI init
qpu_config = {"provider": "qrmi", "backend_name": backend_name or ""}
try:
    backend = QRMIBackend(qpu_config)
    backend.initialize()
except Exception as e:
    print(f"WARNING: QRMI init failed ({e}). Showing manifest only (no live status).")
    backend = None

statuses = {}
for frag_id, entry in sorted(manifest.items(), key=lambda x: int(x[0])):
    job_id   = entry.get("job_id", "—")
    n_orb    = entry.get("n_orb", "?")
    cached   = (Path(entry.get("frag_dir", "")) / "counts.npy").exists() if entry.get("frag_dir") else False

    if cached:
        status = "COUNTS_CACHED"
    elif backend and job_id and job_id != "—":
        try:
            status = backend.get_job_status(job_id)
        except Exception as e:
            status = f"ERROR({e})"
    else:
        status = entry.get("status", "UNKNOWN")

    statuses[frag_id] = status
    icon = "✓" if status in ("COMPLETED", "DONE", "COUNTS_CACHED") else \
           "✗" if status in ("FAILED", "CANCELLED", "ERROR") else "…"
    print(f"  {icon}  frag {frag_id:>4}  n_orb={n_orb:>3}  {job_id:<40}  {status}")

# Summary line
n_done    = sum(1 for s in statuses.values() if s in ("COMPLETED","DONE","COUNTS_CACHED"))
n_running = sum(1 for s in statuses.values() if s in ("RUNNING","QUEUED","VALIDATING"))
n_fail    = sum(1 for s in statuses.values() if "ERROR" in s or s in ("FAILED","CANCELLED"))
n_total   = len(statuses)

print(f"\n  Summary: {n_done}/{n_total} done  {n_running} in-flight  {n_fail} failed")

# Update manifest with live statuses
updated = dict(manifest)
for frag_id, status in statuses.items():
    if frag_id in updated:
        updated[frag_id]["status"] = status
manifest_path.write_text(json.dumps(updated, indent=2))

# Exit code: 0 = all terminal (done or failed), 1 = still running
all_terminal = (n_running == 0)
sys.exit(0 if all_terminal else 1)
PYEOF
}

# ---------------------------------------------------------------------------
# Run once or watch loop
# ---------------------------------------------------------------------------
if [ "$WATCH_INTERVAL" -le 0 ] 2>/dev/null; then
    echo "=== QPU Job Status: $RUN_NAME  [$(date '+%Y-%m-%d %H:%M:%S')] ==="
    poll_once "$MANIFEST" || true
else
    echo "Watching $RUN_NAME every ${WATCH_INTERVAL}s (Ctrl-C to stop)..."
    while true; do
        echo ""
        echo "=== $(date '+%Y-%m-%d %H:%M:%S') ==="
        if poll_once "$MANIFEST"; then
            echo "All jobs terminal. Stopping watch."
            echo "→ Run 03_solve_sqd_collect.py to retrieve results and run SBD postprocessing."
            break
        fi
        sleep "$WATCH_INTERVAL"
    done
fi
