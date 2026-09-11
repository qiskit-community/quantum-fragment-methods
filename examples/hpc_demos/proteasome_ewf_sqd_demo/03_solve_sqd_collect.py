"""
Step 3c: Retrieve QPU counts and run SBD postprocessing for all SQD fragments.
Merge with CCSD results for small fragments to produce the final solver_results.pkl.

This script should be run AFTER all QPU jobs have completed (verified via
03_solve_sqd_monitor.sh).  It:

  1. Reads job_manifest.json to find all fragment job_ids.
  2. For each fragment whose counts.npy is not yet cached, retrieves results
     from IBM Quantum via QRMI.
  3. Runs SBD classical postprocessing on the counts to get energies + RDMs.
     SBD is checkpoint-aware at the per-iteration level: if a run is
     interrupted (TIMEOUT, preemption, OOM) it resumes from the last
     completed iteration rather than restarting from scratch.
  4. Loads ccsd_results.pkl (from 03_solve_ccsd.py) for FCI/CCSD fragments.
  5. Merges everything into solver_results.pkl — the same format consumed by
     04_reconstruct.py with no changes needed downstream.

SBD checkpoint files per fragment:
  sqd_jobs/fragment_<id>/sbd_checkpoint.pkl   — best SCIResult after each iter
  sqd_jobs/fragment_<id>/sbd_result.pkl       — final complete SolverResult

Re-running is fully safe: fragments with sbd_result.pkl are loaded directly;
fragments with sbd_checkpoint.pkl resume from the saved occupancies + CI strings.

Usage:
    python 03_solve_sqd_collect.py --config config_ixazomib_complex_sto-3g.yaml
    python 03_solve_sqd_collect.py --config config_bortezomib_complex_sto-3g.yaml
"""

import argparse
import json
import logging
import os
import pickle
from pathlib import Path

import numpy as np
import yaml
from pyscf import gto, scf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)

from quantum_fragment_methods.application.embedding.base import EmbeddingResult, Fragment
from quantum_fragment_methods.application.solvers.base import SolverResult
from quantum_fragment_methods.application.solvers.quantum_zoo.sqd import SQDSolver
from quantum_fragment_methods.application.solvers.quantum_zoo.sqd import (
    diagonalize_fermionic_hamiltonian,
    counts_to_bit_array,
)
from quantum_fragment_methods.qpu import QRMIBackend

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--config", required=True)
_known, _ = _pre.parse_known_args()
_run_name = Path(_known.config).stem.removeprefix("config_")
_demo_dir = Path(__file__).parent

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config", required=True, help="Path to config YAML")
parser.add_argument(
    "--data-dir",
    default=str(_demo_dir / "data" / _run_name),
    help="Directory containing mf_data.pkl, embedding_data.pkl, ccsd_results.pkl",
)
parser.add_argument(
    "--jobs-dir",
    default=str(_demo_dir / "data" / _run_name / "sqd_jobs"),
    help="Directory containing job_manifest.json and per-fragment checkpoints",
)
parser.add_argument(
    "--output-dir",
    default=str(_demo_dir / "data" / _run_name),
    help="Directory to write solver_results.pkl",
)
parser.add_argument(
    "--skip-incomplete",
    action="store_true",
    help="Skip fragments whose QPU jobs are not yet done instead of raising an error",
)
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Load config
# ---------------------------------------------------------------------------
config_path = Path(args.config).resolve()
with open(config_path) as f:
    config = yaml.safe_load(f)

qpu_config  = config["qpu"]
sqd_config  = config["sqd"]
sel_config  = config["solver_selection"]
orbital_threshold = sel_config.get("orbital_threshold", 15)
strategy = sel_config.get("strategy", "adaptive")

print(f"Config:  {config_path}")

# ---------------------------------------------------------------------------
# Load manifest
# ---------------------------------------------------------------------------
jobs_dir = Path(args.jobs_dir)
manifest_path = jobs_dir / "job_manifest.json"
if not manifest_path.exists():
    raise FileNotFoundError(
        f"job_manifest.json not found at {manifest_path}. "
        "Run 03_solve_sqd_submit.py first."
    )
with open(manifest_path) as f:
    manifest = json.load(f)

print(f"Manifest: {len(manifest)} SQD fragment(s)")

# ---------------------------------------------------------------------------
# Load pickled data
# ---------------------------------------------------------------------------
data_dir = Path(args.data_dir)

with open(data_dir / "mf_data.pkl", "rb") as f:
    mf_data = pickle.load(f)
with open(data_dir / "embedding_data.pkl", "rb") as f:
    emb_data = pickle.load(f)

# Load CCSD/FCI results for non-SQD fragments
ccsd_file = data_dir / "ccsd_results.pkl"
if not ccsd_file.exists():
    raise FileNotFoundError(
        f"ccsd_results.pkl not found at {ccsd_file}. "
        "Run 03_solve_ccsd.py first."
    )
with open(ccsd_file, "rb") as f:
    ccsd_results = pickle.load(f)
print(f"Loaded CCSD/FCI results: {len(ccsd_results)} fragment(s)")

charge = emb_data.get("charge", 0)
spin   = emb_data.get("spin", 0)

# ---------------------------------------------------------------------------
# Reconstruct PySCF mol + mf
# ---------------------------------------------------------------------------
mol = gto.Mole()
mol.atom = emb_data["atom_data"]
mol.unit = "Angstrom"
mol.basis = emb_data["basis"]
mol.charge = charge
mol.spin   = spin
mol.verbose = 0
mol.build()

mf = scf.RHF(mol).density_fit()
mf.mo_coeff  = mf_data["mo_coeff"]
mf.mo_occ    = mf_data["mo_occ"]
mf.mo_energy = mf_data["mo_energy"]
mf.e_tot     = mf_data["hf_energy"]
mf.converged = True

# ---------------------------------------------------------------------------
# Initialize QRMI backend (needed to retrieve any un-cached jobs)
# ---------------------------------------------------------------------------
qrmi_available = bool(os.environ.get("QRMI_JOB_QPU_RESOURCES") or
                      os.environ.get("SLURM_JOB_QPU_RESOURCES"))

backend = None
uncached = [
    fid for fid, entry in manifest.items()
    if not (jobs_dir / f"fragment_{fid}" / "counts.npy").exists()
]

if uncached:
    if not qrmi_available:
        raise EnvironmentError(
            f"{len(uncached)} fragment(s) have no cached counts yet "
            "and QRMI_JOB_QPU_RESOURCES is not set. "
            "Either set QRMI credentials or re-run once all jobs are done "
            "and counts.npy files are present."
        )
    print(f"\n{len(uncached)} fragment(s) need retrieval — initializing QRMI...")
    backend = QRMIBackend(qpu_config)
    backend.initialize()
    backend.get_backend()
    props = backend.get_backend_properties()
    print(f"Resource: {props['backend_name']} ({props['resource_type']})")
else:
    print("\nAll fragments have cached counts — no QRMI connection needed.")

# ---------------------------------------------------------------------------
# SBD checkpoint helpers
# ---------------------------------------------------------------------------

def _sbd_checkpoint_callback(frag_dir: Path):
    """Return a callback that saves the best SCIResult after each SBD iteration.

    diagonalize_fermionic_hamiltonian calls callback(results: list[SCIResult])
    after every configuration-recovery iteration.  We pick the lowest-energy
    result and pickle it to sbd_checkpoint.pkl so a resume can seed
    initial_occupancies and include_configurations from where we left off.
    """
    checkpoint_path = frag_dir / "sbd_checkpoint.pkl"

    def _callback(results):
        best = min(results, key=lambda r: r.energy)
        with open(checkpoint_path, "wb") as f:
            pickle.dump(best, f)

    return _callback


def _load_sbd_checkpoint(frag_dir: Path):
    """Load a partial SBD checkpoint.

    Returns (initial_occupancies, include_configurations, completed_iterations)
    or (None, None, 0) if no checkpoint exists.
    """
    checkpoint_path = frag_dir / "sbd_checkpoint.pkl"
    if not checkpoint_path.exists():
        return None, None, 0

    try:
        with open(checkpoint_path, "rb") as f:
            best = pickle.load(f)
        # initial_occupancies: tuple of (occ_alpha, occ_beta) arrays
        occ = best.orbital_occupancies
        initial_occupancies = (occ, occ) if isinstance(occ, np.ndarray) else tuple(occ)
        # include_configurations: seed the CI subspace from last best state
        include_configurations = (
            best.sci_state.ci_strs_a,
            best.sci_state.ci_strs_b,
        )
        # Count completed iterations from the iteration_N subdirs
        sbd_dir = frag_dir / "sqd_diagonalizer"
        completed = sum(
            1 for p in sbd_dir.iterdir() if p.is_dir() and p.name.startswith("iteration_")
        ) if sbd_dir.exists() else 0
        return initial_occupancies, include_configurations, completed
    except Exception as e:
        print(f"  WARNING: could not load SBD checkpoint ({e}); starting from scratch.")
        return None, None, 0


def _run_sbd_with_checkpoint(
    h1e, h2e, counts, norb, nocc, frag_dir, sqd_config, config
):
    """Run SBD postprocessing with per-iteration checkpointing.

    Resumes from sbd_checkpoint.pkl if present, skipping already-completed
    iterations.  Saves final SolverResult to sbd_result.pkl when done.
    """
    nelec = (nocc, nocc)
    iterations_total = sqd_config.get("iterations", 5)
    sbd_result_path  = frag_dir / "sbd_result.pkl"

    # --- Resume from checkpoint if available ---
    initial_occupancies, include_configurations, completed = _load_sbd_checkpoint(frag_dir)
    remaining = iterations_total - completed

    if completed > 0:
        print(f"  Resuming SBD from iteration {completed + 1}/{iterations_total} "
              f"({remaining} remaining)...", flush=True)
    else:
        print(f"  Starting SBD: {iterations_total} iterations...", flush=True)

    if remaining <= 0:
        # All iterations already done — load from checkpoint directly
        print("  All SBD iterations already complete; loading checkpoint.", flush=True)
        with open(frag_dir / "sbd_checkpoint.pkl", "rb") as f:
            best = pickle.load(f)
    else:
        best = diagonalize_fermionic_hamiltonian(
            one_body_tensor=h1e,
            two_body_tensor=h2e,
            counts=counts,
            samples_per_batch=sqd_config.get("samples_per_batch", 3000),
            norb=norb,
            nelec=nelec,
            num_batches=sqd_config.get("n_batches", 5),
            energy_tol=sqd_config.get("energy_tol", 1e-8),
            occupancies_tol=sqd_config.get("occupancies_tol", 1e-5),
            max_iterations=remaining,
            symmetrize_spin=sqd_config.get("symmetrize_spin", True),
            carryover_threshold=sqd_config.get("carryover_threshold", 1e-4),
            initial_occupancies=initial_occupancies,
            include_configurations=include_configurations,
            callback=_sbd_checkpoint_callback(frag_dir),
            workflow_path=str(frag_dir / "sqd_diagonalizer"),
            sbd_config=sqd_config.get("sbd"),
            classical_backend=sqd_config.get("classical_backend", "python"),
        )

    # Package into SolverResult
    rdm1 = best.rdm1 if best.rdm1 is not None else best.sci_state.rdm(rank=1)
    rdm2 = best.rdm2 if best.rdm2 is not None else best.sci_state.rdm(rank=2)
    result = SolverResult(
        energy=best.energy,
        wavefunction=best.sci_state.amplitudes,
        rdm1=rdm1,
        rdm2=rdm2,
        metadata={
            "ci_strs_a":   best.sci_state.ci_strs_a,
            "ci_strs_b":   best.sci_state.ci_strs_b,
            "occupancies": best.orbital_occupancies,
            "norb":        norb,
            "nelec":       nelec,
        },
    )

    # Save final result so re-runs skip SBD entirely for this fragment
    with open(sbd_result_path, "wb") as f:
        pickle.dump(result, f)

    return result


# ---------------------------------------------------------------------------
# Retrieve counts + SBD postprocessing for SQD fragments
# ---------------------------------------------------------------------------
import h5py

dumpfile = emb_data["dumpfile"]
sqd_results: dict = {}
n_retrieved = n_cached = n_skipped = n_sbd_resumed = 0

print(f"\n{'='*60}")
print(f"Processing {len(manifest)} SQD fragments...")
print(f"{'='*60}")

for frag_id_str, entry in sorted(manifest.items(), key=lambda x: int(x[0])):
    frag_id   = type(next(iter(emb_data["fragment_meta"])))(frag_id_str)  # match key type
    frag_dir  = jobs_dir / f"fragment_{frag_id}"
    job_id    = entry.get("job_id")
    n_orb     = entry.get("n_orb", "?")

    with h5py.File(dumpfile, "r") as hf:
        grp  = hf[f"fragment_{int(frag_id)}"]
        h1e  = grp["heff"][:]
        h2e  = grp["eris"][:]
        norb = int(grp.attrs["norb"])
        nocc = int(grp.attrs["nocc"])

    # --- Stage 0: final SBD result already saved — skip entirely ---
    sbd_result_path = frag_dir / "sbd_result.pkl"
    if sbd_result_path.exists():
        print(f"  Fragment {frag_id}: loading final SBD result from disk", flush=True)
        with open(sbd_result_path, "rb") as f:
            result = pickle.load(f)
        sqd_results[frag_id] = result
        continue

    # --- Stage 1: ensure counts are present ---
    counts_file = frag_dir / "counts.npy"
    if counts_file.exists():
        print(f"  Fragment {frag_id}: counts cached, running SBD...", flush=True)
        n_cached += 1
    elif backend is not None:
        print(f"  Fragment {frag_id}: retrieving job {job_id}...", flush=True)
        try:
            status = backend.get_job_status(job_id)
        except Exception as e:
            print(f"  WARNING: could not check status for {job_id}: {e}")
            status = "UNKNOWN"

        if status not in ("COMPLETED", "DONE"):
            msg = f"  Fragment {frag_id} job {job_id} is {status} — not yet complete."
            if args.skip_incomplete:
                print(f"  {msg} Skipping.")
                n_skipped += 1
                continue
            else:
                raise RuntimeError(
                    f"{msg}\n"
                    "Wait for completion and re-run, or pass --skip-incomplete."
                )
        # Retrieve counts via SQDSolver (writes counts.npy)
        solver = SQDSolver(backend, config=sqd_config)
        solver.solve(
            h1e, h2e, norb, (nocc, nocc),
            mf=mf,
            workflow_path=str(frag_dir),
            wait_for_completion=False,
        )
        n_retrieved += 1
    else:
        print(f"  WARNING: no counts and no backend for fragment {frag_id}; skipping.")
        n_skipped += 1
        continue

    # --- Stage 2: SBD postprocessing with per-iteration checkpointing ---
    _, _, completed = _load_sbd_checkpoint(frag_dir)
    if completed > 0:
        n_sbd_resumed += 1

    counts = np.load(counts_file, allow_pickle=True).item()
    result = _run_sbd_with_checkpoint(
        h1e, h2e, counts, norb, nocc, frag_dir, sqd_config, config
    )

    with h5py.File(dumpfile, "r") as hf:
        grp = hf[f"fragment_{int(frag_id)}"]
        if "c_frag"    in grp: result.metadata["c_frag"]    = grp["c_frag"][:]
        if "c_cluster" in grp: result.metadata["c_cluster"] = grp["c_cluster"][:]
        result.metadata["norb"] = norb
        result.metadata["nocc"] = nocc

    print(f"  → E={result.energy:.8f} Ha  (fragment {frag_id})", flush=True)
    sqd_results[frag_id] = result

    # Update manifest status
    manifest[frag_id_str]["status"] = "COLLECTED"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

print(f"\n  Retrieved: {n_retrieved}  Cached: {n_cached}  "
      f"SBD resumed: {n_sbd_resumed}  Skipped: {n_skipped}")

# ---------------------------------------------------------------------------
# Merge: CCSD/FCI results + SQD results → solver_results.pkl
# ---------------------------------------------------------------------------
fragment_results = dict(ccsd_results)   # FCI + CCSD for small/classical frags
fragment_results.update(sqd_results)    # SQD overrides for quantum-treated frags

print(f"\nMerged results: {len(ccsd_results)} CCSD/FCI + {len(sqd_results)} SQD "
      f"= {len(fragment_results)} total")

output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)
output_file = output_dir / "solver_results.pkl"

with open(output_file, "wb") as f:
    pickle.dump(fragment_results, f)

print(f"\n{'='*60}")
print(f"solver_results.pkl written: {output_file}")
print(f"→ Run 04_reconstruct.py next.")
print(f"{'='*60}")
