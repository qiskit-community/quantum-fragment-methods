"""
Step 3b: Submit SQD QPU jobs for all SQD-eligible fragments, then EXIT.

This script does exactly one thing: build a LUCJ circuit per fragment, submit
it to IBM Quantum via QRMI, record the job_id, and exit.  It does NOT wait for
results.  The HPC node is released immediately after submission.

Checkpointing:
  - Per-fragment: data/<run>/sqd_jobs/fragment_<id>/job_id.txt
  - Summary:      data/<run>/sqd_jobs/job_manifest.json
    { "<frag_id>": {"job_id": "...", "status": "SUBMITTED", "n_orb": N}, ... }

Re-running this script is safe: fragments with an existing job_id.txt are
skipped unless --force-resubmit is passed.

Prerequisites:
  - embedding_data.pkl (step 2)
  - mf_data.pkl        (step 1)
  - QRMI env vars set (QRMI_JOB_QPU_RESOURCES etc.)

Usage:
    python 03_solve_sqd_submit.py --config config_ixazomib_complex_sto-3g.yaml
"""

import argparse
import json
import logging
import os
import pickle
from pathlib import Path

import yaml
from pyscf import gto, scf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)

from quantum_fragment_methods.application.embedding.base import EmbeddingResult, Fragment
from quantum_fragment_methods.application.solvers.quantum_zoo.sqd import SQDSolver
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
    help="Directory containing mf_data.pkl and embedding_data.pkl",
)
parser.add_argument(
    "--jobs-dir",
    default=str(_demo_dir / "data" / _run_name / "sqd_jobs"),
    help="Directory for per-fragment job_id checkpoints and manifest",
)
parser.add_argument(
    "--force-resubmit",
    action="store_true",
    help="Resubmit all fragments even if job_id.txt already exists",
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
strategy    = sel_config.get("strategy", "adaptive")
orbital_threshold = sel_config.get("orbital_threshold", 15)

print(f"Config:            {config_path}")
print(f"Backend:           {qpu_config['backend_name']}")
print(f"Orbital threshold: {orbital_threshold}")

# ---------------------------------------------------------------------------
# Validate QRMI env
# ---------------------------------------------------------------------------
if not (os.environ.get("QRMI_JOB_QPU_RESOURCES") or
        os.environ.get("SLURM_JOB_QPU_RESOURCES")):
    raise EnvironmentError(
        "QRMI_JOB_QPU_RESOURCES not set. "
        "This script must run on an SCC node with a QPU resource allocated, "
        "or have QRMI credentials injected via run_workflow.sh."
    )

# ---------------------------------------------------------------------------
# Load pickled data
# ---------------------------------------------------------------------------
data_dir = Path(args.data_dir)
with open(data_dir / "mf_data.pkl", "rb") as f:
    mf_data = pickle.load(f)
with open(data_dir / "embedding_data.pkl", "rb") as f:
    emb_data = pickle.load(f)

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

fragment_meta = emb_data["fragment_meta"]
fragments = {
    fid: Fragment(
        fragment_id=fid,
        atom_indices=meta["atom_indices"],
        orbital_indices=meta["orbital_indices"],
        n_electrons=meta["n_electrons"],
        metadata={},
    )
    for fid, meta in fragment_meta.items()
}

sqd_fragments = {
    fid: frag for fid, frag in fragments.items()
    if strategy != "adaptive" or frag.n_orbitals >= orbital_threshold
}

print(f"\nTotal fragments:       {len(fragments)}")
print(f"SQD-eligible:          {len(sqd_fragments)}")
print(f"FCI (skipped here):    {len(fragments) - len(sqd_fragments)}")

# ---------------------------------------------------------------------------
# Initialize QRMI backend
# ---------------------------------------------------------------------------
print("\nInitializing QRMI backend...")
backend = QRMIBackend(qpu_config)
backend.initialize()
backend.get_backend()
props = backend.get_backend_properties()
print(f"Resource: {props['backend_name']} ({props['resource_type']})")

# ---------------------------------------------------------------------------
# Submit QPU jobs — one per SQD fragment
# ---------------------------------------------------------------------------
jobs_dir = Path(args.jobs_dir)
jobs_dir.mkdir(parents=True, exist_ok=True)

manifest_path = jobs_dir / "job_manifest.json"
manifest: dict = {}
if manifest_path.exists():
    with open(manifest_path) as f:
        manifest = json.load(f)

import h5py

dumpfile = emb_data["dumpfile"]
n_submitted = n_skipped = 0

print(f"\n{'='*60}")
print(f"Submitting QPU jobs for {len(sqd_fragments)} fragments...")
print(f"{'='*60}")

for frag_id, frag in sqd_fragments.items():
    frag_dir = jobs_dir / f"fragment_{frag_id}"
    frag_dir.mkdir(parents=True, exist_ok=True)
    job_id_file = frag_dir / "job_id.txt"

    if job_id_file.exists() and not args.force_resubmit:
        with open(job_id_file) as f:
            existing_job_id = f.read().strip()
        print(f"  Fragment {frag_id}: skipping (job {existing_job_id} already submitted)")
        manifest[str(frag_id)] = manifest.get(str(frag_id), {
            "job_id": existing_job_id, "status": "SUBMITTED", "n_orb": frag.n_orbitals
        })
        n_skipped += 1
        continue

    with h5py.File(dumpfile, "r") as hf:
        grp  = hf[f"fragment_{int(frag_id)}"]
        h1e  = grp["heff"][:]
        h2e  = grp["eris"][:]
        norb = int(grp.attrs["norb"])
        nocc = int(grp.attrs["nocc"])

    print(f"\n  Fragment {frag_id}: n_orb={norb}  n_occ={nocc}", flush=True)

    # Build solver — submit only (wait_for_completion=False)
    solver = SQDSolver(backend, config=sqd_config)
    solver._fragment_workflow_path = frag_dir

    nelec = (nocc, nocc)
    # Calling solve() with wait_for_completion=False submits and immediately
    # raises RuntimeError("Job ... is still QUEUED") — we catch that and
    # record the job_id from the checkpoint file instead.
    try:
        solver.solve(
            h1e, h2e, norb, nelec,
            mf=mf,
            workflow_path=str(frag_dir),
            wait_for_completion=False,
            force_resubmit=args.force_resubmit,
        )
    except RuntimeError as exc:
        msg = str(exc)
        if "still QUEUED" in msg or "still RUNNING" in msg or "still VALIDATING" in msg:
            pass  # expected — job submitted but not complete
        else:
            raise

    if job_id_file.exists():
        with open(job_id_file) as f:
            job_id = f.read().strip()
        print(f"  → job_id: {job_id}", flush=True)
        manifest[str(frag_id)] = {
            "job_id": job_id,
            "status": "SUBMITTED",
            "n_orb":  norb,
            "n_occ":  nocc,
            "frag_dir": str(frag_dir),
        }
        n_submitted += 1
    else:
        print(f"  WARNING: job_id.txt not created for fragment {frag_id}", flush=True)

    # Update manifest after every submission
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

print(f"\n{'='*60}")
print(f"Submitted: {n_submitted}  Skipped (already queued): {n_skipped}")
print(f"Manifest:  {manifest_path}")
print(f"{'='*60}")
print("\nExiting — HPC node released.  Monitor with 03_solve_sqd_monitor.sh")
