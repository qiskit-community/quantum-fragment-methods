"""
Step 3: Solve each EWF fragment with the configured solver.

Reads solver_selection from config to assign:
  - FCI  for fragments with n_orbitals < orbital_threshold  (strategy: adaptive)
  - SQD  for all other fragments (or all, if strategy: sqd_all)

SQD is checkpoint-aware: re-running this script resumes from job_id.txt / counts.npy
without resubmitting to the QPU.  Pass --force-resubmit to clear checkpoints.

Circuits are saved to disk (QPY format) if sqd.circuit_save_dir is set in config.

Outputs are namespaced under data/<run_name>/ and results/<run_name>/ derived from
the config filename (e.g. config_alanine_sto-3g.yaml → alanine_sto-3g).

Usage:
    python 03_solve.py --config config_alanine_sto-3g.yaml --wait
    python 03_solve.py --config config_alanine_sto-3g.yaml --force-resubmit
"""

import argparse
import logging
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
from quantum_fragment_methods.application.solvers.classical_zoo.fci import FCI

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--config", required=True)
_known, _ = _pre.parse_known_args()
_run_name = Path(_known.config).stem.removeprefix("config_")
_demo_dir = Path(__file__).parent

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True, help="Path to config YAML")
parser.add_argument(
    "--data-dir",
    default=str(_demo_dir / "data" / _run_name),
    help=f"Directory containing mf_data.pkl and embedding_data.pkl (default: <demo_dir>/data/{_run_name}/)",
)
parser.add_argument(
    "--output-dir",
    default=str(_demo_dir / "data" / _run_name),
    help=f"Directory to save solver_results.pkl (default: <demo_dir>/data/{_run_name}/)",
)
parser.add_argument(
    "--results-dir",
    default=str(_demo_dir / "results" / _run_name),
    help=f"Base directory for SQD workflow checkpoints and circuit files (default: <demo_dir>/results/{_run_name}/)",
)
parser.add_argument(
    "--force-resubmit",
    action="store_true",
    help="Ignore existing QPU checkpoints and resubmit all SQD jobs",
)
parser.add_argument(
    "--wait",
    action="store_true",
    help="Poll until QPU jobs complete (may take hours in queue)",
)
parser.add_argument(
    "--max-wait-time",
    type=int,
    default=14400,
    help="Maximum seconds to wait for QPU completion (default: 14400 = 4 hours)",
)
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Load config
# ---------------------------------------------------------------------------
config_path = Path(args.config).resolve()
with open(config_path) as f:
    config = yaml.safe_load(f)

qpu_config = config["qpu"]
sqd_config = config["sqd"]
sel_config = config["solver_selection"]

strategy = sel_config.get("strategy", "adaptive")
orbital_threshold = sel_config.get("orbital_threshold", 15)

print(f"Config:            {config_path}")
print(f"Solver strategy:   {strategy}")
if strategy == "adaptive":
    print(f"Orbital threshold: {orbital_threshold}  (FCI if n_orb < {orbital_threshold}, else SQD)")
print(f"QPU backend:       {qpu_config['backend_name']}")

# ---------------------------------------------------------------------------
# Load pickled data from steps 1 & 2
# ---------------------------------------------------------------------------
data_dir = Path(args.data_dir)

with open(data_dir / "mf_data.pkl", "rb") as f:
    mf_data = pickle.load(f)

with open(data_dir / "embedding_data.pkl", "rb") as f:
    emb_data = pickle.load(f)

print(f"\nLoaded mean-field data:  HF energy = {mf_data['hf_energy']:.8f} Ha")
print(f"Loaded embedding data:   {emb_data['n_fragments']} fragments, dumpfile = {emb_data['dumpfile']}")

# ---------------------------------------------------------------------------
# Reconstruct PySCF mol + mf objects (needed by SQDSolver for CCSD amplitudes)
# ---------------------------------------------------------------------------
mol = gto.Mole()
mol.atom = emb_data["atom_data"]
mol.unit = "Angstrom"
mol.basis = emb_data["basis"]
mol.verbose = 0
mol.build()

mf = scf.RHF(mol).density_fit()
mf.mo_coeff = mf_data["mo_coeff"]
mf.mo_occ = mf_data["mo_occ"]
mf.mo_energy = mf_data["mo_energy"]
mf.e_tot = mf_data["hf_energy"]
mf.converged = True

# ---------------------------------------------------------------------------
# Reconstruct EmbeddingResult directly from saved pkl metadata.
# Step 2 already ran the full Vayesta EWF kernel and wrote the HDF5 dumpfile.
# There is no need to re-run EWF here — we just restore the Fragment objects
# from the serialised metadata so the solve loop below can look up n_orbitals.
# ---------------------------------------------------------------------------
fragment_meta = emb_data["fragment_meta"]

fragments = {}
for frag_id, meta in fragment_meta.items():
    frag = Fragment(
        fragment_id=frag_id,
        atom_indices=meta["atom_indices"],
        orbital_indices=meta["orbital_indices"],
        n_electrons=meta["n_electrons"],
        metadata={},
    )
    fragments[frag_id] = frag

embedding_result = EmbeddingResult(
    fragments=fragments,
    mean_field_energy=mf_data["hf_energy"],
    metadata={"dumpfile": emb_data["dumpfile"]},
)
print(f"Reconstructed {len(fragments)} fragments from pkl (no Vayesta re-kernel).")

# ---------------------------------------------------------------------------
# Determine whether any fragment needs SQD (requires QRMI credentials).
# QRMI is only initialized when:
#   1. At least one fragment exceeds orbital_threshold (would use SQD), AND
#   2. QRMI_JOB_QPU_RESOURCES env var is set (i.e. a QPU resource was allocated)
# In trial mode neither condition holds — large fragments fall back to CCSD.
# ---------------------------------------------------------------------------
import os

n_sqd_fragments = sum(
    1 for frag in embedding_result.fragments.values()
    if strategy != "adaptive" or frag.n_orbitals >= orbital_threshold
)

qrmi_available = bool(os.environ.get("QRMI_JOB_QPU_RESOURCES") or
                      os.environ.get("SLURM_JOB_QPU_RESOURCES"))

backend = None
if n_sqd_fragments > 0 and qrmi_available:
    # Defer QPU imports to here — qiskit_ibm_runtime does network/SSL work on
    # import that hangs the job in FCI-only (trial) mode.
    from quantum_fragment_methods.application.solvers.quantum_zoo.sqd import SQDSolver
    from quantum_fragment_methods.qpu import QRMIBackend
    print(f"\nInitializing QRMI backend ({n_sqd_fragments} SQD fragments)...")
    backend = QRMIBackend(qpu_config)
    backend.initialize()
    backend.get_backend()
    props = backend.get_backend_properties()
    print(f"QRMI resource: {props['backend_name']} ({props['resource_type']})")
elif n_sqd_fragments > 0:
    print(f"\n{n_sqd_fragments} fragment(s) exceed orbital_threshold but no QPU resource allocated.")
    print(f"  → Large fragments will use CCSD fallback (trial mode).")
else:
    print(f"\nAll {len(embedding_result.fragments)} fragments use FCI — skipping QRMI init.")

# ---------------------------------------------------------------------------
# Build QFWorkflow with solver rules, then solve fragments
# ---------------------------------------------------------------------------
from quantum_fragment_methods.workflow import QFWorkflow
from quantum_fragment_methods.application.embedding import EWF

_embedder_stub = EWF(bath_type=emb_data["bath_type"], truncation=emb_data["truncation"])
workflow = QFWorkflow(
    geometry=emb_data["xyz_path"],
    basis=emb_data["basis"],
    embedder=_embedder_stub,
)
# Inject the already-computed embedding result so workflow skips mean-field + fragmentation
workflow.mf = mf
workflow.mol = mol
workflow.embedding_result = embedding_result

results_base = Path(args.results_dir)
results_base.mkdir(parents=True, exist_ok=True)

# Solver rule: FCI for small fragments (adaptive strategy only)
if strategy == "adaptive":
    workflow.add_solver_rule(
        solver_factory=lambda frag: FCI(),
        condition=lambda frag: frag.n_orbitals < orbital_threshold,
        priority=10,
    )

# Solver rule: CCSD fallback for large fragments when no QPU backend is
# available (trial mode).  Sits between FCI (priority 10) and SQD (priority 0)
# so it only catches fragments that exceed orbital_threshold but have no backend.
if backend is None and strategy == "adaptive":
    from quantum_fragment_methods.application.solvers.classical_zoo.ccsd import CCSD
    workflow.add_solver_rule(
        solver_factory=lambda frag: CCSD(),
        condition=lambda frag: frag.n_orbitals >= orbital_threshold,
        priority=5,
    )

# Solver rule: SQD for remaining fragments (or all, if sqd_all).
# Only registered when a backend was initialized.
if backend is not None:
    def _make_sqd_solver(frag):
        """Create an SQDSolver with a fragment-specific workflow path."""
        frag_results_dir = results_base / f"fragment_{frag.fragment_id}"
        frag_results_dir.mkdir(parents=True, exist_ok=True)
        solver = SQDSolver(backend, config=sqd_config)
        solver._fragment_workflow_path = frag_results_dir
        return solver

    workflow.add_solver_rule(
        solver_factory=_make_sqd_solver,
        condition=None,
        priority=0,
    )

# ---------------------------------------------------------------------------
# Solve all fragments
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Solving fragments...")
print("=" * 60)

# Print solver assignment before running
for frag_id, frag in embedding_result.fragments.items():
    if strategy == "adaptive" and frag.n_orbitals < orbital_threshold:
        solver_name = "FCI"
    elif backend is None:
        solver_name = "CCSD"
    else:
        solver_name = "SQD"
    print(f"  Fragment {frag_id}: n_orb={frag.n_orbitals}, n_elec={frag.n_electrons} → {solver_name}")

# Patch solve_fragments to pass per-fragment workflow_path into SQDSolver.solve_from_integrals.
# We monkey-patch the SQD solver instances that were assigned by the rule above so
# that their workflow_path / force_resubmit / wait flags are set correctly.
_orig_solve_fragments = workflow.solve_fragments

def _patched_solve_fragments():
    """Wrap solve_fragments to inject SQD checkpoint args into SQDSolver instances."""
    solvers = workflow._assign_solvers()
    fragment_results = {}

    import h5py
    import numpy as np

    dumpfile = embedding_result.metadata.get("dumpfile")

    timings = {}

    for frag_id, frag in embedding_result.fragments.items():
        solver = solvers[frag_id]
        print(f"\nSolving fragment {frag_id} with {solver.name}...")

        with h5py.File(dumpfile, "r") as hf:
            frag_key = f"fragment_{int(frag_id)}"
            frag_group = hf[frag_key]
            h1e = frag_group["heff"][:]
            h2e = frag_group["eris"][:]
            norb = int(frag_group.attrs["norb"])
            nocc = int(frag_group.attrs["nocc"])

        import time as _time
        _t0 = _time.time()

        if solver.name == "SQD":
            frag_path = getattr(solver, "_fragment_workflow_path", results_base / f"fragment_{frag_id}")
            result = solver.solve_from_integrals(
                h1e, h2e, norb, nocc,
                compute_rdms=True,
                workflow_path=str(frag_path),
                force_resubmit=args.force_resubmit,
                wait_for_completion=args.wait,
                max_wait_time=args.max_wait_time,
            )
        else:
            result = solver.solve_from_integrals(h1e, h2e, norb, nocc, compute_rdms=True)

        timings[frag_id] = _time.time() - _t0

        # Attach c_frag / c_cluster for energy reconstruction
        with h5py.File(dumpfile, "r") as hf:
            frag_key = f"fragment_{int(frag_id)}"
            frag_group = hf[frag_key]
            if "c_frag" in frag_group:
                result.metadata["c_frag"] = frag_group["c_frag"][:]
            if "c_cluster" in frag_group:
                result.metadata["c_cluster"] = frag_group["c_cluster"][:]
            result.metadata["norb"] = norb
            result.metadata["nocc"] = nocc

        if "e_corr" in result.metadata:
            print(f"  Fragment {frag_id} total energy: {result.energy:.8f} Ha  "
                  f"(e_corr={result.metadata['e_corr']:.8f} Ha, {timings[frag_id]:.1f}s)")
        else:
            print(f"  Fragment {frag_id} energy: {result.energy:.8f} Ha  ({timings[frag_id]:.1f}s)")

        fragment_results[frag_id] = result

    return fragment_results, timings

fragment_results, fragment_timings = _patched_solve_fragments()

# ---------------------------------------------------------------------------
# Save solver results
# ---------------------------------------------------------------------------
import numpy as np

output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / "solver_results.pkl"
with open(output_file, "wb") as f:
    pickle.dump(fragment_results, f)

# Save per-fragment RDMs, NOONs, and timing as flat files for easy
# post-processing without unpickling the full result dict.
rdm_dir = output_dir / "rdms"
rdm_dir.mkdir(parents=True, exist_ok=True)

import json as _json

timing_data = {}
for frag_id, result in fragment_results.items():
    # RDMs
    if result.rdm1 is not None:
        np.save(rdm_dir / f"fragment_{frag_id}_rdm1.npy", result.rdm1)
    if result.rdm2 is not None:
        np.save(rdm_dir / f"fragment_{frag_id}_rdm2.npy", result.rdm2)
    # NOONs (natural orbital occupation numbers) — eigenvalues of rdm1
    # Values near 0.5 indicate strong correlation (good SQD candidates)
    if result.rdm1 is not None:
        noons = np.sort(np.linalg.eigvalsh(result.rdm1))[::-1]
        np.save(rdm_dir / f"fragment_{frag_id}_noons.npy", noons)
    # Timing
    timing_data[str(frag_id)] = {
        "solver": "SQD" if (backend is not None and frag_id in fragment_timings) else
                  ("FCI" if embedding_result.fragments[frag_id].n_orbitals < orbital_threshold else "CCSD"),
        "wall_time_s": round(fragment_timings.get(frag_id, 0.0), 3),
        "n_orbitals": embedding_result.fragments[frag_id].n_orbitals,
    }

with open(output_dir / "timing.json", "w") as _f:
    _json.dump(timing_data, _f, indent=2)

print(f"Saved RDMs + NOONs to: {rdm_dir}/")
print(f"Saved timing to:       {output_dir}/timing.json")

print(f"\n{'=' * 60}")
print(f"All fragments solved.")
print(f"Saved solver results to: {output_file}")
print(f"{'=' * 60}")
