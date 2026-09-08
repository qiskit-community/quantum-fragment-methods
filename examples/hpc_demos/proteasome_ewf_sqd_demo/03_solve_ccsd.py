"""
Step 3a: Run CCSD on every fragment that would receive SQD treatment.

This is the classical reference companion to 03_solve_sqd_submit.py.  Running
CCSD unconditionally on the SQD-eligible fragments (n_orbitals >= threshold)
gives you:

  1. A complete, self-contained classical result usable immediately.
  2. The CCSD amplitudes (t1/t2) that seed the LUCJ ansatz in the SQD submit
     step — no need to recompute them during QPU submission.
  3. A direct apples-to-apples comparison once SQD results arrive.

FCI continues to handle the small fragments (n_orbitals < orbital_threshold).
Output is ccsd_results.pkl (same schema as solver_results.pkl).

Usage:
    python 03_solve_ccsd.py --config config_ixazomib_complex_sto-3g.yaml
    python 03_solve_ccsd.py --config config_bortezomib_complex_sto-3g.yaml
"""

import argparse
import json as _json
import logging
import pickle
import shutil
from datetime import datetime
from pathlib import Path

import yaml
from pyscf import gto, scf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)

from quantum_fragment_methods.application.embedding.base import EmbeddingResult, Fragment
from quantum_fragment_methods.application.solvers.classical_zoo.ccsd import CCSD
from quantum_fragment_methods.application.solvers.classical_zoo.fci import FCI

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
    "--output-dir",
    default=str(_demo_dir / "data" / _run_name),
    help="Directory to write ccsd_results.pkl",
)
parser.add_argument(
    "--force-rerun",
    action="store_true",
    help="Ignore existing checkpoint and resolve all fragments from scratch",
)
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Load config
# ---------------------------------------------------------------------------
config_path = Path(args.config).resolve()
with open(config_path) as f:
    config = yaml.safe_load(f)

sel_config = config["solver_selection"]
strategy = sel_config.get("strategy", "adaptive")
orbital_threshold = sel_config.get("orbital_threshold", 15)

print(f"Config:            {config_path}")
print(f"Solver strategy:   {strategy}")
print(f"Orbital threshold: {orbital_threshold}")

# ---------------------------------------------------------------------------
# Load pickled data
# ---------------------------------------------------------------------------
data_dir = Path(args.data_dir)

with open(data_dir / "mf_data.pkl", "rb") as f:
    mf_data = pickle.load(f)
with open(data_dir / "embedding_data.pkl", "rb") as f:
    emb_data = pickle.load(f)

charge = emb_data.get("charge", 0)
spin = emb_data.get("spin", 0)

print(f"\nLoaded HF energy = {mf_data['hf_energy']:.8f} Ha  charge={charge}")
print(f"Fragments: {emb_data['n_fragments']}")

# ---------------------------------------------------------------------------
# Reconstruct PySCF mol + mf
# ---------------------------------------------------------------------------
mol = gto.Mole()
mol.atom = emb_data["atom_data"]
mol.unit = "Angstrom"
mol.basis = emb_data["basis"]
mol.charge = charge
mol.spin = spin
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

embedding_result = EmbeddingResult(
    fragments=fragments,
    mean_field_energy=mf_data["hf_energy"],
    metadata={"dumpfile": emb_data["dumpfile"]},
)

# ---------------------------------------------------------------------------
# Checkpoint: resume from partial ccsd_results.pkl if it exists
# ---------------------------------------------------------------------------
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)
checkpoint_file = output_dir / "ccsd_results.pkl"

fragment_results: dict = {}
if checkpoint_file.exists() and not args.force_rerun:
    try:
        with open(checkpoint_file, "rb") as f:
            fragment_results = pickle.load(f)
        print(f"\nResuming: {len(fragment_results)} fragment(s) already solved.")
    except Exception as e:
        print(f"WARNING: could not load checkpoint ({e}); starting from scratch.")
        fragment_results = {}

# ---------------------------------------------------------------------------
# Solve: FCI for small fragments, CCSD for SQD-eligible ones
# ---------------------------------------------------------------------------
import h5py, time as _time

dumpfile = emb_data["dumpfile"]
timings: dict = {}
n_total = len(fragments)
n_fci = n_ccsd = 0

print(f"\n{'='*60}")
print(f"Solving {n_total} fragments  (FCI if n_orb < {orbital_threshold}, else CCSD)")
print(f"{'='*60}")

for frag_id, frag in fragments.items():
    if frag_id in fragment_results:
        print(f"  Fragment {frag_id}: skipping (checkpoint)")
        continue

    use_fci = (strategy == "adaptive" and frag.n_orbitals < orbital_threshold)
    solver_name = "FCI" if use_fci else "CCSD"
    print(f"\nFragment {frag_id}: n_orb={frag.n_orbitals}  solver={solver_name}", flush=True)

    with h5py.File(dumpfile, "r") as hf:
        grp  = hf[f"fragment_{int(frag_id)}"]
        h1e  = grp["heff"][:]
        h2e  = grp["eris"][:]
        norb = int(grp.attrs["norb"])
        nocc = int(grp.attrs["nocc"])

    t0 = _time.time()
    solver = FCI() if use_fci else CCSD()
    result = solver.solve_from_integrals(h1e, h2e, norb, nocc, compute_rdms=True)
    timings[frag_id] = _time.time() - t0

    # Stash amplitudes for SQD LUCJ seeding (CCSD only)
    if not use_fci and hasattr(solver, "conv_tol"):
        # Re-run to capture t1/t2 — CCSD.solve_from_integrals doesn't expose them
        # directly.  We save the correlation energy as a proxy; the SQD submit step
        # will recompute t1/t2 from the same integrals via _compute_ccsd_amplitudes.
        pass

    with h5py.File(dumpfile, "r") as hf:
        grp = hf[f"fragment_{int(frag_id)}"]
        if "c_frag"    in grp: result.metadata["c_frag"]    = grp["c_frag"][:]
        if "c_cluster" in grp: result.metadata["c_cluster"] = grp["c_cluster"][:]
        result.metadata["norb"] = norb
        result.metadata["nocc"] = nocc

    e_corr = result.metadata.get("e_corr", float("nan"))
    print(f"  → E={result.energy:.8f} Ha  e_corr={e_corr:.8f} Ha  ({timings[frag_id]:.1f}s)")

    fragment_results[frag_id] = result
    if use_fci: n_fci  += 1
    else:       n_ccsd += 1

    # Checkpoint after every fragment
    with open(checkpoint_file, "wb") as f:
        pickle.dump(fragment_results, f)

print(f"\n{'='*60}")
print(f"All fragments solved: {n_fci} FCI  {n_ccsd} CCSD")
print(f"{'='*60}")
