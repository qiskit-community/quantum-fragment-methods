"""
Step 4: Reconstruct total EWF+SQD energy from fragment results.

Loads mf_data.pkl, embedding_data.pkl, and solver_results.pkl produced by
steps 1–3.  Re-builds the Vayesta EWF object (needed for the partitioned
cumulant energy formula) and calls EWF.reconstruct_energy().

Saves a JSON summary with HF energy, total EWF+SQD energy, and per-fragment
correlation energies.

Outputs are namespaced under results/<run_name>/ derived from the config filename.

Usage:
    python 04_reconstruct.py --config config_alanine_sto-3g.yaml
"""

import argparse
import json
import pickle
from pathlib import Path

import yaml
from pyscf import gto, scf

from quantum_fragment_methods.application.embedding import EWF
from quantum_fragment_methods.workflow import QFWorkflow

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
    help=f"Directory containing pkl files from steps 1–3 (default: <demo_dir>/data/{_run_name}/)",
)
parser.add_argument(
    "--results-dir",
    default=str(_demo_dir / "results" / _run_name),
    help=f"Directory to save summary.json (default: <demo_dir>/results/{_run_name}/)",
)
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Load config (for embedder parameters)
# ---------------------------------------------------------------------------
config_path = Path(args.config).resolve()
with open(config_path) as f:
    config = yaml.safe_load(f)

print(f"Config: {config_path}")

# ---------------------------------------------------------------------------
# Load all pickled data
# ---------------------------------------------------------------------------
data_dir = Path(args.data_dir)

with open(data_dir / "mf_data.pkl", "rb") as f:
    mf_data = pickle.load(f)

with open(data_dir / "embedding_data.pkl", "rb") as f:
    emb_data = pickle.load(f)

with open(data_dir / "solver_results.pkl", "rb") as f:
    fragment_results = pickle.load(f)

hf_energy = mf_data["hf_energy"]
n_fragments = emb_data["n_fragments"]

print(f"\nLoaded data:")
print(f"  HF energy:   {hf_energy:.8f} Ha")
print(f"  Fragments:   {n_fragments}")
print(f"  Dumpfile:    {emb_data['dumpfile']}")
print(f"  Solved frags: {len(fragment_results)}")

# ---------------------------------------------------------------------------
# Reconstruct PySCF mol + mf objects
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
mf.e_tot = hf_energy
mf.converged = True

# ---------------------------------------------------------------------------
# Build EmbeddingResult directly from pkl — no Vayesta re-kernel needed.
# reconstruct_energy() only needs vayesta_ewf.mf for get_ovlp()/get_fock(),
# which we already have from the pkl.  Inject a lightweight shim.
# ---------------------------------------------------------------------------
from quantum_fragment_methods.application.embedding.base import EmbeddingResult, Fragment

fragment_meta = emb_data["fragment_meta"]
fragments = {}
for frag_id, meta in fragment_meta.items():
    fragments[frag_id] = Fragment(
        fragment_id=frag_id,
        atom_indices=meta["atom_indices"],
        orbital_indices=meta["orbital_indices"],
        n_electrons=meta["n_electrons"],
        metadata={},
    )

class _VayestaShim:
    """Minimal shim so EWF.reconstruct_energy() can call self.mf.get_ovlp/get_fock."""
    def __init__(self, mf):
        self.mf = mf

embedding_result = EmbeddingResult(
    fragments=fragments,
    mean_field_energy=hf_energy,
    metadata={
        "dumpfile": emb_data["dumpfile"],
        "vayesta_ewf": _VayestaShim(mf),
    },
)
print(f"Reconstructed {len(fragments)} fragments from pkl (no Vayesta re-kernel).")

# ---------------------------------------------------------------------------
# Reconstruct total energy
# ---------------------------------------------------------------------------
print("\nRunning partitioned cumulant energy reconstruction...")
ewf_embedder = EWF(bath_type=emb_data["bath_type"], truncation=emb_data["truncation"])
workflow = QFWorkflow(
    geometry=emb_data["xyz_path"],
    basis=emb_data["basis"],
    embedder=ewf_embedder,
)
workflow.mf = mf
workflow.mol = mol
workflow.embedding_result = embedding_result

total_energy = workflow.reconstruct_energy(fragment_results)
correlation_energy = total_energy - hf_energy

print(f"\n{'=' * 60}")
print(f"EWF+SQD Energy Reconstruction")
print(f"{'=' * 60}")
print(f"  HF energy:          {hf_energy:.8f} Ha")
print(f"  Correlation energy: {correlation_energy:.8f} Ha")
print(f"  Total EWF+SQD:      {total_energy:.8f} Ha")
print(f"{'=' * 60}")

# ---------------------------------------------------------------------------
# Per-fragment summary
# ---------------------------------------------------------------------------
print(f"\nPer-fragment correlation energies:")
for frag_id, result in fragment_results.items():
    if "e_corr" in result.metadata:
        e_corr = result.metadata["e_corr"]
    else:
        e_corr = result.energy  # fallback: solver returned total cluster energy
    print(f"  Fragment {frag_id}: {e_corr:.8f} Ha")

# ---------------------------------------------------------------------------
# Save JSON summary
# ---------------------------------------------------------------------------
results_dir = Path(args.results_dir)
results_dir.mkdir(parents=True, exist_ok=True)

per_fragment = {}
for frag_id, result in fragment_results.items():
    meta = emb_data["fragment_meta"].get(frag_id, {})
    per_fragment[str(frag_id)] = {
        "n_orbitals": meta.get("n_orbitals"),
        "n_electrons": meta.get("n_electrons"),
        "e_corr": float(result.metadata["e_corr"]) if "e_corr" in result.metadata else float(result.energy),
    }

summary = {
    "molecule": "alanine",
    "basis": emb_data["basis"],
    "bath_type": emb_data["bath_type"],
    "fragmentation": emb_data["fragmentation"],
    "n_fragments": n_fragments,
    "hf_energy": hf_energy,
    "correlation_energy": correlation_energy,
    "total_energy": total_energy,
    "fragments": per_fragment,
}

summary_file = results_dir / "summary.json"
with open(summary_file, "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nSummary saved to: {summary_file}")
