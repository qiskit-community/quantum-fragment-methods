"""
Step 2: EWF fragmentation of the alanine molecule.

Loads the mean-field result from 01_meanfield.py, runs Vayesta EWF embedding,
and saves fragment metadata for use by 03_solve.py and 04_reconstruct.py.

All embedder parameters (bath_type, truncation, fragmentation) are read from
the config YAML. The dumpfile path (HDF5, written by Vayesta) is persisted in
embedding_data.pkl so downstream steps can access cluster Hamiltonians.

Usage:
    python 02_fragments.py --config config_alanine_sto-3g.yaml
"""

import argparse
import pickle
from pathlib import Path

import yaml
from pyscf import gto, scf

from quantum_fragment_methods.application.embedding import EWF

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True, help="Path to config YAML")
parser.add_argument(
    "--data-dir",
    default="data",
    help="Directory containing mf_data.pkl from step 1 (default: data/)",
)
parser.add_argument(
    "--output-dir",
    default="data",
    help="Directory to save embedding_data.pkl (default: data/)",
)
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Load config
# ---------------------------------------------------------------------------
config_path = Path(args.config).resolve()
with open(config_path) as f:
    config = yaml.safe_load(f)

ewf_config = config["embedder"]["ewf"]
bath_type = ewf_config["bath_type"]
truncation = ewf_config["truncation"]
fragmentation = ewf_config["fragmentation"]

print(f"Config:        {config_path}")
print(f"Bath type:     {bath_type}")
print(f"Truncation:    {truncation}")
print(f"Fragmentation: {fragmentation}")

# ---------------------------------------------------------------------------
# Load mean-field data from step 1
# ---------------------------------------------------------------------------
data_dir = Path(args.data_dir)
mf_file = data_dir / "mf_data.pkl"

with open(mf_file, "rb") as f:
    mf_data = pickle.load(f)

basis = mf_data["basis"]
xyz_path = mf_data["xyz_path"]
atom_data = mf_data["atom_data"]

print(f"\nLoaded mean-field data from: {mf_file}")
print(f"  Basis:     {basis}")
print(f"  HF energy: {mf_data['hf_energy']:.8f} Ha")

# ---------------------------------------------------------------------------
# Reconstruct PySCF mol + mf objects
# ---------------------------------------------------------------------------
# Re-build mol from the persisted geometry so Vayesta has a live mol object.
# Basis and atom string come from the pickle — no re-reading of config needed.
mol = gto.Mole()
mol.atom = atom_data
mol.unit = "Angstrom"
mol.basis = basis
mol.verbose = 0
mol.build()

mf = scf.RHF(mol).density_fit()
mf.mo_coeff = mf_data["mo_coeff"]
mf.mo_occ = mf_data["mo_occ"]
mf.mo_energy = mf_data["mo_energy"]
mf.e_tot = mf_data["hf_energy"]
mf.converged = True

print(f"\nReconstructed PySCF mol: {mol.natm} atoms, {mol.nao} AOs")

# ---------------------------------------------------------------------------
# Run EWF fragmentation
# ---------------------------------------------------------------------------
print(f"\nRunning EWF fragmentation ({fragmentation} scheme)...")

ewf_embedder = EWF(bath_type=bath_type, truncation=truncation)
embedding_result = ewf_embedder.create_fragments(mf, fragmentation=fragmentation)

fragments = embedding_result.fragments
dumpfile = embedding_result.metadata.get("dumpfile")
n_fragments = len(fragments)

print(f"\nFragmentation complete: {n_fragments} fragments")
print(f"  Dumpfile: {dumpfile}")
print(f"\n{'Frag':>5}  {'n_orb':>6}  {'n_elec':>7}  {'atoms'}")
print("-" * 40)
for frag_id, frag in fragments.items():
    atoms = frag.atom_indices
    print(f"  {frag_id:>3}  {frag.n_orbitals:>6}  {frag.n_electrons:>7}  {atoms}")

# ---------------------------------------------------------------------------
# Save embedding metadata
# ---------------------------------------------------------------------------
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

# Serialize only the lightweight metadata — not the live Vayesta objects.
# 04_reconstruct.py will re-run create_fragments on the existing dumpfile to
# rebuild the Vayesta EWF object needed by reconstruct_energy().
fragment_meta = {
    frag_id: {
        "n_orbitals": frag.n_orbitals,
        "n_electrons": frag.n_electrons,
        "atom_indices": frag.atom_indices,
        "orbital_indices": frag.orbital_indices,
    }
    for frag_id, frag in fragments.items()
}

embedding_data = {
    "dumpfile": dumpfile,
    "mean_field_energy": embedding_result.mean_field_energy,
    "fragment_meta": fragment_meta,
    "n_fragments": n_fragments,
    "bath_type": bath_type,
    "truncation": truncation,
    "fragmentation": fragmentation,
    # Persist mol/mf info so reconstruct can rebuild the Vayesta object
    "atom_data": atom_data,
    "basis": basis,
    "xyz_path": xyz_path,
}

output_file = output_dir / "embedding_data.pkl"
with open(output_file, "wb") as f:
    pickle.dump(embedding_data, f)

print(f"\nSaved embedding data to: {output_file}")
