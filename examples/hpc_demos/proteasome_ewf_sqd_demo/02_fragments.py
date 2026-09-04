"""
Step 2: EWF fragmentation of a proteasome system.

Loads the mean-field result from 01_meanfield.py, runs Vayesta EWF embedding,
and saves fragment metadata for use by 03_solve.py and 04_reconstruct.py.

All embedder parameters (bath_type, truncation, fragmentation) are read from
the config YAML.  The charge/spin carried in mf_data.pkl is used when
reconstructing the PySCF mol object.

Usage:
    python 02_fragments.py --config config_ixazomib_complex_sto-3g.yaml
    python 02_fragments.py --config config_bortezomib_complex_sto-3g.yaml
    python 02_fragments.py --config config_ixazomib_ligand_sto-3g.yaml
    python 02_fragments.py --config config_bortezomib_ligand_sto-3g.yaml
"""

import argparse
import pickle
import sys
from pathlib import Path

import yaml
from pyscf import gto, scf

from quantum_fragment_methods.application.embedding import EWF

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
    help=f"Directory containing mf_data.pkl from step 1 (default: <demo_dir>/data/{_run_name}/)",
)
parser.add_argument(
    "--output-dir",
    default=str(_demo_dir / "data" / _run_name),
    help=f"Directory to save embedding_data.pkl (default: <demo_dir>/data/{_run_name}/)",
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
charge = mf_data.get("charge", 0)
spin = mf_data.get("spin", 0)

print(f"\nLoaded mean-field data from: {mf_file}")
print(f"  Basis:     {basis}")
print(f"  Charge:    {charge}")
print(f"  Spin:      {spin}")
print(f"  HF energy: {mf_data['hf_energy']:.8f} Ha")

# ---------------------------------------------------------------------------
# Reconstruct PySCF mol + mf objects
# ---------------------------------------------------------------------------
mol = gto.Mole()
mol.atom = atom_data
mol.unit = "Angstrom"
mol.basis = basis
mol.charge = charge
mol.spin = spin
mol.verbose = 4
mol.build()

mf = scf.RHF(mol).density_fit()
mf.mo_coeff = mf_data["mo_coeff"]
mf.mo_occ = mf_data["mo_occ"]
mf.mo_energy = mf_data["mo_energy"]
mf.e_tot = mf_data["hf_energy"]
mf.converged = True

print(f"\nReconstructed PySCF mol: {mol.natm} atoms, {mol.nao} AOs, charge={mol.charge}")

# ---------------------------------------------------------------------------
# Run EWF fragmentation
# ---------------------------------------------------------------------------
print(f"\nRunning EWF fragmentation ({fragmentation} scheme)...")
print(f"  Fragments to build: {mol.natm} (one per atom, IAO scheme)")
sys.stdout.flush()

output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)
dumpfile_path = str(output_dir / "ewf_dumpfile.h5")

ewf_embedder = EWF(bath_type=bath_type, truncation=truncation, dumpfile=dumpfile_path)
embedding_result = ewf_embedder.create_fragments(mf, fragmentation=fragmentation)

fragments = embedding_result.fragments
dumpfile = embedding_result.metadata.get("dumpfile")
n_fragments = len(fragments)

print(f"\nFragmentation complete: {n_fragments} fragments")
print(f"  Dumpfile: {dumpfile}")
print(f"\n{'Frag':>5}  {'n_orb':>6}  {'n_bath':>7}  {'n_elec':>7}  {'atoms'}")
print("-" * 50)
for frag_id, frag in fragments.items():
    atoms = frag.atom_indices
    n_bath = frag.metadata.get("bath_orbitals", "?")
    print(f"  {frag_id:>3}  {frag.n_orbitals:>6}  {str(n_bath):>7}  {frag.n_electrons:>7.3f}  {atoms}")

# ---------------------------------------------------------------------------
# Save embedding metadata
# ---------------------------------------------------------------------------
fragment_meta = {
    frag_id: {
        "n_orbitals": frag.n_orbitals,
        "n_bath_orbitals": frag.metadata.get("bath_orbitals"),
        "n_frag_orbitals": (frag.n_orbitals - frag.metadata["bath_orbitals"])
                           if isinstance(frag.metadata.get("bath_orbitals"), int) else None,
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
    "atom_data": atom_data,
    "basis": basis,
    "charge": charge,
    "spin": spin,
    "xyz_path": xyz_path,
}

output_file = output_dir / "embedding_data.pkl"
with open(output_file, "wb") as f:
    pickle.dump(embedding_data, f)

print(f"\nSaved embedding data to: {output_file}")
