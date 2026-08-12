"""
Step 1: N2 molecule setup and mean-field calculation.
Based on sqd_N2_sto-3g_demo.ipynb cells 1-3.

Runs Hartree-Fock and extracts MO-basis Hamiltonian integrals.
Saves results to molecule_data.pkl for use by 02_run_sqd.py.
"""

import argparse
import pickle
from pathlib import Path

from pyscf import ao2mo, gto, scf

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--output-dir", default=".", help="Directory to save molecule_data.pkl")
args = parser.parse_args()

output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Define N2 molecule
# ---------------------------------------------------------------------------
mol = gto.Mole()
mol.atom = """
N 0.0 0.0 0.0
N 1.0 0.0 0.0
"""
mol.basis = "sto-3g"
mol.verbose = 3
mol.build()

print(f"Number of orbitals: {mol.nao}")
print(f"Number of electrons: {mol.nelectron}")

# ---------------------------------------------------------------------------
# Hartree-Fock
# ---------------------------------------------------------------------------
mf = scf.RHF(mol).to_gpu()
mf.kernel()
mf = mf.to_cpu()  # convert back to CPU for NumPy-compatible downstream ops
print(f"HF energy: {mf.e_tot:.8f}")

# ---------------------------------------------------------------------------
# Extract Hamiltonian integrals in MO basis
# ---------------------------------------------------------------------------
norb = mol.nao
nelec = (mol.nelec[0], mol.nelec[1])
mo_coeff = mf.mo_coeff
h1e = mo_coeff.T @ mf.get_hcore() @ mo_coeff
h2e = ao2mo.restore(1, ao2mo.kernel(mol, mo_coeff), norb)
nuc_energy = mf.energy_nuc()

print(f"Number of alpha electrons: {nelec[0]}")
print(f"Number of beta electrons:  {nelec[1]}")
print(f"One-body tensor shape:     {h1e.shape}")
print(f"Two-body tensor shape:     {h2e.shape}")
print(f"Nuclear repulsion energy:  {nuc_energy:.8f}")

# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------
data = {
    "norb": norb,
    "nelec": nelec,
    "h1e": h1e,
    "h2e": h2e,
    "nuc_energy": nuc_energy,
    "hf_energy": mf.e_tot,
    "mo_coeff": mo_coeff,
    "mo_occ": mf.mo_occ,
    "mo_energy": mf.mo_energy,
}

output_file = output_dir / "molecule_data.pkl"
with open(output_file, "wb") as f:
    pickle.dump(data, f)

print(f"\nSaved molecule data to: {output_file}")
