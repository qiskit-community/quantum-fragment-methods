"""
Step 1: Hartree-Fock mean-field calculation on the alanine molecule.

Reads all parameters (basis set, XYZ file) from the config YAML.
Saves mf_data.pkl for use by 02_fragments.py.

Usage:
    python 01_meanfield.py --config config_alanine_sto-3g.yaml
    python 01_meanfield.py --config config_alanine_sto-3g.yaml --xyz /path/to/other.xyz
"""

import argparse
import pickle
from pathlib import Path

import yaml

from quantum_fragment_methods.workflow import QFWorkflow

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True, help="Path to config YAML")
parser.add_argument(
    "--xyz",
    default=None,
    help="Override XYZ file path (default: workflow.xyz_file from config)",
)
parser.add_argument(
    "--output-dir", default="data", help="Directory to save mf_data.pkl (default: data/)"
)
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Load config
# ---------------------------------------------------------------------------
config_path = Path(args.config).resolve()
with open(config_path) as f:
    config = yaml.safe_load(f)

workflow_config = config["workflow"]
basis = workflow_config["basis"]

# Resolve XYZ path: CLI flag overrides config; config path is relative to config file dir
if args.xyz:
    xyz_path = Path(args.xyz).resolve()
else:
    xyz_path = (config_path.parent / workflow_config["xyz_file"]).resolve()

if not xyz_path.exists():
    raise FileNotFoundError(f"XYZ file not found: {xyz_path}")

print(f"Config:   {config_path}")
print(f"XYZ file: {xyz_path}")
print(f"Basis:    {basis}")

# ---------------------------------------------------------------------------
# Run mean-field calculation
# ---------------------------------------------------------------------------
workflow = QFWorkflow(geometry=str(xyz_path), basis=basis)
mf = workflow.run_mean_field()

mol = workflow.mol
print(f"\nMean-field complete:")
print(f"  HF energy:       {mf.e_tot:.8f} Ha")
print(f"  Number of atoms: {mol.natm}")
print(f"  Number of AOs:   {mol.nao}")
print(f"  Basis set:       {basis}")

# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

data = {
    "hf_energy": mf.e_tot,
    "mo_coeff": mf.mo_coeff,
    "mo_occ": mf.mo_occ,
    "mo_energy": mf.mo_energy,
    "atom_data": mol.atom,          # geometry string for mol reconstruction
    "basis": basis,
    "xyz_path": str(xyz_path),
    "natm": mol.natm,
    "nao": mol.nao,
}

output_file = output_dir / "mf_data.pkl"
with open(output_file, "wb") as f:
    pickle.dump(data, f)

print(f"\nSaved mean-field data to: {output_file}")
