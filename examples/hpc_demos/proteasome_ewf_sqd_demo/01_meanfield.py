"""
Step 1: Hartree-Fock mean-field calculation for a proteasome system.

Reads all parameters (basis set, XYZ file, charge, spin) from the config YAML.
Saves mf_data.pkl under data/<run_name>/.

The four proteasome systems each require their own config:
  config_ixazomib_complex_sto-3g.yaml    (533 atoms, charge -1)
  config_bortezomib_complex_sto-3g.yaml  (544 atoms, charge -1)
  config_ixazomib_ligand_sto-3g.yaml     ( 42 atoms, charge  0)
  config_bortezomib_ligand_sto-3g.yaml   ( 53 atoms, charge  0)

Usage:
    python 01_meanfield.py --config config_ixazomib_complex_sto-3g.yaml
    python 01_meanfield.py --config config_bortezomib_complex_sto-3g.yaml
    python 01_meanfield.py --config config_ixazomib_ligand_sto-3g.yaml
    python 01_meanfield.py --config config_bortezomib_ligand_sto-3g.yaml
"""

import argparse
import pickle
from pathlib import Path

import yaml

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
    "--xyz",
    default=None,
    help="Override XYZ file path (default: workflow.xyz_file from config)",
)
parser.add_argument(
    "--output-dir",
    default=str(_demo_dir / "data" / _run_name),
    help=f"Directory to save mf_data.pkl (default: <demo_dir>/data/{_run_name}/)",
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
charge = int(workflow_config.get("charge", 0))
spin = int(workflow_config.get("spin", 0))

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
print(f"Charge:   {charge}")
print(f"Spin:     {spin}")

# ---------------------------------------------------------------------------
# Run mean-field calculation
# ---------------------------------------------------------------------------
workflow = QFWorkflow(geometry=str(xyz_path), basis=basis, charge=charge, spin=spin)
mf = workflow.run_mean_field()

mol = workflow.mol
print(f"\nMean-field complete:")
print(f"  HF energy:       {mf.e_tot:.8f} Ha")
print(f"  Number of atoms: {mol.natm}")
print(f"  Number of AOs:   {mol.nao}")
print(f"  Basis set:       {basis}")
print(f"  Charge:          {mol.charge}")
print(f"  Spin:            {mol.spin}")

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
    "atom_data": mol.atom,
    "basis": basis,
    "charge": charge,
    "spin": spin,
    "xyz_path": str(xyz_path),
    "natm": mol.natm,
    "nao": mol.nao,
}

output_file = output_dir / "mf_data.pkl"
with open(output_file, "wb") as f:
    pickle.dump(data, f)

print(f"\nSaved mean-field data to: {output_file}")
