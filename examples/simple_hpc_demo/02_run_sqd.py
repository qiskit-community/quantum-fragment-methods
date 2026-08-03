"""
Step 2: Run SQD solver using QRMI backend.
Based on sqd_N2_sto-3g_demo.ipynb cells 4-7.

Credentials are NOT required in the config file — they are read from
environment variables set by the Slurm SPANK plugin on SCC.

Usage:
    python 02_run_sqd.py --config config_N2_sto-3g.yaml --data-dir . --wait
"""

import argparse
import json
import pickle
from pathlib import Path

import yaml
from pyscf import fci, gto, scf

from quantum_fragment_methods.application.solvers.quantum_zoo.sqd import SQDSolver
from quantum_fragment_methods.qpu import QRMIBackend

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--config", default="config_N2_sto-3g.yaml", help="Path to config YAML")
parser.add_argument("--data-dir", default=".", help="Directory containing molecule_data.pkl")
parser.add_argument("--results-dir", default="results/sqd_results", help="SQD results directory")
parser.add_argument(
    "--force-resubmit",
    action="store_true",
    help="Ignore existing checkpoints and resubmit QPU job",
)
parser.add_argument(
    "--wait",
    action="store_true",
    help="Poll until QPU job completes (may take hours in queue)",
)
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Load molecule data from step 1
# ---------------------------------------------------------------------------
data_file = Path(args.data_dir) / "molecule_data.pkl"
with open(data_file, "rb") as f:
    data = pickle.load(f)

norb = data["norb"]
nelec = data["nelec"]
h1e = data["h1e"]
h2e = data["h2e"]
nuc_energy = data["nuc_energy"]

print(f"Loaded molecule data: {norb} orbitals, {nelec} electrons")

# Reconstruct mf object for CCSD amplitude computation in SQDSolver
mol = gto.Mole()
mol.atom = """
N 0.0 0.0 0.0
N 1.0 0.0 0.0
"""
mol.basis = "sto-3g"
mol.build()
mf = scf.RHF(mol).to_gpu()
mf.kernel()
mf = mf.to_cpu()  # SQDSolver's CCSD uses PySCF CPU objects

# ---------------------------------------------------------------------------
# Load config
# ---------------------------------------------------------------------------
with open(args.config) as f:
    config = yaml.safe_load(f)

sqd_config = config["sqd"]
qpu_config = config["qpu"]

print(f"Config loaded from: {args.config}")
print(f"QPU backend:    {qpu_config['backend_name']}")
print(f"SQD iterations: {sqd_config['iterations']}")

# ---------------------------------------------------------------------------
# Initialize QRMI backend
#
# Credentials are read from environment variables — no secrets in this file.
# On SCC, the Slurm SPANK plugin sets these automatically when a QPU resource
# is allocated. For local testing, export them manually before running:
#
#   export QRMI_JOB_QPU_RESOURCES=ibm_torino
#   export QRMI_JOB_QPU_TYPES=qiskit-runtime-service
#   export ibm_torino_QRMI_IBM_QRS_ENDPOINT=https://quantum.cloud.ibm.com/api/v1
#   export ibm_torino_QRMI_IBM_QRS_IAM_ENDPOINT=https://iam.cloud.ibm.com
#   export ibm_torino_QRMI_IBM_QRS_IAM_APIKEY=your_apikey
#   export ibm_torino_QRMI_IBM_QRS_SERVICE_CRN=your_crn
# ---------------------------------------------------------------------------
backend = QRMIBackend(qpu_config)
backend.initialize()
backend.get_backend()

props = backend.get_backend_properties()
print(f"\nQRMI resource: {props['backend_name']} ({props['resource_type']})")

# ---------------------------------------------------------------------------
# Run SQD solver
# ---------------------------------------------------------------------------
solver = SQDSolver(backend, config=sqd_config)

results_dir = Path(args.results_dir)
result = solver.solve(
    h1e=h1e,
    h2e=h2e,
    norb=norb,
    nelec=nelec,
    mf=mf,
    workflow_path=str(results_dir),
    force_resubmit=args.force_resubmit,
    wait_for_completion=args.wait,
)

print(f"\nSQD Results:")
print(f"  Electronic energy: {result.energy:.8f} Ha")
print(f"  Total energy:      {result.energy + nuc_energy:.8f} Ha")

# ---------------------------------------------------------------------------
# FCI comparison
# ---------------------------------------------------------------------------
fci_solver = fci.FCI(mf)
fci_solver.kernel()

sqd_total = result.energy + nuc_energy
sqd_error = abs(sqd_total - fci_solver.e_tot)

print(f"\nEnergy Comparison:")
print(f"  HF energy:  {mf.e_tot:.8f} Ha")
print(f"  FCI energy: {fci_solver.e_tot:.8f} Ha")
print(f"  SQD energy: {sqd_total:.8f} Ha")
print(f"  SQD vs FCI: {sqd_error:.8f} Ha ({sqd_error * 627.5:.4f} kcal/mol)")

# ---------------------------------------------------------------------------
# Save summary
# ---------------------------------------------------------------------------
summary = {
    "hf_energy": mf.e_tot,
    "fci_energy": fci_solver.e_tot,
    "sqd_energy": sqd_total,
    "sqd_error_ha": sqd_error,
    "sqd_error_kcal": sqd_error * 627.5,
}
summary_file = results_dir / "summary.json"
with open(summary_file, "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSummary saved to: {summary_file}")
