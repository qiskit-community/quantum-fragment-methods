"""
Step 5: Compute EWF+SQD binding energies for ixazomib and bortezomib.

Loads the four summary.json files produced by 04_reconstruct.py and assembles
the matched-receptor binding energy observable:

    ΔE_bind(inhibitor) = E(complex) - E(ligand)
                       = [E(complex) - E(receptor)] - E(ligand)

Because the common receptor is EXACTLY identical in both complexes (same
coordinates, same atom order, by construction), the receptor contribution
cancels exactly in the relative observable:

    ΔΔE_bind = ΔE_bind(ixazomib) - ΔE_bind(bortezomib)
             = E(ixa_complex) - E(ixa_ligand) - E(bo2_complex) + E(bo2_ligand)

No explicit receptor calculation is needed for ΔΔE_bind.

The common-receptor contribution does appear in the individual ΔE_bind values,
but the receptor charge (-1) must be treated consistently:
    E(receptor) is NOT computed here since it cancels.  Only ΔΔE_bind is
    the directly comparable observable.

Usage:
    python 05_binding_energy.py
    python 05_binding_energy.py --results-dir results
"""

import argparse
import json
from pathlib import Path

HARTREE_TO_KCAL = 627.509474

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
_demo_dir = Path(__file__).parent

parser = argparse.ArgumentParser()
parser.add_argument(
    "--results-dir",
    default=str(_demo_dir / "results"),
    help="Base results directory containing per-run subfolders (default: <demo_dir>/results/)",
)
parser.add_argument(
    "--output",
    default=str(_demo_dir / "results" / "binding_energies.json"),
    help="Path to write binding energy summary JSON",
)
args = parser.parse_args()

results_dir = Path(args.results_dir)

# ---------------------------------------------------------------------------
# Load per-system summaries
# ---------------------------------------------------------------------------
SYSTEMS = {
    "ixazomib_complex":   "ixazomib_complex_sto-3g",
    "ixazomib_ligand":    "ixazomib_ligand_sto-3g",
    "bortezomib_complex": "bortezomib_complex_sto-3g",
    "bortezomib_ligand":  "bortezomib_ligand_sto-3g",
}

energies = {}
for label, run_name in SYSTEMS.items():
    summary_file = results_dir / run_name / "summary.json"
    if not summary_file.exists():
        raise FileNotFoundError(
            f"Summary not found for {label}: {summary_file}\n"
            f"Run 04_reconstruct.py --config config_{run_name.replace('_sto-3g', '')}_sto-3g.yaml first."
        )
    with open(summary_file) as f:
        data = json.load(f)
    energies[label] = {
        "hf_energy":    data["hf_energy"],
        "total_energy": data["total_energy"],
        "n_fragments":  data["n_fragments"],
        "basis":        data["basis"],
        "charge":       data.get("charge", 0),
    }
    print(f"  {label:30s}  E = {data['total_energy']:16.8f} Ha  "
          f"({data['n_fragments']} frags, charge {data.get('charge', 0)})")

# ---------------------------------------------------------------------------
# Matched-receptor ΔΔE_bind
#
# ΔΔE = ΔE_bind(ixa) - ΔE_bind(bo2)
#     = [E(ixa_complex) - E(receptor) - E(ixa_lig)]
#       - [E(bo2_complex) - E(receptor) - E(bo2_lig)]
#     = E(ixa_complex) - E(ixa_lig) - E(bo2_complex) + E(bo2_lig)
#
# E(receptor) cancels exactly by the matched-receptor construction.
# ---------------------------------------------------------------------------
E_ixa_cpx = energies["ixazomib_complex"]["total_energy"]
E_ixa_lig = energies["ixazomib_ligand"]["total_energy"]
E_bo2_cpx = energies["bortezomib_complex"]["total_energy"]
E_bo2_lig = energies["bortezomib_ligand"]["total_energy"]

# Individual ΔE_bind (receptor contribution cancels in ΔΔ, not in individual)
dE_ixa_ha = E_ixa_cpx - E_ixa_lig   # missing -E(receptor); valid for ΔΔ
dE_bo2_ha = E_bo2_cpx - E_bo2_lig

ddE_ha    = dE_ixa_ha - dE_bo2_ha
ddE_kcal  = ddE_ha * HARTREE_TO_KCAL

print(f"\n{'=' * 66}")
print(f"EWF+SQD Binding Energetics  (matched-receptor cancellation)")
print(f"{'=' * 66}")
print(f"  ΔE_bind(ixazomib)   = {dE_ixa_ha:+.8f} Ha  ({dE_ixa_ha * HARTREE_TO_KCAL:+.3f} kcal/mol)")
print(f"  ΔE_bind(bortezomib) = {dE_bo2_ha:+.8f} Ha  ({dE_bo2_ha * HARTREE_TO_KCAL:+.3f} kcal/mol)")
print(f"  ΔΔE_bind (ixa - bo2)= {ddE_ha:+.8f} Ha  ({ddE_kcal:+.3f} kcal/mol)")
print(f"{'=' * 66}")
print(f"  Positive ΔΔE → bortezomib binds more strongly")
print(f"  Negative ΔΔE → ixazomib binds more strongly")
print(f"{'=' * 66}")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
output_path = Path(args.output)
output_path.parent.mkdir(parents=True, exist_ok=True)

result = {
    "method": "EWF+SQD",
    "basis": energies["ixazomib_complex"]["basis"],
    "note": (
        "Individual delta_E values include the common receptor contribution "
        "with implicit sign; receptor cancels exactly in delta_delta_E."
    ),
    "energies_Ha": {k: v["total_energy"] for k, v in energies.items()},
    "delta_E_ixazomib_Ha":            dE_ixa_ha,
    "delta_E_bortezomib_Ha":          dE_bo2_ha,
    "delta_delta_E_Ha":               ddE_ha,
    "delta_E_ixazomib_kcal_mol":      dE_ixa_ha * HARTREE_TO_KCAL,
    "delta_E_bortezomib_kcal_mol":    dE_bo2_ha * HARTREE_TO_KCAL,
    "delta_delta_E_kcal_mol":         ddE_kcal,
}

with open(output_path, "w") as f:
    json.dump(result, f, indent=2)

print(f"\nBinding energy summary saved to: {output_path}")
