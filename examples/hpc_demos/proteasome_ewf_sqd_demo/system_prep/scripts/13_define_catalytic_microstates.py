#!/usr/bin/env python3
"""
13_define_catalytic_microstates.py

Proteasome Challenge
Stage 8d: define a controlled, same-composition catalytic protonation screen.

This script does NOT add hydrogens. It creates a machine-readable registry of
the catalytic microstates that will be built next.

Assumption for the FIRST screen
-------------------------------
The crystallographic tetrahedral Thr1-O-B adduct is preserved:
    B26 bonded to Thr1 OG1, two ligand O atoms, and C21.

Both ligand B-bound oxygens are provisionally treated as B-OH groups.
The proton removed from Thr1-OH during formation of the Thr1-O-B bond is then
placed at one of three catalytic acceptor sites:

  MS_A_THRN:
      Thr1 N-terminus = NH3+
      Lys33 = neutral NH2
      Asp17 = COO-

  MS_B_LYS33:
      Thr1 N-terminus = neutral NH2
      Lys33 = NH3+
      Asp17 = COO-

  MS_C_ASP17:
      Thr1 N-terminus = neutral NH2
      Lys33 = neutral NH2
      Asp17 = COOH

The tetrahedral boronate adduct is assigned -1 formal charge in all three
states, consistent with four-coordinate B bearing C, Thr-O, OH, OH ligands.

With the current v2 receptor:
  fixed background excluding Thr1/Lys33/Asp17/ligand = 0

Therefore all three first-screen complexes have the SAME total charge:
  -1

This is a model registry, not an energetic claim.

A more chemically rearranged state in which Thr1 NH3+ transfers a proton to a
B-bound hydroxyl (creating a water-like B ligand) is recorded as DEFERRED and
should be tested only after the fixed-tetrahedral microstate screen.

Example:
python3 scripts/13_define_catalytic_microstates.py \
  --protonation validation/protonation_audit_v3.json \
  --boronate validation/boronate_coordination_audit.json \
  --out validation/catalytic_microstates.json \
  --tsv validation/catalytic_microstates.tsv
"""

import argparse
import csv
import json
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--protonation", required=True)
    p.add_argument("--boronate", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tsv", required=True)
    return p.parse_args()


def identify_oxygen_roles(boronate):
    """
    Map atom-name-specific B-bound oxygens onto functional roles:
      O_NTERM: closest to Thr1 N
      O_OXY:   closest to Gly47 N

    Require the two roles to identify distinct oxygens.
    """
    role_map = {}

    for inhibitor, data in boronate["inhibitors"].items():
        rows = data["boronate_oxygen_environment"]
        if len(rows) != 2:
            raise RuntimeError(
                f"{inhibitor}: expected exactly two B-bound ligand oxygens, "
                f"found {len(rows)}"
            )

        nterm = min(rows, key=lambda r: r["Thr1_N_A"])
        oxy = min(rows, key=lambda r: r["Gly47_N_A"])

        if nterm["oxygen_atom"] == oxy["oxygen_atom"]:
            raise RuntimeError(
                f"{inhibitor}: functional oxygen-role assignment is ambiguous"
            )

        role_map[inhibitor] = {
            "O_NTERM": {
                "atom": nterm["oxygen_atom"],
                "B26_O_A": nterm["B26_O_A"],
                "Thr1_N_A": nterm["Thr1_N_A"],
                "Gly47_N_A": nterm["Gly47_N_A"],
            },
            "O_OXY": {
                "atom": oxy["oxygen_atom"],
                "B26_O_A": oxy["B26_O_A"],
                "Thr1_N_A": oxy["Thr1_N_A"],
                "Gly47_N_A": oxy["Gly47_N_A"],
            },
        }

    return role_map


def main():
    args = parse_args()
    prot = json.loads(Path(args.protonation).read_text())
    bor = json.loads(Path(args.boronate).read_text())

    fixed_key = "ordinary_fixed_charge_sum_excluding_Thr1_Lys33_and_ligand"
    if fixed_key not in prot:
        raise RuntimeError(
            "Protonation audit does not contain the corrected fixed-charge field. "
            "Use protonation_audit_v3.json from 11c."
        )

    # protonation_audit_v3 excluded Thr1 and Lys33 but INCLUDED Asp17=-1.
    # Remove Asp17 from the fixed background so it can participate in MS_C.
    fixed_excluding_thr_lys_lig = int(prot[fixed_key])
    asp17_default_charge = None
    for r in prot["ionizable_groups"]:
        if r["residue"].startswith("Y:17 ASP"):
            asp17_default_charge = r["default_charge"]
            break

    if asp17_default_charge != -1:
        raise RuntimeError(
            f"Expected default Asp17 charge -1, found {asp17_default_charge}"
        )

    fixed_background = fixed_excluding_thr_lys_lig - asp17_default_charge

    role_map = identify_oxygen_roles(bor)

    microstates = [
        {
            "id": "MS_A_THRN",
            "description":
                "Transferred Thr1 hydroxyl proton resides on the Thr1 N-terminal amine.",
            "Thr1_N": {"form": "NH3+", "formal_charge": +1},
            "Lys33_NZ": {"form": "NH2", "formal_charge": 0},
            "Asp17": {"form": "COO-", "formal_charge": -1},
            "boronate": {
                "form": "tetrahedral_B_C_ThrO_OH_OH",
                "formal_charge": -1,
                "O_NTERM": "OH",
                "O_OXY": "OH",
            },
            "transferred_proton_location": "Thr1_N",
        },
        {
            "id": "MS_B_LYS33",
            "description":
                "Transferred Thr1 hydroxyl proton resides on Lys33 NZ.",
            "Thr1_N": {"form": "NH2", "formal_charge": 0},
            "Lys33_NZ": {"form": "NH3+", "formal_charge": +1},
            "Asp17": {"form": "COO-", "formal_charge": -1},
            "boronate": {
                "form": "tetrahedral_B_C_ThrO_OH_OH",
                "formal_charge": -1,
                "O_NTERM": "OH",
                "O_OXY": "OH",
            },
            "transferred_proton_location": "Lys33_NZ",
        },
        {
            "id": "MS_C_ASP17",
            "description":
                "Sensitivity state: transferred Thr1 hydroxyl proton resides on Asp17.",
            "Thr1_N": {"form": "NH2", "formal_charge": 0},
            "Lys33_NZ": {"form": "NH2", "formal_charge": 0},
            "Asp17": {
                "form": "COOH",
                "formal_charge": 0,
                "proton_oxygen": "choose_by_local_H_bond_geometry_during_build",
            },
            "boronate": {
                "form": "tetrahedral_B_C_ThrO_OH_OH",
                "formal_charge": -1,
                "O_NTERM": "OH",
                "O_OXY": "OH",
            },
            "transferred_proton_location": "Asp17",
        },
    ]

    for ms in microstates:
        catalytic_charge = (
            ms["Thr1_N"]["formal_charge"]
            + ms["Lys33_NZ"]["formal_charge"]
            + ms["Asp17"]["formal_charge"]
            + ms["boronate"]["formal_charge"]
        )
        ms["fixed_background_charge"] = fixed_background
        ms["catalytic_subsystem_charge"] = catalytic_charge
        ms["total_complex_charge"] = fixed_background + catalytic_charge

    charges = {ms["total_complex_charge"] for ms in microstates}
    if len(charges) != 1:
        raise RuntimeError(f"Microstates are not isocharged: {sorted(charges)}")

    report = {
        "stage": "8d_catalytic_microstate_registry",
        "purpose":
            "Define a small, same-composition, same-charge proton-location screen before hydrogen placement.",
        "fixed_background_charge_excluding_Thr1_Lys33_Asp17_ligand":
            fixed_background,
        "boronate_oxygen_functional_roles": role_map,
        "first_screen_microstates": microstates,
        "comparison_invariants": {
            "same_heavy_atom_coordinates": True,
            "same_number_of_atoms": True,
            "same_number_of_hydrogens": True,
            "same_total_charge": next(iter(charges)),
            "same_boron_connectivity": "B26--Thr1_OG1, B26--O_NTERM, B26--O_OXY, B26--C21",
            "both_boronate_oxygens_provisionally_protonated": True,
        },
        "deferred_sensitivity_state": {
            "id": "MS_D_BORONATE_REARRANGED",
            "status": "DEFERRED",
            "description":
                "Explore transfer of a proton from Thr1 NH3+ to a B-bound hydroxyl, producing a water-like B-bound oxygen and potentially altered B-O connectivity/coordination.",
            "reason_deferred":
                "This is a more substantial chemical rearrangement than simple proton relocation within the fixed crystallographic tetrahedral adduct and should not be mixed into the first controlled screen.",
        },
        "selection_plan": [
            "Build MS_A, MS_B, and MS_C for both 6V8 and BO2 using identical receptor heavy atoms.",
            "Add all ordinary hydrogens consistently.",
            "Place only the catalytic proton differently between microstates.",
            "Initially hold all heavy atoms fixed.",
            "Optimize hydrogen positions only with the same method for all states.",
            "Compare energies separately for 6V8 and BO2.",
            "Prefer one common receptor/catalytic microstate convention for both inhibitors unless evidence strongly supports ligand-specific protonation.",
            "Treat MS_D only as a later sensitivity calculation.",
        ],
    }

    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")

    rows = []
    for ms in microstates:
        rows.append({
            "microstate": ms["id"],
            "transferred_proton_location": ms["transferred_proton_location"],
            "Thr1_N": ms["Thr1_N"]["form"],
            "Lys33_NZ": ms["Lys33_NZ"]["form"],
            "Asp17": ms["Asp17"]["form"],
            "boronate": ms["boronate"]["form"],
            "fixed_background_charge": ms["fixed_background_charge"],
            "catalytic_subsystem_charge": ms["catalytic_subsystem_charge"],
            "total_complex_charge": ms["total_complex_charge"],
        })

    with open(args.tsv, "w", newline="") as f:
        fields = list(rows[0].keys())
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {args.out}")
    print(f"Wrote {args.tsv}")
    print()
    print("Stage-8d catalytic microstate registry")
    print(
        "  fixed background charge "
        "(excluding Thr1 + Lys33 + Asp17 + ligand): "
        f"{fixed_background:+d}"
    )
    print()

    print("Boronate oxygen functional roles")
    for inhibitor, roles in role_map.items():
        print(
            f"  {inhibitor}: "
            f"O_NTERM={roles['O_NTERM']['atom']} "
            f"(Thr1N {roles['O_NTERM']['Thr1_N_A']:.3f} A), "
            f"O_OXY={roles['O_OXY']['atom']} "
            f"(Gly47N {roles['O_OXY']['Gly47_N_A']:.3f} A)"
        )

    print()
    print("First-screen microstates")
    for ms in microstates:
        print(
            f"  {ms['id']:12s} proton->{ms['transferred_proton_location']:9s}  "
            f"Thr1={ms['Thr1_N']['form']:4s}  "
            f"Lys33={ms['Lys33_NZ']['form']:4s}  "
            f"Asp17={ms['Asp17']['form']:4s}  "
            f"q_total={ms['total_complex_charge']:+d}"
        )

    print()
    print("All first-screen microstates are same-composition and isocharged.")
    print("No hydrogen coordinates have been generated yet.")


if __name__ == "__main__":
    main()
