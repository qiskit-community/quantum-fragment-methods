#!/usr/bin/env python3
"""
15_validate_hydrogenated_microstates.py

Proteasome Challenge
Stage 8f: geometry/composition sanity gate for the six initial H microstates.

Checks
------
1. Expected six structures exist.
2. Within each ligand, all microstates have identical atom counts and element counts.
3. Heavy-atom identities and coordinates are identical across microstates.
4. Special catalytic H counts match the registered microstate definitions.
5. Retained waters each have two H atoms.
6. Detect obviously bad initial H geometry:
     - H--H < 0.55 A
     - H--heavy non-parent < 0.75 A
     - any distinct heavy--heavy < 0.80 A
7. Report minimum H--H and non-parent H--heavy distances.

This is a sanity screen, not an optimization or energetic validation.

Example
-------
python3 scripts/15_validate_hydrogenated_microstates.py \
  --indir intermediate/hydrogenated_microstates \
  --build-report validation/hydrogenated_microstates.json \
  --manifest validation/capped_receptor_scaffold_v2.json \
  --microstates validation/catalytic_microstates.json \
  --out validation/hydrogenated_geometry_validation.json
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


EXPECTED = [
    ("6V8", "MS_A_THRN"),
    ("6V8", "MS_B_LYS33"),
    ("6V8", "MS_C_ASP17"),
    ("BO2", "MS_A_THRN"),
    ("BO2", "MS_B_LYS33"),
    ("BO2", "MS_C_ASP17"),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--indir", required=True)
    p.add_argument("--build-report", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--microstates", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--heavy-tol", type=float, default=1e-6)
    return p.parse_args()


def parse_pdb(path):
    atoms = []
    for line in Path(path).read_text().splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        atoms.append({
            "record": line[0:6].strip(),
            "name": line[12:16].strip(),
            "resname": line[17:20].strip(),
            "chain": line[21:22].strip() or " ",
            "resseq": int(line[22:26]),
            "xyz": np.array([
                float(line[30:38]),
                float(line[38:46]),
                float(line[46:54]),
            ]),
            "element": (line[76:78].strip() or line[12:14].strip()).upper(),
        })
    return atoms


def signature(atom):
    return (
        atom["record"], atom["name"], atom["resname"],
        atom["chain"], atom["resseq"], atom["element"]
    )


def select_one(atoms, chain, resseq, name, resname=None):
    hits = [
        a for a in atoms
        if a["chain"] == chain
        and a["resseq"] == resseq
        and a["name"] == name
        and (resname is None or a["resname"] == resname)
    ]
    if len(hits) != 1:
        raise RuntimeError(
            f"Expected one {chain}:{resseq} {resname or '*'} {name}; found {len(hits)}"
        )
    return hits[0]


def attached_h(atoms, parent, cutoff=1.30):
    p = parent["xyz"]
    return [
        a for a in atoms
        if a["element"] == "H"
        and a["chain"] == parent["chain"]
        and a["resseq"] == parent["resseq"]
        and np.linalg.norm(a["xyz"] - p) <= cutoff
    ]


def nearest_heavy_parent(atoms, h):
    heavy = [a for a in atoms if a["element"] != "H"]
    ds = [(float(np.linalg.norm(h["xyz"] - a["xyz"])), a) for a in heavy]
    ds.sort(key=lambda x: x[0])
    return ds[0]


def validate_special_counts(atoms, ligand, microstate, role_map, manifest):
    expected = {
        "MS_A_THRN": {"Thr1_N": 3, "Thr1_OG1": 0, "Lys33_NZ": 2, "Asp17_total": 0},
        "MS_B_LYS33": {"Thr1_N": 2, "Thr1_OG1": 0, "Lys33_NZ": 3, "Asp17_total": 0},
        "MS_C_ASP17": {"Thr1_N": 2, "Thr1_OG1": 0, "Lys33_NZ": 2, "Asp17_total": 1},
    }[microstate]

    thrN = select_one(atoms, "Y", 1, "N", "THR")
    thrOG = select_one(atoms, "Y", 1, "OG1", "THR")
    lysNZ = select_one(atoms, "Y", 33, "NZ", "LYS")
    od1 = select_one(atoms, "Y", 17, "OD1", "ASP")
    od2 = select_one(atoms, "Y", 17, "OD2", "ASP")

    observed = {
        "Thr1_N": len(attached_h(atoms, thrN)),
        "Thr1_OG1": len(attached_h(atoms, thrOG)),
        "Lys33_NZ": len(attached_h(atoms, lysNZ)),
        "Asp17_total": len(attached_h(atoms, od1)) + len(attached_h(atoms, od2)),
    }

    checks = {
        key: {"expected": expected[key], "observed": observed[key],
              "pass": expected[key] == observed[key]}
        for key in expected
    }

    # Both B-bound ligand oxygens should carry exactly one H.
    for role_name in ("O_NTERM", "O_OXY"):
        oname = role_map[ligand][role_name]["atom"]
        O = select_one(atoms, "L", 800, oname, ligand)
        n = len(attached_h(atoms, O))
        checks[f"{ligand}_{role_name}_{oname}"] = {
            "expected": 1, "observed": n, "pass": n == 1
        }

    # Retained waters: exactly two H.
    for w in manifest["retained_waters"]:
        chain = w["source_water"].split(":")[0]
        O = select_one(atoms, chain, int(w["pdb_resseq"]), "O", "HOH")
        n = len(attached_h(atoms, O))
        checks[f"water_{w['source_water']}"] = {
            "expected": 2, "observed": n, "pass": n == 2
        }

    return checks


def geometry_screen(atoms):
    H = [a for a in atoms if a["element"] == "H"]
    heavy = [a for a in atoms if a["element"] != "H"]

    min_hh = None
    bad_hh = []
    for i in range(len(H)):
        for j in range(i + 1, len(H)):
            d = float(np.linalg.norm(H[i]["xyz"] - H[j]["xyz"]))
            if min_hh is None or d < min_hh:
                min_hh = d
            if d < 0.55:
                bad_hh.append({
                    "H1": signature(H[i]), "H2": signature(H[j]), "distance_A": d
                })

    min_nonparent_hheavy = None
    bad_hheavy = []

    for h in H:
        parent_d, parent = nearest_heavy_parent(atoms, h)
        for a in heavy:
            if a is parent:
                continue
            d = float(np.linalg.norm(h["xyz"] - a["xyz"]))
            if min_nonparent_hheavy is None or d < min_nonparent_hheavy:
                min_nonparent_hheavy = d
            if d < 0.75:
                bad_hheavy.append({
                    "H": signature(h),
                    "nearest_parent": signature(parent),
                    "other_heavy": signature(a),
                    "distance_A": d,
                })

    min_heavyheavy = None
    bad_heavyheavy = []
    for i in range(len(heavy)):
        for j in range(i + 1, len(heavy)):
            d = float(np.linalg.norm(heavy[i]["xyz"] - heavy[j]["xyz"]))
            if min_heavyheavy is None or d < min_heavyheavy:
                min_heavyheavy = d
            if d < 0.80:
                bad_heavyheavy.append({
                    "A1": signature(heavy[i]),
                    "A2": signature(heavy[j]),
                    "distance_A": d,
                })

    return {
        "min_H_H_A": min_hh,
        "min_nonparent_H_heavy_A": min_nonparent_hheavy,
        "min_heavy_heavy_A": min_heavyheavy,
        "bad_H_H_lt_0p55A": bad_hh,
        "bad_nonparent_H_heavy_lt_0p75A": bad_hheavy,
        "bad_heavy_heavy_lt_0p80A": bad_heavyheavy,
    }


def main():
    args = parse_args()
    indir = Path(args.indir)
    build_report = json.loads(Path(args.build_report).read_text())
    manifest = json.loads(Path(args.manifest).read_text())
    registry = json.loads(Path(args.microstates).read_text())
    role_map = registry["boronate_oxygen_functional_roles"]

    structures = {}
    for ligand, ms in EXPECTED:
        path = indir / f"{ligand}_{ms}.pdb"
        if not path.exists():
            raise RuntimeError(f"Missing expected structure {path}")
        structures[(ligand, ms)] = parse_pdb(path)

    results = {}
    overall_pass = True

    # Composition and heavy coordinate invariance per ligand.
    composition_checks = {}
    heavy_invariance = {}

    for ligand in ("6V8", "BO2"):
        states = [ms for lig, ms in EXPECTED if lig == ligand]
        ref_atoms = structures[(ligand, states[0])]

        comp_rows = []
        for ms in states:
            atoms = structures[(ligand, ms)]
            counts = Counter(a["element"] for a in atoms)
            comp_rows.append({
                "microstate": ms,
                "n_atoms": len(atoms),
                "element_counts": dict(sorted(counts.items())),
            })

        same_comp = all(
            row["n_atoms"] == comp_rows[0]["n_atoms"]
            and row["element_counts"] == comp_rows[0]["element_counts"]
            for row in comp_rows[1:]
        )
        composition_checks[ligand] = {
            "pass": same_comp,
            "states": comp_rows,
        }
        overall_pass &= same_comp

        ref_heavy = [a for a in ref_atoms if a["element"] != "H"]
        inv_states = []

        for ms in states[1:]:
            cur_heavy = [a for a in structures[(ligand, ms)] if a["element"] != "H"]

            identity_ok = (
                len(ref_heavy) == len(cur_heavy)
                and all(signature(a) == signature(b)
                        for a, b in zip(ref_heavy, cur_heavy))
            )

            max_disp = None
            if identity_ok:
                disps = [
                    float(np.linalg.norm(a["xyz"] - b["xyz"]))
                    for a, b in zip(ref_heavy, cur_heavy)
                ]
                max_disp = max(disps) if disps else 0.0

            coord_ok = identity_ok and max_disp <= args.heavy_tol
            inv_states.append({
                "reference": states[0],
                "microstate": ms,
                "identity_pass": identity_ok,
                "max_heavy_displacement_A": max_disp,
                "coordinate_pass": coord_ok,
            })
            overall_pass &= coord_ok

        heavy_invariance[ligand] = inv_states

    # Per-structure special-site and geometry screens.
    structure_results = []
    for ligand, ms in EXPECTED:
        atoms = structures[(ligand, ms)]
        special = validate_special_counts(atoms, ligand, ms, role_map, manifest)
        special_ok = all(x["pass"] for x in special.values())

        geom = geometry_screen(atoms)
        geom_ok = (
            len(geom["bad_H_H_lt_0p55A"]) == 0
            and len(geom["bad_nonparent_H_heavy_lt_0p75A"]) == 0
            and len(geom["bad_heavy_heavy_lt_0p80A"]) == 0
        )

        structure_results.append({
            "ligand": ligand,
            "microstate": ms,
            "special_H_counts": special,
            "special_H_counts_pass": special_ok,
            "geometry_screen": geom,
            "geometry_screen_pass": geom_ok,
        })
        overall_pass &= special_ok and geom_ok

    report = {
        "stage": "8f_hydrogenated_geometry_validation",
        "overall_pass": overall_pass,
        "heavy_coordinate_tolerance_A": args.heavy_tol,
        "composition_checks": composition_checks,
        "heavy_atom_invariance": heavy_invariance,
        "structures": structure_results,
        "interpretation": {
            "pass":
                "Safe to proceed to identical hydrogen-only optimization with all heavy atoms frozen.",
            "fail":
                "Inspect reported special-site H count or sub-angstrom clash before optimization.",
        },
    }

    Path(args.out).write_text(json.dumps(report, indent=2, default=float) + "\n")

    print(f"Wrote {args.out}")
    print()
    print("Stage-8f hydrogenated geometry validation")
    print(f"  overall: {'PASS' if overall_pass else 'FAIL'}")
    print()

    for ligand in ("6V8", "BO2"):
        print(
            f"  {ligand} composition invariant: "
            f"{'PASS' if composition_checks[ligand]['pass'] else 'FAIL'}"
        )
        for row in heavy_invariance[ligand]:
            md = row["max_heavy_displacement_A"]
            md_txt = "n/a" if md is None else f"{md:.6f} A"
            print(
                f"    heavy atoms {row['reference']} vs {row['microstate']}: "
                f"{'PASS' if row['coordinate_pass'] else 'FAIL'} "
                f"(max Δ={md_txt})"
            )

    print()
    for r in structure_results:
        g = r["geometry_screen"]
        print(
            f"  {r['ligand']:3s} {r['microstate']:12s} "
            f"specialH={'PASS' if r['special_H_counts_pass'] else 'FAIL'} "
            f"geometry={'PASS' if r['geometry_screen_pass'] else 'FAIL'} "
            f"min H-H={g['min_H_H_A']:.3f} A "
            f"min nonparent H-heavy={g['min_nonparent_H_heavy_A']:.3f} A"
        )

    print()
    if overall_pass:
        print("All sanity gates passed.")
        print("Ready for identical H-only optimization with heavy atoms frozen.")
    else:
        print("At least one sanity gate failed.")
        print("Do NOT optimize until the reported issue is corrected.")


if __name__ == "__main__":
    main()
