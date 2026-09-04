#!/usr/bin/env python3
"""
24_validate_production_models.py

Independent final audit of the four frozen challenge XYZ inputs.

Checks
------
1. Expected atom counts and elements.
2. Complex = common receptor prefix + ligand suffix.
3. Receptor atoms are EXACTLY identical between the two complexes.
4. Ligand coordinates/order are EXACTLY identical to the ligand suffix of each
   corresponding complex.
5. Thr1 OG1-B covalent distance remains chemically sensible.
6. No catastrophic receptor-ligand atom overlap other than intended Thr1-B.
7. Isolated ligand boron is three-coordinate (C,O,O).
8. Expected charge metadata is written into the validation report:
       complexes -1
       isolated ligands 0

This script intentionally does not infer electronic charges from XYZ.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--indir", type=Path, default=Path("production_models")
    )
    p.add_argument(
        "--report",
        type=Path,
        default=Path("validation/final_production_model_audit.json"),
    )
    p.add_argument("--coord-tol", type=float, default=1e-10)
    p.add_argument("--clash-threshold", type=float, default=0.70)
    return p.parse_args()


def read_xyz(path):
    lines = path.read_text().splitlines()
    n = int(lines[0])
    atoms = []
    for i, line in enumerate(lines[2:2+n]):
        s = line.split()
        atoms.append({
            "index": i,
            "element": s[0].upper(),
            "xyz": np.array([float(s[1]), float(s[2]), float(s[3])]),
        })
    if len(atoms) != n:
        raise RuntimeError(f"{path}: expected {n} atoms, found {len(atoms)}")
    return atoms


def coord_delta(a, b):
    return float(np.linalg.norm(a["xyz"] - b["xyz"]))


def exact_block_equal(a, b, tol):
    if len(a) != len(b):
        return False, float("inf"), "count"
    mx = 0.0
    for i, (x, y) in enumerate(zip(a, b)):
        if x["element"] != y["element"]:
            return False, mx, f"element at {i}"
        dd = coord_delta(x, y)
        mx = max(mx, dd)
        if dd > tol:
            return False, mx, f"coordinate at {i}"
    return True, mx, None


def b_neighbors(lig):
    bs = [a for a in lig if a["element"] == "B"]
    if len(bs) != 1:
        raise RuntimeError(f"Expected exactly one B, found {len(bs)}")
    b = bs[0]
    neigh = []
    for a in lig:
        if a is b or a["element"] == "H":
            continue
        dd = coord_delta(a, b)
        if dd <= 1.90:
            neigh.append((dd, a))
    neigh.sort(key=lambda x: x[0])
    return b, neigh


def min_cross(rec, lig, b):
    best = None
    # Without PDB metadata, identify the intended Thr-O-B bond as the closest
    # receptor oxygen to B in the 1.30-1.65 A range and exclude just that pair.
    intended = None
    candidates = []
    for i, r in enumerate(rec):
        if r["element"] != "O":
            continue
        dd = coord_delta(r, b)
        if 1.30 <= dd <= 1.65:
            candidates.append((dd, i))
    if candidates:
        candidates.sort()
        intended = candidates[0][1]

    for i, r in enumerate(rec):
        for l in lig:
            if l is b and i == intended:
                continue
            dd = coord_delta(r, l)
            if best is None or dd < best[0]:
                best = (dd, i, l["index"])
    return best, intended


def main():
    args = parse_args()
    args.report.parent.mkdir(parents=True, exist_ok=True)

    rec = read_xyz(args.indir / "common_receptor.xyz")
    ix_c = read_xyz(args.indir / "ixazomib_complex.xyz")
    ix_l = read_xyz(args.indir / "ixazomib_ligand.xyz")
    bo_c = read_xyz(args.indir / "bortezomib_complex.xyz")
    bo_l = read_xyz(args.indir / "bortezomib_ligand.xyz")

    nrec = len(rec)
    report = {
        "stage": "final_production_model_audit",
        "expected_charges": {
            "ixazomib_complex": -1,
            "bortezomib_complex": -1,
            "ixazomib_ligand": 0,
            "bortezomib_ligand": 0,
        },
        "counts": {
            "common_receptor": len(rec),
            "ixazomib_complex": len(ix_c),
            "ixazomib_ligand": len(ix_l),
            "bortezomib_complex": len(bo_c),
            "bortezomib_ligand": len(bo_l),
        },
        "checks": {},
        "per_ligand": {},
    }

    # Receptor prefix equality in both complexes.
    ok_ix_rec, mx_ix_rec, why_ix_rec = exact_block_equal(
        rec, ix_c[:nrec], args.coord_tol
    )
    ok_bo_rec, mx_bo_rec, why_bo_rec = exact_block_equal(
        rec, bo_c[:nrec], args.coord_tol
    )
    ok_rec_cross, mx_rec_cross, why_rec_cross = exact_block_equal(
        ix_c[:nrec], bo_c[:nrec], args.coord_tol
    )

    report["checks"]["ixazomib_receptor_equals_common"] = {
        "pass": ok_ix_rec, "max_delta_A": mx_ix_rec, "failure": why_ix_rec
    }
    report["checks"]["bortezomib_receptor_equals_common"] = {
        "pass": ok_bo_rec, "max_delta_A": mx_bo_rec, "failure": why_bo_rec
    }
    report["checks"]["complex_receptors_exactly_matched"] = {
        "pass": ok_rec_cross, "max_delta_A": mx_rec_cross,
        "failure": why_rec_cross
    }

    for drug, comp, lig in [
        ("ixazomib", ix_c, ix_l),
        ("bortezomib", bo_c, bo_l),
    ]:
        suffix = comp[nrec:]
        ok_lig, mx_lig, why_lig = exact_block_equal(
            suffix, lig, args.coord_tol
        )

        b, neigh = b_neighbors(lig)
        elems = sorted(a["element"] for _, a in neigh)
        b3 = len(neigh) == 3 and elems == ["C", "O", "O"]

        cross, intended_o_idx = min_cross(rec, lig, b)
        clash_ok = cross[0] >= args.clash_threshold

        thr_b = None
        if intended_o_idx is not None:
            thr_b = coord_delta(rec[intended_o_idx], b)

        report["per_ligand"][drug] = {
            "complex_equals_receptor_plus_ligand": ok_lig,
            "ligand_suffix_max_delta_A": mx_lig,
            "ligand_suffix_failure": why_lig,
            "isolated_ligand_B_three_coordinate_COO": b3,
            "B_heavy_neighbors": [
                {"element": a["element"], "distance_A": dd}
                for dd, a in neigh
            ],
            "intended_receptor_O_B_distance_A": thr_b,
            "minimum_other_receptor_ligand_distance_A": cross[0],
            "catastrophic_overlap_absent": clash_ok,
        }

    # Expected counts from the frozen chemistry.
    count_ok = (
        len(rec) == 491
        and len(ix_l) == 42
        and len(ix_c) == 533
        and len(bo_l) == 53
        and len(bo_c) == 544
    )
    report["checks"]["expected_atom_counts"] = {"pass": count_ok}

    booleans = [count_ok, ok_ix_rec, ok_bo_rec, ok_rec_cross]
    for d in report["per_ligand"].values():
        booleans += [
            d["complex_equals_receptor_plus_ligand"],
            d["isolated_ligand_B_three_coordinate_COO"],
            d["catastrophic_overlap_absent"],
        ]

    report["overall"] = "PASS" if all(booleans) else "FAIL"
    args.report.write_text(json.dumps(report, indent=2) + "\n")

    print("Final production-model audit")
    print("----------------------------")
    print(f"common receptor      {len(rec):4d} atoms")
    print(f"ixazomib complex     {len(ix_c):4d} atoms")
    print(f"ixazomib ligand      {len(ix_l):4d} atoms")
    print(f"bortezomib complex   {len(bo_c):4d} atoms")
    print(f"bortezomib ligand    {len(bo_l):4d} atoms")
    print()
    print(
        "matched receptor      "
        f"{'PASS' if ok_rec_cross else 'FAIL'} "
        f"max_delta={mx_rec_cross:.3e} A"
    )
    for drug, dct in report["per_ligand"].items():
        print(
            f"{drug:12s} suffix={'PASS' if dct['complex_equals_receptor_plus_ligand'] else 'FAIL'} "
            f"B3={'PASS' if dct['isolated_ligand_B_three_coordinate_COO'] else 'FAIL'} "
            f"ThrO-B={dct['intended_receptor_O_B_distance_A']} "
            f"min_other={dct['minimum_other_receptor_ligand_distance_A']:.3f} A"
        )
    print()
    print(f"Overall: {report['overall']}")
    print(f"Report : {args.report}")

    raise SystemExit(0 if report["overall"] == "PASS" else 2)


if __name__ == "__main__":
    main()
