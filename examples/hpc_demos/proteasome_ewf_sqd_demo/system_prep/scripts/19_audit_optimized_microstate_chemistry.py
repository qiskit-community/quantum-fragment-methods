#!/usr/bin/env python3
"""
19_audit_optimized_microstate_chemistry.py

Proteasome Challenge
Stage 8h: post-relaxation chemical audit for the six H-only optimized
BO2/6V8 catalytic microstates.

Inputs
------
- Optimized XYZ files from ASE + GFN-FF.
- Matching PDB files from the final pre-optimization repaired-v2 inputs.
  The PDB files are used only as atom/residue metadata; optimized coordinates
  are always read from the XYZ files.
- catalytic_microstates.json for functional boronate oxygen roles.

Checks
------
1. Atom order/composition preserved.
2. Every heavy atom is unchanged from the pre-optimization structure.
3. Heavy atoms are identical across A/B/C microstates for each inhibitor.
4. The receptor heavy-atom set/ordering/coordinates are identical between
   BO2 and 6V8.
5. No pathological H-H or non-parent H-heavy contacts.
6. MS_A/MS_B/MS_C protonation identities are preserved.
7. Thr1-OG1--B coordination remains intact and B is four-coordinate.
8. Both B-bound ligand oxygens remain O-H.
9. Arg19 guanidinium NE-H remains chemically sensible.
10. All three retained waters remain two-proton waters with sensible geometry.
11. Catalytic-network H-bond geometry is reported.

This is a validation script. It does not modify coordinates.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


LIGAND_NAMES = {"6V8", "BO2"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--optimized",
        type=Path,
        default=Path("intermediate/hydrogen_optimized_ase_gfnff"),
    )
    p.add_argument(
        "--metadata",
        type=Path,
        default=Path("intermediate/hydrogenated_microstates_repaired_v2"),
    )
    p.add_argument(
        "--microstates",
        type=Path,
        default=Path("validation/catalytic_microstates.json"),
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("validation/hydrogen_optimized_chemical_audit.json"),
    )
    p.add_argument(
        "--tsv",
        type=Path,
        default=Path("validation/hydrogen_optimized_chemical_audit.tsv"),
    )
    p.add_argument("--heavy-tol", type=float, default=1e-10)
    p.add_argument("--min-hh", type=float, default=1.20)
    p.add_argument("--min-nonparent-hheavy", type=float, default=1.20)
    p.add_argument("--xh-min", type=float, default=0.80)
    p.add_argument("--xh-max", type=float, default=1.25)
    return p.parse_args()


def read_xyz(path: Path):
    lines = path.read_text().splitlines()
    n = int(lines[0])
    rows = []
    for line in lines[2 : 2 + n]:
        s = line.split()
        rows.append(
            {
                "element": s[0].upper(),
                "xyz": np.array([float(s[1]), float(s[2]), float(s[3])]),
            }
        )
    if len(rows) != n:
        raise RuntimeError(f"{path}: expected {n} atoms, parsed {len(rows)}")
    return rows


def read_pdb_metadata(path: Path):
    rows = []
    for line in path.read_text().splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        elem = line[76:78].strip()
        if not elem:
            name = line[12:16].strip()
            elem = "".join(ch for ch in name if ch.isalpha())[:1]
        rows.append(
            {
                "record": line[0:6].strip(),
                "name": line[12:16].strip(),
                "resname": line[17:20].strip(),
                "chain": line[21:22].strip(),
                "resseq": line[22:26].strip(),
                "element": elem.upper(),
                "xyz_input": np.array(
                    [
                        float(line[30:38]),
                        float(line[38:46]),
                        float(line[46:54]),
                    ]
                ),
            }
        )
    return rows


def load_case(stem: str, optimized: Path, metadata: Path):
    xyz = read_xyz(optimized / f"{stem}.xyz")
    meta = read_pdb_metadata(metadata / f"{stem}.pdb")
    if len(xyz) != len(meta):
        raise RuntimeError(
            f"{stem}: XYZ/PDB atom count mismatch {len(xyz)} != {len(meta)}"
        )
    atoms = []
    for i, (x, m) in enumerate(zip(xyz, meta)):
        if x["element"] != m["element"]:
            raise RuntimeError(
                f"{stem}: atom-order mismatch at {i}: "
                f"{x['element']} vs {m['element']}"
            )
        a = dict(m)
        a["xyz"] = x["xyz"]
        a["index"] = i
        atoms.append(a)
    return atoms


def label(a):
    return (
        f"{a['chain']}:{a['resseq']} {a['resname']} "
        f"{a['name']} ({a['element']}) idx={a['index']}"
    )


def select(atoms, chain=None, resseq=None, resname=None, name=None, element=None):
    out = []
    for a in atoms:
        if chain is not None and a["chain"] != str(chain):
            continue
        if resseq is not None and a["resseq"] != str(resseq):
            continue
        if resname is not None and a["resname"] != resname:
            continue
        if name is not None and a["name"] != name:
            continue
        if element is not None and a["element"] != element.upper():
            continue
        out.append(a)
    return out


def one(atoms, **kw):
    hits = select(atoms, **kw)
    if len(hits) != 1:
        raise RuntimeError(f"Expected one atom for {kw}, found {len(hits)}")
    return hits[0]


def dist(a, b):
    return float(np.linalg.norm(a["xyz"] - b["xyz"]))


def angle(a, b, c):
    u = a["xyz"] - b["xyz"]
    v = c["xyz"] - b["xyz"]
    u /= np.linalg.norm(u)
    v /= np.linalg.norm(v)
    return float(np.degrees(np.arccos(np.clip(np.dot(u, v), -1.0, 1.0))))


def nearest_heavy(atoms, h):
    cand = [
        (dist(a, h), a)
        for a in atoms
        if a["element"] != "H"
    ]
    cand.sort(key=lambda x: x[0])
    return cand


def attached_h(atoms, parent, cutoff=1.25):
    hs = [
        (dist(parent, h), h)
        for h in atoms
        if h["element"] == "H"
        and h["chain"] == parent["chain"]
        and h["resseq"] == parent["resseq"]
        and h["resname"] == parent["resname"]
        and dist(parent, h) <= cutoff
    ]
    hs.sort(key=lambda x: x[0])
    return [h for _, h in hs]


def min_hh(atoms):
    hs = [a for a in atoms if a["element"] == "H"]
    best = (math.inf, None, None)
    for i in range(len(hs) - 1):
        p = hs[i]["xyz"]
        for j in range(i + 1, len(hs)):
            d = float(np.linalg.norm(p - hs[j]["xyz"]))
            if d < best[0]:
                best = (d, hs[i], hs[j])
    return best


def min_nonparent_hheavy(atoms):
    best = (math.inf, None, None)
    for h in atoms:
        if h["element"] != "H":
            continue
        nh = nearest_heavy(atoms, h)
        if len(nh) < 2:
            continue
        # Nearest heavy is defined as the covalent parent.
        for d, a in nh[1:]:
            if d < best[0]:
                best = (d, h, a)
            break
    return best


def heavy_displacement(atoms):
    ds = [
        float(np.linalg.norm(a["xyz"] - a["xyz_input"]))
        for a in atoms
        if a["element"] != "H"
    ]
    return max(ds) if ds else 0.0, float(np.sqrt(np.mean(np.square(ds)))) if ds else 0.0


def all_xh_parent_distances(atoms):
    rows = []
    for h in atoms:
        if h["element"] != "H":
            continue
        d, parent = nearest_heavy(atoms, h)[0]
        rows.append((d, h, parent))
    return rows


def check_same_heavy(cases: Dict[str, List[Dict]], stems: List[str], tol: float):
    ref = cases[stems[0]]
    refh = [a for a in ref if a["element"] != "H"]
    out = {"reference": stems[0], "comparisons": {}, "status": "PASS"}

    for stem in stems[1:]:
        curh = [a for a in cases[stem] if a["element"] != "H"]
        if len(curh) != len(refh):
            out["comparisons"][stem] = {
                "status": "FAIL",
                "reason": "heavy atom count mismatch",
            }
            out["status"] = "FAIL"
            continue

        labels_match = all(
            (a["chain"], a["resseq"], a["resname"], a["name"], a["element"])
            ==
            (b["chain"], b["resseq"], b["resname"], b["name"], b["element"])
            for a, b in zip(refh, curh)
        )
        ds = [dist(a, b) for a, b in zip(refh, curh)] if labels_match else [math.inf]
        md = max(ds)
        st = "PASS" if labels_match and md <= tol else "FAIL"
        if st == "FAIL":
            out["status"] = "FAIL"
        out["comparisons"][stem] = {
            "labels_match": labels_match,
            "max_heavy_delta_A": md,
            "status": st,
        }
    return out


def receptor_heavy(atoms):
    return [
        a for a in atoms
        if a["element"] != "H" and a["resname"] not in LIGAND_NAMES
    ]


def check_matched_receptor(a, b, tol):
    ra, rb = receptor_heavy(a), receptor_heavy(b)
    labels_match = len(ra) == len(rb) and all(
        (x["chain"], x["resseq"], x["resname"], x["name"], x["element"])
        ==
        (y["chain"], y["resseq"], y["resname"], y["name"], y["element"])
        for x, y in zip(ra, rb)
    )
    md = math.inf
    if labels_match:
        md = max(dist(x, y) for x, y in zip(ra, rb))
    return {
        "n_receptor_heavy_A": len(ra),
        "n_receptor_heavy_B": len(rb),
        "labels_match": labels_match,
        "max_coordinate_delta_A": md,
        "status": "PASS" if labels_match and md <= tol else "FAIL",
    }


def microstate_expected_counts(stem):
    if "_MS_A_THRN" in stem:
        return {"Thr1_N": 3, "Lys33_NZ": 2, "Asp17_OD1": 0, "Asp17_OD2": 0}
    if "_MS_B_LYS33" in stem:
        return {"Thr1_N": 2, "Lys33_NZ": 3, "Asp17_OD1": 0, "Asp17_OD2": 0}
    if "_MS_C_ASP17" in stem:
        return {"Thr1_N": 2, "Lys33_NZ": 2, "Asp17_OD1": 0, "Asp17_OD2": 1}
    raise RuntimeError(f"Cannot identify microstate from {stem}")


def audit_case(stem, atoms, roles, args):
    failures = []
    warnings = []

    heavy_max, heavy_rms = heavy_displacement(atoms)
    if heavy_max > args.heavy_tol:
        failures.append(f"heavy atoms moved by {heavy_max:.3e} A")

    hh_d, hh1, hh2 = min_hh(atoms)
    if hh_d < args.min_hh:
        failures.append(f"H-H clash {hh_d:.3f} A: {label(hh1)} / {label(hh2)}")

    nh_d, nh_h, nh_heavy = min_nonparent_hheavy(atoms)
    if nh_d < args.min_nonparent_hheavy:
        failures.append(
            f"non-parent H-heavy clash {nh_d:.3f} A: "
            f"{label(nh_h)} / {label(nh_heavy)}"
        )

    xh = all_xh_parent_distances(atoms)
    bad_xh = [
        (d, h, p) for d, h, p in xh
        if not (args.xh_min <= d <= args.xh_max)
    ]
    if bad_xh:
        d, h, p = sorted(bad_xh, key=lambda x: abs(x[0] - 1.0), reverse=True)[0]
        failures.append(
            f"{len(bad_xh)} X-H parent distances outside "
            f"[{args.xh_min},{args.xh_max}] A; worst {d:.3f} A "
            f"{label(h)} -> {label(p)}"
        )

    # Core atoms.
    thrN = one(atoms, chain="Y", resseq=1, resname="THR", name="N")
    thrOG = one(atoms, chain="Y", resseq=1, resname="THR", name="OG1")
    lysNZ = one(atoms, chain="Y", resseq=33, resname="LYS", name="NZ")
    aspOD1 = one(atoms, chain="Y", resseq=17, resname="ASP", name="OD1")
    aspOD2 = one(atoms, chain="Y", resseq=17, resname="ASP", name="OD2")
    argNE = one(atoms, chain="Y", resseq=19, resname="ARG", name="NE")
    argCD = one(atoms, chain="Y", resseq=19, resname="ARG", name="CD")
    argCZ = one(atoms, chain="Y", resseq=19, resname="ARG", name="CZ")

    ligand = "6V8" if stem.startswith("6V8") else "BO2"
    B = one(atoms, chain="L", resseq=800, resname=ligand, name="B26")

    # Microstate H-count identity.
    exp = microstate_expected_counts(stem)
    obs = {
        "Thr1_N": len(attached_h(atoms, thrN)),
        "Lys33_NZ": len(attached_h(atoms, lysNZ)),
        "Asp17_OD1": len(attached_h(atoms, aspOD1)),
        "Asp17_OD2": len(attached_h(atoms, aspOD2)),
    }
    if obs != exp:
        failures.append(f"catalytic protonation mismatch expected={exp} observed={obs}")

    # Thr1 OG1 should have no H.
    thrOG_h = len(attached_h(atoms, thrOG))
    if thrOG_h != 0:
        failures.append(f"Thr1 OG1 unexpectedly has {thrOG_h} H")

    # B coordination.
    heavy_neighbors = sorted(
        (
            dist(B, a),
            a,
        )
        for a in atoms
        if a["element"] != "H" and a is not B
    )
    bonded = [(d, a) for d, a in heavy_neighbors if d <= 1.90]
    expected_labels = {
        ("Y", "1", "THR", "OG1"),
    }
    b_neighbor_labels = {
        (a["chain"], a["resseq"], a["resname"], a["name"])
        for d, a in bonded
    }
    if len(bonded) != 4:
        failures.append(
            f"B26 coordination expected 4 heavy neighbors <=1.90 A, found {len(bonded)}"
        )
    if not expected_labels.issubset(b_neighbor_labels):
        failures.append("B26 no longer bonded to Thr1 OG1 by distance criterion")

    # Both functional B-bound oxygens should be OH.
    boronate_oh = {}
    for role_name in ("O_NTERM", "O_OXY"):
        oname = roles[ligand][role_name]["atom"]
        O = one(atoms, chain="L", resseq=800, resname=ligand, name=oname)
        hs = attached_h(atoms, O)
        boronate_oh[role_name] = {
            "atom": oname,
            "B_O_A": dist(B, O),
            "n_H": len(hs),
            "O_H_A": dist(O, hs[0]) if hs else None,
        }
        if len(hs) != 1:
            failures.append(
                f"{ligand} {role_name} {oname} expected one H, found {len(hs)}"
            )

    # Arg19 NE-H.
    arg_hs = attached_h(atoms, argNE)
    arg_info = {"n_NE_H": len(arg_hs)}
    if len(arg_hs) != 1:
        failures.append(f"Arg19 NE expected 1 H, found {len(arg_hs)}")
    else:
        h = arg_hs[0]
        arg_info.update(
            {
                "NE_H_A": dist(argNE, h),
                "H_CD_A": dist(h, argCD),
                "H_CZ_A": dist(h, argCZ),
                "CD_NE_H_deg": angle(argCD, argNE, h),
                "CZ_NE_H_deg": angle(argCZ, argNE, h),
            }
        )
        if dist(h, argCD) < 1.70 or dist(h, argCZ) < 1.70:
            failures.append("Arg19 NE-H points into guanidinium carbon skeleton")
        for ang in (angle(argCD, argNE, h), angle(argCZ, argNE, h)):
            if not (90.0 <= ang <= 150.0):
                warnings.append(f"Arg19 NE-H trigonal angle unusual: {ang:.1f} deg")

    # Retained water geometry.
    water_O = select(atoms, resname="HOH", element="O")
    waters = []
    if len(water_O) != 3:
        failures.append(f"expected 3 retained water oxygens, found {len(water_O)}")
    for O in water_O:
        hs = [
            h for h in atoms
            if h["element"] == "H"
            and h["chain"] == O["chain"]
            and h["resseq"] == O["resseq"]
            and h["resname"] == "HOH"
        ]
        entry = {
            "water": f"{O['chain']}:{O['resseq']}",
            "n_H": len(hs),
        }
        if len(hs) == 2:
            entry.update(
                {
                    "OH1_A": dist(O, hs[0]),
                    "OH2_A": dist(O, hs[1]),
                    "HOH_deg": angle(hs[0], O, hs[1]),
                }
            )
            if not (0.80 <= entry["OH1_A"] <= 1.20 and 0.80 <= entry["OH2_A"] <= 1.20):
                failures.append(f"water {entry['water']} has abnormal O-H distance")
            if not (80.0 <= entry["HOH_deg"] <= 130.0):
                warnings.append(
                    f"water {entry['water']} HOH angle {entry['HOH_deg']:.1f} deg"
                )
        else:
            failures.append(f"water {entry['water']} expected 2 H, found {len(hs)}")

        # Report nearest non-water polar heavy contacts.
        polar = [
            (dist(O, a), a)
            for a in atoms
            if a["element"] in {"N", "O", "S"}
            and a["resname"] != "HOH"
            and a is not O
        ]
        polar.sort(key=lambda x: x[0])
        entry["nearest_polar_contacts"] = [
            {"distance_A": d, "atom": label(a)}
            for d, a in polar[:5]
        ]
        waters.append(entry)

    # Catalytic H-bond geometry.
    lys_h = attached_h(atoms, lysNZ)
    thr_h = attached_h(atoms, thrN)
    asp_h1 = attached_h(atoms, aspOD1)
    asp_h2 = attached_h(atoms, aspOD2)

    network = {
        "Lys33_NZ_Thr1_OG1_A": dist(lysNZ, thrOG),
        "Lys33_NZ_Asp17_OD1_A": dist(lysNZ, aspOD1),
        "Lys33_NZ_Asp17_OD2_A": dist(lysNZ, aspOD2),
        "Thr1_N_Asp17_OD1_A": dist(thrN, aspOD1),
        "Thr1_N_Asp17_OD2_A": dist(thrN, aspOD2),
        "closest_Lys33H_to_Thr1_OG1_A":
            min((dist(h, thrOG) for h in lys_h), default=None),
        "closest_Lys33H_to_Asp17_OD1_A":
            min((dist(h, aspOD1) for h in lys_h), default=None),
        "closest_Lys33H_to_Asp17_OD2_A":
            min((dist(h, aspOD2) for h in lys_h), default=None),
        "closest_Thr1NH_to_boronate_ONTERM_A": None,
        "Asp17_OD1_H_A": dist(aspOD1, asp_h1[0]) if asp_h1 else None,
        "Asp17_OD2_H_A": dist(aspOD2, asp_h2[0]) if asp_h2 else None,
    }
    oname = roles[ligand]["O_NTERM"]["atom"]
    O_NTERM = one(atoms, chain="L", resseq=800, resname=ligand, name=oname)
    network["closest_Thr1NH_to_boronate_ONTERM_A"] = min(
        (dist(h, O_NTERM) for h in thr_h), default=None
    )

    return {
        "stem": stem,
        "ligand": ligand,
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "warnings": warnings,
        "n_atoms": len(atoms),
        "n_heavy": sum(a["element"] != "H" for a in atoms),
        "n_H": sum(a["element"] == "H" for a in atoms),
        "heavy_max_displacement_A": heavy_max,
        "heavy_rms_displacement_A": heavy_rms,
        "min_HH_A": hh_d,
        "min_HH_pair": [label(hh1), label(hh2)],
        "min_nonparent_Hheavy_A": nh_d,
        "min_nonparent_pair": [label(nh_h), label(nh_heavy)],
        "worst_XH_range_violations": [
            {
                "distance_A": d,
                "H": label(h),
                "parent": label(p),
            }
            for d, h, p in bad_xh[:10]
        ],
        "microstate_expected_H_counts": exp,
        "microstate_observed_H_counts": obs,
        "Thr1_OG1_H_count": thrOG_h,
        "Thr1_OG1_B26_A": dist(thrOG, B),
        "B26_coordination": [
            {"distance_A": d, "atom": label(a)}
            for d, a in bonded
        ],
        "boronate_OH": boronate_oh,
        "Arg19_NEH": arg_info,
        "waters": waters,
        "catalytic_network": network,
    }


def main():
    args = parse_args()
    registry = json.loads(args.microstates.read_text())
    roles = registry["boronate_oxygen_functional_roles"]

    stems = [
        "6V8_MS_A_THRN",
        "6V8_MS_B_LYS33",
        "6V8_MS_C_ASP17",
        "BO2_MS_A_THRN",
        "BO2_MS_B_LYS33",
        "BO2_MS_C_ASP17",
    ]

    cases = {
        stem: load_case(stem, args.optimized, args.metadata)
        for stem in stems
    }

    results = [
        audit_case(stem, cases[stem], roles, args)
        for stem in stems
    ]

    within_6v8 = check_same_heavy(
        cases,
        ["6V8_MS_A_THRN", "6V8_MS_B_LYS33", "6V8_MS_C_ASP17"],
        args.heavy_tol,
    )
    within_bo2 = check_same_heavy(
        cases,
        ["BO2_MS_A_THRN", "BO2_MS_B_LYS33", "BO2_MS_C_ASP17"],
        args.heavy_tol,
    )

    matched = {}
    for ms in ("MS_A_THRN", "MS_B_LYS33", "MS_C_ASP17"):
        matched[ms] = check_matched_receptor(
            cases[f"6V8_{ms}"],
            cases[f"BO2_{ms}"],
            args.heavy_tol,
        )

    invariant_status = (
        within_6v8["status"] == "PASS"
        and within_bo2["status"] == "PASS"
        and all(v["status"] == "PASS" for v in matched.values())
    )

    overall = (
        "PASS"
        if invariant_status and all(r["status"] == "PASS" for r in results)
        else "FAIL"
    )

    payload = {
        "stage": "8h_post_relaxation_chemical_audit",
        "role": (
            "Validate H-only relaxed structures before protonation-state "
            "energetic screening; no GFN-FF energies are interpreted."
        ),
        "thresholds": {
            "heavy_tolerance_A": args.heavy_tol,
            "min_HH_A": args.min_hh,
            "min_nonparent_Hheavy_A": args.min_nonparent_hheavy,
            "XH_parent_range_A": [args.xh_min, args.xh_max],
        },
        "within_ligand_heavy_invariance": {
            "6V8": within_6v8,
            "BO2": within_bo2,
        },
        "matched_receptor_invariance_between_ligands": matched,
        "results": results,
        "overall": overall,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")

    rows = []
    for r in results:
        rows.append(
            {
                "stem": r["stem"],
                "status": r["status"],
                "n_atoms": r["n_atoms"],
                "n_heavy": r["n_heavy"],
                "n_H": r["n_H"],
                "heavy_max_A": r["heavy_max_displacement_A"],
                "min_HH_A": r["min_HH_A"],
                "min_nonparent_Hheavy_A": r["min_nonparent_Hheavy_A"],
                "Thr1_OG1_B26_A": r["Thr1_OG1_B26_A"],
                "Thr1_N_H": r["microstate_observed_H_counts"]["Thr1_N"],
                "Lys33_NZ_H": r["microstate_observed_H_counts"]["Lys33_NZ"],
                "Asp17_OD1_H": r["microstate_observed_H_counts"]["Asp17_OD1"],
                "Asp17_OD2_H": r["microstate_observed_H_counts"]["Asp17_OD2"],
                "failures": "; ".join(r["failures"]),
                "warnings": "; ".join(r["warnings"]),
            }
        )

    with args.tsv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(rows)

    print("Stage 8h post-relaxation chemical audit")
    print()
    for r in results:
        print(
            f"{r['stem']:18s} {r['status']:4s} "
            f"heavy={r['heavy_max_displacement_A']:.3e} A  "
            f"minHH={r['min_HH_A']:.3f} A  "
            f"ThrOG-B={r['Thr1_OG1_B26_A']:.3f} A  "
            f"Hcounts={r['microstate_observed_H_counts']}"
        )
        for f in r["failures"]:
            print(f"    FAIL: {f}")
        for w in r["warnings"]:
            print(f"    WARN: {w}")

    print()
    print(
        "Within-ligand heavy invariance: "
        f"6V8={within_6v8['status']} BO2={within_bo2['status']}"
    )
    print(
        "Matched receptor invariance: "
        + ", ".join(f"{k}={v['status']}" for k, v in matched.items())
    )
    print(f"Overall: {overall}")
    print(f"JSON: {args.out}")
    print(f"TSV : {args.tsv}")

    raise SystemExit(0 if overall == "PASS" else 2)


if __name__ == "__main__":
    main()
