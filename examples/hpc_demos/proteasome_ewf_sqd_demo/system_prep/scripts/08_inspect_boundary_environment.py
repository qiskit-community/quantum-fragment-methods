#!/usr/bin/env python3
"""
08_inspect_boundary_environment.py

Proteasome Challenge
Stage 7b: boundary-environment screening before the molecular scalpel cuts.

Purpose
-------
For every proposed peptide-segment boundary from the network-closure model,
measure whether the retained endpoint and the immediately excluded neighboring
residue are actually close to the active-site chemistry.

Targets measured against:
  * native ixazomib / MLN2238 (6V8) in canonical 5LF7
  * bortezomib (BO2) transplanted from 5LF3 into the 5LF7-Y receptor frame
  * the three retained crystallographic water oxygens
  * catalytic Thr1
  * nonlocal functional-core residues

Both whole-residue heavy-atom and side-chain-only distances are reported.
This matters particularly for ionizable residues: a peptide backbone can be
close while an Asp/Glu/Lys/Arg side chain points away, or vice versa.

This stage is diagnostic only:
  * no atoms are deleted
  * no caps are added
  * no hydrogens/protonation states are assigned
  * no geometry is optimized

Example
-------
python3 scripts/08_inspect_boundary_environment.py \
  raw/5LF3.cif raw/5LF7.cif \
  --network validation/network_closure_report.json \
  --out validation/boundary_environment_report.json \
  --tsv validation/boundary_environment.tsv
"""

import argparse
import csv
import json
from pathlib import Path

import gemmi
import numpy as np


BACKBONE = {"N", "CA", "C", "O", "OXT"}
IONIZABLE = {"ASP", "GLU", "LYS", "ARG", "HIS", "CYS", "TYR"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("pdb_5lf3", help="5LF3 mmCIF containing BO2")
    p.add_argument("pdb_5lf7", help="5LF7 mmCIF containing 6V8; canonical receptor")
    p.add_argument("--network", required=True,
                   help="network_closure_report.json")
    p.add_argument("--out", required=True,
                   help="Output JSON report")
    p.add_argument("--tsv", required=True,
                   help="Output TSV")
    p.add_argument("--chain", default="Y",
                   help="Canonical beta5 label chain (default: Y)")
    return p.parse_args()


def clean(x):
    if x is None:
        return None
    x = str(x).strip()
    if x in ("", ".", "?"):
        return None
    return x


def as_int(x):
    x = clean(x)
    if x is None:
        return None
    try:
        return int(x)
    except ValueError:
        return None


def euclidean(a, b):
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


def min_distance(coords_a, coords_b):
    if not coords_a or not coords_b:
        return None
    A = np.asarray(coords_a, dtype=float)
    B = np.asarray(coords_b, dtype=float)
    d2 = ((A[:, None, :] - B[None, :, :]) ** 2).sum(axis=2)
    return float(np.sqrt(d2.min()))


def parse_atom_site(cif_path):
    doc = gemmi.cif.read_file(str(cif_path))
    block = doc.sole_block()

    tags = [
        "_atom_site.group_PDB",
        "_atom_site.type_symbol",
        "_atom_site.label_atom_id",
        "_atom_site.label_alt_id",
        "_atom_site.label_comp_id",
        "_atom_site.label_asym_id",
        "_atom_site.label_seq_id",
        "_atom_site.auth_asym_id",
        "_atom_site.auth_seq_id",
        "_atom_site.Cartn_x",
        "_atom_site.Cartn_y",
        "_atom_site.Cartn_z",
        "_atom_site.occupancy",
    ]
    table = block.find(tags)
    if not table:
        raise RuntimeError(f"Required _atom_site columns not found in {cif_path}")

    polymer = {}
    hetero = {}

    for row in table:
        group = clean(row[0])
        elem = clean(row[1])
        atom_name = clean(row[2])
        alt = clean(row[3])
        comp = clean(row[4])
        label_chain = clean(row[5])
        label_seq = as_int(row[6])
        auth_chain = clean(row[7])
        auth_seq = clean(row[8])

        if alt not in (None, "A"):
            continue

        xyz = np.array([float(row[9]), float(row[10]), float(row[11])], dtype=float)

        if group == "ATOM" and label_chain is not None and label_seq is not None:
            key = (label_chain, label_seq)
            rec = polymer.setdefault(key, {
                "label_chain": label_chain,
                "label_seq": label_seq,
                "auth_chain": auth_chain,
                "auth_seq": auth_seq,
                "comp_id": comp,
                "atoms": {},
                "elements": {},
            })
            if atom_name not in rec["atoms"]:
                rec["atoms"][atom_name] = xyz
                rec["elements"][atom_name] = elem

        elif group == "HETATM":
            key = (auth_chain, auth_seq, comp)
            rec = hetero.setdefault(key, {
                "auth_chain": auth_chain,
                "auth_seq": auth_seq,
                "label_chain": label_chain,
                "label_seq": label_seq,
                "comp_id": comp,
                "atoms": {},
                "elements": {},
            })
            if atom_name not in rec["atoms"]:
                rec["atoms"][atom_name] = xyz
                rec["elements"][atom_name] = elem

    return polymer, hetero


def heavy_coords(residue, sidechain=False):
    if residue is None:
        return []
    out = []
    for name, xyz in residue["atoms"].items():
        elem = residue["elements"].get(name)
        if elem and elem.upper() == "H":
            continue
        if sidechain and name in BACKBONE:
            continue
        out.append(xyz)
    return out


def align_5lf3_y_to_5lf7_y(poly3, poly7, chain):
    P, Q = [], []
    for key in sorted(set(poly3) & set(poly7)):
        c, _ = key
        if c != chain:
            continue
        r3, r7 = poly3[key], poly7[key]
        if r3["comp_id"] != r7["comp_id"]:
            continue
        for atom_name in ("N", "CA", "C", "O"):
            if atom_name in r3["atoms"] and atom_name in r7["atoms"]:
                P.append(r3["atoms"][atom_name])
                Q.append(r7["atoms"][atom_name])

    if len(P) < 12:
        raise RuntimeError("Too few matched backbone atoms for 5LF3->5LF7 alignment")

    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)
    cP, cQ = P.mean(axis=0), Q.mean(axis=0)
    P0, Q0 = P - cP, Q - cQ

    H = P0.T @ Q0
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    def transform(x):
        return R @ (np.asarray(x, dtype=float) - cP) + cQ

    Pfit = np.vstack([transform(x) for x in P])
    rmsd = float(np.sqrt(np.mean(np.sum((Pfit - Q) ** 2, axis=1))))
    return transform, rmsd, len(P)


def choose_active_site_ligand(hetero, comp_id, polymer, chain):
    thr1 = polymer.get((chain, 1))
    if thr1 is None or "OG1" not in thr1["atoms"]:
        raise RuntimeError(f"Could not find catalytic {chain}:1 Thr OG1")
    og1 = thr1["atoms"]["OG1"]

    candidates = []
    for _, lig in hetero.items():
        if lig["comp_id"] != comp_id:
            continue
        if "B26" in lig["atoms"]:
            d = euclidean(lig["atoms"]["B26"], og1)
        else:
            d = min_distance(heavy_coords(lig), [og1])
        candidates.append((d, lig))

    if not candidates:
        raise RuntimeError(f"No {comp_id} ligand copies found")
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1], candidates[0][0]


def locate_retained_waters(hetero7, network):
    waters = {}
    for w in network["retained_water_candidates"]:
        auth_chain = str(w["auth_chain"])
        auth_seq = str(w["auth_seq"])
        matches = [
            rec for (c, s, comp), rec in hetero7.items()
            if c == auth_chain and s == auth_seq and comp in ("HOH", "WAT", "H2O")
        ]
        if not matches:
            matches = [
                rec for (c, s, _), rec in hetero7.items()
                if c == auth_chain and s == auth_seq
            ]
        if not matches:
            raise RuntimeError(f"Could not locate retained water {auth_chain}:{auth_seq}")
        coords = heavy_coords(matches[0])
        if not coords:
            raise RuntimeError(f"No heavy atom for retained water {auth_chain}:{auth_seq}")
        waters[f"{auth_chain}:{auth_seq}"] = coords
    return waters


def residue_key_from_string(s):
    chain, seq = s.split(":")
    return chain, int(seq)


def core_coords(poly7, core_keys, exclude_key=None, exclude_local_neighbors=False):
    coords = []
    for key in core_keys:
        if exclude_key is not None and key == exclude_key:
            continue
        if exclude_local_neighbors and exclude_key is not None:
            if key[0] == exclude_key[0] and abs(key[1] - exclude_key[1]) <= 1:
                continue
        coords.extend(heavy_coords(poly7.get(key)))
    return coords


def safe_round(x, n=3):
    return None if x is None else round(float(x), n)


def classify(rec, metrics):
    whole_lig_vals = [x for x in (metrics["whole_to_6V8_A"], metrics["whole_to_BO2_A"]) if x is not None]
    whole_lig = min(whole_lig_vals) if whole_lig_vals else None
    whole_water = metrics["whole_to_any_water_A"]
    nonlocal_core = metrics["whole_to_nonlocal_core_A"]

    side_candidates = [
        metrics["sidechain_to_6V8_A"],
        metrics["sidechain_to_BO2_A"],
        metrics["sidechain_to_any_water_A"],
        metrics["sidechain_to_Thr1_A"],
    ]
    side_candidates = [x for x in side_candidates if x is not None]
    side_min = min(side_candidates) if side_candidates else None

    if ((whole_lig is not None and whole_lig <= 4.5)
        or (whole_water is not None and whole_water <= 3.5)
        or (nonlocal_core is not None and nonlocal_core <= 4.0)
        or (side_min is not None and side_min <= 4.0)):
        return "network_relevant_review"

    if rec["comp_id"] in IONIZABLE and side_min is not None and side_min <= 8.0:
        return "charge_sensitive_review"

    if ((whole_lig is not None and whole_lig <= 6.0)
        or (whole_water is not None and whole_water <= 5.0)
        or (nonlocal_core is not None and nonlocal_core <= 5.0)):
        return "second_shell_review"

    return "peripheral_by_geometry"


def analyze_residue(rec, lig6, ligbo2, waters, thr1_coords, nonlocal_core):
    whole = heavy_coords(rec)
    side = heavy_coords(rec, sidechain=True)

    water_whole = {wid: min_distance(whole, coords) for wid, coords in waters.items()}
    water_side = {wid: (min_distance(side, coords) if side else None) for wid, coords in waters.items()}

    whole_water_vals = [x for x in water_whole.values() if x is not None]
    side_water_vals = [x for x in water_side.values() if x is not None]

    m = {
        "whole_to_6V8_A": min_distance(whole, lig6),
        "whole_to_BO2_A": min_distance(whole, ligbo2),
        "whole_to_any_water_A": min(whole_water_vals) if whole_water_vals else None,
        "whole_to_Thr1_A": min_distance(whole, thr1_coords),
        "whole_to_nonlocal_core_A": min_distance(whole, nonlocal_core),
        "sidechain_to_6V8_A": min_distance(side, lig6) if side else None,
        "sidechain_to_BO2_A": min_distance(side, ligbo2) if side else None,
        "sidechain_to_any_water_A": min(side_water_vals) if side_water_vals else None,
        "sidechain_to_Thr1_A": min_distance(side, thr1_coords) if side else None,
        "water_distances_whole_A": water_whole,
        "water_distances_sidechain_A": water_side,
    }
    m["screen"] = classify(rec, m)
    return m


def main():
    args = parse_args()
    network = json.loads(Path(args.network).read_text())
    poly3, hetero3 = parse_atom_site(Path(args.pdb_5lf3))
    poly7, hetero7 = parse_atom_site(Path(args.pdb_5lf7))

    transform, alignment_rmsd, nfit = align_5lf3_y_to_5lf7_y(poly3, poly7, args.chain)
    bo2, bo2_native_thr_dist = choose_active_site_ligand(hetero3, "BO2", poly3, args.chain)
    lig6_rec, lig6_thr_dist = choose_active_site_ligand(hetero7, "6V8", poly7, args.chain)

    lig6 = heavy_coords(lig6_rec)
    ligbo2 = [transform(x) for x in heavy_coords(bo2)]
    waters = locate_retained_waters(hetero7, network)

    thr1 = poly7.get((args.chain, 1))
    if thr1 is None:
        raise RuntimeError(f"Could not find {args.chain}:1 in 5LF7")
    thr1_coords = heavy_coords(thr1)

    core_keys = [residue_key_from_string(s) for s in network["functional_network_core"]["residues"]]
    segments = network["peptide_coherent_reference"]["segments"]
    results = []

    for seg in segments:
        chain = seg["label_chain"]
        start = int(seg["label_start"])
        end = int(seg["label_end"])
        boundary_specs = [
            ("N", (chain, start), (chain, start - 1)),
            ("C", (chain, end), (chain, end + 1)),
        ]

        for side, retained_key, outside_key in boundary_specs:
            retained = poly7.get(retained_key)
            outside = poly7.get(outside_key)
            if retained is None:
                raise RuntimeError(f"Missing retained residue {retained_key}")

            boundary = {
                "segment": f"{chain}:{start}-{end}",
                "side": side,
                "retained_endpoint": None,
                "outside_neighbor": None,
            }

            for role, key, rec in (
                ("retained_endpoint", retained_key, retained),
                ("outside_neighbor", outside_key, outside),
            ):
                if rec is None:
                    boundary[role] = None
                    continue

                nonlocal_core = core_coords(poly7, core_keys, exclude_key=key, exclude_local_neighbors=True)
                metrics = analyze_residue(rec, lig6, ligbo2, waters, thr1_coords, nonlocal_core)
                boundary[role] = {
                    "label_chain": rec["label_chain"],
                    "label_seq": rec["label_seq"],
                    "auth_chain": rec["auth_chain"],
                    "auth_seq": rec["auth_seq"],
                    "comp_id": rec["comp_id"],
                    "ionizable_or_charge_sensitive_type": rec["comp_id"] in IONIZABLE,
                    "metrics": metrics,
                }

            results.append(boundary)

    rows = []
    for b in results:
        for role in ("retained_endpoint", "outside_neighbor"):
            r = b[role]
            if r is None:
                continue
            m = r["metrics"]
            rows.append({
                "segment": b["segment"],
                "boundary_side": b["side"],
                "role": role,
                "residue": f'{r["label_chain"]}:{r["label_seq"]} {r["comp_id"]}',
                "ionizable": r["ionizable_or_charge_sensitive_type"],
                "screen": m["screen"],
                "whole_to_6V8_A": safe_round(m["whole_to_6V8_A"]),
                "whole_to_BO2_A": safe_round(m["whole_to_BO2_A"]),
                "whole_to_any_water_A": safe_round(m["whole_to_any_water_A"]),
                "whole_to_Thr1_A": safe_round(m["whole_to_Thr1_A"]),
                "whole_to_nonlocal_core_A": safe_round(m["whole_to_nonlocal_core_A"]),
                "sidechain_to_6V8_A": safe_round(m["sidechain_to_6V8_A"]),
                "sidechain_to_BO2_A": safe_round(m["sidechain_to_BO2_A"]),
                "sidechain_to_any_water_A": safe_round(m["sidechain_to_any_water_A"]),
                "sidechain_to_Thr1_A": safe_round(m["sidechain_to_Thr1_A"]),
            })

    report = {
        "purpose": "Boundary-environment screening before final peptide cut/cap selection",
        "alignment": {
            "5LF3_chain": args.chain,
            "5LF7_chain": args.chain,
            "matched_backbone_atoms": nfit,
            "rmsd_A": alignment_rmsd,
            "note": "5LF3 BO2 coordinates transformed into the canonical 5LF7-Y receptor frame",
        },
        "ligand_selection": {
            "BO2_native_Thr1_B_distance_A": bo2_native_thr_dist,
            "6V8_native_Thr1_B_distance_A": lig6_thr_dist,
        },
        "retained_waters": list(waters.keys()),
        "boundaries": results,
        "screen_meanings": {
            "network_relevant_review": "Geometry places this residue within first-shell/network-like distance of ligand, retained water, or nonlocal functional core.",
            "charge_sensitive_review": "An ionizable/charge-sensitive side chain lies within 8 A of the modeled chemistry; do not discard solely because it is outside the direct-contact shell.",
            "second_shell_review": "Not first-shell by this screen, but geometrically close enough to deserve one explicit modeling decision.",
            "peripheral_by_geometry": "No close geometric signal in this diagnostic; this supports, but does not prove, safe exclusion.",
        },
        "important_caveats": [
            "These are geometric screens, not energetic calculations.",
            "Charge effects are longer ranged than hydrogen-bond/contact distances.",
            "Side-chain protonation states remain unresolved.",
            "Backbone adjacency to a cut is intentionally excluded from the nonlocal-core metric.",
            "No keep/delete/cap decision is made automatically.",
            "No coordinates are modified.",
        ],
        "next_step": "Use these distances to finalize segment endpoints, then construct the first explicit capped receptor topology.",
    }

    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")

    fields = [
        "segment", "boundary_side", "role", "residue", "ionizable", "screen",
        "whole_to_6V8_A", "whole_to_BO2_A", "whole_to_any_water_A",
        "whole_to_Thr1_A", "whole_to_nonlocal_core_A",
        "sidechain_to_6V8_A", "sidechain_to_BO2_A",
        "sidechain_to_any_water_A", "sidechain_to_Thr1_A",
    ]
    with open(args.tsv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {args.out}")
    print(f"Wrote {args.tsv}")
    print()
    print("Stage-7b boundary-environment screen")
    print(f"  5LF3 -> 5LF7 chain {args.chain} alignment RMSD: {alignment_rmsd:.3f} A over {nfit} backbone atoms")
    print(f"  retained waters: {', '.join(waters.keys())}")
    print()
    print("Boundary residue summary")
    print("  distances = whole-residue heavy atoms; side-chain shown for ionizable residues")
    print()

    for row in rows:
        base = (
            f'  {row["segment"]:10s} {row["boundary_side"]}-side '
            f'{row["role"][:3]:3s} {row["residue"]:12s} '
            f'screen={row["screen"]:24s} '
            f'lig6={str(row["whole_to_6V8_A"]):>5s} '
            f'BO2={str(row["whole_to_BO2_A"]):>5s} '
            f'water={str(row["whole_to_any_water_A"]):>5s} '
            f'nonlocal_core={str(row["whole_to_nonlocal_core_A"]):>5s}'
        )
        if row["ionizable"]:
            base += (
                f' | sidechain: lig6={str(row["sidechain_to_6V8_A"]):>5s}'
                f' BO2={str(row["sidechain_to_BO2_A"]):>5s}'
                f' water={str(row["sidechain_to_any_water_A"]):>5s}'
                f' Thr1={str(row["sidechain_to_Thr1_A"]):>5s}'
            )
        print(base)

    print()
    print("Ionizable / charge-sensitive boundary residues")
    ion_rows = [r for r in rows if r["ionizable"]]
    if not ion_rows:
        print("  none")
    else:
        for r in ion_rows:
            print(
                f'  {r["residue"]:12s} at {r["segment"]} {r["boundary_side"]}-side '
                f'({r["role"]}): {r["screen"]}; '
                f'sidechain->6V8={r["sidechain_to_6V8_A"]} A, '
                f'BO2={r["sidechain_to_BO2_A"]} A, '
                f'water={r["sidechain_to_any_water_A"]} A, '
                f'Thr1={r["sidechain_to_Thr1_A"]} A'
            )


if __name__ == "__main__":
    main()
