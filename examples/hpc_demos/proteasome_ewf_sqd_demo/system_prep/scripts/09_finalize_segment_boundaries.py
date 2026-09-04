#!/usr/bin/env python3
"""
09_finalize_segment_boundaries.py

Proteasome Challenge
Stage 7c: final boundary-extension check before cap construction.

Candidate topology reviewed here:
  Y:1-3
  Y:17-22
  Y:46-50
  Y:129-131
  Y:167-170
  Z:124-126

This stage is deliberately narrow. It checks the residues immediately outside
those revised peptide spans against:
  - native ixazomib/MLN2238 (6V8) in 5LF7
  - bortezomib (BO2) transformed from 5LF3 into the canonical 5LF7-Y frame
  - retained waters Y:437, Z:406, Y:462
  - catalytic Thr1

It reports whole-residue and side-chain heavy-atom distances and gives a
conservative "expand / review / stop" suggestion.

NO coordinates are modified.
NO caps/hydrogens/protonation states are created.

Example
-------
python3 scripts/09_finalize_segment_boundaries.py \
  raw/5LF3.cif raw/5LF7.cif \
  --network validation/network_closure_report.json \
  --out validation/final_boundary_review.json \
  --tsv validation/final_boundary_review.tsv
"""

import argparse
import csv
import json
import math
from pathlib import Path

import gemmi
import numpy as np

BACKBONE = {"N", "CA", "C", "O", "OXT"}
IONIZABLE = {"ASP", "GLU", "LYS", "ARG", "HIS", "CYS", "TYR"}

DEFAULT_SEGMENTS = [
    ("Y", 1, 3),
    ("Y", 17, 22),
    ("Y", 46, 50),
    ("Y", 129, 131),
    ("Y", 167, 170),
    ("Z", 124, 126),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("pdb_5lf3")
    p.add_argument("pdb_5lf7")
    p.add_argument("--network", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tsv", required=True)
    p.add_argument("--chain", default="Y")
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


def parse_cif(path):
    doc = gemmi.cif.read_file(str(path))
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
    ]
    table = block.find(tags)
    if not table:
        raise RuntimeError(f"Could not read _atom_site from {path}")

    polymer = {}
    hetero = {}

    for row in table:
        group = clean(row[0])
        elem = clean(row[1])
        atom = clean(row[2])
        alt = clean(row[3])
        comp = clean(row[4])
        lchain = clean(row[5])
        lseq = as_int(row[6])
        achain = clean(row[7])
        aseq = clean(row[8])
        xyz = np.array([float(row[9]), float(row[10]), float(row[11])])

        if alt not in (None, "A"):
            continue

        if group == "ATOM" and lchain is not None and lseq is not None:
            key = (lchain, lseq)
            rec = polymer.setdefault(key, {
                "label_chain": lchain,
                "label_seq": lseq,
                "auth_chain": achain,
                "auth_seq": aseq,
                "comp_id": comp,
                "atoms": {},
                "elements": {},
            })
            rec["atoms"].setdefault(atom, xyz)
            rec["elements"].setdefault(atom, elem)

        elif group == "HETATM":
            key = (achain, aseq, comp)
            rec = hetero.setdefault(key, {
                "auth_chain": achain,
                "auth_seq": aseq,
                "label_chain": lchain,
                "label_seq": lseq,
                "comp_id": comp,
                "atoms": {},
                "elements": {},
            })
            rec["atoms"].setdefault(atom, xyz)
            rec["elements"].setdefault(atom, elem)

    return polymer, hetero


def heavy_coords(rec, sidechain=False):
    if rec is None:
        return []
    coords = []
    for name, xyz in rec["atoms"].items():
        elem = rec["elements"].get(name, "")
        if elem.upper() == "H":
            continue
        if sidechain and name in BACKBONE:
            continue
        coords.append(np.asarray(xyz, dtype=float))
    return coords


def min_dist(a, b):
    if not a or not b:
        return None
    A = np.vstack(a)
    B = np.vstack(b)
    d2 = ((A[:, None, :] - B[None, :, :]) ** 2).sum(axis=2)
    return float(np.sqrt(d2.min()))


def align(poly3, poly7, chain):
    P, Q = [], []

    for key in sorted(set(poly3) & set(poly7)):
        if key[0] != chain:
            continue
        r3, r7 = poly3[key], poly7[key]
        if r3["comp_id"] != r7["comp_id"]:
            continue
        for atom in ("N", "CA", "C", "O"):
            if atom in r3["atoms"] and atom in r7["atoms"]:
                P.append(r3["atoms"][atom])
                Q.append(r7["atoms"][atom])

    P = np.asarray(P)
    Q = np.asarray(Q)
    if len(P) < 12:
        raise RuntimeError("Too few matched atoms for alignment")

    cp, cq = P.mean(0), Q.mean(0)
    P0, Q0 = P - cp, Q - cq
    U, S, Vt = np.linalg.svd(P0.T @ Q0)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    def transform(x):
        return R @ (np.asarray(x) - cp) + cq

    fit = np.vstack([transform(x) for x in P])
    rmsd = float(np.sqrt(np.mean(np.sum((fit - Q) ** 2, axis=1))))
    return transform, rmsd, len(P)


def choose_ligand(hetero, comp_id, polymer, chain):
    thr1 = polymer[(chain, 1)]
    og1 = thr1["atoms"]["OG1"]

    candidates = []
    for rec in hetero.values():
        if rec["comp_id"] != comp_id:
            continue
        if "B26" in rec["atoms"]:
            d = float(np.linalg.norm(rec["atoms"]["B26"] - og1))
        else:
            d = min_dist(heavy_coords(rec), [og1])
        candidates.append((d, rec))

    if not candidates:
        raise RuntimeError(f"No {comp_id} found")
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1], candidates[0][0]


def locate_waters(hetero7, network):
    waters = {}
    for w in network["retained_water_candidates"]:
        achain = str(w["auth_chain"])
        aseq = str(w["auth_seq"])
        found = None
        for (c, s, comp), rec in hetero7.items():
            if c == achain and s == aseq:
                found = rec
                break
        if found is None:
            raise RuntimeError(f"Retained water not found: {achain}:{aseq}")
        waters[f"{achain}:{aseq}"] = heavy_coords(found)
    return waters


def sidechain_min(rec, target):
    side = heavy_coords(rec, sidechain=True)
    return min_dist(side, target) if side else None


def analyze(rec, lig6, ligbo2, waters, thr1):
    whole = heavy_coords(rec)
    side = heavy_coords(rec, sidechain=True)

    water_whole = {
        wid: min_dist(whole, coords) for wid, coords in waters.items()
    }
    water_side = {
        wid: min_dist(side, coords) if side else None
        for wid, coords in waters.items()
    }

    vals_whole = [v for v in water_whole.values() if v is not None]
    vals_side = [v for v in water_side.values() if v is not None]

    return {
        "whole_to_6V8_A": min_dist(whole, lig6),
        "whole_to_BO2_A": min_dist(whole, ligbo2),
        "whole_to_any_water_A": min(vals_whole) if vals_whole else None,
        "whole_to_Thr1_A": min_dist(whole, thr1),
        "sidechain_to_6V8_A": min_dist(side, lig6) if side else None,
        "sidechain_to_BO2_A": min_dist(side, ligbo2) if side else None,
        "sidechain_to_any_water_A": min(vals_side) if vals_side else None,
        "sidechain_to_Thr1_A": min_dist(side, thr1) if side else None,
        "water_distances_whole_A": water_whole,
        "water_distances_sidechain_A": water_side,
    }


def recommendation(rec, m):
    """
    Conservative final-boundary heuristic.

    Expand:
      ionizable side chain <=4.5 A to Thr1 / ligand / retained water,
      OR any residue <=3.5 A to retained water.

    Review:
      ionizable side chain 4.5-6.0 A to modeled chemistry,
      OR whole-residue <=5.0 A to ligand/water.

    Stop:
      no compelling first-/near-second-shell signal by these criteria.
    """
    side_targets = [
        m["sidechain_to_6V8_A"],
        m["sidechain_to_BO2_A"],
        m["sidechain_to_any_water_A"],
        m["sidechain_to_Thr1_A"],
    ]
    side_targets = [x for x in side_targets if x is not None]
    side_min = min(side_targets) if side_targets else None

    whole_chem = [
        m["whole_to_6V8_A"],
        m["whole_to_BO2_A"],
        m["whole_to_any_water_A"],
        m["whole_to_Thr1_A"],
    ]
    whole_min = min(x for x in whole_chem if x is not None)

    if rec["comp_id"] in IONIZABLE and side_min is not None and side_min <= 4.5:
        return "EXPAND", "ionizable side chain <=4.5 A from modeled chemistry"

    if m["whole_to_any_water_A"] is not None and m["whole_to_any_water_A"] <= 3.5:
        return "EXPAND", "whole residue <=3.5 A from retained water"

    if rec["comp_id"] in IONIZABLE and side_min is not None and side_min <= 6.0:
        return "REVIEW", "ionizable side chain within 4.5-6.0 A of modeled chemistry"

    if whole_min <= 5.0:
        return "REVIEW", "whole residue within 5.0 A of modeled chemistry"

    return "STOP", "no compelling first-/near-second-shell signal by this diagnostic"


def r3(x):
    return None if x is None else round(float(x), 3)


def main():
    args = parse_args()
    network = json.loads(Path(args.network).read_text())

    poly3, het3 = parse_cif(Path(args.pdb_5lf3))
    poly7, het7 = parse_cif(Path(args.pdb_5lf7))

    transform, rmsd, nfit = align(poly3, poly7, args.chain)

    bo2, bo2_thr = choose_ligand(het3, "BO2", poly3, args.chain)
    v8, v8_thr = choose_ligand(het7, "6V8", poly7, args.chain)

    ligbo2 = [transform(x) for x in heavy_coords(bo2)]
    lig6 = heavy_coords(v8)
    waters = locate_waters(het7, network)
    thr1 = heavy_coords(poly7[(args.chain, 1)])

    results = []

    for chain, start, end in DEFAULT_SEGMENTS:
        checks = []

        # No N-side outside residue for the natural chain start.
        if start > 1:
            checks.append(("N", (chain, start - 1)))
        checks.append(("C", (chain, end + 1)))

        for side, key in checks:
            rec = poly7.get(key)
            if rec is None:
                results.append({
                    "segment": f"{chain}:{start}-{end}",
                    "side": side,
                    "outside_neighbor": None,
                    "recommendation": "STOP",
                    "reason": "natural polymer terminus / no adjacent polymer residue",
                    "metrics": None,
                })
                continue

            m = analyze(rec, lig6, ligbo2, waters, thr1)
            decision, reason = recommendation(rec, m)

            results.append({
                "segment": f"{chain}:{start}-{end}",
                "side": side,
                "outside_neighbor": {
                    "label_chain": rec["label_chain"],
                    "label_seq": rec["label_seq"],
                    "auth_chain": rec["auth_chain"],
                    "auth_seq": rec["auth_seq"],
                    "comp_id": rec["comp_id"],
                    "ionizable_or_charge_sensitive_type":
                        rec["comp_id"] in IONIZABLE,
                },
                "recommendation": decision,
                "reason": reason,
                "metrics": {k: (
                    {wk: r3(wv) for wk, wv in v.items()}
                    if isinstance(v, dict) else r3(v)
                ) for k, v in m.items()},
            })

    report = {
        "purpose":
            "Final boundary-extension check before freezing peptide segment endpoints",
        "candidate_segments": [
            f"{c}:{s}-{e}" for c, s, e in DEFAULT_SEGMENTS
        ],
        "alignment": {
            "5LF3_to_5LF7_chain": args.chain,
            "backbone_atoms": nfit,
            "rmsd_A": round(rmsd, 4),
        },
        "retained_waters": list(waters.keys()),
        "results": results,
        "decision_rule_note":
            "EXPAND/REVIEW/STOP is a conservative geometry heuristic, not an energetic result.",
        "important_caveats": [
            "Charge effects may extend beyond simple distance thresholds.",
            "Protonation states are not assigned.",
            "No coordinates are modified.",
            "A STOP recommendation means no compelling geometric reason to expand again; it does not prove zero energetic effect.",
        ],
        "next_step":
            "If no new outside neighbor earns EXPAND, freeze the candidate spans and construct the first capped receptor topology.",
    }

    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")

    rows = []
    for r in results:
        out = r["outside_neighbor"]
        m = r["metrics"]
        rows.append({
            "segment": r["segment"],
            "side": r["side"],
            "outside_neighbor": "" if out is None else (
                f'{out["label_chain"]}:{out["label_seq"]} {out["comp_id"]}'
            ),
            "ionizable": "" if out is None else out["ionizable_or_charge_sensitive_type"],
            "recommendation": r["recommendation"],
            "reason": r["reason"],
            "whole_to_6V8_A": "" if m is None else m["whole_to_6V8_A"],
            "whole_to_BO2_A": "" if m is None else m["whole_to_BO2_A"],
            "whole_to_any_water_A": "" if m is None else m["whole_to_any_water_A"],
            "whole_to_Thr1_A": "" if m is None else m["whole_to_Thr1_A"],
            "sidechain_to_6V8_A": "" if m is None else m["sidechain_to_6V8_A"],
            "sidechain_to_BO2_A": "" if m is None else m["sidechain_to_BO2_A"],
            "sidechain_to_any_water_A": "" if m is None else m["sidechain_to_any_water_A"],
            "sidechain_to_Thr1_A": "" if m is None else m["sidechain_to_Thr1_A"],
        })

    fields = list(rows[0].keys())
    with open(args.tsv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {args.out}")
    print(f"Wrote {args.tsv}")
    print()
    print("Stage-7c final boundary-extension review")
    print(f"  5LF3 -> 5LF7 alignment RMSD: {rmsd:.3f} A over {nfit} backbone atoms")
    print(f"  candidate segments: {', '.join(report['candidate_segments'])}")
    print()

    for row in rows:
        print(
            f'  {row["segment"]:10s} {row["side"]}-side  '
            f'outside={row["outside_neighbor"] or "NATURAL TERMINUS":12s}  '
            f'{row["recommendation"]:6s}  '
            f'lig6={str(row["whole_to_6V8_A"]):>5s} '
            f'BO2={str(row["whole_to_BO2_A"]):>5s} '
            f'water={str(row["whole_to_any_water_A"]):>5s} '
            f'Thr1={str(row["whole_to_Thr1_A"]):>5s}  '
            f'| {row["reason"]}'
        )

    print()
    print("Decision summary")
    counts = {"EXPAND": 0, "REVIEW": 0, "STOP": 0}
    for r in results:
        counts[r["recommendation"]] += 1
    for k in ("EXPAND", "REVIEW", "STOP"):
        print(f"  {k}: {counts[k]}")

    if counts["EXPAND"] == 0:
        print()
        print("  No outside neighbor triggered automatic expansion.")
        print("  Candidate topology is ready for human review and boundary freeze.")
    else:
        print()
        print("  At least one outside neighbor triggered expansion.")
        print("  Review those residue(s) before freezing the topology.")


if __name__ == "__main__":
    main()
