#!/usr/bin/env python3
"""
07_inspect_cut_boundaries.py

Proteasome Challenge
Stage 7a: inspect proposed peptide-segment cut boundaries before capping.

This script is intentionally diagnostic. It DOES NOT:
  - modify coordinates
  - add caps
  - add hydrogens
  - assign protonation states
  - optimize geometry

Inputs
------
1. canonical receptor mmCIF (5LF7)
2. network_closure_report.json from Stage 6c

Outputs
-------
- JSON report with every N- and C-terminal boundary
- TSV summary for easy inspection

For each proposed segment, the report records:
  - retained endpoint residue
  - immediately adjacent excluded residue, if present
  - peptide C--N distance across the proposed cut
  - whether the boundary is a natural polymer terminus
  - whether Pro/Gly or ionizable residues are involved
  - a conservative cap-planning note

The cap notes are planning heuristics only. Final cap chemistry should be
chosen after review of the report and active-site topology.

Example
-------
python3 scripts/07_inspect_cut_boundaries.py \
  raw/5LF7.cif \
  --network validation/network_closure_report.json \
  --out validation/cut_boundary_report.json \
  --tsv validation/cut_boundaries.tsv
"""

import argparse
import csv
import json
import math
from pathlib import Path

import gemmi


IONIZABLE = {"ASP", "GLU", "LYS", "ARG", "HIS", "CYS", "TYR"}
SPECIAL = {"PRO", "GLY"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("canonical_cif", help="Canonical receptor mmCIF (5LF7)")
    p.add_argument("--network", required=True,
                   help="network_closure_report.json")
    p.add_argument("--out", required=True,
                   help="Output JSON report")
    p.add_argument("--tsv", required=True,
                   help="Output TSV summary")
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


def distance(a, b):
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def load_polymer_residues(cif_path):
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
        raise RuntimeError("Required _atom_site columns not found")

    residues = {}

    for row in table:
        group = clean(row[0])
        if group != "ATOM":
            continue

        atom = clean(row[2])
        alt = clean(row[3])
        comp = clean(row[4])
        label_chain = clean(row[5])
        label_seq = as_int(row[6])
        auth_chain = clean(row[7])
        auth_seq = clean(row[8])

        if label_chain is None or label_seq is None:
            continue

        # Prefer blank / primary conformer. For our selected region there
        # should not be relevant alternate conformers, but guard anyway.
        if alt not in (None, "A"):
            continue

        xyz = (float(row[9]), float(row[10]), float(row[11]))
        key = (label_chain, label_seq)

        rec = residues.setdefault(key, {
            "label_chain": label_chain,
            "label_seq": label_seq,
            "auth_chain": auth_chain,
            "auth_seq": auth_seq,
            "comp_id": comp,
            "atoms": {},
        })

        if atom not in rec["atoms"]:
            rec["atoms"][atom] = xyz

    return residues


def residue_id(rec):
    if rec is None:
        return None
    return f'{rec["label_chain"]}:{rec["label_seq"]} {rec["comp_id"]}'


def boundary_flags(retained, outside):
    flags = []

    if retained["comp_id"] in IONIZABLE:
        flags.append("retained_endpoint_ionizable")
    if retained["comp_id"] in SPECIAL:
        flags.append(f'retained_endpoint_{retained["comp_id"].lower()}')

    if outside is not None:
        if outside["comp_id"] in IONIZABLE:
            flags.append("outside_neighbor_ionizable")
        if outside["comp_id"] in SPECIAL:
            flags.append(f'outside_neighbor_{outside["comp_id"].lower()}')

    return flags


def bond_quality(d):
    if d is None:
        return "unavailable"
    # Peptide C--N is normally ~1.32-1.35 A; use a broad diagnostic window.
    if 1.20 <= d <= 1.50:
        return "normal_peptide_geometry"
    return "inspect_geometry"


def cap_note(side, natural, retained, outside):
    if natural:
        if side == "N":
            return (
                "Preserve the native polymer N-terminus; do NOT add an ACE-like "
                "cap here. For Y:1 this is chemically essential because catalytic "
                "Thr1 requires its authentic N-terminal amino group."
            )
        return "Natural polymer C-terminus; no artificial peptide cap required."

    if side == "N":
        note = (
            "Internal N-side peptide cut. Initial reference choice: consider an "
            "ACE-like carbonyl cap on the retained residue N to preserve peptide-"
            "amide electronic structure rather than simply replacing the severed "
            "bond with H."
        )
        if retained["comp_id"] == "PRO":
            note += (
                " Retained endpoint is Pro; preserve its cyclic N substitution "
                "and inspect the cap geometry explicitly."
            )
        return note

    note = (
        "Internal C-side peptide cut. Initial reference choice: consider an "
        "N-methylamide/NME-like continuation on the retained carbonyl C to "
        "preserve peptide-amide character."
    )
    if outside is not None and outside["comp_id"] == "PRO":
        note += (
            " The excluded next residue is Pro, whose peptide N is substituted; "
            "a generic NME cap is a poorer mimic here, so extension across Pro "
            "should be considered."
        )
    return note


def find_prev_next(residues, chain, seq):
    prev_rec = residues.get((chain, seq - 1))
    next_rec = residues.get((chain, seq + 1))
    return prev_rec, next_rec


def make_boundary(side, segment, retained, outside, natural, d):
    return {
        "segment": f'{segment["label_chain"]}:{segment["label_start"]}-{segment["label_end"]}',
        "side": side,
        "retained_endpoint": {
            "label_chain": retained["label_chain"],
            "label_seq": retained["label_seq"],
            "auth_chain": retained["auth_chain"],
            "auth_seq": retained["auth_seq"],
            "comp_id": retained["comp_id"],
        },
        "outside_neighbor": None if outside is None else {
            "label_chain": outside["label_chain"],
            "label_seq": outside["label_seq"],
            "auth_chain": outside["auth_chain"],
            "auth_seq": outside["auth_seq"],
            "comp_id": outside["comp_id"],
        },
        "natural_polymer_terminus": natural,
        "peptide_CN_distance_A": None if d is None else round(d, 4),
        "geometry_screen": bond_quality(d),
        "flags": boundary_flags(retained, outside),
        "cap_planning_note": cap_note(side, natural, retained, outside),
    }


def main():
    args = parse_args()
    residues = load_polymer_residues(Path(args.canonical_cif))
    network = json.loads(Path(args.network).read_text())

    segments = network["peptide_coherent_reference"]["segments"]

    boundaries = []

    for seg in segments:
        chain = seg["label_chain"]
        start = int(seg["label_start"])
        end = int(seg["label_end"])

        start_rec = residues.get((chain, start))
        end_rec = residues.get((chain, end))
        if start_rec is None or end_rec is None:
            raise RuntimeError(f"Missing retained segment endpoint for {chain}:{start}-{end}")

        prev_rec, _ = find_prev_next(residues, chain, start)
        _, next_rec = find_prev_next(residues, chain, end)

        # N-side cut is prev C -- retained N
        n_natural = prev_rec is None
        n_dist = None
        if prev_rec is not None:
            if "C" in prev_rec["atoms"] and "N" in start_rec["atoms"]:
                n_dist = distance(prev_rec["atoms"]["C"], start_rec["atoms"]["N"])

        # C-side cut is retained C -- next N
        c_natural = next_rec is None
        c_dist = None
        if next_rec is not None:
            if "C" in end_rec["atoms"] and "N" in next_rec["atoms"]:
                c_dist = distance(end_rec["atoms"]["C"], next_rec["atoms"]["N"])

        boundaries.append(
            make_boundary("N", seg, start_rec, prev_rec, n_natural, n_dist)
        )
        boundaries.append(
            make_boundary("C", seg, end_rec, next_rec, c_natural, c_dist)
        )

    # High-level review categories
    special_review = []
    for b in boundaries:
        if (
            b["geometry_screen"] != "normal_peptide_geometry"
            or b["flags"]
            or b["natural_polymer_terminus"]
        ):
            special_review.append({
                "segment": b["segment"],
                "side": b["side"],
                "reason": (
                    (["natural_polymer_terminus"] if b["natural_polymer_terminus"] else [])
                    + ([b["geometry_screen"]] if b["geometry_screen"] != "normal_peptide_geometry" else [])
                    + b["flags"]
                ),
            })

    report = {
        "purpose":
            "Inspect peptide-coherent reference boundaries before cap construction",
        "canonical_receptor": network["canonical_receptor"],
        "n_segments": len(segments),
        "n_boundaries": len(boundaries),
        "boundaries": boundaries,
        "special_review": special_review,
        "important_principles": [
            "Catalytic Thr1 must retain its authentic N-terminal amino group.",
            "Do not choose cap chemistry merely to minimize atom count.",
            "Prefer preserving peptide-amide electronic structure at internal cuts.",
            "A cap is part of the model Hamiltonian and must be identical between the two ligand complexes.",
            "Cuts near Pro require extra scrutiny because proline peptide nitrogens are substituted.",
            "Ionizable side chains near segment boundaries require protonation review but are not automatically bad cut locations.",
            "This stage is diagnostic only; no cap atoms are created.",
        ],
        "next_step":
            "Review boundary report, adjust any awkward segment endpoints, then construct a first explicit capped receptor with a documented cap scheme.",
    }

    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")

    rows = []
    for b in boundaries:
        out = b["outside_neighbor"]
        rows.append({
            "segment": b["segment"],
            "side": b["side"],
            "retained": (
                f'{b["retained_endpoint"]["label_chain"]}:'
                f'{b["retained_endpoint"]["label_seq"]} '
                f'{b["retained_endpoint"]["comp_id"]}'
            ),
            "outside_neighbor": "" if out is None else (
                f'{out["label_chain"]}:{out["label_seq"]} {out["comp_id"]}'
            ),
            "natural_terminus": b["natural_polymer_terminus"],
            "peptide_CN_distance_A": (
                "" if b["peptide_CN_distance_A"] is None
                else b["peptide_CN_distance_A"]
            ),
            "geometry_screen": b["geometry_screen"],
            "flags": ";".join(b["flags"]),
            "cap_planning_note": b["cap_planning_note"],
        })

    with open(args.tsv, "w", newline="") as f:
        fields = [
            "segment", "side", "retained", "outside_neighbor",
            "natural_terminus", "peptide_CN_distance_A",
            "geometry_screen", "flags", "cap_planning_note",
        ]
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {args.out}")
    print(f"Wrote {args.tsv}")
    print()
    print("Stage-7a cut-boundary inspection")
    print(f"  segments:   {len(segments)}")
    print(f"  boundaries: {len(boundaries)}")
    print()

    for b in boundaries:
        out = b["outside_neighbor"]
        outtxt = "NATURAL TERMINUS" if out is None else (
            f'{out["label_chain"]}:{out["label_seq"]} {out["comp_id"]}'
        )
        d = b["peptide_CN_distance_A"]
        dtxt = "n/a" if d is None else f"{d:.3f} A"
        flags = ", ".join(b["flags"]) if b["flags"] else "-"
        print(
            f'  {b["segment"]:10s} {b["side"]}-side  '
            f'retained={b["retained_endpoint"]["label_chain"]}:'
            f'{b["retained_endpoint"]["label_seq"]} '
            f'{b["retained_endpoint"]["comp_id"]:3s}  '
            f'outside={outtxt:18s}  C-N={dtxt:8s}  '
            f'{b["geometry_screen"]:24s}  flags={flags}'
        )

    print()
    print("Special-review boundaries")
    for x in special_review:
        print(
            f'  {x["segment"]} {x["side"]}-side: '
            + ", ".join(x["reason"])
        )


if __name__ == "__main__":
    main()
