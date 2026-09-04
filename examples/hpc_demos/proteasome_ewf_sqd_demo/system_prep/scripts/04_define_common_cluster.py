#!/usr/bin/env python3
"""Stage 6: define and rank a common receptor-cluster candidate set.

This is a *screening / bookkeeping* step.  It does not protonate, cap, optimize,
or otherwise modify the authoritative receptor geometry.

The canonical scaffold is expected to be 5LF7 chain Y from Stage 5.  The script:

1. Places native 6V8 and the rigidly transplanted BO2 pose in the 5LF7 frame.
2. Builds the UNION of protein residues contacting either ligand at
   3.0/3.5/4.0/4.5/5.0 A.
3. Reports full-residue heavy-atom lower bounds for each cutoff.
4. Ranks residues by proximity and flags the catalytic Thr residue.
5. Inventories nearby 5LF7 crystallographic waters and gives a simple spatial
   conservation check against waters from 5LF3 after alignment.
6. Writes visualization-only PDB snapshots with the SAME receptor residue set
   for the native 6V8 and transplanted BO2 systems.

Important: whole-residue atom counts are deliberately conservative and are NOT
an instruction to keep whole residues.  Stage 6 ends by identifying which
chemically meaningful residue fragments are worth considering; capping,
protonation, and final all-atom budgets are later stages.

Example
-------
python scripts/04_define_common_cluster.py \
  raw/5LF3.cif raw/5LF7.cif \
  --alignment validation/alignment_report.json \
  --canonical validation/canonical_receptor_report.json \
  --stage4-script scripts/02_align_active_sites.py \
  --chain Y \
  --out validation/cluster_candidate_report.json \
  --tsv validation/cluster_candidates.tsv \
  --snapshots intermediate/cluster_design
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

CUTOFFS = (3.0, 3.5, 4.0, 4.5, 5.0)
WATER_NAMES = {"HOH", "WAT", "DOD"}


def load_stage4_module(path: Path):
    spec = importlib.util.spec_from_file_location("stage4_alignment", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import Stage-4 helpers from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def inverse_transform(r: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ri = r.T
    ti = -ri @ t
    return ri, ti


def transformed_copy(mod, atoms: Iterable[dict[str, Any]], r: np.ndarray, t: np.ndarray,
                     *, auth_chain: str | None = None, auth_seq: str | None = None) -> list[dict[str, Any]]:
    out = []
    for a in atoms:
        b = dict(a)
        b["xyz"] = mod.apply_xyz(a["xyz"], r, t)
        if auth_chain is not None:
            b["auth_chain"] = auth_chain
        if auth_seq is not None:
            b["auth_seq"] = auth_seq
        out.append(b)
    return out


def heavy(atoms: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [a for a in atoms if (a.get("element") or "").upper() not in {"H", "D"}]


def protein_groups(mod, atoms: Iterable[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for a in mod.protein_atoms(atoms):
        if a.get("label_chain") is None or a.get("label_seq") is None:
            continue
        groups[(a["label_chain"], a["label_seq"])].append(a)
    # Dedupe atom names within each residue defensively.
    return {
        k: list(mod.dedupe(v, lambda a: a["atom"]).values())
        for k, v in groups.items()
    }


def water_atoms(mod, atoms: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    aa = [
        a for a in atoms
        if a.get("group") == "HETATM"
        and (a.get("comp") or "").upper() in WATER_NAMES
        and (a.get("element") or "").upper() == "O"
    ]
    return list(mod.dedupe(aa, lambda a: (a.get("label_chain"), a.get("auth_chain"), a.get("auth_seq"), a.get("atom"))).values())


def closest_pair(res_atoms: Iterable[dict[str, Any]], lig_atoms: Iterable[dict[str, Any]]) -> dict[str, Any] | None:
    rr = heavy(res_atoms)
    ll = heavy(lig_atoms)
    best = None
    for a in rr:
        for b in ll:
            d = float(np.linalg.norm(a["xyz"] - b["xyz"]))
            if best is None or d < best[0]:
                best = (d, a, b)
    if best is None:
        return None
    d, a, b = best
    return {
        "distance_A": round(d, 4),
        "receptor_atom": a.get("atom"),
        "receptor_element": a.get("element"),
        "ligand_atom": b.get("atom"),
        "ligand_element": b.get("element"),
    }


def min_distance(res_atoms: Iterable[dict[str, Any]], lig_atoms: Iterable[dict[str, Any]]) -> float:
    p = closest_pair(res_atoms, lig_atoms)
    return p["distance_A"] if p else math.inf


def tier(d: float, catalytic: bool) -> str:
    if catalytic:
        return "0_catalytic"
    if d <= 3.0:
        return "1_within_3.0A"
    if d <= 3.5:
        return "2_within_3.5A"
    if d <= 4.0:
        return "3_within_4.0A"
    if d <= 4.5:
        return "4_within_4.5A"
    return "5_within_5.0A"


def residue_record(key: tuple[str, str], aa: list[dict[str, Any]], lig6, ligbo2, chain: str) -> dict[str, Any]:
    p6 = closest_pair(aa, lig6)
    pb = closest_pair(aa, ligbo2)
    d6 = p6["distance_A"] if p6 else math.inf
    db = pb["distance_A"] if pb else math.inf
    dmin = min(d6, db)
    ex = aa[0]
    catalytic = key == (chain, "1")
    return {
        "label_chain": key[0],
        "label_seq": key[1],
        "auth_chain": ex.get("auth_chain"),
        "auth_seq": ex.get("auth_seq"),
        "comp_id": ex.get("label_comp") or ex.get("comp"),
        "full_residue_heavy_atoms": len(heavy(aa)),
        "min_to_native_6V8_A": round(d6, 4) if math.isfinite(d6) else None,
        "min_to_transplanted_BO2_A": round(db, 4) if math.isfinite(db) else None,
        "min_to_either_A": round(dmin, 4),
        "closest_native_6V8_pair": p6,
        "closest_transplanted_BO2_pair": pb,
        "contacts_both_within_4A": bool(d6 <= 4.0 and db <= 4.0),
        "catalytic_thr": catalytic,
        "priority_tier": tier(dmin, catalytic),
    }


def nearest_transformed_water(mod, w7: dict[str, Any], waters3: list[dict[str, Any]], r3to7, t3to7) -> dict[str, Any] | None:
    best = None
    for w3 in waters3:
        p = mod.apply_xyz(w3["xyz"], r3to7, t3to7)
        d = float(np.linalg.norm(w7["xyz"] - p))
        if best is None or d < best[0]:
            best = (d, w3)
    if best is None:
        return None
    d, w3 = best
    return {
        "nearest_5LF3_water_distance_A": round(d, 4),
        "5LF3_auth_chain": w3.get("auth_chain"),
        "5LF3_auth_seq": w3.get("auth_seq"),
        "spatially_conserved_le_1.0A": d <= 1.0,
    }


def write_tsv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "priority_tier", "label_chain", "label_seq", "auth_chain", "auth_seq", "comp_id",
        "full_residue_heavy_atoms", "min_to_native_6V8_A", "min_to_transplanted_BO2_A",
        "min_to_either_A", "contacts_both_within_4A", "catalytic_thr",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        for row in records:
            w.writerow(row)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cif_5lf3", type=Path)
    ap.add_argument("cif_5lf7", type=Path)
    ap.add_argument("--alignment", type=Path, default=Path("validation/alignment_report.json"))
    ap.add_argument("--canonical", type=Path, default=Path("validation/canonical_receptor_report.json"))
    ap.add_argument("--stage4-script", type=Path, default=Path("scripts/02_align_active_sites.py"))
    ap.add_argument("--chain", default="Y")
    ap.add_argument("--out", type=Path, default=Path("validation/cluster_candidate_report.json"))
    ap.add_argument("--tsv", type=Path, default=Path("validation/cluster_candidates.tsv"))
    ap.add_argument("--snapshots", type=Path, default=Path("intermediate/cluster_design"))
    ap.add_argument(
        "--planning-bortezomib-total-atoms", type=int, default=53,
        help="Planning estimate only; used to display the approximate receptor/cap/water budget under a 100-atom complex cap.",
    )
    args = ap.parse_args()

    mod = load_stage4_module(args.stage4_script)
    alignment = load_json(args.alignment)
    canonical = load_json(args.canonical)

    # Re-verify that the raw coordinate files are the same immutable inputs used in Stage 4.
    expected3 = alignment.get("reference", {}).get("sha256")
    expected7 = alignment.get("mobile", {}).get("sha256")
    actual3 = mod.sha256(args.cif_5lf3)
    actual7 = mod.sha256(args.cif_5lf7)
    if expected3 and actual3 != expected3:
        raise RuntimeError(f"5LF3 SHA256 mismatch: {actual3} != Stage-4 {expected3}")
    if expected7 and actual7 != expected7:
        raise RuntimeError(f"5LF7 SHA256 mismatch: {actual7} != Stage-4 {expected7}")

    rec = canonical.get("recommendation", {}).get("recommended_entry")
    cchain = canonical.get("canonical_beta5_copy")
    if rec != "5LF7":
        raise RuntimeError(f"Stage-5 report does not freeze 5LF7 as canonical receptor (found {rec!r}).")
    if cchain != args.chain:
        raise RuntimeError(f"Stage-5 canonical beta5 copy is {cchain!r}, but --chain={args.chain!r}.")
    if args.chain not in alignment.get("chains", {}):
        raise RuntimeError(f"Chain {args.chain} not present in Stage-4 alignment report.")

    _b3, a3_all = mod.read_atoms(args.cif_5lf3)
    _b7, a7_all = mod.read_atoms(args.cif_5lf7)
    a3 = mod.model1(a3_all)
    a7 = mod.model1(a7_all)

    bo2_3 = mod.ligand_atoms(a3, "BO2", args.chain)
    v8_7 = mod.ligand_atoms(a7, "6V8", args.chain)

    tr = alignment["chains"][args.chain]["transform_5LF7_to_5LF3"]
    r7to3 = np.array(tr["rotation"], dtype=float)
    t7to3 = np.array(tr["translation_A"], dtype=float)
    r3to7, t3to7 = inverse_transform(r7to3, t7to3)
    bo2_7 = transformed_copy(mod, bo2_3, r3to7, t3to7, auth_chain="Q", auth_seq="901")

    groups7 = protein_groups(mod, a7)
    records_all = [residue_record(k, aa, v8_7, bo2_7, args.chain) for k, aa in groups7.items()]
    records = [r for r in records_all if r["min_to_either_A"] <= 5.0]
    records.sort(key=lambda r: (r["priority_tier"], r["min_to_either_A"], r["label_chain"], int(r["label_seq"]) if str(r["label_seq"]).isdigit() else 999999))

    cutoff_summary = {}
    for cutoff in CUTOFFS:
        rr = [r for r in records if r["min_to_either_A"] <= cutoff]
        cutoff_summary[f"{cutoff:.1f}"] = {
            "union_residue_count": len(rr),
            "full_residue_heavy_atom_lower_bound": sum(r["full_residue_heavy_atoms"] for r in rr),
            "residues": [f"{r['label_chain']}:{r['label_seq']}:{r['comp_id']}" for r in rr],
        }

    # Nearby canonical waters. These remain candidates only; hydrogen orientation and
    # chemical relevance are explicitly deferred.
    waters7 = water_atoms(mod, a7)
    waters3 = water_atoms(mod, a3)
    water_rows = []
    for w in waters7:
        p6 = closest_pair([w], v8_7)
        pb = closest_pair([w], bo2_7)
        d6 = p6["distance_A"] if p6 else math.inf
        db = pb["distance_A"] if pb else math.inf
        if min(d6, db) > 5.0:
            continue
        conservation = nearest_transformed_water(mod, w, waters3, r3to7, t3to7)
        water_rows.append({
            "5LF7_auth_chain": w.get("auth_chain"),
            "5LF7_auth_seq": w.get("auth_seq"),
            "5LF7_label_chain": w.get("label_chain"),
            "min_to_native_6V8_A": round(d6, 4) if math.isfinite(d6) else None,
            "min_to_transplanted_BO2_A": round(db, 4) if math.isfinite(db) else None,
            "min_to_either_A": round(min(d6, db), 4),
            "nearest_5LF3_water": conservation,
        })
    water_rows.sort(key=lambda r: r["min_to_either_A"])

    max_receptor_est = 100 - args.planning_bortezomib_total_atoms
    budget = {
        "complex_atom_cap": 100,
        "planning_bortezomib_total_atoms_estimate": args.planning_bortezomib_total_atoms,
        "approximate_receptor_plus_caps_plus_waters_capacity": max_receptor_est,
        "warning": (
            "The 47-ish atom capacity is a planning estimate, not a final count. "
            "Deposited CIFs omit hydrogens; final receptor fragments, caps, waters, protonation, and ligand chemistry must be counted explicitly later."
        ),
        "whole_residue_heavy_atom_test": {
            c: {
                "heavy_atoms": x["full_residue_heavy_atom_lower_bound"],
                "already_impossible_under_planning_all_atom_receptor_capacity": x["full_residue_heavy_atom_lower_bound"] > max_receptor_est,
            }
            for c, x in cutoff_summary.items()
        },
    }

    # Visualization snapshots: SAME receptor union at 5 A for both ligand poses.
    keys5 = {(r["label_chain"], r["label_seq"]) for r in records}
    receptor5 = [a for k in keys5 for a in groups7[k]]
    water5 = [
        w for w in waters7
        if any(
            w.get("auth_chain") == row["5LF7_auth_chain"] and w.get("auth_seq") == row["5LF7_auth_seq"]
            for row in water_rows
        )
    ]
    args.snapshots.mkdir(parents=True, exist_ok=True)
    snap_v8 = args.snapshots / "5LF7_Y_union5A_native_6V8.pdb"
    snap_bo2 = args.snapshots / "5LF7_Y_union5A_transplanted_BO2.pdb"
    mod.write_pdb_snapshot(
        snap_v8,
        receptor5 + water5 + v8_7,
        remark="Stage-6 visualization: canonical 5LF7-Y 5A union receptor + nearby waters + native 6V8; no chemistry modification",
    )
    mod.write_pdb_snapshot(
        snap_bo2,
        receptor5 + water5 + bo2_7,
        remark="Stage-6 visualization: SAME canonical 5LF7-Y 5A union receptor + nearby waters + rigidly transplanted BO2; no chemistry modification",
    )

    report = {
        "purpose": "Stage-6 common receptor-cluster candidate inventory; no capping/protonation/optimization",
        "canonical_receptor": {"entry_id": "5LF7", "beta5_copy": args.chain, "sha256": actual7},
        "source_5LF3_sha256": actual3,
        "ligand_poses_in_canonical_frame": {
            "6V8": "native deposited 5LF7 pose",
            "BO2": "5LF3 pose rigidly transformed into the 5LF7 frame using the inverse Stage-4 alignment transform",
        },
        "definition": "Union of canonical 5LF7 protein residues contacting either ligand pose at each cutoff.",
        "cutoff_summary": cutoff_summary,
        "ranked_residue_candidates": records,
        "water_candidates": {
            "note": "Canonical 5LF7 crystallographic waters within 5 A of either ligand pose. Spatial conservation to transformed 5LF3 waters is a screening clue only, not a retention decision.",
            "count": len(water_rows),
            "rows": water_rows,
        },
        "planning_atom_budget": budget,
        "selection_guidance": [
            "Catalytic Thr must be represented with chemically correct N-terminal/covalent-boron chemistry.",
            "Do not keep whole residues merely because they fall within a distance cutoff; whole-residue counts are lower-bound stress tests.",
            "Prioritize contacts shared by both ligands and chemically specific polar/covalent interactions before distal hydrophobic shell atoms.",
            "A conserved crystallographic water can be worth more than a distal residue fragment, but water hydrogens/orientation must be resolved later.",
            "Every protein covalent cut will require an explicit, identical capping rule in both complexes.",
            "Do not finalize the cluster until protonation, total charge, caps, and all-atom counts are explicit.",
        ],
        "visualization_snapshots": {
            "native_6V8": str(snap_v8),
            "transplanted_BO2": str(snap_bo2),
            "note": "The receptor residue and water sets are identical between these visualization snapshots; only the ligand differs.",
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    write_tsv(args.tsv, records)

    print(f"Wrote {args.out}")
    print(f"Wrote {args.tsv}")
    print("\nStage-6 common-cluster screening")
    print(f"  canonical receptor = 5LF7 chain {args.chain}")
    print(f"  planning receptor+caps+waters capacity ~ {max_receptor_est} atoms (100 - {args.planning_bortezomib_total_atoms})")
    for c in CUTOFFS:
        x = cutoff_summary[f"{c:.1f}"]
        print(f"  union <= {c:.1f} A: {x['union_residue_count']:2d} residues, {x['full_residue_heavy_atom_lower_bound']:3d} full-residue heavy atoms")
    print(f"  candidate crystallographic waters <=5 A of either pose: {len(water_rows)}")
    print("\nHighest-priority residues")
    for r in records[:12]:
        print(
            f"  {r['priority_tier']:<16s} {r['label_chain']}:{r['label_seq']} {r['comp_id']:<3s} "
            f"d6V8={r['min_to_native_6V8_A']:.3f} A  dBO2={r['min_to_transplanted_BO2_A']:.3f} A  "
            f"heavy={r['full_residue_heavy_atoms']}"
        )
    print("\nThis report ranks candidates; it does NOT yet choose final residue fragments or caps.")


if __name__ == "__main__":
    main()
