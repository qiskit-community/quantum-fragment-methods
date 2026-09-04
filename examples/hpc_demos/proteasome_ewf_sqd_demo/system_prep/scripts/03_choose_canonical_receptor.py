#!/usr/bin/env python3
"""Stage 5: compare 5LF3-Y vs 5LF7-Y as the canonical receptor geometry.

This script does NOT modify the deposited structures. It uses the rigid transform
from Stage 4 and asks which receptor can host the *other* inhibitor pose with the
least geometric distortion.

Primary diagnostics
-------------------
1. reciprocal hard-clash counts (from Stage 4),
2. catalytic Thr-OG1--B26 bond-length distortion after transplantation,
3. residue-contact distortion for the transplanted ligand over the 5 A pocket,
4. crystallographic resolution only as a tie-breaker.

Example
-------
python scripts/03_choose_canonical_receptor.py \
  raw/5LF3.cif raw/5LF7.cif \
  --alignment validation/alignment_report.json \
  --inventory validation/structure_inventory.json \
  --chain Y \
  --out validation/canonical_receptor_report.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def load_stage4_module(path: Path):
    spec = importlib.util.spec_from_file_location("stage4_align", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load Stage-4 helpers from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def resolution_map(inventory: dict[str, Any]) -> dict[str, float | None]:
    return {e["entry_id"]: e.get("resolution_A") for e in inventory.get("entries", [])}


def residue_groups(mod, atoms: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    out: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for a in mod.protein_atoms(atoms):
        if a["label_chain"] is not None and a["label_seq"] is not None:
            out[(a["label_chain"], a["label_seq"])].append(a)
    return out


def transformed_ligand_xyz(mod, lig: list[dict[str, Any]], r: np.ndarray | None, t: np.ndarray | None) -> np.ndarray:
    xyz = []
    for a in mod.heavy(lig):
        p = a["xyz"]
        if r is not None and t is not None:
            p = mod.apply_xyz(p, r, t)
        xyz.append(p)
    return np.array(xyz)


def transformed_residue_min_distance(mod, atoms: list[dict[str, Any]], lig_xyz: np.ndarray,
                                     r: np.ndarray | None = None, t: np.ndarray | None = None) -> float:
    vals: list[float] = []
    for a in mod.heavy(atoms):
        p = a["xyz"]
        if r is not None and t is not None:
            p = mod.apply_xyz(p, r, t)
        vals.append(float(np.linalg.norm(lig_xyz - p, axis=1).min()))
    return min(vals) if vals else math.inf


def contact_distortion(mod, *, native_receptor, host_receptor, ligand,
                       residue_keys: set[tuple[str, str]],
                       ligand_transform: tuple[np.ndarray, np.ndarray] | None = None,
                       host_transform: tuple[np.ndarray, np.ndarray] | None = None,
                       native_cutoff: float = 5.0) -> dict[str, Any]:
    native_groups = residue_groups(mod, native_receptor)
    host_groups = residue_groups(mod, host_receptor)

    native_lig_xyz = transformed_ligand_xyz(mod, ligand, None, None)
    lr, lt = ligand_transform if ligand_transform else (None, None)
    transplanted_lig_xyz = transformed_ligand_xyz(mod, ligand, lr, lt)
    hr, ht = host_transform if host_transform else (None, None)

    rows = []
    deltas = []
    for key in sorted(residue_keys):
        if key not in native_groups or key not in host_groups:
            continue
        d_native = transformed_residue_min_distance(mod, native_groups[key], native_lig_xyz)
        if d_native > native_cutoff:
            continue
        d_host = transformed_residue_min_distance(mod, host_groups[key], transplanted_lig_xyz, hr, ht)
        delta = d_host - d_native
        rows.append({
            "label_chain": key[0],
            "label_seq": key[1],
            "native_min_A": round(d_native, 4),
            "transplanted_min_A": round(d_host, 4),
            "delta_A": round(delta, 4),
        })
        deltas.append(delta)

    arr = np.array(deltas, dtype=float)
    return {
        "residue_count": len(rows),
        "mean_absolute_delta_A": round(float(np.mean(np.abs(arr))), 4) if len(arr) else None,
        "rms_delta_A": round(float(np.sqrt(np.mean(arr**2))), 4) if len(arr) else None,
        "max_absolute_delta_A": round(float(np.max(np.abs(arr))), 4) if len(arr) else None,
        "per_residue": rows,
    }


def inverse_transform(r: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ri = r.T
    ti = -ri @ t
    return ri, ti


def recommendation(c3: dict[str, Any], c7: dict[str, Any], res: dict[str, float | None]) -> dict[str, Any]:
    """Conservative, transparent heuristic. No hidden weighted score."""
    reasons = []

    h3 = c3["hard_clashes_lt_1.8A"]
    h7 = c7["hard_clashes_lt_1.8A"]
    if h3 != h7:
        pick = "5LF3" if h3 < h7 else "5LF7"
        reasons.append(f"{pick} has fewer <1.8 A nonbonded heavy-atom clashes after transplantation.")
        return {"recommended_entry": pick, "basis": reasons, "confidence": "high"}

    b3 = c3["transplanted_thr_B26_deviation_from_donor_native_A"]
    b7 = c7["transplanted_thr_B26_deviation_from_donor_native_A"]
    m3 = c3["contact_distortion"]["mean_absolute_delta_A"]
    m7 = c7["contact_distortion"]["mean_absolute_delta_A"]

    # A >0.10 A advantage in the covalent geometry is considered material at this screening stage.
    if abs(b3 - b7) > 0.10:
        pick = "5LF3" if b3 < b7 else "5LF7"
        reasons.append(f"{pick} preserves the transplanted Thr-OG1--B26 distance materially better.")
        return {"recommended_entry": pick, "basis": reasons, "confidence": "moderate"}

    # A >0.05 A advantage in mean contact-distance distortion is a useful secondary discriminator.
    if m3 is not None and m7 is not None and abs(m3 - m7) > 0.05:
        pick = "5LF3" if m3 < m7 else "5LF7"
        reasons.append(f"{pick} better preserves the donor ligand's native 5 A residue-contact pattern.")
        return {"recommended_entry": pick, "basis": reasons, "confidence": "moderate"}

    r3, r7 = res.get("5LF3"), res.get("5LF7")
    if r3 is not None and r7 is not None and r3 != r7:
        pick = "5LF3" if r3 < r7 else "5LF7"
        reasons.append("The transplantation metrics are effectively tied at screening resolution.")
        reasons.append(f"{pick} is used as a deterministic tie-breaker because its deposited resolution is slightly higher ({min(r3,r7):.2f} A vs {max(r3,r7):.2f} A).")
        return {"recommended_entry": pick, "basis": reasons, "confidence": "low-to-moderate"}

    return {
        "recommended_entry": None,
        "basis": ["The two receptor candidates remain effectively indistinguishable by the Stage-5 screening metrics."],
        "confidence": "undetermined",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cif_5lf3", type=Path)
    ap.add_argument("cif_5lf7", type=Path)
    ap.add_argument("--alignment", type=Path, default=Path("validation/alignment_report.json"))
    ap.add_argument("--inventory", type=Path, default=Path("validation/structure_inventory.json"))
    ap.add_argument("--stage4-script", type=Path, default=Path("scripts/02_align_active_sites.py"))
    ap.add_argument("--chain", default="Y")
    ap.add_argument("--pocket-cutoff", type=float, default=5.0)
    ap.add_argument("--out", type=Path, default=Path("validation/canonical_receptor_report.json"))
    args = ap.parse_args()

    mod = load_stage4_module(args.stage4_script)
    alignment = load_json(args.alignment)
    inventory = load_json(args.inventory)
    chain = args.chain

    if chain not in alignment.get("chains", {}):
        raise RuntimeError(f"Chain {chain} not found in {args.alignment}")

    _b3, a3_all = mod.read_atoms(args.cif_5lf3)
    _b7, a7_all = mod.read_atoms(args.cif_5lf7)
    a3 = mod.model1(a3_all)
    a7 = mod.model1(a7_all)
    bo2 = mod.ligand_atoms(a3, "BO2", chain)
    v8 = mod.ligand_atoms(a7, "6V8", chain)

    ch = alignment["chains"][chain]
    tr = ch["transform_5LF7_to_5LF3"]
    r = np.array(tr["rotation"], dtype=float)
    t = np.array(tr["translation_A"], dtype=float)
    ri, ti = inverse_transform(r, t)

    thr3 = mod.catalytic_thr_og1(a3, chain, bo2)
    thr7 = mod.catalytic_thr_og1(a7, chain, v8)
    b3 = mod.find_atom(bo2, "B26")
    b7 = mod.find_atom(v8, "B26")

    native_bo2_bond = mod.dist(thr3["xyz"], b3["xyz"])
    native_v8_bond = mod.dist(thr7["xyz"], b7["xyz"])

    # Candidate 5LF3 host: transplant 6V8 from 5LF7 into 5LF3 frame.
    b7_into_3 = mod.apply_xyz(b7["xyz"], r, t)
    v8_into_3_bond = mod.dist(thr3["xyz"], b7_into_3)

    # Candidate 5LF7 host: transplant BO2 from 5LF3 into native 5LF7 frame.
    b3_into_7 = mod.apply_xyz(b3["xyz"], ri, ti)
    bo2_into_7_bond = mod.dist(thr7["xyz"], b3_into_7)

    env3 = mod.environment_residues(a3, bo2, args.pocket_cutoff)
    env7 = mod.environment_residues(a7, v8, args.pocket_cutoff)
    env_union = env3 | env7

    contacts_v8_into_3 = contact_distortion(
        mod,
        native_receptor=a7,
        host_receptor=a3,
        ligand=v8,
        residue_keys=env_union,
        ligand_transform=(r, t),
        host_transform=None,
        native_cutoff=args.pocket_cutoff,
    )
    contacts_bo2_into_7 = contact_distortion(
        mod,
        native_receptor=a3,
        host_receptor=a7,
        ligand=bo2,
        residue_keys=env_union,
        ligand_transform=(ri, ti),
        host_transform=None,
        native_cutoff=args.pocket_cutoff,
    )

    cl = ch["transplant_clashes"]
    c3_clash = cl["6V8_aligned_into_5LF3_receptor"]
    c7_clash = cl["BO2_into_aligned_5LF7_receptor"]

    candidates = {
        "5LF3": {
            "host_chain": chain,
            "native_ligand": "BO2",
            "transplanted_ligand": "6V8",
            "hard_clashes_lt_1.8A": c3_clash["counts"]["lt_1.8A"],
            "closest_nonbonded_heavy_pair_A": c3_clash["closest_nonbonded_heavy_pair"]["distance_A"] if c3_clash.get("closest_nonbonded_heavy_pair") else None,
            "donor_native_thr_B26_A": round(native_v8_bond, 4),
            "transplanted_thr_B26_A": round(v8_into_3_bond, 4),
            "transplanted_thr_B26_deviation_from_donor_native_A": round(abs(v8_into_3_bond - native_v8_bond), 4),
            "contact_distortion": contacts_v8_into_3,
        },
        "5LF7": {
            "host_chain": chain,
            "native_ligand": "6V8",
            "transplanted_ligand": "BO2",
            "hard_clashes_lt_1.8A": c7_clash["counts"]["lt_1.8A"],
            "closest_nonbonded_heavy_pair_A": c7_clash["closest_nonbonded_heavy_pair"]["distance_A"] if c7_clash.get("closest_nonbonded_heavy_pair") else None,
            "donor_native_thr_B26_A": round(native_bo2_bond, 4),
            "transplanted_thr_B26_A": round(bo2_into_7_bond, 4),
            "transplanted_thr_B26_deviation_from_donor_native_A": round(abs(bo2_into_7_bond - native_bo2_bond), 4),
            "contact_distortion": contacts_bo2_into_7,
        },
    }

    res = resolution_map(inventory)
    rec = recommendation(candidates["5LF3"], candidates["5LF7"], res)

    report = {
        "purpose": "Stage-5 canonical receptor screening; no coordinate modification",
        "canonical_beta5_copy": chain,
        "pocket_cutoff_A": args.pocket_cutoff,
        "source_resolution_A": {"5LF3": res.get("5LF3"), "5LF7": res.get("5LF7")},
        "stage4_context": {
            "local_environment_heavy_rmsd_A": ch["rmsd_A"].get("matched_local_environment_heavy_atoms_after_global_fit"),
            "aligned_B26_displacement_A": ch["covalent_warhead_geometry"].get("aligned_B26_displacement_A"),
            "warhead_vector_angle_deg": ch["covalent_warhead_geometry"].get("ThrOG1_to_B26_vector_angle_deg"),
        },
        "candidates": candidates,
        "recommendation": rec,
        "selection_rule": [
            "Reject a host if reciprocal transplantation creates severe hard clashes.",
            "Prefer a host that preserves the donor ligand's catalytic Thr-OG1--B26 geometry.",
            "Then prefer smaller distortion of the donor ligand's native residue-contact pattern.",
            "Use deposited resolution only as a deterministic tie-breaker when transplantation metrics are effectively tied.",
            "Do not interpret a tie-break selection as evidence that the other receptor is biologically inferior.",
        ],
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")

    print(f"Wrote {args.out}")
    print("\nStage-5 canonical receptor screening")
    for eid in ("5LF3", "5LF7"):
        c = candidates[eid]
        print(
            f"  {eid}: clashes<1.8A={c['hard_clashes_lt_1.8A']}, "
            f"Thr-B transplant deviation={c['transplanted_thr_B26_deviation_from_donor_native_A']:.3f} A, "
            f"contact MAE={c['contact_distortion']['mean_absolute_delta_A']:.3f} A"
        )
    print(f"  recommendation={rec['recommended_entry']} ({rec['confidence']})")
    for reason in rec["basis"]:
        print(f"    - {reason}")


if __name__ == "__main__":
    main()
