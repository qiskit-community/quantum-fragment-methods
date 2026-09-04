#!/usr/bin/env python3
"""Immutable structural inventory for the proteasome challenge.

Usage:
    python scripts/01_inventory.py raw/5LF3.cif raw/5LF7.cif \
        --out validation/structure_inventory.json

The script reads deposited mmCIF data only. It does not modify coordinates,
add atoms, infer protonation, or repair chemistry.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import gemmi

TARGETS = {"5LF3": "BO2", "5LF7": "6V8"}
CUTOFFS = (3.0, 3.5, 4.0, 4.5, 5.0)
WATER_NAMES = {"HOH", "WAT", "DOD"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def norm(v: str | None) -> str | None:
    if v is None or v in {".", "?", ""}:
        return None
    return v


def col(block: gemmi.cif.Block, tag: str) -> list[str]:
    return [str(x) for x in block.find_values(tag)]


def rows(block: gemmi.cif.Block, tags: list[str]) -> list[dict[str, str | None]]:
    columns = [col(block, t) for t in tags]
    n = max((len(c) for c in columns), default=0)
    out = []
    for i in range(n):
        out.append({t: norm(columns[j][i]) if i < len(columns[j]) else None
                    for j, t in enumerate(tags)})
    return out


def fnum(x: str | None) -> float | None:
    if x is None:
        return None
    try:
        return float(x.split("(")[0])
    except ValueError:
        return None


def atom_rows(block: gemmi.cif.Block) -> list[dict[str, Any]]:
    tags = [
        "_atom_site.group_PDB", "_atom_site.id", "_atom_site.type_symbol",
        "_atom_site.label_atom_id", "_atom_site.label_alt_id",
        "_atom_site.label_comp_id", "_atom_site.label_asym_id",
        "_atom_site.label_seq_id", "_atom_site.auth_atom_id",
        "_atom_site.auth_comp_id", "_atom_site.auth_asym_id",
        "_atom_site.auth_seq_id", "_atom_site.Cartn_x", "_atom_site.Cartn_y",
        "_atom_site.Cartn_z", "_atom_site.occupancy",
        "_atom_site.pdbx_PDB_model_num",
    ]
    raw = rows(block, tags)
    atoms = []
    for r in raw:
        try:
            xyz = (float(r["_atom_site.Cartn_x"]), float(r["_atom_site.Cartn_y"]), float(r["_atom_site.Cartn_z"]))
        except (TypeError, ValueError):
            continue
        atoms.append({
            "group": r["_atom_site.group_PDB"],
            "id": r["_atom_site.id"],
            "element": r["_atom_site.type_symbol"],
            "atom": r["_atom_site.auth_atom_id"] or r["_atom_site.label_atom_id"],
            "alt": r["_atom_site.label_alt_id"],
            "comp": r["_atom_site.auth_comp_id"] or r["_atom_site.label_comp_id"],
            "label_comp": r["_atom_site.label_comp_id"],
            "label_chain": r["_atom_site.label_asym_id"],
            "label_seq": r["_atom_site.label_seq_id"],
            "auth_chain": r["_atom_site.auth_asym_id"],
            "auth_seq": r["_atom_site.auth_seq_id"],
            "xyz": xyz,
            "occ": fnum(r["_atom_site.occupancy"]),
            "model": r["_atom_site.pdbx_PDB_model_num"],
        })
    return atoms


def residue_key(a: dict[str, Any]) -> tuple[str | None, str | None, str | None, str | None]:
    return (a["auth_chain"], a["auth_seq"], a["comp"], a["label_chain"])


def dist(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.dist(a, b)


def beta5_chains(block: gemmi.cif.Block, atoms: list[dict[str, Any]]) -> dict[str, Any]:
    ent = rows(block, ["_entity.id", "_entity.pdbx_description"])
    beta_entities = {
        r["_entity.id"] for r in ent
        if r["_entity.pdbx_description"] and "proteasome subunit beta type-5" in r["_entity.pdbx_description"].lower()
    }
    asym = rows(block, ["_struct_asym.id", "_struct_asym.entity_id"])
    labels = sorted({r["_struct_asym.id"] for r in asym if r["_struct_asym.entity_id"] in beta_entities})
    auth_map: dict[str, set[str]] = defaultdict(set)
    for a in atoms:
        if a["label_chain"] in labels and a["auth_chain"]:
            auth_map[a["label_chain"]].add(a["auth_chain"])
    return {
        "entity_ids": sorted(x for x in beta_entities if x),
        "label_asym_ids": labels,
        "auth_asym_ids": sorted({x for s in auth_map.values() for x in s}),
        "label_to_auth": {k: sorted(v) for k, v in auth_map.items()},
    }


def ligand_instances(atoms: list[dict[str, Any]], ligand: str) -> list[dict[str, Any]]:
    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for a in atoms:
        if a["comp"] == ligand or a["label_comp"] == ligand:
            groups[residue_key(a)].append(a)
    out = []
    for key, aa in sorted(groups.items(), key=lambda kv: str(kv[0])):
        heavy = [a for a in aa if (a["element"] or "").upper() not in {"H", "D"}]
        out.append({
            "auth_chain": key[0], "auth_seq": key[1], "comp_id": key[2], "label_chain": key[3],
            "deposited_atom_count": len(aa),
            "deposited_heavy_atom_count": len(heavy),
            "elements": dict(sorted(_counts(a["element"] for a in aa).items())),
            "alt_ids": sorted({a["alt"] for a in aa if a["alt"]}),
            "occupancy_min": min((a["occ"] for a in aa if a["occ"] is not None), default=None),
            "occupancy_max": max((a["occ"] for a in aa if a["occ"] is not None), default=None),
            "atom_ids": [a["id"] for a in aa],
        })
    return out


def _counts(xs: Iterable[str | None]) -> dict[str, int]:
    d: dict[str, int] = defaultdict(int)
    for x in xs:
        d[x or "?"] += 1
    return dict(d)


def neighborhoods(atoms: list[dict[str, Any]], ligand: str) -> list[dict[str, Any]]:
    lig_groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for a in atoms:
        if a["comp"] == ligand or a["label_comp"] == ligand:
            lig_groups[residue_key(a)].append(a)

    result = []
    for lk, lig_atoms in lig_groups.items():
        lig_xyz = [a["xyz"] for a in lig_atoms]
        per_cutoff = {}
        for cutoff in CUTOFFS:
            residues: dict[tuple, dict[str, Any]] = {}
            waters: dict[tuple, dict[str, Any]] = {}
            for a in atoms:
                rk = residue_key(a)
                if rk == lk:
                    continue
                mind = min(dist(a["xyz"], p) for p in lig_xyz)
                if mind > cutoff:
                    continue
                entry = {
                    "auth_chain": a["auth_chain"], "auth_seq": a["auth_seq"], "comp_id": a["comp"],
                    "label_chain": a["label_chain"], "min_distance_A": round(mind, 3)
                }
                target = waters if a["comp"] in WATER_NAMES else residues
                if rk not in target or mind < target[rk]["min_distance_A"]:
                    target[rk] = entry
            per_cutoff[f"{cutoff:.1f}"] = {
                "residues": sorted(residues.values(), key=lambda x: x["min_distance_A"]),
                "waters": sorted(waters.values(), key=lambda x: x["min_distance_A"]),
            }
        result.append({
            "ligand_instance": {"auth_chain": lk[0], "auth_seq": lk[1], "label_chain": lk[3]},
            "cutoffs_A": per_cutoff,
        })
    return result


def struct_connections(block: gemmi.cif.Block, ligand: str) -> list[dict[str, Any]]:
    tags = [
        "_struct_conn.id", "_struct_conn.conn_type_id",
        "_struct_conn.ptnr1_label_asym_id", "_struct_conn.ptnr1_auth_asym_id",
        "_struct_conn.ptnr1_label_comp_id", "_struct_conn.ptnr1_auth_seq_id",
        "_struct_conn.ptnr1_label_atom_id",
        "_struct_conn.ptnr2_label_asym_id", "_struct_conn.ptnr2_auth_asym_id",
        "_struct_conn.ptnr2_label_comp_id", "_struct_conn.ptnr2_auth_seq_id",
        "_struct_conn.ptnr2_label_atom_id", "_struct_conn.pdbx_dist_value",
    ]
    out = []
    for r in rows(block, tags):
        if ligand not in {r["_struct_conn.ptnr1_label_comp_id"], r["_struct_conn.ptnr2_label_comp_id"]}:
            continue
        out.append({k.removeprefix("_struct_conn."): v for k, v in r.items()})
    return out


def occupancy_anomalies(atoms: list[dict[str, Any]], ligand: str, near_A: float = 5.0) -> dict[str, Any]:
    lig_xyz = [a["xyz"] for a in atoms if a["comp"] == ligand or a["label_comp"] == ligand]
    partial, alt = [], []
    for a in atoms:
        near = bool(lig_xyz) and min(dist(a["xyz"], p) for p in lig_xyz) <= near_A
        rec = {k: a[k] for k in ("id", "atom", "comp", "auth_chain", "auth_seq", "label_chain", "alt", "occ")}
        if a["occ"] is not None and a["occ"] < 0.999:
            partial.append({**rec, "within_5A_of_ligand": near})
        if a["alt"]:
            alt.append({**rec, "within_5A_of_ligand": near})
    return {"partial_occupancy_atoms": partial, "alternate_conformer_atoms": alt}


def unobserved(block: gemmi.cif.Block) -> dict[str, Any]:
    residue_tags = [
        "_pdbx_unobs_or_zero_occ_residues.auth_asym_id",
        "_pdbx_unobs_or_zero_occ_residues.auth_comp_id",
        "_pdbx_unobs_or_zero_occ_residues.auth_seq_id",
        "_pdbx_unobs_or_zero_occ_residues.PDB_model_num",
        "_pdbx_unobs_or_zero_occ_residues.occupancy_flag",
    ]
    atom_tags = [
        "_pdbx_unobs_or_zero_occ_atoms.auth_asym_id",
        "_pdbx_unobs_or_zero_occ_atoms.auth_comp_id",
        "_pdbx_unobs_or_zero_occ_atoms.auth_seq_id",
        "_pdbx_unobs_or_zero_occ_atoms.auth_atom_id",
        "_pdbx_unobs_or_zero_occ_atoms.PDB_model_num",
        "_pdbx_unobs_or_zero_occ_atoms.occupancy_flag",
    ]
    return {"residues": rows(block, residue_tags), "atoms": rows(block, atom_tags)}


def entry_id(block: gemmi.cif.Block, path: Path) -> str:
    ids = col(block, "_entry.id")
    return (ids[0] if ids else path.stem).upper()


def resolution(block: gemmi.cif.Block) -> float | None:
    for tag in ("_refine.ls_d_res_high", "_reflns.d_resolution_high"):
        vals = col(block, tag)
        if vals:
            v = fnum(vals[0])
            if v is not None:
                return v
    return None


def inspect(path: Path) -> dict[str, Any]:
    doc = gemmi.cif.read_file(str(path))
    block = doc.sole_block()
    eid = entry_id(block, path)
    ligand = TARGETS.get(eid)
    if not ligand:
        raise ValueError(f"No configured target ligand for {eid}; expected one of {sorted(TARGETS)}")
    atoms = atom_rows(block)
    beta = beta5_chains(block, atoms)
    unobs = unobserved(block)
    beta_auth = set(beta["auth_asym_ids"])
    return {
        "entry_id": eid,
        "source_file": str(path),
        "sha256": sha256(path),
        "resolution_A": resolution(block),
        "target_ligand": ligand,
        "atom_site_rows": len(atoms),
        "beta5": beta,
        "ligand_instances": ligand_instances(atoms, ligand),
        "ligand_struct_conn": struct_connections(block, ligand),
        "neighborhoods": neighborhoods(atoms, ligand),
        "occupancy_and_altloc": occupancy_anomalies(atoms, ligand),
        "unobserved_all": unobs,
        "unobserved_beta5": {
            "residues": [r for r in unobs["residues"] if r.get("_pdbx_unobs_or_zero_occ_residues.auth_asym_id") in beta_auth],
            "atoms": [r for r in unobs["atoms"] if r.get("_pdbx_unobs_or_zero_occ_atoms.auth_asym_id") in beta_auth],
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cif", nargs="+", type=Path, help="Deposited mmCIF files (5LF3/5LF7)")
    ap.add_argument("--out", type=Path, default=Path("validation/structure_inventory.json"))
    args = ap.parse_args()

    report = {
        "purpose": "read-only inventory; no coordinate or chemistry modifications",
        "cutoffs_A": CUTOFFS,
        "entries": [inspect(p) for p in args.cif],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
