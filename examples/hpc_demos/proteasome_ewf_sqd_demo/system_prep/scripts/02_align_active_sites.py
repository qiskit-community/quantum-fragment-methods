#!/usr/bin/env python3
"""Stage 4: sequence-aware alignment of the 5LF3/5LF7 beta5 active sites.

This script performs *rigid-body analysis only*. It does not alter the raw mmCIF
files, add atoms, infer bonds/protonation, or optimize geometry.

Key design choice
-----------------
Protein atom correspondence is based on mmCIF ``label_asym_id`` +
``label_seq_id`` + atom name, not author residue numbers.  This avoids the
known author-numbering offset between 5LF3 and 5LF7 around beta5.

Outputs
-------
- validation/alignment_report.json
- local PDB snapshots for visual inspection under intermediate/alignment/

Example
-------
python scripts/02_align_active_sites.py \
    raw/5LF3.cif raw/5LF7.cif \
    --inventory validation/structure_inventory.json \
    --out validation/alignment_report.json \
    --snapshots intermediate/alignment
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
import numpy as np

REFERENCE_ID = "5LF3"
MOBILE_ID = "5LF7"
REFERENCE_LIGAND = "BO2"
MOBILE_LIGAND = "6V8"
DEFAULT_CHAINS = ("K", "Y")
FIT_ATOMS = ("N", "CA", "C")
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
    return [
        {
            t: norm(columns[j][i]) if i < len(columns[j]) else None
            for j, t in enumerate(tags)
        }
        for i in range(n)
    ]


def fnum(x: str | None) -> float | None:
    if x is None:
        return None
    try:
        return float(x.split("(")[0])
    except ValueError:
        return None


def read_atoms(path: Path) -> tuple[gemmi.cif.Block, list[dict[str, Any]]]:
    block = gemmi.cif.read_file(str(path)).sole_block()
    tags = [
        "_atom_site.group_PDB",
        "_atom_site.id",
        "_atom_site.type_symbol",
        "_atom_site.label_atom_id",
        "_atom_site.label_alt_id",
        "_atom_site.label_comp_id",
        "_atom_site.label_asym_id",
        "_atom_site.label_seq_id",
        "_atom_site.auth_atom_id",
        "_atom_site.auth_comp_id",
        "_atom_site.auth_asym_id",
        "_atom_site.auth_seq_id",
        "_atom_site.Cartn_x",
        "_atom_site.Cartn_y",
        "_atom_site.Cartn_z",
        "_atom_site.occupancy",
        "_atom_site.pdbx_PDB_model_num",
    ]
    out: list[dict[str, Any]] = []
    for r in rows(block, tags):
        try:
            xyz = np.array(
                [
                    float(r["_atom_site.Cartn_x"]),
                    float(r["_atom_site.Cartn_y"]),
                    float(r["_atom_site.Cartn_z"]),
                ],
                dtype=float,
            )
        except (TypeError, ValueError):
            continue
        out.append(
            {
                "group": r["_atom_site.group_PDB"],
                "id": r["_atom_site.id"],
                "element": (r["_atom_site.type_symbol"] or "").upper(),
                "atom": r["_atom_site.auth_atom_id"] or r["_atom_site.label_atom_id"],
                "label_atom": r["_atom_site.label_atom_id"],
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
            }
        )
    return block, out


def model1(atoms: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [a for a in atoms if a["model"] in {None, "1"}]


def atom_preference(a: dict[str, Any]) -> tuple[float, int, int]:
    """Higher tuple wins when duplicate/altloc atom records exist."""
    occ = a["occ"] if a["occ"] is not None else 0.0
    alt = a["alt"]
    no_alt = 1 if alt is None else 0
    alt_a = 1 if alt == "A" else 0
    return (occ, no_alt, alt_a)


def dedupe(atoms: Iterable[dict[str, Any]], key_fn) -> dict[Any, dict[str, Any]]:
    best: dict[Any, dict[str, Any]] = {}
    for a in atoms:
        k = key_fn(a)
        if k not in best or atom_preference(a) > atom_preference(best[k]):
            best[k] = a
    return best


def protein_atoms(atoms: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [a for a in atoms if a["group"] == "ATOM"]


def heavy(atoms: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [a for a in atoms if a["element"] not in {"H", "D"}]


def ligand_atoms(atoms: Iterable[dict[str, Any]], ligand: str, auth_chain: str) -> list[dict[str, Any]]:
    groups: dict[tuple[str | None, str | None, str | None], list[dict[str, Any]]] = defaultdict(list)
    for a in atoms:
        if a["auth_chain"] == auth_chain and (a["comp"] == ligand or a["label_comp"] == ligand):
            groups[(a["auth_chain"], a["auth_seq"], a["label_chain"])].append(a)
    if len(groups) != 1:
        raise RuntimeError(
            f"Expected exactly one {ligand} instance on auth chain {auth_chain}; found {len(groups)}: {list(groups)}"
        )
    aa = next(iter(groups.values()))
    # Ligands in these entries have no relevant altloc ambiguity, but dedupe defensively.
    return list(dedupe(aa, lambda a: a["atom"]).values())


def beta5_atom_map(atoms: Iterable[dict[str, Any]], chain: str, atom_names: Iterable[str] | None = None):
    names = set(atom_names) if atom_names is not None else None
    aa = [
        a
        for a in protein_atoms(atoms)
        if a["label_chain"] == chain
        and a["label_seq"] is not None
        and (names is None or a["atom"] in names)
    ]
    return dedupe(aa, lambda a: (a["label_seq"], a["atom"]))


def global_protein_map(atoms: Iterable[dict[str, Any]]):
    aa = [a for a in protein_atoms(atoms) if a["label_seq"] is not None]
    return dedupe(aa, lambda a: (a["label_chain"], a["label_seq"], a["atom"]))


def kabsch(mobile: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Return R,t,rmsd for column-vector transform x' = R @ x + t."""
    if mobile.shape != reference.shape or mobile.ndim != 2 or mobile.shape[1] != 3:
        raise ValueError(f"Bad Kabsch shapes: {mobile.shape} vs {reference.shape}")
    if len(mobile) < 3:
        raise ValueError("Need at least three matched points for a rigid fit")

    cm = mobile.mean(axis=0)
    cr = reference.mean(axis=0)
    pm = mobile - cm
    pr = reference - cr
    h = pm.T @ pr
    u, _s, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1
        r = vt.T @ u.T
    t = cr - r @ cm
    fitted = (r @ mobile.T).T + t
    rmsd = float(np.sqrt(np.mean(np.sum((fitted - reference) ** 2, axis=1))))
    return r, t, rmsd


def apply_xyz(xyz: np.ndarray, r: np.ndarray, t: np.ndarray) -> np.ndarray:
    return r @ xyz + t


def rmsd_after_transform(
    mobile_atoms: list[dict[str, Any]],
    reference_atoms: list[dict[str, Any]],
    r: np.ndarray,
    t: np.ndarray,
) -> float | None:
    if not mobile_atoms:
        return None
    m = np.array([apply_xyz(a["xyz"], r, t) for a in mobile_atoms])
    q = np.array([a["xyz"] for a in reference_atoms])
    return float(np.sqrt(np.mean(np.sum((m - q) ** 2, axis=1))))


def dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def min_distance_to_ligand(res_atoms: Iterable[dict[str, Any]], lig_xyz: np.ndarray) -> float:
    vals = []
    for a in heavy(res_atoms):
        d = np.linalg.norm(lig_xyz - a["xyz"], axis=1)
        vals.append(float(d.min()))
    return min(vals) if vals else math.inf


def environment_residues(
    atoms: list[dict[str, Any]], lig: list[dict[str, Any]], cutoff: float
) -> set[tuple[str, str]]:
    lig_xyz = np.array([a["xyz"] for a in heavy(lig)])
    residues: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for a in heavy(protein_atoms(atoms)):
        if a["label_chain"] is None or a["label_seq"] is None:
            continue
        residues[(a["label_chain"], a["label_seq"])].append(a)
    return {
        key
        for key, aa in residues.items()
        if min_distance_to_ligand(aa, lig_xyz) <= cutoff
    }


def matched_environment_atoms(
    ref_atoms: list[dict[str, Any]],
    mob_atoms: list[dict[str, Any]],
    residue_union: set[tuple[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    ref_map = global_protein_map(ref_atoms)
    mob_map = global_protein_map(mob_atoms)
    keys = sorted(
        k for k in (set(ref_map) & set(mob_map)) if (k[0], k[1]) in residue_union
    )
    rr: list[dict[str, Any]] = []
    mm: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    for k in keys:
        a = ref_map[k]
        b = mob_map[k]
        if a["label_comp"] != b["label_comp"]:
            mismatches.append(
                {
                    "label_chain": k[0],
                    "label_seq": k[1],
                    "atom": k[2],
                    "5LF3_comp": a["label_comp"],
                    "5LF7_comp": b["label_comp"],
                }
            )
            continue
        rr.append(a)
        mm.append(b)
    return rr, mm, mismatches


def residue_contact_table(
    ref_atoms: list[dict[str, Any]],
    mob_atoms: list[dict[str, Any]],
    ref_lig: list[dict[str, Any]],
    mob_lig: list[dict[str, Any]],
    residue_union: set[tuple[str, str]],
) -> list[dict[str, Any]]:
    ref_res: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    mob_res: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for a in protein_atoms(ref_atoms):
        if a["label_chain"] is not None and a["label_seq"] is not None:
            ref_res[(a["label_chain"], a["label_seq"])].append(a)
    for a in protein_atoms(mob_atoms):
        if a["label_chain"] is not None and a["label_seq"] is not None:
            mob_res[(a["label_chain"], a["label_seq"])].append(a)

    q_ref = np.array([a["xyz"] for a in heavy(ref_lig)])
    q_mob = np.array([a["xyz"] for a in heavy(mob_lig)])
    out = []
    for key in sorted(residue_union):
        a = ref_res.get(key, [])
        b = mob_res.get(key, [])
        d3 = min_distance_to_ligand(a, q_ref) if a else math.inf
        d7 = min_distance_to_ligand(b, q_mob) if b else math.inf
        exemplar3 = a[0] if a else None
        exemplar7 = b[0] if b else None
        out.append(
            {
                "label_chain": key[0],
                "label_seq": key[1],
                "5LF3_auth_seq": exemplar3["auth_seq"] if exemplar3 else None,
                "5LF7_auth_seq": exemplar7["auth_seq"] if exemplar7 else None,
                "5LF3_comp": exemplar3["label_comp"] if exemplar3 else None,
                "5LF7_comp": exemplar7["label_comp"] if exemplar7 else None,
                "5LF3_native_min_distance_A": round(d3, 3) if math.isfinite(d3) else None,
                "5LF7_native_min_distance_A": round(d7, 3) if math.isfinite(d7) else None,
                "native_distance_delta_A": round(d7 - d3, 3)
                if math.isfinite(d3) and math.isfinite(d7)
                else None,
            }
        )
    out.sort(
        key=lambda x: min(
            x["5LF3_native_min_distance_A"] if x["5LF3_native_min_distance_A"] is not None else 999,
            x["5LF7_native_min_distance_A"] if x["5LF7_native_min_distance_A"] is not None else 999,
        )
    )
    return out


def find_atom(atoms: Iterable[dict[str, Any]], atom_name: str) -> dict[str, Any]:
    hits = [a for a in atoms if a["atom"] == atom_name]
    if len(hits) != 1:
        raise RuntimeError(f"Expected one atom named {atom_name}; found {len(hits)}")
    return hits[0]


def catalytic_thr_og1(
    atoms: list[dict[str, Any]], chain: str, lig: list[dict[str, Any]]
) -> dict[str, Any]:
    boron = find_atom(lig, "B26")
    candidates = [
        a
        for a in protein_atoms(atoms)
        if a["auth_chain"] == chain and a["label_comp"] == "THR" and a["atom"] == "OG1"
    ]
    if not candidates:
        raise RuntimeError(f"No THR OG1 found on beta5 auth chain {chain}")
    winner = min(candidates, key=lambda a: dist(a["xyz"], boron["xyz"]))
    d = dist(winner["xyz"], boron["xyz"])
    if d > 2.0:
        raise RuntimeError(
            f"Closest THR OG1 to {chain}:{boron['auth_seq']} B26 is unexpectedly far ({d:.3f} A)"
        )
    return winner


def angle_deg(v1: np.ndarray, v2: np.ndarray) -> float:
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return float("nan")
    c = float(np.dot(v1, v2) / (n1 * n2))
    c = max(-1.0, min(1.0, c))
    return float(np.degrees(np.arccos(c)))


def clash_report(
    ligand: list[dict[str, Any]],
    receptor: list[dict[str, Any]],
    ligand_transform: tuple[np.ndarray, np.ndarray] | None,
    receptor_transform: tuple[np.ndarray, np.ndarray] | None,
    catalytic_og1: dict[str, Any],
    thresholds=(1.6, 1.8, 2.0, 2.2),
) -> dict[str, Any]:
    lig = heavy(ligand)
    rec = heavy(protein_atoms(receptor))
    r_l, t_l = ligand_transform if ligand_transform else (None, None)
    r_r, t_r = receptor_transform if receptor_transform else (None, None)

    best = (math.inf, None)
    counts = {f"lt_{x:.1f}A": 0 for x in thresholds}
    examples: list[dict[str, Any]] = []
    for la in lig:
        lp = apply_xyz(la["xyz"], r_l, t_l) if r_l is not None else la["xyz"]
        for ra in rec:
            # The catalytic Thr-Ogamma -- B26 bond is expected, not a clash.
            if la["atom"] == "B26" and ra is catalytic_og1:
                continue
            rp = apply_xyz(ra["xyz"], r_r, t_r) if r_r is not None else ra["xyz"]
            d = dist(lp, rp)
            if d < best[0]:
                best = (d, (la, ra))
            for x in thresholds:
                if d < x:
                    counts[f"lt_{x:.1f}A"] += 1
            if d < 2.0 and len(examples) < 20:
                examples.append(
                    {
                        "distance_A": round(d, 3),
                        "ligand_atom": la["atom"],
                        "receptor": {
                            "label_chain": ra["label_chain"],
                            "label_seq": ra["label_seq"],
                            "auth_chain": ra["auth_chain"],
                            "auth_seq": ra["auth_seq"],
                            "comp": ra["label_comp"],
                            "atom": ra["atom"],
                        },
                    }
                )
    examples.sort(key=lambda x: x["distance_A"])
    closest = None
    if best[1] is not None:
        la, ra = best[1]
        closest = {
            "distance_A": round(best[0], 3),
            "ligand_atom": la["atom"],
            "receptor": {
                "label_chain": ra["label_chain"],
                "label_seq": ra["label_seq"],
                "auth_chain": ra["auth_chain"],
                "auth_seq": ra["auth_seq"],
                "comp": ra["label_comp"],
                "atom": ra["atom"],
            },
        }
    return {"counts": counts, "closest_nonbonded_heavy_pair": closest, "examples_lt_2A": examples}


def within_snapshot_cutoff(
    atoms: list[dict[str, Any]], lig: list[dict[str, Any]], cutoff: float
) -> list[dict[str, Any]]:
    lig_xyz = np.array([a["xyz"] for a in heavy(lig)])
    keep = []
    lig_ids = {id(a) for a in lig}
    for a in atoms:
        if id(a) in lig_ids:
            keep.append(a)
            continue
        if a["group"] not in {"ATOM", "HETATM"}:
            continue
        if a["comp"] not in WATER_NAMES and a["group"] != "ATOM":
            continue
        d = float(np.linalg.norm(lig_xyz - a["xyz"], axis=1).min())
        if d <= cutoff:
            keep.append(a)
    # explicitly append ligand, because identity can be lost across list construction
    keys = {(a["id"], a["label_chain"]) for a in keep}
    for a in lig:
        if (a["id"], a["label_chain"]) not in keys:
            keep.append(a)
    return keep


def pdb_atom_name(name: str, element: str) -> str:
    # Sufficient for visualization snapshots, not intended as authoritative PDB export.
    if len(name) >= 4:
        return name[:4]
    if len(element) == 1:
        return f" {name:<3}"
    return f"{name:<4}"


def write_pdb_snapshot(
    path: Path,
    atoms: list[dict[str, Any]],
    transform: tuple[np.ndarray, np.ndarray] | None = None,
    remark: str = "",
) -> None:
    r, t = transform if transform else (None, None)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"REMARK {remark}".rstrip()]
    serial = 1
    for a in atoms:
        p = apply_xyz(a["xyz"], r, t) if r is not None else a["xyz"]
        rec = "ATOM  " if a["group"] == "ATOM" else "HETATM"
        name = pdb_atom_name(a["atom"] or "X", a["element"])
        alt = (a["alt"] or " ")[:1]
        resn = (a["comp"] or "UNK")[:3]
        chain = (a["auth_chain"] or " ")[:1]
        try:
            resi = int(str(a["auth_seq"]).split(".")[0])
        except (TypeError, ValueError):
            resi = 0
        occ = a["occ"] if a["occ"] is not None else 1.0
        elem = (a["element"] or "")[:2].rjust(2)
        lines.append(
            f"{rec}{serial:5d} {name}{alt}{resn:>3} {chain}{resi:4d}    "
            f"{p[0]:8.3f}{p[1]:8.3f}{p[2]:8.3f}{occ:6.2f}{0.0:6.2f}          {elem}"
        )
        serial += 1
    lines.extend(["TER", "END"])
    path.write_text("\n".join(lines) + "\n")


def inventory_hashes(path: Path) -> dict[str, str]:
    data = json.loads(path.read_text())
    return {e["entry_id"].upper(): e["sha256"] for e in data.get("entries", [])}


def analyze_chain(
    chain: str,
    ref_atoms: list[dict[str, Any]],
    mob_atoms: list[dict[str, Any]],
    pocket_cutoff: float,
    snapshot_cutoff: float,
    snapshot_dir: Path,
) -> dict[str, Any]:
    ref_lig = ligand_atoms(ref_atoms, REFERENCE_LIGAND, chain)
    mob_lig = ligand_atoms(mob_atoms, MOBILE_LIGAND, chain)

    ref_fit_map = beta5_atom_map(ref_atoms, chain, FIT_ATOMS)
    mob_fit_map = beta5_atom_map(mob_atoms, chain, FIT_ATOMS)
    fit_keys = sorted(set(ref_fit_map) & set(mob_fit_map))

    # Reject residue-identity mismatches from the fit rather than silently matching them.
    good_fit_keys = []
    fit_mismatches = []
    for k in fit_keys:
        a = ref_fit_map[k]
        b = mob_fit_map[k]
        if a["label_comp"] != b["label_comp"]:
            fit_mismatches.append(
                {
                    "label_seq": k[0],
                    "atom": k[1],
                    "5LF3_comp": a["label_comp"],
                    "5LF7_comp": b["label_comp"],
                }
            )
        else:
            good_fit_keys.append(k)

    ref_fit = [ref_fit_map[k] for k in good_fit_keys]
    mob_fit = [mob_fit_map[k] for k in good_fit_keys]
    r, t, fit_rmsd = kabsch(
        np.array([a["xyz"] for a in mob_fit]),
        np.array([a["xyz"] for a in ref_fit]),
    )

    ref_env = environment_residues(ref_atoms, ref_lig, pocket_cutoff)
    mob_env = environment_residues(mob_atoms, mob_lig, pocket_cutoff)
    env_union = ref_env | mob_env

    env_ref_atoms, env_mob_atoms, env_mismatches = matched_environment_atoms(
        ref_atoms, mob_atoms, env_union
    )
    env_rmsd = rmsd_after_transform(env_mob_atoms, env_ref_atoms, r, t)

    local_keys = [
        k
        for k in good_fit_keys
        if (chain, k[0]) in env_union
    ]
    local_ref = [ref_fit_map[k] for k in local_keys]
    local_mob = [mob_fit_map[k] for k in local_keys]
    local_after_global = rmsd_after_transform(local_mob, local_ref, r, t)
    local_fit_rmsd = None
    if len(local_keys) >= 3:
        _lr, _lt, local_fit_rmsd = kabsch(
            np.array([a["xyz"] for a in local_mob]),
            np.array([a["xyz"] for a in local_ref]),
        )

    b3 = find_atom(ref_lig, "B26")
    b7 = find_atom(mob_lig, "B26")
    thr3 = catalytic_thr_og1(ref_atoms, chain, ref_lig)
    thr7 = catalytic_thr_og1(mob_atoms, chain, mob_lig)
    b7_aligned = apply_xyz(b7["xyz"], r, t)
    thr7_aligned = apply_xyz(thr7["xyz"], r, t)

    bond3 = dist(thr3["xyz"], b3["xyz"])
    bond7 = dist(thr7["xyz"], b7["xyz"])
    warhead_angle = angle_deg(b3["xyz"] - thr3["xyz"], b7_aligned - thr7_aligned)

    # Candidate receptor = 5LF3: transplant aligned 5LF7 ligand into 5LF3 protein.
    clash_into_5lf3 = clash_report(
        ligand=mob_lig,
        receptor=ref_atoms,
        ligand_transform=(r, t),
        receptor_transform=None,
        catalytic_og1=thr3,
    )
    # Candidate receptor = 5LF7: compare 5LF3 ligand with aligned 5LF7 protein in common frame.
    # The expected catalytic bond is to the *mobile* Thr atom.  Because clash_report uses
    # object identity before applying coordinates, pass thr7 here.
    clash_into_5lf7 = clash_report(
        ligand=ref_lig,
        receptor=mob_atoms,
        ligand_transform=None,
        receptor_transform=(r, t),
        catalytic_og1=thr7,
    )

    contacts = residue_contact_table(ref_atoms, mob_atoms, ref_lig, mob_lig, env_union)

    snap3 = within_snapshot_cutoff(ref_atoms, ref_lig, snapshot_cutoff)
    snap7 = within_snapshot_cutoff(mob_atoms, mob_lig, snapshot_cutoff)
    write_pdb_snapshot(
        snapshot_dir / f"5LF3_{chain}_BO2_pocket.pdb",
        snap3,
        remark=f"5LF3 chain {chain} native pocket; no coordinate modification",
    )
    write_pdb_snapshot(
        snapshot_dir / f"5LF7_{chain}_6V8_pocket_aligned_to_5LF3.pdb",
        snap7,
        transform=(r, t),
        remark=f"5LF7 chain {chain} rigidly aligned to 5LF3 chain {chain}; visualization only",
    )

    return {
        "chain": chain,
        "matching": {
            "strategy": "label_seq_id + atom name; require matching label_comp_id",
            "fit_atom_names": list(FIT_ATOMS),
            "matched_fit_atoms": len(good_fit_keys),
            "matched_fit_residues": len({k[0] for k in good_fit_keys}),
            "fit_identity_mismatches_excluded": fit_mismatches,
        },
        "transform_5LF7_to_5LF3": {
            "convention": "x_5LF3_frame = R @ x_5LF7 + t",
            "rotation": np.round(r, 12).tolist(),
            "translation_A": np.round(t, 12).tolist(),
            "det_rotation": float(np.linalg.det(r)),
        },
        "rmsd_A": {
            "global_beta5_backbone_fit": round(fit_rmsd, 4),
            "local_beta5_backbone_after_global_fit": round(local_after_global, 4)
            if local_after_global is not None
            else None,
            "local_beta5_backbone_best_local_fit": round(local_fit_rmsd, 4)
            if local_fit_rmsd is not None
            else None,
            "matched_local_environment_heavy_atoms_after_global_fit": round(env_rmsd, 4)
            if env_rmsd is not None
            else None,
            "matched_local_environment_heavy_atom_count": len(env_ref_atoms),
        },
        "pocket_definition": {
            "cutoff_A": pocket_cutoff,
            "5LF3_residue_count": len(ref_env),
            "5LF7_residue_count": len(mob_env),
            "union_residue_count": len(env_union),
            "environment_identity_mismatches_excluded": env_mismatches,
        },
        "covalent_warhead_geometry": {
            "5LF3_catalytic_thr": {
                "label_seq": thr3["label_seq"],
                "auth_seq": thr3["auth_seq"],
                "OG1_B26_A": round(bond3, 4),
            },
            "5LF7_catalytic_thr": {
                "label_seq": thr7["label_seq"],
                "auth_seq": thr7["auth_seq"],
                "OG1_B26_A": round(bond7, 4),
            },
            "aligned_catalytic_OG1_displacement_A": round(dist(thr3["xyz"], thr7_aligned), 4),
            "aligned_B26_displacement_A": round(dist(b3["xyz"], b7_aligned), 4),
            "ThrOG1_to_B26_vector_angle_deg": round(warhead_angle, 3),
        },
        "transplant_clashes": {
            "6V8_aligned_into_5LF3_receptor": clash_into_5lf3,
            "BO2_into_aligned_5LF7_receptor": clash_into_5lf7,
            "note": "Expected catalytic Thr-OG1--B26 covalent pair is excluded from clash counts.",
        },
        "native_residue_contacts": contacts,
        "visualization_snapshots": {
            "5LF3": str(snapshot_dir / f"5LF3_{chain}_BO2_pocket.pdb"),
            "5LF7_aligned": str(snapshot_dir / f"5LF7_{chain}_6V8_pocket_aligned_to_5LF3.pdb"),
            "snapshot_cutoff_A": snapshot_cutoff,
        },
    }


def compact_summary(chain_result: dict[str, Any]) -> dict[str, Any]:
    c = chain_result["transplant_clashes"]
    return {
        "chain": chain_result["chain"],
        "global_backbone_rmsd_A": chain_result["rmsd_A"]["global_beta5_backbone_fit"],
        "local_backbone_rmsd_A": chain_result["rmsd_A"]["local_beta5_backbone_after_global_fit"],
        "local_environment_heavy_rmsd_A": chain_result["rmsd_A"][
            "matched_local_environment_heavy_atoms_after_global_fit"
        ],
        "B26_displacement_A": chain_result["covalent_warhead_geometry"]["aligned_B26_displacement_A"],
        "warhead_vector_angle_deg": chain_result["covalent_warhead_geometry"][
            "ThrOG1_to_B26_vector_angle_deg"
        ],
        "6V8_into_5LF3_clashes_lt_1.8A": c["6V8_aligned_into_5LF3_receptor"]["counts"]["lt_1.8A"],
        "BO2_into_5LF7_clashes_lt_1.8A": c["BO2_into_aligned_5LF7_receptor"]["counts"]["lt_1.8A"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("reference_cif", type=Path, help="5LF3 deposited mmCIF")
    ap.add_argument("mobile_cif", type=Path, help="5LF7 deposited mmCIF")
    ap.add_argument(
        "--inventory", type=Path, default=Path("validation/structure_inventory.json")
    )
    ap.add_argument(
        "--out", type=Path, default=Path("validation/alignment_report.json")
    )
    ap.add_argument(
        "--snapshots", type=Path, default=Path("intermediate/alignment")
    )
    ap.add_argument("--chains", nargs="+", default=list(DEFAULT_CHAINS))
    ap.add_argument("--pocket-cutoff", type=float, default=5.0)
    ap.add_argument("--snapshot-cutoff", type=float, default=8.0)
    ap.add_argument(
        "--skip-hash-check",
        action="store_true",
        help="Not recommended. Skip verification against Stage-3 inventory SHA256 values.",
    )
    args = ap.parse_args()

    actual = {
        REFERENCE_ID: sha256(args.reference_cif),
        MOBILE_ID: sha256(args.mobile_cif),
    }
    if not args.skip_hash_check:
        expected = inventory_hashes(args.inventory)
        for eid, digest in actual.items():
            if eid not in expected:
                raise RuntimeError(f"{eid} missing from inventory {args.inventory}")
            if digest != expected[eid]:
                raise RuntimeError(
                    f"SHA256 mismatch for {eid}: current {digest}, inventory {expected[eid]}"
                )

    _ref_block, ref_all = read_atoms(args.reference_cif)
    _mob_block, mob_all = read_atoms(args.mobile_cif)
    ref = model1(ref_all)
    mob = model1(mob_all)

    results = [
        analyze_chain(
            chain=chain,
            ref_atoms=ref,
            mob_atoms=mob,
            pocket_cutoff=args.pocket_cutoff,
            snapshot_cutoff=args.snapshot_cutoff,
            snapshot_dir=args.snapshots,
        )
        for chain in args.chains
    ]

    report = {
        "purpose": "Stage-4 rigid alignment and active-site comparison; no chemistry modification",
        "reference": {"entry_id": REFERENCE_ID, "ligand": REFERENCE_LIGAND, "sha256": actual[REFERENCE_ID]},
        "mobile": {"entry_id": MOBILE_ID, "ligand": MOBILE_LIGAND, "sha256": actual[MOBILE_ID]},
        "important_numbering_rule": "Protein matching uses label_seq_id, never raw auth_seq_id.",
        "chains": {r["chain"]: r for r in results},
        "screening_summary": [compact_summary(r) for r in results],
        "selection_guidance": [
            "Do not choose K versus Y from a single RMSD alone.",
            "First reject a copy if reciprocal transplantation creates severe nonbonded heavy-atom clashes.",
            "Then prefer lower local environment RMSD and better conserved warhead geometry.",
            "Inspect the paired PDB snapshots before freezing the canonical copy.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")

    print(f"Wrote {args.out}")
    print("\nStage-4 screening summary")
    for x in report["screening_summary"]:
        print(
            f"  {x['chain']}: global={x['global_backbone_rmsd_A']:.3f} A, "
            f"local={x['local_backbone_rmsd_A']:.3f} A, "
            f"env={x['local_environment_heavy_rmsd_A']:.3f} A, "
            f"B26 shift={x['B26_displacement_A']:.3f} A, "
            f"clashes<1.8A={x['6V8_into_5LF3_clashes_lt_1.8A']}/"
            f"{x['BO2_into_5LF7_clashes_lt_1.8A']}"
        )


if __name__ == "__main__":
    main()
