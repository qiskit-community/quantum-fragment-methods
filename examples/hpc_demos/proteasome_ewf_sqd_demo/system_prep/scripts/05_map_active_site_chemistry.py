#!/usr/bin/env python3
"""Stage 6B/7: chemistry-first active-site interaction map.

This script follows the Stage-6 distance inventory after the atom-count ceiling
has been relaxed.  Its purpose is to identify *what each nearby residue/water is
actually doing* before any molecular scalpel decisions are made.

It does NOT:
  - delete receptor atoms,
  - choose final residue fragments,
  - add caps or hydrogens,
  - assign protonation states,
  - optimize coordinates,
  - claim hydrogen bonds from heavy-atom distances alone.

It DOES:
  1. Reconstruct native 6V8 and rigidly transplanted BO2 in the frozen
     canonical 5LF7 chain-Y receptor frame.
  2. Enumerate all protein--ligand heavy-atom contacts <= 4.0 A.
  3. Distinguish backbone vs side-chain receptor atoms.
  4. Flag the catalytic Thr-OG1--B26 contact explicitly.
  5. Flag heteroatom/polar-contact candidates without over-assigning H-bonds.
  6. Summarize each residue by proximity, contact counts, contacted atoms,
     residue chemistry, and an initial retention signal.
  7. Analyze nearby crystallographic waters for ligand/protein bridging and
     spatial conservation with 5LF3 waters after alignment.
  8. Write visualization-only snapshots of a <=4 A chemistry shell.

Example
-------
python3 scripts/05_map_active_site_chemistry.py \
  raw/5LF3.cif raw/5LF7.cif \
  --alignment validation/alignment_report.json \
  --canonical validation/canonical_receptor_report.json \
  --stage4-script scripts/02_align_active_sites.py \
  --chain Y \
  --out validation/active_site_chemistry_report.json \
  --contacts validation/active_site_contacts.tsv \
  --waters validation/water_bridge_candidates.tsv \
  --snapshots intermediate/chemistry_map
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

CONTACT_CUTOFF = 4.0
POLAR_CUTOFF = 3.5
STRONG_CONTACT_CUTOFF = 3.2
WATER_CUTOFF = 5.0
WATER_BRIDGE_CUTOFF = 3.5
WATER_CONSERVATION_CUTOFF = 1.0
WATER_NAMES = {"HOH", "WAT", "DOD"}
BACKBONE_ATOMS = {"N", "CA", "C", "O", "OXT"}
HETERO_ELEMENTS = {"N", "O", "S"}
POLAR_LIGAND_ELEMENTS = {"N", "O", "S", "B"}

RESIDUE_CHEMISTRY = {
    "ASP": "acidic/polar side chain",
    "GLU": "acidic/polar side chain",
    "LYS": "basic/polar side chain",
    "ARG": "basic/polar side chain",
    "HIS": "ionizable/polar aromatic side chain",
    "SER": "polar hydroxyl side chain",
    "THR": "polar hydroxyl side chain",
    "TYR": "polar aromatic hydroxyl side chain",
    "ASN": "polar amide side chain",
    "GLN": "polar amide side chain",
    "CYS": "polarizable thiol side chain",
    "MET": "polarizable thioether side chain",
    "GLY": "small/nonpolar backbone-dominated residue",
    "ALA": "small/nonpolar side chain",
    "VAL": "hydrophobic side chain",
    "LEU": "hydrophobic side chain",
    "ILE": "hydrophobic side chain",
    "PHE": "hydrophobic aromatic side chain",
    "TRP": "aromatic side chain",
    "PRO": "hydrophobic conformationally constrained residue",
}


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
    out: list[dict[str, Any]] = []
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
    return {k: list(mod.dedupe(v, lambda a: a["atom"]).values()) for k, v in groups.items()}


def water_oxygens(mod, atoms: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    aa = [
        a for a in atoms
        if a.get("group") == "HETATM"
        and (a.get("comp") or "").upper() in WATER_NAMES
        and (a.get("element") or "").upper() == "O"
    ]
    return list(mod.dedupe(
        aa,
        lambda a: (a.get("label_chain"), a.get("auth_chain"), a.get("auth_seq"), a.get("atom")),
    ).values())


def is_backbone(atom: dict[str, Any]) -> bool:
    return (atom.get("atom") or "").upper() in BACKBONE_ATOMS


def distance(a: dict[str, Any], b: dict[str, Any]) -> float:
    return float(np.linalg.norm(a["xyz"] - b["xyz"]))


def contact_type(receptor_atom: dict[str, Any], ligand_atom: dict[str, Any], d: float,
                 *, catalytic: bool = False) -> str:
    ra = (receptor_atom.get("atom") or "").upper()
    la = (ligand_atom.get("atom") or "").upper()
    re = (receptor_atom.get("element") or "").upper()
    le = (ligand_atom.get("element") or "").upper()

    if catalytic and ra == "OG1" and la == "B26":
        return "catalytic_reversible_covalent_ThrO-B"
    if d <= POLAR_CUTOFF and re in HETERO_ELEMENTS and le in POLAR_LIGAND_ELEMENTS:
        return "polar/heteroatom_contact_candidate"
    if d <= CONTACT_CUTOFF and re == "C" and le == "C":
        return "close_nonpolar_contact_candidate"
    if d <= CONTACT_CUTOFF and (re in HETERO_ELEMENTS or le in POLAR_LIGAND_ELEMENTS):
        return "close_mixed_contact_candidate"
    return "close_contact_candidate"


def enumerate_contacts(res_atoms: Iterable[dict[str, Any]], lig_atoms: Iterable[dict[str, Any]],
                       *, ligand_name: str, residue_key: tuple[str, str], catalytic: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rr = heavy(res_atoms)
    ll = heavy(lig_atoms)
    for a in rr:
        for b in ll:
            d = distance(a, b)
            if d > CONTACT_CUTOFF:
                continue
            rows.append({
                "ligand": ligand_name,
                "label_chain": residue_key[0],
                "label_seq": residue_key[1],
                "auth_chain": a.get("auth_chain"),
                "auth_seq": a.get("auth_seq"),
                "comp_id": a.get("label_comp") or a.get("comp"),
                "receptor_atom": a.get("atom"),
                "receptor_element": a.get("element"),
                "receptor_region": "backbone" if is_backbone(a) else "side_chain",
                "ligand_atom": b.get("atom"),
                "ligand_element": b.get("element"),
                "distance_A": round(d, 4),
                "interaction_screen": contact_type(a, b, d, catalytic=catalytic),
            })
    rows.sort(key=lambda x: x["distance_A"])
    return rows


def min_or_none(values: Iterable[float]) -> float | None:
    vals = list(values)
    return round(min(vals), 4) if vals else None


def residue_retention_signal(summary: dict[str, Any]) -> str:
    if summary["catalytic_thr"]:
        return "mandatory_catalytic"

    d6 = summary["min_to_6V8_A"]
    db = summary["min_to_BO2_A"]
    polar6 = summary["polar_contact_candidates_6V8"]
    polarb = summary["polar_contact_candidates_BO2"]
    chemistry = summary["residue_chemistry"]

    both_32 = d6 is not None and db is not None and d6 <= STRONG_CONTACT_CUTOFF and db <= STRONG_CONTACT_CUTOFF
    both_35 = d6 is not None and db is not None and d6 <= POLAR_CUTOFF and db <= POLAR_CUTOFF
    both_40 = d6 is not None and db is not None and d6 <= CONTACT_CUTOFF and db <= CONTACT_CUTOFF

    if both_32:
        return "strong_initial_retain"
    if polar6 > 0 and polarb > 0:
        return "strong_initial_retain"
    if both_35:
        return "strong_initial_retain"
    if both_40 and any(word in chemistry for word in ("acidic", "basic", "polar", "ionizable", "polarizable")):
        return "moderate_to_strong_inspect"
    if both_40:
        return "moderate_inspect"
    return "contextual_inspect"


def summarize_residue(key: tuple[str, str], atoms: list[dict[str, Any]], contacts6: list[dict[str, Any]],
                      contactsb: list[dict[str, Any]], *, chain: str) -> dict[str, Any]:
    ex = atoms[0]
    comp = ex.get("label_comp") or ex.get("comp") or "UNK"
    catalytic = key == (chain, "1")

    def side_min(rows, region):
        return min_or_none(r["distance_A"] for r in rows if r["receptor_region"] == region)

    summary = {
        "label_chain": key[0],
        "label_seq": key[1],
        "auth_chain": ex.get("auth_chain"),
        "auth_seq": ex.get("auth_seq"),
        "comp_id": comp,
        "residue_chemistry": RESIDUE_CHEMISTRY.get(comp, "unclassified residue chemistry"),
        "catalytic_thr": catalytic,
        "full_residue_heavy_atoms": len(heavy(atoms)),
        "min_to_6V8_A": min_or_none(r["distance_A"] for r in contacts6),
        "min_to_BO2_A": min_or_none(r["distance_A"] for r in contactsb),
        "contacts_le_3.2A_6V8": sum(r["distance_A"] <= STRONG_CONTACT_CUTOFF for r in contacts6),
        "contacts_le_3.2A_BO2": sum(r["distance_A"] <= STRONG_CONTACT_CUTOFF for r in contactsb),
        "contacts_le_3.5A_6V8": sum(r["distance_A"] <= POLAR_CUTOFF for r in contacts6),
        "contacts_le_3.5A_BO2": sum(r["distance_A"] <= POLAR_CUTOFF for r in contactsb),
        "contacts_le_4.0A_6V8": len(contacts6),
        "contacts_le_4.0A_BO2": len(contactsb),
        "polar_contact_candidates_6V8": sum(r["interaction_screen"] == "polar/heteroatom_contact_candidate" for r in contacts6),
        "polar_contact_candidates_BO2": sum(r["interaction_screen"] == "polar/heteroatom_contact_candidate" for r in contactsb),
        "backbone_min_6V8_A": side_min(contacts6, "backbone"),
        "backbone_min_BO2_A": side_min(contactsb, "backbone"),
        "side_chain_min_6V8_A": side_min(contacts6, "side_chain"),
        "side_chain_min_BO2_A": side_min(contactsb, "side_chain"),
        "receptor_atoms_contacting_6V8": sorted({r["receptor_atom"] for r in contacts6}),
        "receptor_atoms_contacting_BO2": sorted({r["receptor_atom"] for r in contactsb}),
        "ligand_atoms_contacted_6V8": sorted({r["ligand_atom"] for r in contacts6}),
        "ligand_atoms_contacted_BO2": sorted({r["ligand_atom"] for r in contactsb}),
        "closest_contact_6V8": contacts6[0] if contacts6 else None,
        "closest_contact_BO2": contactsb[0] if contactsb else None,
    }
    summary["initial_retention_signal"] = residue_retention_signal(summary)
    return summary


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
        "spatially_conserved_le_1.0A": d <= WATER_CONSERVATION_CUTOFF,
    }


def nearest_pairs(origin: dict[str, Any], targets: Iterable[dict[str, Any]], cutoff: float,
                  *, allowed_elements: set[str] | None = None) -> list[dict[str, Any]]:
    rows = []
    for a in heavy(targets):
        el = (a.get("element") or "").upper()
        if allowed_elements is not None and el not in allowed_elements:
            continue
        d = distance(origin, a)
        if d <= cutoff:
            rows.append({
                "atom": a.get("atom"),
                "element": a.get("element"),
                "comp_id": a.get("label_comp") or a.get("comp"),
                "label_chain": a.get("label_chain"),
                "label_seq": a.get("label_seq"),
                "auth_chain": a.get("auth_chain"),
                "auth_seq": a.get("auth_seq"),
                "distance_A": round(d, 4),
            })
    rows.sort(key=lambda x: x["distance_A"])
    return rows


def analyze_water(mod, w: dict[str, Any], *, lig6, ligb, protein_heavy, waters3, r3to7, t3to7) -> dict[str, Any] | None:
    lig6_all = nearest_pairs(w, lig6, WATER_CUTOFF)
    ligb_all = nearest_pairs(w, ligb, WATER_CUTOFF)
    if not lig6_all and not ligb_all:
        return None

    lig6_polar = nearest_pairs(w, lig6, WATER_BRIDGE_CUTOFF, allowed_elements=POLAR_LIGAND_ELEMENTS)
    ligb_polar = nearest_pairs(w, ligb, WATER_BRIDGE_CUTOFF, allowed_elements=POLAR_LIGAND_ELEMENTS)
    prot_polar = nearest_pairs(w, protein_heavy, WATER_BRIDGE_CUTOFF, allowed_elements=HETERO_ELEMENTS)

    conserved = nearest_transformed_water(mod, w, waters3, r3to7, t3to7)
    conserved_bool = bool(conserved and conserved["spatially_conserved_le_1.0A"])

    bridge6 = bool(lig6_polar and prot_polar)
    bridgeb = bool(ligb_polar and prot_polar)
    bridge_both = bridge6 and bridgeb

    ligand_boron6 = [x for x in lig6_polar if (x.get("element") or "").upper() == "B"]
    ligand_boronb = [x for x in ligb_polar if (x.get("element") or "").upper() == "B"]

    if bridge_both and conserved_bool:
        priority = "high_conserved_bridge_both"
    elif bridge_both:
        priority = "high_bridge_both"
    elif (bridge6 or bridgeb) and conserved_bool:
        priority = "high_conserved_bridge_one"
    elif bridge6 or bridgeb:
        priority = "medium_bridge_one"
    elif conserved_bool and (lig6_polar or ligb_polar):
        priority = "medium_conserved_polar_contact"
    else:
        priority = "low_contextual"

    return {
        "5LF7_auth_chain": w.get("auth_chain"),
        "5LF7_auth_seq": w.get("auth_seq"),
        "5LF7_label_chain": w.get("label_chain"),
        "min_to_6V8_A": lig6_all[0]["distance_A"] if lig6_all else None,
        "min_to_BO2_A": ligb_all[0]["distance_A"] if ligb_all else None,
        "6V8_polar_atoms_within_3.5A": lig6_polar,
        "BO2_polar_atoms_within_3.5A": ligb_polar,
        "protein_heteroatoms_within_3.5A": prot_polar,
        "bridge_candidate_6V8": bridge6,
        "bridge_candidate_BO2": bridgeb,
        "bridge_candidate_both": bridge_both,
        "near_boron_6V8": bool(ligand_boron6),
        "near_boron_BO2": bool(ligand_boronb),
        "nearest_5LF3_water": conserved,
        "priority": priority,
        "caveat": "Heavy-atom geometry only; not a hydrogen-bond assignment. Water H orientations/protonation remain unresolved.",
    }


def write_tsv(path: Path, rows: list[dict[str, Any]], cols: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        for row in rows:
            out = dict(row)
            for k, v in list(out.items()):
                if isinstance(v, (list, dict)):
                    out[k] = json.dumps(v, separators=(",", ":"))
            w.writerow(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cif_5lf3", type=Path)
    ap.add_argument("cif_5lf7", type=Path)
    ap.add_argument("--alignment", type=Path, default=Path("validation/alignment_report.json"))
    ap.add_argument("--canonical", type=Path, default=Path("validation/canonical_receptor_report.json"))
    ap.add_argument("--stage4-script", type=Path, default=Path("scripts/02_align_active_sites.py"))
    ap.add_argument("--chain", default="Y")
    ap.add_argument("--out", type=Path, default=Path("validation/active_site_chemistry_report.json"))
    ap.add_argument("--contacts", type=Path, default=Path("validation/active_site_contacts.tsv"))
    ap.add_argument("--waters", type=Path, default=Path("validation/water_bridge_candidates.tsv"))
    ap.add_argument("--snapshots", type=Path, default=Path("intermediate/chemistry_map"))
    args = ap.parse_args()

    mod = load_stage4_module(args.stage4_script)
    alignment = load_json(args.alignment)
    canonical = load_json(args.canonical)

    if canonical.get("recommendation", {}).get("recommended_entry") != "5LF7":
        raise RuntimeError("Stage-5 report does not freeze 5LF7 as canonical receptor.")
    if canonical.get("canonical_beta5_copy") != args.chain:
        raise RuntimeError(
            f"Stage-5 canonical beta5 copy is {canonical.get('canonical_beta5_copy')!r}, but --chain={args.chain!r}."
        )

    expected3 = alignment.get("reference", {}).get("sha256")
    expected7 = alignment.get("mobile", {}).get("sha256")
    actual3 = mod.sha256(args.cif_5lf3)
    actual7 = mod.sha256(args.cif_5lf7)
    if expected3 and actual3 != expected3:
        raise RuntimeError(f"5LF3 SHA256 mismatch: {actual3} != Stage-4 {expected3}")
    if expected7 and actual7 != expected7:
        raise RuntimeError(f"5LF7 SHA256 mismatch: {actual7} != Stage-4 {expected7}")

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

    groups = protein_groups(mod, a7)
    all_contacts: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    for key, atoms in groups.items():
        catalytic = key == (args.chain, "1")
        c6 = enumerate_contacts(atoms, v8_7, ligand_name="6V8_native", residue_key=key, catalytic=catalytic)
        cb = enumerate_contacts(atoms, bo2_7, ligand_name="BO2_transplanted", residue_key=key, catalytic=catalytic)
        if not c6 and not cb:
            continue
        all_contacts.extend(c6)
        all_contacts.extend(cb)
        summaries.append(summarize_residue(key, atoms, c6, cb, chain=args.chain))

    rank = {
        "mandatory_catalytic": 0,
        "strong_initial_retain": 1,
        "moderate_to_strong_inspect": 2,
        "moderate_inspect": 3,
        "contextual_inspect": 4,
    }
    summaries.sort(key=lambda x: (
        rank.get(x["initial_retention_signal"], 99),
        min(v for v in [x["min_to_6V8_A"], x["min_to_BO2_A"]] if v is not None),
        x["label_chain"],
        int(x["label_seq"]) if str(x["label_seq"]).isdigit() else 999999,
    ))
    all_contacts.sort(key=lambda x: (x["distance_A"], x["ligand"], x["label_chain"], str(x["label_seq"])))

    # Water analysis uses all canonical protein heavy atoms so that potential
    # protein--water--ligand bridges can involve neighboring subunits as needed.
    protein_heavy = heavy(mod.protein_atoms(a7))
    waters7 = water_oxygens(mod, a7)
    waters3 = water_oxygens(mod, a3)
    water_rows: list[dict[str, Any]] = []
    for w in waters7:
        row = analyze_water(
            mod, w, lig6=v8_7, ligb=bo2_7, protein_heavy=protein_heavy,
            waters3=waters3, r3to7=r3to7, t3to7=t3to7,
        )
        if row is not None:
            water_rows.append(row)

    water_rank = {
        "high_conserved_bridge_both": 0,
        "high_bridge_both": 1,
        "high_conserved_bridge_one": 2,
        "medium_bridge_one": 3,
        "medium_conserved_polar_contact": 4,
        "low_contextual": 5,
    }
    water_rows.sort(key=lambda x: (
        water_rank.get(x["priority"], 99),
        min(v for v in [x["min_to_6V8_A"], x["min_to_BO2_A"]] if v is not None),
    ))

    # Visualization-only chemistry shell: residues with any <=4 A contact,
    # plus medium/high water candidates. No atom editing or chemistry changes.
    keys = {(s["label_chain"], s["label_seq"]) for s in summaries}
    receptor_shell = [a for k in keys for a in groups[k]]
    keep_water_ids = {
        (w["5LF7_auth_chain"], w["5LF7_auth_seq"])
        for w in water_rows
        if not w["priority"].startswith("low_")
    }
    shell_waters = [w for w in waters7 if (w.get("auth_chain"), w.get("auth_seq")) in keep_water_ids]

    args.snapshots.mkdir(parents=True, exist_ok=True)
    snap6 = args.snapshots / "5LF7_Y_chemistry_shell_native_6V8.pdb"
    snapb = args.snapshots / "5LF7_Y_chemistry_shell_transplanted_BO2.pdb"
    mod.write_pdb_snapshot(
        snap6, receptor_shell + shell_waters + v8_7,
        remark="Chemistry-map visualization only: canonical 5LF7-Y contact shell + screened waters + native 6V8; no chemistry modification",
    )
    mod.write_pdb_snapshot(
        snapb, receptor_shell + shell_waters + bo2_7,
        remark="Chemistry-map visualization only: SAME canonical 5LF7-Y contact shell + screened waters + rigidly transplanted BO2; no chemistry modification",
    )

    report = {
        "purpose": "Chemistry-first active-site interaction map before cluster cutting/capping/protonation",
        "canonical_receptor": {"entry_id": "5LF7", "beta5_copy": args.chain, "sha256": actual7},
        "source_5LF3_sha256": actual3,
        "ligand_poses": {
            "6V8": "native deposited 5LF7 pose",
            "BO2": "5LF3 pose rigidly transformed into frozen 5LF7-Y frame using inverse Stage-4 transform",
        },
        "screening_cutoffs_A": {
            "all_heavy_atom_contacts": CONTACT_CUTOFF,
            "polar_heteroatom_candidate": POLAR_CUTOFF,
            "strong_contact": STRONG_CONTACT_CUTOFF,
            "water_inventory": WATER_CUTOFF,
            "water_bridge_candidate": WATER_BRIDGE_CUTOFF,
            "water_spatial_conservation": WATER_CONSERVATION_CUTOFF,
        },
        "important_caveats": [
            "Heavy-atom distances alone do not establish hydrogen bonds; protonation and H orientations are unresolved.",
            "Residue retention signals are screening labels, not final quantum-cluster membership decisions.",
            "Charged/ionizable residue labels describe chemical capability; actual protonation/charge is deferred.",
            "No coordinates or chemical identities are modified in this step.",
        ],
        "residue_summaries": summaries,
        "contact_rows": all_contacts,
        "water_candidates": water_rows,
        "selection_principle": [
            "Retain catalytic covalent chemistry first.",
            "Then favor contacts shared by both inhibitors, especially specific polar/heteroatom interactions.",
            "Do not discard a residue merely because it belongs to a neighboring subunit.",
            "Backbone-only contacts may motivate peptide/backbone fragments rather than whole side chains.",
            "Side-chain-specific contacts may motivate retaining the chemically active side chain with a defensible cap/linkage.",
            "Conserved protein--water--ligand bridge candidates deserve explicit evaluation before deletion.",
            "Final cluster size will be determined by chemical completeness and convergence, not a preset atom ceiling.",
        ],
        "visualization_snapshots": {"native_6V8": str(snap6), "transplanted_BO2": str(snapb)},
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")

    contact_cols = [
        "ligand", "label_chain", "label_seq", "auth_chain", "auth_seq", "comp_id",
        "receptor_atom", "receptor_element", "receptor_region", "ligand_atom", "ligand_element",
        "distance_A", "interaction_screen",
    ]
    write_tsv(args.contacts, all_contacts, contact_cols)

    water_cols = [
        "priority", "5LF7_auth_chain", "5LF7_auth_seq", "5LF7_label_chain",
        "min_to_6V8_A", "min_to_BO2_A", "bridge_candidate_6V8", "bridge_candidate_BO2",
        "bridge_candidate_both", "near_boron_6V8", "near_boron_BO2",
        "6V8_polar_atoms_within_3.5A", "BO2_polar_atoms_within_3.5A",
        "protein_heteroatoms_within_3.5A", "nearest_5LF3_water", "caveat",
    ]
    write_tsv(args.waters, water_rows, water_cols)

    print(f"Wrote {args.out}")
    print(f"Wrote {args.contacts}")
    print(f"Wrote {args.waters}")
    print("\nChemistry-first active-site screening")
    print(f"  canonical receptor = 5LF7 chain {args.chain}")
    print(f"  residues with >=1 heavy-atom contact <= {CONTACT_CUTOFF:.1f} A: {len(summaries)}")
    print(f"  protein-ligand atom-pair contacts <= {CONTACT_CUTOFF:.1f} A: {len(all_contacts)}")
    print(f"  crystallographic waters <= {WATER_CUTOFF:.1f} A of either pose: {len(water_rows)}")
    print(f"  water bridge candidates for both poses: {sum(w['bridge_candidate_both'] for w in water_rows)}")
    print("\nHighest-priority residue signals")
    for s in summaries[:15]:
        print(
            f"  {s['initial_retention_signal']:<28s} "
            f"{s['label_chain']}:{s['label_seq']} {s['comp_id']:<3s} "
            f"d6V8={s['min_to_6V8_A'] if s['min_to_6V8_A'] is not None else float('nan'):.3f} A  "
            f"dBO2={s['min_to_BO2_A'] if s['min_to_BO2_A'] is not None else float('nan'):.3f} A  "
            f"polar={s['polar_contact_candidates_6V8']}/{s['polar_contact_candidates_BO2']}"
        )
    print("\nHighest-priority water signals")
    for w in water_rows[:10]:
        conserved = w.get("nearest_5LF3_water", {}) or {}
        print(
            f"  {w['priority']:<32s} water {w['5LF7_auth_chain']}:{w['5LF7_auth_seq']} "
            f"d6V8={w['min_to_6V8_A'] if w['min_to_6V8_A'] is not None else float('nan'):.3f} A  "
            f"dBO2={w['min_to_BO2_A'] if w['min_to_BO2_A'] is not None else float('nan'):.3f} A  "
            f"conserved={conserved.get('spatially_conserved_le_1.0A', False)}"
        )
    print("\nNo cluster atoms were deleted or modified. Use this report to design the first chemically complete cluster.")


if __name__ == "__main__":
    main()
