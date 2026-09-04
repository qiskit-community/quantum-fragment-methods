#!/usr/bin/env python3
"""
21_build_catalytic_core_models.py

Build compact, chemically controlled catalytic-core models for higher-level
A/B/C proton-microstate screening.

Core definition
---------------
Keep:
  1. Full Y:1-3 segment (contains the natural N-terminal Thr1) plus any ACE/NME
     cap covalently attached to that segment.
  2. Asp17 side chain: CB, CG, OD1, OD2 + hydrogens attached to those atoms.
  3. Lys33 side chain: CB, CG, CD, CE, NZ + hydrogens attached to those atoms.
  4. Entire inhibitor (6V8 or BO2).

For Asp17 and Lys33, the removed CA-CB bond is replaced by one link hydrogen
placed along the original CB->CA direction at 1.09 Angstrom.  This converts
the isolated side chains into chemically closed acetate-/alkylamine-like
fragments while preserving the experimentally derived heavy-atom geometry.

Scientific role
---------------
This is a *minimal catalytic-core convergence model* used to test whether the
GFN2 proton-state ordering survives a higher-level DFT treatment.  It is not
the final production binding model.

All heavy atoms are taken directly from the validated H-only optimized
structures.  The script does not move any retained heavy atom.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np


STEMS = [
    "6V8_MS_A_THRN",
    "6V8_MS_B_LYS33",
    "6V8_MS_C_ASP17",
    "BO2_MS_A_THRN",
    "BO2_MS_B_LYS33",
    "BO2_MS_C_ASP17",
]


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
        "--outdir",
        type=Path,
        default=Path("intermediate/catalytic_core_dft"),
    )
    p.add_argument(
        "--report",
        type=Path,
        default=Path("validation/catalytic_core_models.json"),
    )
    p.add_argument("--link-ch", type=float, default=1.09)
    return p.parse_args()


def read_xyz(path: Path):
    lines = path.read_text().splitlines()
    n = int(lines[0])
    atoms = []
    for line in lines[2:2+n]:
        s = line.split()
        atoms.append(
            {"element": s[0].upper(),
             "xyz": np.array([float(s[1]), float(s[2]), float(s[3])])}
        )
    if len(atoms) != n:
        raise RuntimeError(f"{path}: XYZ parse mismatch")
    return atoms


def read_pdb(path: Path):
    atoms = []
    for line in path.read_text().splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        elem = line[76:78].strip()
        if not elem:
            name = line[12:16].strip()
            elem = "".join(c for c in name if c.isalpha())[:1]
        atoms.append(
            {
                "name": line[12:16].strip(),
                "resname": line[17:20].strip(),
                "chain": line[21:22].strip(),
                "resseq": line[22:26].strip(),
                "element": elem.upper(),
                "xyz_input": np.array([
                    float(line[30:38]),
                    float(line[38:46]),
                    float(line[46:54]),
                ]),
            }
        )
    return atoms


def load_case(stem, optimized, metadata):
    xyz = read_xyz(optimized / f"{stem}.xyz")
    meta = read_pdb(metadata / f"{stem}.pdb")
    if len(xyz) != len(meta):
        raise RuntimeError(
            f"{stem}: XYZ/PDB count mismatch {len(xyz)} != {len(meta)}"
        )
    out = []
    for i, (x, m) in enumerate(zip(xyz, meta)):
        if x["element"] != m["element"]:
            raise RuntimeError(
                f"{stem}: atom-order mismatch at {i}: "
                f"{x['element']} vs {m['element']}"
            )
        a = dict(m)
        a["xyz"] = x["xyz"]
        a["index"] = i
        out.append(a)
    return out


def dist(a, b):
    return float(np.linalg.norm(a["xyz"] - b["xyz"]))


def select(atoms, **kw):
    hits = []
    for a in atoms:
        ok = True
        for k, v in kw.items():
            if a[k] != str(v) if k == "resseq" else a[k] != v:
                ok = False
                break
        if ok:
            hits.append(a)
    return hits


def one(atoms, **kw):
    hits = select(atoms, **kw)
    if len(hits) != 1:
        raise RuntimeError(f"Expected one atom for {kw}, found {len(hits)}")
    return hits[0]


def nearest_heavy_in_residue(atoms, h):
    cand = []
    for a in atoms:
        if a["element"] == "H":
            continue
        if (a["chain"], a["resseq"], a["resname"]) != (
            h["chain"], h["resseq"], h["resname"]
        ):
            continue
        cand.append((dist(a, h), a))
    if not cand:
        return None
    cand.sort(key=lambda x: x[0])
    return cand[0][1]


def attached_caps(atoms, selected_indices):
    """
    Include complete ACE/NME residues if any cap heavy atom is covalently
    adjacent (<1.85 A) to a selected heavy atom.
    """
    selected_heavy = [
        atoms[i] for i in selected_indices if atoms[i]["element"] != "H"
    ]
    cap_keys = set()

    for a in atoms:
        if a["resname"] not in {"ACE", "NME"} or a["element"] == "H":
            continue
        if any(dist(a, b) < 1.85 for b in selected_heavy):
            cap_keys.add((a["chain"], a["resseq"], a["resname"]))

    out = set()
    for i, a in enumerate(atoms):
        if (a["chain"], a["resseq"], a["resname"]) in cap_keys:
            out.add(i)
    return out


def make_link_h(parent_cb, removed_ca, bond=1.09):
    v = removed_ca["xyz"] - parent_cb["xyz"]
    v /= np.linalg.norm(v)
    return parent_cb["xyz"] + bond * v


def build_case(stem, atoms, link_ch):
    ligand = "6V8" if stem.startswith("6V8") else "BO2"
    keep = set()
    reasons: Dict[int, str] = {}

    # Full Thr1-containing Y:1-3 segment.
    for i, a in enumerate(atoms):
        if a["chain"] == "Y" and a["resseq"] in {"1", "2", "3"}:
            keep.add(i)
            reasons[i] = "Thr1_segment"

    # Add any cap covalently attached to Y:1-3.
    for i in attached_caps(atoms, keep):
        keep.add(i)
        reasons[i] = "Thr1_segment_cap"

    # Entire ligand.
    for i, a in enumerate(atoms):
        if a["resname"] == ligand:
            keep.add(i)
            reasons[i] = "ligand"

    # Sidechain heavy atoms.
    side_defs = {
        ("Y", "17", "ASP"): {"CB", "CG", "OD1", "OD2"},
        ("Y", "33", "LYS"): {"CB", "CG", "CD", "CE", "NZ"},
    }

    selected_side_heavy = set()

    for key, names in side_defs.items():
        chain, resseq, resname = key
        for i, a in enumerate(atoms):
            if (
                a["chain"] == chain
                and a["resseq"] == resseq
                and a["resname"] == resname
                and a["element"] != "H"
                and a["name"] in names
            ):
                keep.add(i)
                selected_side_heavy.add(i)
                reasons[i] = f"{resname}{resseq}_sidechain"

    # Hydrogens whose nearest heavy atom in the residue is retained.
    for i, h in enumerate(atoms):
        if h["element"] != "H":
            continue
        key = (h["chain"], h["resseq"], h["resname"])
        if key not in side_defs:
            continue
        parent = nearest_heavy_in_residue(atoms, h)
        if parent is not None and parent["index"] in selected_side_heavy:
            keep.add(i)
            reasons[i] = f"{h['resname']}{h['resseq']}_sidechain_H"

    # Link hydrogens replacing CA-CB bonds for Asp17 and Lys33.
    links = []
    for chain, resseq, resname in side_defs:
        ca = one(
            atoms, chain=chain, resseq=resseq, resname=resname, name="CA"
        )
        cb = one(
            atoms, chain=chain, resseq=resseq, resname=resname, name="CB"
        )
        links.append(
            {
                "element": "H",
                "xyz": make_link_h(cb, ca, link_ch),
                "label": f"LINK_{resname}{resseq}_CB",
                "parent": f"{chain}:{resseq} {resname} CB",
            }
        )

    ordered = [atoms[i] for i in sorted(keep)]

    # Validate all retained heavy atoms are unmodified from metadata.
    max_heavy_delta = 0.0
    for a in ordered:
        if a["element"] != "H":
            max_heavy_delta = max(
                max_heavy_delta,
                float(np.linalg.norm(a["xyz"] - a["xyz_input"]))
            )

    return ordered, links, ligand, max_heavy_delta, reasons


def write_xyz(path, atoms, links, comment):
    rows = []
    for a in atoms:
        rows.append((a["element"], a["xyz"]))
    for a in links:
        rows.append((a["element"], a["xyz"]))

    lines = [str(len(rows)), comment]
    lines += [
        f"{el:<2s} {r[0]:18.10f} {r[1]:18.10f} {r[2]:18.10f}"
        for el, r in rows
    ]
    path.write_text("\n".join(lines) + "\n")


def main():
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "stage": "9b_build_minimal_catalytic_core_models",
        "role": (
            "Minimal catalytic-core convergence model for higher-level "
            "proton-microstate screening; not the final binding model."
        ),
        "definition": {
            "Thr1_region": "full Y:1-3 segment plus attached ACE/NME cap",
            "Asp17": "sidechain CB/CG/OD1/OD2 + attached H + CB link H",
            "Lys33": "sidechain CB/CG/CD/CE/NZ + attached H + CB link H",
            "ligand": "entire 6V8 or BO2",
            "waters": "excluded at this first core level",
            "link_CH_A": args.link_ch,
        },
        "cases": {},
    }

    for stem in STEMS:
        atoms = load_case(stem, args.optimized, args.metadata)
        core, links, ligand, heavy_delta, reasons = build_case(
            stem, atoms, args.link_ch
        )

        out = args.outdir / f"{stem}_core.xyz"
        write_xyz(
            out,
            core,
            links,
            (
                f"{stem} minimal catalytic core | "
                f"Y1-3 + Asp17 sidechain + Lys33 sidechain + {ligand} | "
                f"2 link H"
            ),
        )

        n_heavy = sum(a["element"] != "H" for a in core)
        n_h = sum(a["element"] == "H" for a in core) + len(links)
        elements = {}
        for a in core:
            elements[a["element"]] = elements.get(a["element"], 0) + 1
        elements["H"] = elements.get("H", 0) + len(links)

        report["cases"][stem] = {
            "output": str(out),
            "ligand": ligand,
            "n_atoms": len(core) + len(links),
            "n_heavy": n_heavy,
            "n_H": n_h,
            "elements": elements,
            "n_link_H": len(links),
            "heavy_max_delta_from_metadata_A": heavy_delta,
            "status": "PASS" if heavy_delta <= 1e-10 else "FAIL",
        }

        print(
            f"{stem:18s} atoms={len(core)+len(links):3d} "
            f"heavy={n_heavy:3d} H={n_h:3d} "
            f"heavy_delta={heavy_delta:.3e} A "
            f"{report['cases'][stem]['status']}"
        )

    # Composition must match across microstates of same ligand.
    for ligand in ("6V8", "BO2"):
        vals = [
            report["cases"][s]["elements"]
            for s in STEMS if s.startswith(ligand + "_")
        ]
        ok = all(v == vals[0] for v in vals[1:])
        report[f"{ligand}_composition_invariant"] = ok
        print(f"{ligand} composition invariant: {'PASS' if ok else 'FAIL'}")

    overall = all(
        c["status"] == "PASS" for c in report["cases"].values()
    ) and report["6V8_composition_invariant"] and report["BO2_composition_invariant"]

    report["overall"] = "PASS" if overall else "FAIL"
    args.report.write_text(json.dumps(report, indent=2) + "\n")

    print()
    print(f"Overall: {report['overall']}")
    print(f"Report : {args.report}")

    raise SystemExit(0 if overall else 2)


if __name__ == "__main__":
    main()
