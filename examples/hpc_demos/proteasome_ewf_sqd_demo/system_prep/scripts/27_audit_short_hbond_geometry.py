#!/usr/bin/env python3
"""
27_audit_short_hbond_geometry.py

Audit the shortest H-heavy receptor-ligand contact in the refined production
models by identifying the hydrogen's bonded heavy-atom parent and computing
donor-H...acceptor geometry.

This is intended as the final chemical sanity check before freezing the
production models.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np


CASES = {
    "ixazomib": {
        "stem": "6V8_MS_B_LYS33",
        "ligand_resname": "6V8",
        "complex": "ixazomib_complex.xyz",
    },
    "bortezomib": {
        "stem": "BO2_MS_B_LYS33",
        "ligand_resname": "BO2",
        "complex": "bortezomib_complex.xyz",
    },
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--production",
        type=Path,
        default=Path("production_models_refined"),
    )
    p.add_argument(
        "--metadata",
        type=Path,
        default=Path("intermediate/hydrogenated_microstates_repaired_v2"),
    )
    p.add_argument("--n-receptor", type=int, default=491)
    return p.parse_args()


def read_xyz(path):
    lines = path.read_text().splitlines()
    n = int(lines[0])
    atoms = []
    for i, line in enumerate(lines[2:2+n]):
        s = line.split()
        atoms.append({
            "index": i,
            "element": s[0].upper(),
            "xyz": np.array([float(s[1]), float(s[2]), float(s[3])]),
        })
    return atoms


def read_pdb(path):
    atoms = []
    for line in path.read_text().splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        elem = line[76:78].strip().upper()
        if not elem:
            name = line[12:16].strip()
            elem = "".join(c for c in name if c.isalpha())[:1].upper()
        atoms.append({
            "name": line[12:16].strip(),
            "resname": line[17:20].strip(),
            "chain": line[21:22].strip(),
            "resseq": line[22:26].strip(),
            "element": elem,
        })
    return atoms


def merge(xyz, meta):
    if len(xyz) != len(meta):
        raise RuntimeError(f"count mismatch {len(xyz)} != {len(meta)}")
    out = []
    for i, (x, m) in enumerate(zip(xyz, meta)):
        if x["element"] != m["element"]:
            raise RuntimeError(
                f"element mismatch at {i}: {x['element']} != {m['element']}"
            )
        a = dict(m)
        a["xyz"] = x["xyz"]
        a["index"] = i
        out.append(a)
    return out


def dist(a, b):
    return float(np.linalg.norm(a["xyz"] - b["xyz"]))


def angle(a, b, c):
    """
    angle ABC in degrees.
    """
    v1 = a["xyz"] - b["xyz"]
    v2 = c["xyz"] - b["xyz"]
    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)
    x = np.clip(np.dot(v1, v2), -1.0, 1.0)
    return float(np.degrees(np.arccos(x)))


def label(a):
    return (
        f"{a['chain']}:{a['resseq']} {a['resname']} "
        f"{a['name']}({a['element']})"
    )


def nearest_heavy_parent(h, lig):
    cand = [
        (dist(h, a), a)
        for a in lig
        if a["element"] != "H"
    ]
    cand.sort(key=lambda x: x[0])
    return cand[0]


def main():
    args = parse_args()

    for drug, cfg in CASES.items():
        xyz = read_xyz(args.production / cfg["complex"])
        meta = read_pdb(args.metadata / f"{cfg['stem']}.pdb")

        # Metadata order is receptor then ligand for these source models.
        rec_meta = [a for a in meta if a["resname"] != cfg["ligand_resname"]]
        lig_meta = [a for a in meta if a["resname"] == cfg["ligand_resname"]]

        rec = merge(xyz[:args.n_receptor], rec_meta)
        lig = merge(xyz[args.n_receptor:], lig_meta)

        # Find shortest receptor-ligand H-heavy contact excluding intended Thr1-B.
        pairs = []
        for r in rec:
            for l in lig:
                intended = (
                    r["chain"] == "Y"
                    and r["resseq"] == "1"
                    and r["resname"] == "THR"
                    and r["name"] == "OG1"
                    and l["element"] == "B"
                )
                if intended:
                    continue

                rh = r["element"] == "H"
                lh = l["element"] == "H"
                if rh == lh:
                    continue  # only exactly one H

                pairs.append((dist(r, l), r, l))

        pairs.sort(key=lambda x: x[0])
        dd, r, l = pairs[0]

        if l["element"] == "H":
            h = l
            acceptor = r
            parent_dist, donor = nearest_heavy_parent(h, lig)
            side = "ligand donor -> receptor acceptor"
        else:
            h = r
            acceptor = l
            # receptor parent search within same residue first
            cand = [
                (dist(h, a), a)
                for a in rec
                if a["element"] != "H"
                and (a["chain"], a["resseq"], a["resname"])
                == (h["chain"], h["resseq"], h["resname"])
            ]
            cand.sort(key=lambda x: x[0])
            parent_dist, donor = cand[0]
            side = "receptor donor -> ligand acceptor"

        dha = angle(donor, h, acceptor)
        da = dist(donor, acceptor)

        print()
        print("=" * 86)
        print(drug.upper())
        print("=" * 86)
        print(f"shortest H-heavy contact : {dd:.3f} A")
        print(f"type                     : {side}")
        print(f"donor heavy atom         : {label(donor)}")
        print(f"hydrogen                 : {label(h)}")
        print(f"acceptor                  : {label(acceptor)}")
        print(f"D-H distance             : {parent_dist:.3f} A")
        print(f"H...A distance           : {dd:.3f} A")
        print(f"D...A distance           : {da:.3f} A")
        print(f"D-H...A angle            : {dha:.1f} deg")

        plausible = (
            0.85 <= parent_dist <= 1.20
            and 1.40 <= dd <= 2.30
            and 2.30 <= da <= 3.40
            and dha >= 120.0
        )

        if plausible:
            print("assessment               : PLAUSIBLE HYDROGEN BOND")
        else:
            print("assessment               : REVIEW GEOMETRY")

    print()
    print("Interpretation guide:")
    print("  Strong protein-ligand H bonds often have H...A ~1.5-2.0 A")
    print("  and D-H...A angles approaching linearity.")
    print("  The geometry is more important than H...A distance alone.")


if __name__ == "__main__":
    main()
