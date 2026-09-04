#!/usr/bin/env python3
"""
25_inspect_short_receptor_ligand_contacts.py

Identify and chemically label the shortest receptor-ligand contacts in the
frozen production models.

Purpose
-------
The final production-model audit passed, but the bortezomib complex contains a
non-Thr1-B receptor/ligand distance of 1.302 A. This script determines exactly
which atoms create that contact and whether it is H-H, H-heavy, or heavy-heavy.

It uses the original PDB metadata only for atom labels; coordinates come from
the frozen production XYZ files.
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
        "ligand": "ixazomib_ligand.xyz",
    },
    "bortezomib": {
        "stem": "BO2_MS_B_LYS33",
        "ligand_resname": "BO2",
        "complex": "bortezomib_complex.xyz",
        "ligand": "bortezomib_ligand.xyz",
    },
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--production",
        type=Path,
        default=Path("production_models"),
    )
    p.add_argument(
        "--metadata",
        type=Path,
        default=Path("intermediate/hydrogenated_microstates_repaired_v2"),
    )
    p.add_argument("--cutoff", type=float, default=2.0)
    p.add_argument("--top", type=int, default=30)
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
    if len(atoms) != n:
        raise RuntimeError(f"{path}: expected {n} atoms, found {len(atoms)}")
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


def dist(a, b):
    return float(np.linalg.norm(a["xyz"] - b["xyz"]))


def label(a):
    chain = a.get("chain", "")
    resseq = a.get("resseq", "")
    resname = a.get("resname", "")
    name = a.get("name", "")
    return f"{chain}:{resseq} {resname} {name}({a['element']})"


def merge_xyz_meta(xyz, meta):
    if len(xyz) != len(meta):
        raise RuntimeError(f"XYZ/PDB mismatch {len(xyz)} != {len(meta)}")
    out = []
    for i, (x, m) in enumerate(zip(xyz, meta)):
        if x["element"] != m["element"]:
            raise RuntimeError(
                f"atom-order mismatch at {i}: {x['element']} != {m['element']}"
            )
        a = dict(m)
        a["xyz"] = x["xyz"]
        a["index"] = i
        out.append(a)
    return out


def classify(a, b):
    ah = a["element"] == "H"
    bh = b["element"] == "H"
    if ah and bh:
        return "H-H"
    if ah or bh:
        return "H-heavy"
    return "heavy-heavy"


def find_b(lig):
    bs = [a for a in lig if a["element"] == "B"]
    if len(bs) != 1:
        raise RuntimeError(f"Expected one ligand B, found {len(bs)}")
    return bs[0]


def is_intended_thr_b(r, l):
    return (
        r["chain"] == "Y"
        and r["resseq"] == "1"
        and r["resname"] == "THR"
        and r["name"] == "OG1"
        and l["element"] == "B"
    )


def main():
    args = parse_args()

    all_case_data = {}

    # Frozen common receptor is the first 491 atoms of each complex.
    for drug, cfg in CASES.items():
        comp_xyz = read_xyz(args.production / cfg["complex"])
        lig_xyz = read_xyz(args.production / cfg["ligand"])
        meta = read_pdb(args.metadata / f"{cfg['stem']}.pdb")

        # Metadata from source full complex. Split using residue identity.
        if len(meta) != len(comp_xyz):
            # Final BO2 complex uses canonical receptor from 6V8, so source metadata
            # still has the same receptor atom ordering/count; labels remain valid.
            if len(meta) != 491 + len(lig_xyz):
                raise RuntimeError(
                    f"{drug}: metadata count {len(meta)} incompatible with "
                    f"production complex {len(comp_xyz)}"
                )

        rec_meta = [a for a in meta if a["resname"] != cfg["ligand_resname"]]
        lig_meta = [a for a in meta if a["resname"] == cfg["ligand_resname"]]

        rec_xyz = comp_xyz[:491]
        comp_lig_xyz = comp_xyz[491:]

        rec = merge_xyz_meta(rec_xyz, rec_meta)
        lig = merge_xyz_meta(comp_lig_xyz, lig_meta)

        contacts = []
        for r in rec:
            for l in lig:
                if is_intended_thr_b(r, l):
                    continue
                dd = dist(r, l)
                contacts.append((dd, r, l))

        contacts.sort(key=lambda x: x[0])
        all_case_data[drug] = (rec, lig, contacts)

        thr = next(
            r for r in rec
            if r["chain"] == "Y" and r["resseq"] == "1"
            and r["resname"] == "THR" and r["name"] == "OG1"
        )
        b = find_b(lig)

        print()
        print("=" * 84)
        print(drug.upper())
        print("=" * 84)
        print(f"Thr1 OG1-B intended covalent distance: {dist(thr, b):.3f} A")
        print(f"Shortest non-Thr-B contact: {contacts[0][0]:.3f} A")
        print()
        print(f"Contacts <= {args.cutoff:.2f} A (up to top {args.top}):")
        print("-" * 84)

        shown = 0
        for dd, r, l in contacts:
            if dd > args.cutoff or shown >= args.top:
                break
            print(
                f"{dd:7.3f} A  {classify(r,l):11s}  "
                f"{label(r):28s}  <->  {label(l)}"
            )
            shown += 1

        if shown == 0:
            print("none")

    # Compare the receptor atom involved in BO2's shortest contact against the
    # ixazomib ligand environment.
    bo_rec, bo_lig, bo_contacts = all_case_data["bortezomib"]
    ix_rec, ix_lig, ix_contacts = all_case_data["ixazomib"]

    dd0, r0, l0 = bo_contacts[0]
    key0 = (r0["chain"], r0["resseq"], r0["resname"], r0["name"], r0["element"])

    print()
    print("=" * 84)
    print("SHORTEST BO2 CONTACT DIAGNOSIS")
    print("=" * 84)
    print(f"BO2 pair : {label(r0)} <-> {label(l0)}")
    print(f"distance : {dd0:.3f} A")
    print(f"class    : {classify(r0,l0)}")

    if classify(r0, l0) == "H-H":
        print("initial interpretation: likely hydrogen-hydrogen steric clash")
    elif classify(r0, l0) == "H-heavy":
        print(
            "initial interpretation: could be a legitimate strong H-bond/contact "
            "or an H-placement clash; inspect donor-parent geometry"
        )
    else:
        print(
            "initial interpretation: unusually short heavy-heavy contact; requires "
            "manual chemistry inspection before freezing"
        )

    # Same named receptor atom in ixazomib; report its nearest ligand atom.
    ix_r_hits = [
        r for r in ix_rec
        if (r["chain"], r["resseq"], r["resname"], r["name"], r["element"]) == key0
    ]
    if len(ix_r_hits) == 1:
        ix_r = ix_r_hits[0]
        nearest = sorted((dist(ix_r, l), l) for l in ix_lig)[0]
        print()
        print(
            f"Same receptor atom in ixazomib: nearest ligand contact "
            f"{nearest[0]:.3f} A to {label(nearest[1])}"
        )

    print()
    print("Decision rule:")
    print("  H-H < ~1.2 A       -> repair required")
    print("  H-heavy ~1.3 A     -> inspect donor-parent geometry before deciding")
    print("  heavy-heavy 1.3 A  -> repair/model review required")
    print("  chemically sensible X-H...Y contact -> no repair; freeze models")


if __name__ == "__main__":
    main()
