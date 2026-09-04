#!/usr/bin/env python3
"""
23_freeze_production_models.py

Freeze the full production geometries for the Proteasome Challenge using the
common MS_B_LYS33 proton microstate.

Construction
------------
Canonical receptor:
    receptor atoms/coordinates from 6V8_MS_B_LYS33

Ixazomib complex:
    canonical receptor + 6V8 ligand from 6V8_MS_B_LYS33

Bortezomib complex:
    SAME canonical receptor + BO2 ligand from BO2_MS_B_LYS33

This guarantees exact receptor coordinate/order identity in the two final
complexes, which is required for the matched relative-energy construction.

The isolated ligand XYZ files use exactly the same ligand coordinates and atom
order as in their respective complexes.

Charge bookkeeping
------------------
Full complexes:
    q = -1

Extracted isolated ligand:
    q = 0

Why q(ligand)=0?
After removing the Thr1-OG1 -> B coordinate bond, the inhibitor boron returns
to a three-coordinate R-B(OH)2-like fragment.  The two B-bound ligand oxygens
retain their hydrogens.  The -1 formal charge of the tetrahedral covalent
boronate adduct therefore belongs to the *bound* four-coordinate state, not
the extracted three-coordinate ligand.

The implied common receptor fragment produced by simply cutting the Thr1-O-B
bond without moving protons is q = -1 (Thr1 alkoxide in MS_B).  We do not need
to compute its standalone energy because it cancels exactly in the matched
observable.

This script does not perform any geometry optimization.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

CASES = {
    "ixazomib": ("6V8_MS_B_LYS33", "6V8"),
    "bortezomib": ("BO2_MS_B_LYS33", "BO2"),
}


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
        default=Path("production_models"),
    )
    p.add_argument(
        "--manifest",
        type=Path,
        default=Path("validation/production_model_manifest.json"),
    )
    p.add_argument(
        "--canonical-receptor",
        choices=["6V8", "BO2"],
        default="6V8",
        help="Use 6V8/MS_B by default because the canonical receptor is 5LF7-Y.",
    )
    p.add_argument("--heavy-tol", type=float, default=1e-8)
    p.add_argument("--clash-threshold", type=float, default=0.70)
    return p.parse_args()


def read_xyz(path: Path):
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
        raise RuntimeError(f"{path}: expected {n} XYZ atoms, found {len(atoms)}")
    return atoms


def read_pdb(path: Path):
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


def load_case(stem: str, optimized: Path, metadata: Path):
    xyz_path = optimized / f"{stem}.xyz"
    pdb_path = metadata / f"{stem}.pdb"
    if not xyz_path.exists():
        raise FileNotFoundError(xyz_path)
    if not pdb_path.exists():
        raise FileNotFoundError(pdb_path)

    xyz = read_xyz(xyz_path)
    meta = read_pdb(pdb_path)
    if len(xyz) != len(meta):
        raise RuntimeError(
            f"{stem}: XYZ/PDB atom count mismatch {len(xyz)} != {len(meta)}"
        )

    out = []
    for i, (x, m) in enumerate(zip(xyz, meta)):
        if x["element"] != m["element"]:
            raise RuntimeError(
                f"{stem}: atom-order mismatch at {i}: "
                f"{x['element']} != {m['element']}"
            )
        a = dict(m)
        a["xyz"] = x["xyz"]
        a["source_index"] = i
        out.append(a)
    return out


def atom_key(a):
    return (a["chain"], a["resseq"], a["resname"], a["name"], a["element"])


def split_case(atoms, ligand_resname):
    lig = [a for a in atoms if a["resname"] == ligand_resname]
    rec = [a for a in atoms if a["resname"] != ligand_resname]
    if not lig:
        raise RuntimeError(f"No ligand residue {ligand_resname} found")
    return rec, lig


def write_xyz(path: Path, atoms, comment: str):
    lines = [str(len(atoms)), comment]
    for a in atoms:
        x, y, z = a["xyz"]
        lines.append(
            f"{a['element']:<2s} {x:18.10f} {y:18.10f} {z:18.10f}"
        )
    path.write_text("\n".join(lines) + "\n")


def d(a, b):
    return float(np.linalg.norm(a["xyz"] - b["xyz"]))


def find_one(atoms, *, chain=None, resseq=None, resname=None, name=None, element=None):
    out = []
    for a in atoms:
        if chain is not None and a["chain"] != chain:
            continue
        if resseq is not None and a["resseq"] != str(resseq):
            continue
        if resname is not None and a["resname"] != resname:
            continue
        if name is not None and a["name"] != name:
            continue
        if element is not None and a["element"] != element:
            continue
        out.append(a)
    if len(out) != 1:
        raise RuntimeError(
            f"Expected exactly one atom for chain={chain} resseq={resseq} "
            f"resname={resname} name={name} element={element}; found {len(out)}"
        )
    return out[0]


def b_coordination(ligand):
    borons = [a for a in ligand if a["element"] == "B"]
    if len(borons) != 1:
        raise RuntimeError(f"Expected one B atom in ligand, found {len(borons)}")
    b = borons[0]
    neigh = []
    for a in ligand:
        if a is b or a["element"] == "H":
            continue
        dd = d(a, b)
        if dd <= 1.90:
            neigh.append((dd, a))
    neigh.sort(key=lambda x: x[0])
    return b, neigh


def attached_h_count(ligand, heavy, cutoff=1.25):
    return sum(
        1 for h in ligand
        if h["element"] == "H" and d(h, heavy) <= cutoff
    )


def min_cross_distance(receptor, ligand, exclude_thr_b=True):
    best = None
    for r in receptor:
        for l in ligand:
            # Exclude the intended Thr1 OG1-B covalent contact.
            if exclude_thr_b:
                intended = (
                    r["chain"] == "Y"
                    and r["resseq"] == "1"
                    and r["resname"] == "THR"
                    and r["name"] == "OG1"
                    and l["element"] == "B"
                )
                if intended:
                    continue
            dd = d(r, l)
            if best is None or dd < best[0]:
                best = (dd, atom_key(r), atom_key(l))
    return best


def main():
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)

    loaded = {}
    split = {}

    for drug, (stem, ligand_resname) in CASES.items():
        atoms = load_case(stem, args.optimized, args.metadata)
        loaded[drug] = atoms
        split[drug] = split_case(atoms, ligand_resname)

    # Select canonical receptor.
    canonical_drug = (
        "ixazomib" if args.canonical_receptor == "6V8" else "bortezomib"
    )
    canonical_rec = split[canonical_drug][0]

    # Verify the two source receptor atom lists are chemically identical.
    ix_rec = split["ixazomib"][0]
    bo_rec = split["bortezomib"][0]
    if len(ix_rec) != len(bo_rec):
        raise RuntimeError(
            f"Source receptor atom counts differ: {len(ix_rec)} vs {len(bo_rec)}"
        )

    key_mismatch = []
    max_heavy_source_delta = 0.0
    max_all_source_delta = 0.0
    for i, (a, b) in enumerate(zip(ix_rec, bo_rec)):
        if atom_key(a) != atom_key(b):
            key_mismatch.append((i, atom_key(a), atom_key(b)))
            continue
        dd = d(a, b)
        max_all_source_delta = max(max_all_source_delta, dd)
        if a["element"] != "H":
            max_heavy_source_delta = max(max_heavy_source_delta, dd)

    if key_mismatch:
        raise RuntimeError(
            f"Receptor atom identity/order mismatch. First: {key_mismatch[0]}"
        )

    if max_heavy_source_delta > args.heavy_tol:
        raise RuntimeError(
            f"Source receptor heavy atoms differ by {max_heavy_source_delta:.3e} A "
            f"(tol={args.heavy_tol:.3e})"
        )

    outputs = {}
    manifest = {
        "stage": "freeze_production_models",
        "selected_microstate": "MS_B_LYS33",
        "selection_basis": {
            "GFN2_whole_cluster": {
                "6V8_delta_B_minus_A_kcal_mol": -7.2281,
                "BO2_delta_B_minus_A_kcal_mol": 25.2008,
                "note": "Semiempirical prescreen; ligand-dependent switch not confirmed by DFT.",
            },
            "PBE0_def2SVP_catalytic_core": {
                "6V8_delta_B_minus_A_kcal_mol": -25.15,
                "BO2_delta_B_minus_A_kcal_mol": -19.34,
                "note": "Both inhibitors strongly favor MS_B in the reduced catalytic core.",
            },
        },
        "canonical_receptor_source": canonical_drug,
        "charges": {
            "complex": -1,
            "isolated_ligand": 0,
            "implied_cut_receptor": -1,
        },
        "source_receptor_comparison": {
            "n_atoms": len(ix_rec),
            "max_heavy_coordinate_delta_A": max_heavy_source_delta,
            "max_all_atom_coordinate_delta_A": max_all_source_delta,
            "note": (
                "Final complexes use a single canonical receptor coordinate set, "
                "so final receptor identity is exact by construction."
            ),
        },
        "models": {},
    }

    # Optional reference receptor file.
    receptor_path = args.outdir / "common_receptor.xyz"
    write_xyz(
        receptor_path,
        canonical_rec,
        "Common canonical receptor | MS_B_LYS33 | implied cut-fragment charge -1",
    )
    outputs["common_receptor"] = str(receptor_path)

    for drug, (stem, ligand_resname) in CASES.items():
        ligand = split[drug][1]
        complex_atoms = canonical_rec + ligand

        complex_path = args.outdir / f"{drug}_complex.xyz"
        ligand_path = args.outdir / f"{drug}_ligand.xyz"

        write_xyz(
            complex_path,
            complex_atoms,
            f"{drug} production complex | MS_B_LYS33 | charge -1",
        )
        write_xyz(
            ligand_path,
            ligand,
            (
                f"{drug} extracted bound-geometry ligand | "
                f"three-coordinate boronic acid reference | charge 0"
            ),
        )

        # Core chemistry checks.
        thr_og = find_one(
            canonical_rec,
            chain="Y", resseq="1", resname="THR", name="OG1",
        )
        b, neigh = b_coordination(ligand)
        thr_b = d(thr_og, b)

        # The isolated ligand B should have exactly 3 heavy neighbors:
        # two O plus one C.
        neigh_summary = [
            {
                "distance_A": dd,
                "atom": atom_key(a),
                "attached_H": attached_h_count(ligand, a)
                if a["element"] == "O" else None,
            }
            for dd, a in neigh
        ]
        neigh_elements = sorted(a["element"] for _, a in neigh)
        b3_ok = len(neigh) == 3 and neigh_elements == ["C", "O", "O"]

        # Both B-bound ligand oxygens should still be protonated.
        bo_oxygens = [a for _, a in neigh if a["element"] == "O"]
        bo_oh_ok = len(bo_oxygens) == 2 and all(
            attached_h_count(ligand, o) == 1 for o in bo_oxygens
        )

        cross = min_cross_distance(canonical_rec, ligand)
        clash_ok = cross[0] >= args.clash_threshold

        status = "PASS" if (b3_ok and bo_oh_ok and clash_ok) else "FAIL"

        manifest["models"][drug] = {
            "source": stem,
            "ligand_resname": ligand_resname,
            "complex_file": str(complex_path),
            "ligand_file": str(ligand_path),
            "complex_charge": -1,
            "ligand_charge": 0,
            "n_receptor_atoms": len(canonical_rec),
            "n_ligand_atoms": len(ligand),
            "n_complex_atoms": len(complex_atoms),
            "n_receptor_heavy": sum(a["element"] != "H" for a in canonical_rec),
            "n_ligand_heavy": sum(a["element"] != "H" for a in ligand),
            "n_complex_heavy": sum(a["element"] != "H" for a in complex_atoms),
            "Thr1_OG1_B_distance_A": thr_b,
            "isolated_ligand_B_heavy_coordination": neigh_summary,
            "isolated_ligand_B_is_three_coordinate_COO": b3_ok,
            "both_B_bound_O_are_OH": bo_oh_ok,
            "min_non_covalent_receptor_ligand_distance_A": cross[0],
            "min_non_covalent_contact_receptor": cross[1],
            "min_non_covalent_contact_ligand": cross[2],
            "status": status,
        }

        print(
            f"{drug:12s} complex={len(complex_atoms):3d} "
            f"receptor={len(canonical_rec):3d} ligand={len(ligand):2d} "
            f"ThrOG-B={thr_b:.3f} A "
            f"min_cross={cross[0]:.3f} A {status}"
        )

    # Exact matched receptor guarantee in outputs.
    manifest["final_matched_receptor"] = {
        "atom_count_equal": True,
        "atom_order_equal": True,
        "coordinates_equal": True,
        "max_coordinate_delta_A": 0.0,
        "by_construction": True,
    }

    overall = all(
        m["status"] == "PASS" for m in manifest["models"].values()
    )
    manifest["overall"] = "PASS" if overall else "FAIL"
    manifest["outputs"] = outputs

    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")

    print()
    print(f"Overall: {manifest['overall']}")
    print(f"Manifest: {args.manifest}")
    print(f"Models  : {args.outdir}")

    raise SystemExit(0 if overall else 2)


if __name__ == "__main__":
    main()
