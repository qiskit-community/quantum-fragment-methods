#!/usr/bin/env python3
"""
18_repair_special_hydrogen_geometries.py

Second-pass repair for hydrogenated proteasome microstates.

Starting point:
    intermediate/hydrogenated_microstates_repaired
which already contains the corrected Arg19 NE-H geometry.

Repairs performed:
1. Reorient the retained water nearest Z:125 ASP OD2 so that:
   - one O-H points toward Asp125 OD2,
   - the other is placed at a 104.5 degree HOH angle and points as closely
     as possible toward ligand O8.
   This targets the conserved Asp125/water/ligand network and removes the
   unconverged free water-rotation mode observed in BO2 MS_A.

2. For MS_C_ASP17 only, move the Asp17 carboxylic-acid proton from OD1 to OD2.
   OD1 is the oxygen closest to Lys33 NZ and should remain available as the
   H-bond acceptor in this tautomer.  The new OD2-H geometry is generated in
   the carboxyl plane at a 110 degree C-O-H angle; of the two planar
   orientations, the one with better steric clearance is chosen.

Heavy atoms are never changed.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--indir",
        type=Path,
        default=Path("intermediate/hydrogenated_microstates_repaired"),
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=Path("intermediate/hydrogenated_microstates_repaired_v2"),
    )
    return p.parse_args()


def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError("Cannot normalize near-zero vector")
    return v / n


def angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle ABC in degrees."""
    u = unit(a - b)
    v = unit(c - b)
    return float(np.degrees(np.arccos(np.clip(np.dot(u, v), -1.0, 1.0))))


def read_xyz(path: Path):
    lines = path.read_text().splitlines()
    n = int(lines[0])
    rows = []
    for line in lines[2 : 2 + n]:
        s = line.split()
        rows.append([s[0], np.array(list(map(float, s[1:4])), dtype=float)])
    if len(rows) != n:
        raise RuntimeError(f"{path}: XYZ atom count mismatch")
    return rows


def read_pdb(path: Path):
    lines = path.read_text().splitlines()
    atoms = []
    for lineno, line in enumerate(lines):
        if line.startswith(("ATOM  ", "HETATM")):
            elem = line[76:78].strip()
            if not elem:
                # safe enough for these generated structures
                name = line[12:16].strip()
                elem = "".join(ch for ch in name if ch.isalpha())[:1]
            atoms.append(
                {
                    "lineno": lineno,
                    "line": line,
                    "name": line[12:16].strip(),
                    "resname": line[17:20].strip(),
                    "chain": line[21:22].strip(),
                    "resseq": line[22:26].strip(),
                    "element": elem,
                    "xyz": np.array(
                        [
                            float(line[30:38]),
                            float(line[38:46]),
                            float(line[46:54]),
                        ],
                        dtype=float,
                    ),
                }
            )
    return lines, atoms


def label(a: Dict, i: int | None = None) -> str:
    prefix = f"idx={i} " if i is not None else ""
    return (
        f"{prefix}{a['chain']}:{a['resseq']} "
        f"{a['resname']} {a['name']} ({a['element']})"
    )


def find_atoms(atoms: List[Dict], **kw) -> List[Tuple[int, Dict]]:
    out = []
    for i, a in enumerate(atoms):
        good = True
        for k, v in kw.items():
            if a[k] != v:
                good = False
                break
        if good:
            out.append((i, a))
    return out


def unique_atom(atoms: List[Dict], **kw) -> Tuple[int, Dict]:
    hits = find_atoms(atoms, **kw)
    if len(hits) != 1:
        raise RuntimeError(f"Expected one atom for {kw}, found {len(hits)}")
    return hits[0]


def set_atom_position(
    atom_index: int,
    new_xyz: np.ndarray,
    pdb_lines: List[str],
    pdb_atoms: List[Dict],
    xyz_rows,
):
    a = pdb_atoms[atom_index]
    old = pdb_lines[a["lineno"]]
    pdb_lines[a["lineno"]] = (
        old[:30]
        + f"{new_xyz[0]:8.3f}{new_xyz[1]:8.3f}{new_xyz[2]:8.3f}"
        + old[54:]
    )
    a["xyz"] = new_xyz.copy()
    xyz_rows[atom_index][1] = new_xyz.copy()


def validate_correspondence(pdb_atoms, xyz_rows, stem):
    if len(pdb_atoms) != len(xyz_rows):
        raise RuntimeError(
            f"{stem}: PDB/XYZ atom count mismatch "
            f"{len(pdb_atoms)} != {len(xyz_rows)}"
        )
    for i, (p, x) in enumerate(zip(pdb_atoms, xyz_rows)):
        if p["element"].upper() != x[0].upper():
            raise RuntimeError(
                f"{stem}: PDB/XYZ element mismatch at {i}: "
                f"{p['element']} vs {x[0]}"
            )


def choose_bridge_water(atoms: List[Dict], asp_od2_xyz: np.ndarray):
    candidates = []
    for i, a in enumerate(atoms):
        if a["resname"] == "HOH" and a["element"].upper() == "O":
            d = float(np.linalg.norm(a["xyz"] - asp_od2_xyz))
            candidates.append((d, i, a))
    if not candidates:
        raise RuntimeError("No HOH oxygen atoms found")
    candidates.sort(key=lambda x: x[0])
    return candidates[0]


def orient_bridge_water(
    atoms: List[Dict],
    pdb_lines: List[str],
    xyz_rows,
):
    # Target acceptor 1: Z:125 ASP OD2
    _, asp_od2 = unique_atom(
        atoms,
        chain="Z",
        resseq="125",
        resname="ASP",
        name="OD2",
    )

    # Target acceptor 2: ligand O8
    ligand_o8_hits = [
        (i, a)
        for i, a in enumerate(atoms)
        if a["name"] == "O8" and a["resname"] in {"6V8", "BO2"}
    ]
    if len(ligand_o8_hits) != 1:
        raise RuntimeError(
            f"Expected exactly one ligand O8, found {len(ligand_o8_hits)}"
        )
    _, ligand_o8 = ligand_o8_hits[0]

    d_asp, i_ow, ow = choose_bridge_water(atoms, asp_od2["xyz"])

    # Find the two hydrogens belonging to this same water residue.
    wh = [
        (i, a)
        for i, a in enumerate(atoms)
        if a["chain"] == ow["chain"]
        and a["resseq"] == ow["resseq"]
        and a["resname"] == "HOH"
        and a["element"].upper() == "H"
    ]
    if len(wh) != 2:
        raise RuntimeError(
            f"{label(ow, i_ow)}: expected 2 water H atoms, found {len(wh)}"
        )
    wh = sorted(wh, key=lambda t: t[1]["name"])

    O = ow["xyz"]
    a = unit(asp_od2["xyz"] - O)
    b = unit(ligand_o8["xyz"] - O)

    OH = 0.970
    theta = math.radians(104.5)

    # H1 points directly toward Asp125 OD2.
    v1 = a

    # H2 must be 104.5 degrees from H1.  Within that cone, choose the
    # direction with maximum alignment to ligand O8.
    b_perp = b - np.dot(b, a) * a
    if np.linalg.norm(b_perp) < 1e-8:
        # Fallback: construct a deterministic perpendicular vector.
        trial = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(trial, a)) > 0.9:
            trial = np.array([0.0, 1.0, 0.0])
        b_perp = trial - np.dot(trial, a) * a
    p = unit(b_perp)

    v2a = math.cos(theta) * a + math.sin(theta) * p
    v2b = math.cos(theta) * a - math.sin(theta) * p
    v2 = v2a if np.dot(v2a, b) >= np.dot(v2b, b) else v2b
    v2 = unit(v2)

    H1 = O + OH * v1
    H2 = O + OH * v2

    set_atom_position(wh[0][0], H1, pdb_lines, atoms, xyz_rows)
    set_atom_position(wh[1][0], H2, pdb_lines, atoms, xyz_rows)

    return {
        "water": label(ow, i_ow),
        "O_AspOD2_A": float(np.linalg.norm(O - asp_od2["xyz"])),
        "O_ligandO8_A": float(np.linalg.norm(O - ligand_o8["xyz"])),
        "OH1_A": float(np.linalg.norm(H1 - O)),
        "OH2_A": float(np.linalg.norm(H2 - O)),
        "HOH_deg": angle_deg(H1, O, H2),
        "H1_AspOD2_A": float(np.linalg.norm(H1 - asp_od2["xyz"])),
        "H2_ligandO8_A": float(np.linalg.norm(H2 - ligand_o8["xyz"])),
    }


def steric_score(
    candidate: np.ndarray,
    atoms: List[Dict],
    exclude_indices: set[int],
) -> float:
    """Larger is better: minimum distance to all nonexcluded atoms."""
    distances = []
    for i, a in enumerate(atoms):
        if i in exclude_indices:
            continue
        distances.append(float(np.linalg.norm(candidate - a["xyz"])))
    return min(distances)


def repair_ms_c_asp17(
    atoms: List[Dict],
    pdb_lines: List[str],
    xyz_rows,
):
    i_cg, cg = unique_atom(
        atoms, chain="Y", resseq="17", resname="ASP", name="CG"
    )
    i_od1, od1 = unique_atom(
        atoms, chain="Y", resseq="17", resname="ASP", name="OD1"
    )
    i_od2, od2 = unique_atom(
        atoms, chain="Y", resseq="17", resname="ASP", name="OD2"
    )
    i_hd, hd = unique_atom(
        atoms, chain="Y", resseq="17", resname="ASP", name="HD"
    )
    i_nz, nz = unique_atom(
        atoms, chain="Y", resseq="33", resname="LYS", name="NZ"
    )

    # Carboxyl plane from OD1-CG-OD2.
    u = unit(cg["xyz"] - od2["xyz"])  # OD2 -> CG
    plane_normal = unit(
        np.cross(od1["xyz"] - cg["xyz"], od2["xyz"] - cg["xyz"])
    )
    p = unit(np.cross(plane_normal, u))

    OH = 0.990
    theta = math.radians(110.0)  # CG-OD2-H

    v_plus = unit(math.cos(theta) * u + math.sin(theta) * p)
    v_minus = unit(math.cos(theta) * u - math.sin(theta) * p)

    cand_plus = od2["xyz"] + OH * v_plus
    cand_minus = od2["xyz"] + OH * v_minus

    # Exclude the proton itself and its new parent oxygen from steric score.
    exclude = {i_hd, i_od2}
    score_plus = steric_score(cand_plus, atoms, exclude)
    score_minus = steric_score(cand_minus, atoms, exclude)

    new_h = cand_plus if score_plus >= score_minus else cand_minus
    chosen_score = max(score_plus, score_minus)

    old_parent_od1 = float(np.linalg.norm(hd["xyz"] - od1["xyz"]))
    old_parent_od2 = float(np.linalg.norm(hd["xyz"] - od2["xyz"]))

    set_atom_position(i_hd, new_h, pdb_lines, atoms, xyz_rows)

    # Distances to Lys33 hydrogens are useful diagnostics.
    lys_h = [
        (i, a)
        for i, a in enumerate(atoms)
        if a["chain"] == "Y"
        and a["resseq"] == "33"
        and a["resname"] == "LYS"
        and a["element"].upper() == "H"
    ]
    lys_h_d = sorted(
        float(np.linalg.norm(new_h - a["xyz"])) for _, a in lys_h
    )

    return {
        "old_HD_OD1_A": old_parent_od1,
        "old_HD_OD2_A": old_parent_od2,
        "new_HD_OD2_A": float(np.linalg.norm(new_h - od2["xyz"])),
        "CG_OD2_HD_deg": angle_deg(cg["xyz"], od2["xyz"], new_h),
        "HD_Lys33_NZ_A": float(np.linalg.norm(new_h - nz["xyz"])),
        "nearest_HD_Lys33H_A": lys_h_d[0] if lys_h_d else None,
        "steric_clearance_A": chosen_score,
    }


def main():
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    pdbs = sorted(args.indir.glob("*.pdb"))
    if not pdbs:
        raise SystemExit(f"No PDB files in {args.indir}")

    for pdb in pdbs:
        stem = pdb.stem
        xyz = args.indir / f"{stem}.xyz"
        if not xyz.exists():
            raise RuntimeError(f"Missing {xyz}")

        pdb_lines, atoms = read_pdb(pdb)
        xyz_rows = read_xyz(xyz)
        validate_correspondence(atoms, xyz_rows, stem)

        heavy_before = np.array(
            [a["xyz"].copy() for a in atoms if a["element"].upper() != "H"]
        )

        water_info = orient_bridge_water(atoms, pdb_lines, xyz_rows)
        asp_info = None
        if "_MS_C_ASP17" in stem:
            asp_info = repair_ms_c_asp17(
                atoms, pdb_lines, xyz_rows
            )

        heavy_after = np.array(
            [a["xyz"].copy() for a in atoms if a["element"].upper() != "H"]
        )
        heavy_delta = np.linalg.norm(heavy_after - heavy_before, axis=1)
        heavy_max = float(heavy_delta.max()) if len(heavy_delta) else 0.0
        if heavy_max != 0.0:
            raise RuntimeError(
                f"{stem}: heavy atoms changed unexpectedly: {heavy_max}"
            )

        out_xyz = [
            str(len(xyz_rows)),
            f"{stem} | Arg19 repaired + controlled bridge water"
            + (" + Asp17 OD2-H tautomer" if asp_info else ""),
        ]
        out_xyz += [
            f"{sym:<2s} {r[0]:18.10f} {r[1]:18.10f} {r[2]:18.10f}"
            for sym, r in xyz_rows
        ]

        (args.outdir / pdb.name).write_text(
            "\n".join(pdb_lines) + "\n"
        )
        (args.outdir / xyz.name).write_text(
            "\n".join(out_xyz) + "\n"
        )

        print()
        print(stem)
        print(f"  heavy max displacement: {heavy_max:.3e} A")
        print(
            f"  bridge water: {water_info['water']}, "
            f"O-Asp125={water_info['O_AspOD2_A']:.3f} A, "
            f"O-ligO8={water_info['O_ligandO8_A']:.3f} A"
        )
        print(
            f"  water geometry: "
            f"OH={water_info['OH1_A']:.3f}/{water_info['OH2_A']:.3f} A, "
            f"HOH={water_info['HOH_deg']:.2f} deg, "
            f"H->Asp={water_info['H1_AspOD2_A']:.3f} A, "
            f"H->ligO8={water_info['H2_ligandO8_A']:.3f} A"
        )
        if asp_info:
            print(
                f"  MS_C Asp17: old HD-OD1={asp_info['old_HD_OD1_A']:.3f} A, "
                f"new HD-OD2={asp_info['new_HD_OD2_A']:.3f} A, "
                f"CG-OD2-HD={asp_info['CG_OD2_HD_deg']:.2f} deg"
            )
            print(
                f"  MS_C clearance: HD-Lys33NZ="
                f"{asp_info['HD_Lys33_NZ_A']:.3f} A, "
                f"nearest HD-Lys33H="
                f"{asp_info['nearest_HD_Lys33H_A']:.3f} A"
            )

    print()
    print("Wrote repaired v2 structures to:", args.outdir)


if __name__ == "__main__":
    main()
