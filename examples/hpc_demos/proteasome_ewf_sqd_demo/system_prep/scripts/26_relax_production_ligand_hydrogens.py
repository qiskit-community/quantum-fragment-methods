#!/usr/bin/env python3
"""
26_relax_production_ligand_hydrogens.py

Final hydrogen refinement for the frozen production models.

Problem addressed
-----------------
After imposing the canonical 6V8 receptor on the BO2 complex, the final contact
audit found a 1.302 A H-H contact:
    common receptor water Y:953 H2  <->  BO2 boronate HO27

The heavy-atom model is already validated and MUST NOT change.  The common
receptor (including all receptor hydrogens) must also remain exactly identical
between the two complexes.

Solution
--------
For BOTH ixazomib and bortezomib complexes, optimize only ligand hydrogens with:
    ASE optimizer + GFN-FF forces
while fixing:
    - every receptor atom, including receptor H
    - every ligand heavy atom

This preserves:
    - exact matched receptor identity
    - all experimental/model heavy-atom coordinates
    - ligand composition and atom order
    - proton parent identities

The resulting ligand XYZ files are re-extracted directly from the refined
complex suffixes, so complex/ligand coordinate identity remains exact.

GFN-FF energies are geometry-preparation quantities only and are not used for
binding energetics.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Tuple

import numpy as np

try:
    from ase.io import read, write
    from ase.constraints import FixAtoms
    from ase.optimize import FIRE, LBFGS
except ImportError as exc:
    raise SystemExit(
        'ASE is required. Install with: python3 -m pip install "gfnff[ase]"'
    ) from exc

try:
    from gfnff import GFNFF
except ImportError as exc:
    raise SystemExit(
        'gfnff is required. Install with: python3 -m pip install "gfnff[ase]"'
    ) from exc


CASES = {
    "ixazomib": {
        "complex": "ixazomib_complex.xyz",
        "ligand": "ixazomib_ligand.xyz",
    },
    "bortezomib": {
        "complex": "bortezomib_complex.xyz",
        "ligand": "bortezomib_ligand.xyz",
    },
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--indir", type=Path, default=Path("production_models")
    )
    p.add_argument(
        "--outdir", type=Path, default=Path("production_models_refined")
    )
    p.add_argument(
        "--report",
        type=Path,
        default=Path("validation/production_ligand_H_refinement.json"),
    )
    p.add_argument("--n-receptor", type=int, default=491)
    p.add_argument("--charge", type=int, default=-1)

    p.add_argument("--fire-fmax", type=float, default=0.20)
    p.add_argument("--fire-steps", type=int, default=300)
    p.add_argument("--lbfgs-fmax", type=float, default=0.05)
    p.add_argument("--lbfgs-steps", type=int, default=1000)
    p.add_argument("--maxstep", type=float, default=0.03)

    p.add_argument("--fixed-tol", type=float, default=1e-10)
    p.add_argument("--parent-min", type=float, default=0.65)
    p.add_argument("--parent-max", type=float, default=1.35)
    p.add_argument("--parent-change-max", type=float, default=0.25)
    p.add_argument("--min-hh", type=float, default=1.20)
    p.add_argument("--min-nonparent-hheavy", type=float, default=1.20)
    return p.parse_args()


def nearest_heavy_parents(
    positions: np.ndarray,
    h_idx: np.ndarray,
    heavy_idx: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    parents, distances = [], []
    hp = positions[heavy_idx]
    for hi in h_idx:
        d = np.linalg.norm(hp - positions[hi], axis=1)
        j = int(np.argmin(d))
        parents.append(int(heavy_idx[j]))
        distances.append(float(d[j]))
    return np.asarray(parents, dtype=int), np.asarray(distances, dtype=float)


def min_hh(positions, h_idx):
    if len(h_idx) < 2:
        return math.inf
    best = math.inf
    hp = positions[h_idx]
    for i in range(len(hp) - 1):
        d = np.linalg.norm(hp[i+1:] - hp[i], axis=1)
        if len(d):
            best = min(best, float(d.min()))
    return best


def min_nonparent_hheavy(positions, h_idx, heavy_idx, parents):
    best = math.inf
    heavy_pos = positions[heavy_idx]
    lookup = {int(idx): j for j, idx in enumerate(heavy_idx)}
    for hi, parent in zip(h_idx, parents):
        d = np.linalg.norm(heavy_pos - positions[hi], axis=1)
        d[lookup[int(parent)]] = np.inf
        best = min(best, float(d.min()))
    return best


def max_force_on_indices(atoms, idx):
    if not len(idx):
        return 0.0
    f = atoms.get_forces(apply_constraint=True)
    return float(np.linalg.norm(f[idx], axis=1).max())


class Monitor:
    def __init__(
        self, atoms, pos0, fixed_idx, movable_h, ligand_heavy,
        parents0, parent_dist0, args
    ):
        self.atoms = atoms
        self.pos0 = pos0
        self.fixed_idx = fixed_idx
        self.movable_h = movable_h
        self.ligand_heavy = ligand_heavy
        self.parents0 = parents0
        self.parent_dist0 = parent_dist0
        self.args = args

    def __call__(self):
        pos = self.atoms.positions

        if len(self.fixed_idx):
            d = np.linalg.norm(
                pos[self.fixed_idx] - self.pos0[self.fixed_idx], axis=1
            )
            mx = float(d.max())
            if mx > self.args.fixed_tol:
                raise RuntimeError(
                    f"Fixed-atom invariant violated: {mx:.3e} A"
                )

        parents, pd = nearest_heavy_parents(
            pos, self.movable_h, self.ligand_heavy
        )
        if np.any(parents != self.parents0):
            j = int(np.where(parents != self.parents0)[0][0])
            raise RuntimeError(
                f"Ligand proton parent changed for atom index "
                f"{int(self.movable_h[j])}"
            )

        if np.any(pd < self.args.parent_min) or np.any(pd > self.args.parent_max):
            j = int(np.argmax(
                np.maximum(self.args.parent_min - pd, pd - self.args.parent_max)
            ))
            raise RuntimeError(
                f"Ligand H-parent bond out of range for atom "
                f"{int(self.movable_h[j])}: {pd[j]:.3f} A"
            )

        delta = np.abs(pd - self.parent_dist0)
        if float(delta.max()) > self.args.parent_change_max:
            j = int(np.argmax(delta))
            raise RuntimeError(
                f"Ligand H-parent bond changed too much for atom "
                f"{int(self.movable_h[j])}: "
                f"{self.parent_dist0[j]:.3f} -> {pd[j]:.3f} A"
            )


def optimize_case(drug, cfg, args):
    path = args.indir / cfg["complex"]
    atoms = read(path)
    symbols = atoms.get_chemical_symbols()
    pos0 = atoms.positions.copy()

    n = len(atoms)
    if args.n_receptor >= n:
        raise RuntimeError(f"{drug}: n_receptor >= total atoms")

    ligand_idx = np.arange(args.n_receptor, n, dtype=int)
    receptor_idx = np.arange(0, args.n_receptor, dtype=int)

    ligand_h = np.asarray(
        [i for i in ligand_idx if symbols[i].upper() == "H"], dtype=int
    )
    ligand_heavy = np.asarray(
        [i for i in ligand_idx if symbols[i].upper() != "H"], dtype=int
    )

    # Everything except ligand H is fixed:
    # common receptor (including H) + ligand heavy atoms.
    fixed_idx = np.asarray(
        sorted(set(range(n)) - set(ligand_h.tolist())), dtype=int
    )

    parents0, parent_dist0 = nearest_heavy_parents(
        pos0, ligand_h, ligand_heavy
    )

    atoms.set_constraint(FixAtoms(indices=fixed_idx.tolist()))
    atoms.info["charge"] = args.charge
    atoms.calc = GFNFF(charge=args.charge, printlevel=0)

    monitor = Monitor(
        atoms, pos0, fixed_idx, ligand_h, ligand_heavy,
        parents0, parent_dist0, args
    )

    work = args.outdir / f"{drug}_work"
    work.mkdir(parents=True, exist_ok=True)

    fire = FIRE(
        atoms,
        logfile=str(work / "fire.log"),
        trajectory=str(work / "fire.traj"),
        maxstep=args.maxstep,
    )
    fire.attach(monitor, interval=1)
    fire_ok = bool(fire.run(fmax=args.fire_fmax, steps=args.fire_steps))
    monitor()

    lb = LBFGS(
        atoms,
        logfile=str(work / "lbfgs.log"),
        trajectory=str(work / "lbfgs.traj"),
        maxstep=args.maxstep,
    )
    lb.attach(monitor, interval=1)
    lb_ok = bool(lb.run(fmax=args.lbfgs_fmax, steps=args.lbfgs_steps))
    monitor()

    pos1 = atoms.positions.copy()

    fixed_delta = np.linalg.norm(
        pos1[fixed_idx] - pos0[fixed_idx], axis=1
    )
    fixed_max = float(fixed_delta.max()) if len(fixed_delta) else 0.0

    parents1, parent_dist1 = nearest_heavy_parents(
        pos1, ligand_h, ligand_heavy
    )
    parent_changes = int(np.count_nonzero(parents1 != parents0))
    max_parent_change = float(
        np.max(np.abs(parent_dist1 - parent_dist0))
    ) if len(parent_dist1) else 0.0

    # Global contact metrics.
    all_h = np.asarray(
        [i for i, s in enumerate(symbols) if s.upper() == "H"], dtype=int
    )
    all_heavy = np.asarray(
        [i for i, s in enumerate(symbols) if s.upper() != "H"], dtype=int
    )
    all_parents, _ = nearest_heavy_parents(pos1, all_h, all_heavy)
    hh = min_hh(pos1, all_h)
    hheavy = min_nonparent_hheavy(
        pos1, all_h, all_heavy, all_parents
    )
    fmax = max_force_on_indices(atoms, ligand_h)

    status = "PASS"
    notes = []
    if fixed_max > args.fixed_tol:
        status = "FAIL"
        notes.append("fixed atoms moved")
    if parent_changes:
        status = "FAIL"
        notes.append("ligand proton parent changed")
    if max_parent_change > args.parent_change_max:
        status = "FAIL"
        notes.append("ligand H-parent bond changed too much")
    if hh < args.min_hh:
        status = "FAIL"
        notes.append(f"global min H-H {hh:.3f} A < {args.min_hh:.3f}")
    if hheavy < args.min_nonparent_hheavy:
        status = "FAIL"
        notes.append(
            f"global min nonparent H-heavy {hheavy:.3f} A "
            f"< {args.min_nonparent_hheavy:.3f}"
        )
    if fmax > args.lbfgs_fmax * 1.001:
        status = "FAIL"
        notes.append(
            f"movable-H fmax {fmax:.4f} > {args.lbfgs_fmax:.4f} eV/A"
        )
    if not lb_ok:
        status = "FAIL"
        notes.append("LBFGS did not converge")

    # Write refined complex.
    out_complex = args.outdir / cfg["complex"]
    write(out_complex, atoms, format="xyz")

    # Re-extract exact ligand suffix.
    ligand_atoms = atoms[args.n_receptor:]
    out_ligand = args.outdir / cfg["ligand"]
    write(out_ligand, ligand_atoms, format="xyz")

    # extxyz provenance.
    write(work / "final.extxyz", atoms, format="extxyz")

    ligand_h_disp = np.linalg.norm(
        pos1[ligand_h] - pos0[ligand_h], axis=1
    )

    return {
        "drug": drug,
        "input": str(path),
        "output_complex": str(out_complex),
        "output_ligand": str(out_ligand),
        "n_atoms": n,
        "n_receptor_atoms_fixed": len(receptor_idx),
        "n_ligand_heavy_fixed": len(ligand_heavy),
        "n_ligand_H_movable": len(ligand_h),
        "charge": args.charge,
        "fire_converged": fire_ok,
        "lbfgs_converged": lb_ok,
        "movable_H_final_fmax_eV_A": fmax,
        "fixed_atom_max_displacement_A": fixed_max,
        "ligand_H_rms_displacement_A": float(
            np.sqrt(np.mean(ligand_h_disp**2))
        ) if len(ligand_h_disp) else 0.0,
        "ligand_H_max_displacement_A": float(
            ligand_h_disp.max()
        ) if len(ligand_h_disp) else 0.0,
        "ligand_parent_identity_changes": parent_changes,
        "max_ligand_parent_bond_change_A": max_parent_change,
        "global_min_HH_A": hh,
        "global_min_nonparent_Hheavy_A": hheavy,
        "status": status,
        "notes": notes,
    }


def main():
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    # Copy common receptor exactly, never regenerate it.
    src_rec = args.indir / "common_receptor.xyz"
    dst_rec = args.outdir / "common_receptor.xyz"
    shutil.copy2(src_rec, dst_rec)

    results = []

    print("Final production ligand-H refinement")
    print("-------------------------------------")
    print(f"input       : {args.indir}")
    print(f"output      : {args.outdir}")
    print(f"receptor    : first {args.n_receptor} atoms, completely fixed")
    print(f"charge      : {args.charge}")
    print()

    for drug, cfg in CASES.items():
        print(f"[{drug}] optimizing ligand H only...")
        r = optimize_case(drug, cfg, args)
        results.append(r)
        print(
            f"  {r['status']} "
            f"fmax={r['movable_H_final_fmax_eV_A']:.5f} eV/A "
            f"fixed_max={r['fixed_atom_max_displacement_A']:.3e} A "
            f"minHH={r['global_min_HH_A']:.3f} A "
            f"minH-heavy={r['global_min_nonparent_Hheavy_A']:.3f} A"
        )
        if r["notes"]:
            for note in r["notes"]:
                print(f"    note: {note}")

    overall = "PASS" if all(r["status"] == "PASS" for r in results) else "FAIL"

    payload = {
        "stage": "final_production_ligand_hydrogen_refinement",
        "method": "ASE-controlled ligand-H-only relaxation with GFN-FF",
        "scientific_role": (
            "Final hydrogen geometry cleanup after imposing the common canonical "
            "receptor. GFN-FF energies are not used scientifically."
        ),
        "fixed": [
            "all 491 common-receptor atoms, including receptor hydrogens",
            "all ligand heavy atoms",
        ],
        "movable": "ligand hydrogens only",
        "settings": vars(args) | {
            "indir": str(args.indir),
            "outdir": str(args.outdir),
            "report": str(args.report),
        },
        "results": results,
        "overall": overall,
    }
    args.report.write_text(json.dumps(payload, indent=2) + "\n")

    print()
    print(f"Overall: {overall}")
    print(f"Report : {args.report}")
    print(f"Models : {args.outdir}")

    raise SystemExit(0 if overall == "PASS" else 2)


if __name__ == "__main__":
    main()
