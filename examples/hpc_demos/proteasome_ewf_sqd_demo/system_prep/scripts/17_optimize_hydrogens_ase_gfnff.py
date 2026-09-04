#!/usr/bin/env python3
"""
17_optimize_hydrogens_ase_gfnff.py

Hydrogen-only relaxation of proteasome microstates using:
  - GFN-FF for energies/forces
  - ASE for optimization and exact heavy-atom constraints

Design goals
------------
1. Heavy atoms are immutable.
2. Atom order/composition are immutable.
3. Protonation microstate identity is monitored throughout optimization.
4. GFN-FF is used only as a geometry-preparation force model.
5. Each structure gets a machine-readable validation record.

The script intentionally writes to a NEW output directory so that failed
xTB-internal-optimizer geometries cannot be mixed with accepted structures.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    from ase.io import read, write
    from ase.constraints import FixAtoms
    from ase.optimize import FIRE, LBFGS
except ImportError as exc:
    raise SystemExit(
        "ASE is not installed. Install with:\n"
        '  python3 -m pip install "gfnff[ase]"\n'
    ) from exc

try:
    from gfnff import GFNFF
except ImportError as exc:
    raise SystemExit(
        "gfnff is not installed. Install with:\n"
        '  python3 -m pip install "gfnff[ase]"\n'
    ) from exc


@dataclass
class Result:
    name: str
    input_file: str
    output_file: str
    n_atoms: int
    n_heavy: int
    n_h: int
    charge: int
    fire_converged: bool
    lbfgs_converged: bool
    final_fmax_eV_A: float
    heavy_max_displacement_A: float
    heavy_rms_displacement_A: float
    h_rms_displacement_A: float
    h_max_displacement_A: float
    min_HH_A: float
    min_nonparent_Hheavy_A: float
    max_parent_bond_change_A: float
    parent_identity_changes: int
    status: str
    notes: List[str]


def parse_args():
    p = argparse.ArgumentParser(
        description="Optimize hydrogens only with ASE + GFN-FF."
    )
    p.add_argument(
        "--indir",
        type=Path,
        default=Path("intermediate/hydrogenated_microstates"),
        help="Directory containing input XYZ files.",
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=Path("intermediate/hydrogen_optimized_ase_gfnff"),
        help="Output directory.",
    )
    p.add_argument(
        "--report",
        type=Path,
        default=Path("validation/hydrogen_optimization_ase_gfnff.json"),
        help="JSON summary report.",
    )
    p.add_argument("--pattern", default="*.xyz")
    p.add_argument("--charge", type=int, default=-1)

    # Conservative two-stage relaxation.
    p.add_argument(
        "--fire-fmax",
        type=float,
        default=0.20,
        help="FIRE pre-relaxation force threshold in eV/Angstrom.",
    )
    p.add_argument(
        "--fire-steps",
        type=int,
        default=400,
        help="Maximum FIRE steps.",
    )
    p.add_argument(
        "--lbfgs-fmax",
        type=float,
        default=0.05,
        help="Final LBFGS force threshold in eV/Angstrom.",
    )
    p.add_argument(
        "--lbfgs-steps",
        type=int,
        default=1500,
        help="Maximum LBFGS steps.",
    )
    p.add_argument(
        "--maxstep",
        type=float,
        default=0.03,
        help="Maximum per-atom optimizer step in Angstrom.",
    )

    # Hard invariants / sanity thresholds.
    p.add_argument(
        "--heavy-tol",
        type=float,
        default=1e-10,
        help="Maximum allowed heavy-atom displacement in Angstrom.",
    )
    p.add_argument(
        "--parent-min",
        type=float,
        default=0.65,
        help="Minimum allowed H-parent-heavy distance in Angstrom.",
    )
    p.add_argument(
        "--parent-max",
        type=float,
        default=1.35,
        help="Maximum allowed H-parent-heavy distance in Angstrom.",
    )
    p.add_argument(
        "--parent-change-max",
        type=float,
        default=0.25,
        help="Maximum allowed change in H-parent distance in Angstrom.",
    )
    p.add_argument(
        "--min-hh",
        type=float,
        default=0.70,
        help="Hard-fail threshold for minimum H-H distance in Angstrom.",
    )
    p.add_argument(
        "--min-nonparent-hheavy",
        type=float,
        default=0.70,
        help="Hard-fail threshold for H to non-parent-heavy distance in Angstrom.",
    )
    return p.parse_args()


def nearest_heavy_parents(
    positions: np.ndarray,
    h_idx: np.ndarray,
    heavy_idx: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """For each H, return nearest heavy index and distance."""
    parents = []
    distances = []
    heavy_pos = positions[heavy_idx]
    for hi in h_idx:
        d = np.linalg.norm(heavy_pos - positions[hi], axis=1)
        j = int(np.argmin(d))
        parents.append(int(heavy_idx[j]))
        distances.append(float(d[j]))
    return np.asarray(parents, dtype=int), np.asarray(distances, dtype=float)


def min_hh_distance(positions: np.ndarray, h_idx: np.ndarray) -> float:
    if len(h_idx) < 2:
        return math.inf
    hp = positions[h_idx]
    best = math.inf
    for i in range(len(hp) - 1):
        d = np.linalg.norm(hp[i + 1 :] - hp[i], axis=1)
        if len(d):
            best = min(best, float(d.min()))
    return best


def min_nonparent_hheavy(
    positions: np.ndarray,
    h_idx: np.ndarray,
    heavy_idx: np.ndarray,
    parents: np.ndarray,
) -> float:
    best = math.inf
    heavy_pos = positions[heavy_idx]
    heavy_lookup = {int(idx): j for j, idx in enumerate(heavy_idx)}
    for hi, parent in zip(h_idx, parents):
        d = np.linalg.norm(heavy_pos - positions[hi], axis=1)
        parent_local = heavy_lookup[int(parent)]
        d[parent_local] = np.inf
        best = min(best, float(d.min()))
    return best


def max_force_on_h(atoms, h_idx: np.ndarray) -> float:
    forces = atoms.get_forces(apply_constraint=True)
    if len(h_idx) == 0:
        return 0.0
    norms = np.linalg.norm(forces[h_idx], axis=1)
    return float(norms.max())


class InvariantMonitor:
    """Abort immediately if a protected invariant is violated."""

    def __init__(
        self,
        atoms,
        initial_positions: np.ndarray,
        heavy_idx: np.ndarray,
        h_idx: np.ndarray,
        parents0: np.ndarray,
        parent_dist0: np.ndarray,
        heavy_tol: float,
        parent_min: float,
        parent_max: float,
        parent_change_max: float,
    ):
        self.atoms = atoms
        self.initial_positions = initial_positions
        self.heavy_idx = heavy_idx
        self.h_idx = h_idx
        self.parents0 = parents0
        self.parent_dist0 = parent_dist0
        self.heavy_tol = heavy_tol
        self.parent_min = parent_min
        self.parent_max = parent_max
        self.parent_change_max = parent_change_max

    def __call__(self):
        pos = self.atoms.positions

        heavy_d = np.linalg.norm(
            pos[self.heavy_idx] - self.initial_positions[self.heavy_idx], axis=1
        )
        max_heavy = float(heavy_d.max()) if len(heavy_d) else 0.0
        if max_heavy > self.heavy_tol:
            raise RuntimeError(
                f"Heavy-atom invariant violated: max displacement "
                f"{max_heavy:.6e} A > {self.heavy_tol:.6e} A"
            )

        parents, parent_dist = nearest_heavy_parents(
            pos, self.h_idx, self.heavy_idx
        )
        changed = np.where(parents != self.parents0)[0]
        if len(changed):
            h_atom = int(self.h_idx[int(changed[0])])
            raise RuntimeError(
                f"Protonation-topology invariant violated: H atom index "
                f"{h_atom} changed nearest heavy parent."
            )

        if np.any(parent_dist < self.parent_min) or np.any(
            parent_dist > self.parent_max
        ):
            j = int(
                np.argmax(
                    np.maximum(
                        self.parent_min - parent_dist,
                        parent_dist - self.parent_max,
                    )
                )
            )
            raise RuntimeError(
                f"H-parent bond-length invariant violated for H index "
                f"{int(self.h_idx[j])}: {parent_dist[j]:.4f} A"
            )

        delta = np.abs(parent_dist - self.parent_dist0)
        if float(delta.max()) > self.parent_change_max:
            j = int(np.argmax(delta))
            raise RuntimeError(
                f"H-parent bond changed too much for H index "
                f"{int(self.h_idx[j])}: initial {self.parent_dist0[j]:.4f} A, "
                f"current {parent_dist[j]:.4f} A"
            )


def optimize_one(path: Path, outdir: Path, args) -> Result:
    name = path.stem
    workdir = outdir / f"{name}_work"
    workdir.mkdir(parents=True, exist_ok=True)

    atoms = read(path)
    symbols0 = atoms.get_chemical_symbols()
    pos0 = atoms.positions.copy()

    heavy_idx = np.asarray(
        [i for i, s in enumerate(symbols0) if s.upper() != "H"], dtype=int
    )
    h_idx = np.asarray(
        [i for i, s in enumerate(symbols0) if s.upper() == "H"], dtype=int
    )

    if len(heavy_idx) == 0 or len(h_idx) == 0:
        raise RuntimeError(
            f"{path}: expected both heavy atoms and hydrogens."
        )

    parents0, parent_dist0 = nearest_heavy_parents(pos0, h_idx, heavy_idx)

    # Exact coordinate-level protection of every heavy atom.
    atoms.set_constraint(FixAtoms(indices=heavy_idx.tolist()))

    # Charge is explicitly attached both to the calculator and Atoms metadata.
    atoms.info["charge"] = args.charge
    atoms.calc = GFNFF(charge=args.charge, printlevel=0)

    monitor = InvariantMonitor(
        atoms=atoms,
        initial_positions=pos0,
        heavy_idx=heavy_idx,
        h_idx=h_idx,
        parents0=parents0,
        parent_dist0=parent_dist0,
        heavy_tol=args.heavy_tol,
        parent_min=args.parent_min,
        parent_max=args.parent_max,
        parent_change_max=args.parent_change_max,
    )

    fire_log = workdir / "fire.log"
    fire_traj = workdir / "fire.traj"
    lbfgs_log = workdir / "lbfgs.log"
    lbfgs_traj = workdir / "lbfgs.traj"

    fire = FIRE(
        atoms,
        logfile=str(fire_log),
        trajectory=str(fire_traj),
        maxstep=args.maxstep,
    )
    fire.attach(monitor, interval=1)
    fire_converged = bool(
        fire.run(fmax=args.fire_fmax, steps=args.fire_steps)
    )

    # Re-check before switching optimizers.
    monitor()

    lbfgs = LBFGS(
        atoms,
        logfile=str(lbfgs_log),
        trajectory=str(lbfgs_traj),
        maxstep=args.maxstep,
    )
    lbfgs.attach(monitor, interval=1)
    lbfgs_converged = bool(
        lbfgs.run(fmax=args.lbfgs_fmax, steps=args.lbfgs_steps)
    )

    monitor()

    symbols1 = atoms.get_chemical_symbols()
    if symbols1 != symbols0:
        raise RuntimeError(f"{path}: atom identity/order changed.")

    pos1 = atoms.positions.copy()
    heavy_d = np.linalg.norm(
        pos1[heavy_idx] - pos0[heavy_idx], axis=1
    )
    h_d = np.linalg.norm(pos1[h_idx] - pos0[h_idx], axis=1)

    parents1, parent_dist1 = nearest_heavy_parents(
        pos1, h_idx, heavy_idx
    )
    parent_changes = int(np.count_nonzero(parents1 != parents0))
    max_parent_change = float(np.max(np.abs(parent_dist1 - parent_dist0)))

    hh = min_hh_distance(pos1, h_idx)
    nonparent = min_nonparent_hheavy(
        pos1, h_idx, heavy_idx, parents1
    )
    final_fmax = max_force_on_h(atoms, h_idx)

    notes = []
    status = "PASS"

    heavy_max = float(heavy_d.max())
    heavy_rms = float(np.sqrt(np.mean(heavy_d**2)))
    h_max = float(h_d.max())
    h_rms = float(np.sqrt(np.mean(h_d**2)))

    if heavy_max > args.heavy_tol:
        status = "FAIL"
        notes.append("heavy atoms moved")
    if parent_changes:
        status = "FAIL"
        notes.append("H nearest-heavy parent identity changed")
    if max_parent_change > args.parent_change_max:
        status = "FAIL"
        notes.append("H-parent bond changed too much")
    if hh < args.min_hh:
        status = "FAIL"
        notes.append(f"minimum H-H distance too short: {hh:.3f} A")
    if nonparent < args.min_nonparent_hheavy:
        status = "FAIL"
        notes.append(
            f"minimum non-parent H-heavy distance too short: "
            f"{nonparent:.3f} A"
        )
    if final_fmax > args.lbfgs_fmax * 1.001:
        status = "FAIL"
        notes.append(
            f"final H force max {final_fmax:.4f} eV/A exceeds "
            f"{args.lbfgs_fmax:.4f} eV/A"
        )
    if not lbfgs_converged:
        status = "FAIL"
        notes.append("LBFGS did not report convergence")

    outfile = outdir / f"{name}.xyz"
    write(outfile, atoms, format="xyz")

    # Also write extxyz containing ASE metadata/energy/forces when possible.
    write(workdir / "final.extxyz", atoms, format="extxyz")

    return Result(
        name=name,
        input_file=str(path),
        output_file=str(outfile),
        n_atoms=len(atoms),
        n_heavy=len(heavy_idx),
        n_h=len(h_idx),
        charge=args.charge,
        fire_converged=fire_converged,
        lbfgs_converged=lbfgs_converged,
        final_fmax_eV_A=final_fmax,
        heavy_max_displacement_A=heavy_max,
        heavy_rms_displacement_A=heavy_rms,
        h_rms_displacement_A=h_rms,
        h_max_displacement_A=h_max,
        min_HH_A=hh,
        min_nonparent_Hheavy_A=nonparent,
        max_parent_bond_change_A=max_parent_change,
        parent_identity_changes=parent_changes,
        status=status,
        notes=notes,
    )


def check_group_invariance(files: List[Path]) -> Dict[str, dict]:
    """
    Verify that all microstates belonging to the same ligand prefix
    (e.g. 6V8_* or BO2_*) have the same atom order and identical heavy atoms.
    """
    groups: Dict[str, List[Path]] = {}
    for p in files:
        prefix = p.stem.split("_MS_")[0]
        groups.setdefault(prefix, []).append(p)

    report = {}

    for prefix, members in groups.items():
        members = sorted(members)
        ref = read(members[0])
        ref_symbols = ref.get_chemical_symbols()
        heavy = np.asarray(
            [i for i, s in enumerate(ref_symbols) if s.upper() != "H"],
            dtype=int,
        )
        ref_pos = ref.positions.copy()

        max_heavy_delta = 0.0
        symbol_order_match = True

        for p in members[1:]:
            a = read(p)
            syms = a.get_chemical_symbols()
            if syms != ref_symbols:
                symbol_order_match = False
                continue
            d = np.linalg.norm(
                a.positions[heavy] - ref_pos[heavy], axis=1
            )
            max_heavy_delta = max(max_heavy_delta, float(d.max()))

        report[prefix] = {
            "files": [str(p) for p in members],
            "symbol_order_match": symbol_order_match,
            "input_heavy_max_delta_A": max_heavy_delta,
            "status": (
                "PASS"
                if symbol_order_match and max_heavy_delta <= 1e-10
                else "FAIL"
            ),
        }

    return report


def main():
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(args.indir.glob(args.pattern))
    if not files:
        raise SystemExit(
            f"No input files matching {args.pattern!r} in {args.indir}"
        )

    group_report = check_group_invariance(files)
    bad_groups = [
        k for k, v in group_report.items() if v["status"] != "PASS"
    ]
    if bad_groups:
        raise SystemExit(
            "Input heavy-atom invariance failed for group(s): "
            + ", ".join(bad_groups)
        )

    results = []
    print(
        f"ASE + GFN-FF hydrogen-only optimization\n"
        f"  inputs : {args.indir}\n"
        f"  outputs: {args.outdir}\n"
        f"  charge : {args.charge}\n"
        f"  files  : {len(files)}\n"
    )

    for i, path in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {path.name}")
        try:
            r = optimize_one(path, args.outdir, args)
            results.append(r)
            print(
                f"  {r.status}: fmax={r.final_fmax_eV_A:.5f} eV/A, "
                f"heavy_max={r.heavy_max_displacement_A:.3e} A, "
                f"min_HH={r.min_HH_A:.3f} A, "
                f"min_nonparent_H-heavy={r.min_nonparent_Hheavy_A:.3f} A"
            )
        except Exception as exc:
            print(f"  FAIL: {exc}", file=sys.stderr)
            results.append(
                Result(
                    name=path.stem,
                    input_file=str(path),
                    output_file="",
                    n_atoms=0,
                    n_heavy=0,
                    n_h=0,
                    charge=args.charge,
                    fire_converged=False,
                    lbfgs_converged=False,
                    final_fmax_eV_A=math.inf,
                    heavy_max_displacement_A=math.inf,
                    heavy_rms_displacement_A=math.inf,
                    h_rms_displacement_A=math.inf,
                    h_max_displacement_A=math.inf,
                    min_HH_A=math.nan,
                    min_nonparent_Hheavy_A=math.nan,
                    max_parent_bond_change_A=math.inf,
                    parent_identity_changes=-1,
                    status="FAIL",
                    notes=[str(exc)],
                )
            )

    overall = (
        "PASS"
        if all(r.status == "PASS" for r in results)
        and all(v["status"] == "PASS" for v in group_report.values())
        else "FAIL"
    )

    payload = {
        "method": "ASE-controlled hydrogen-only relaxation with GFN-FF",
        "scientific_role": (
            "geometry preparation only; GFN-FF energies are not used "
            "for final microstate ranking or binding energetics"
        ),
        "settings": {
            "charge": args.charge,
            "fire_fmax_eV_A": args.fire_fmax,
            "fire_steps": args.fire_steps,
            "lbfgs_fmax_eV_A": args.lbfgs_fmax,
            "lbfgs_steps": args.lbfgs_steps,
            "maxstep_A": args.maxstep,
            "heavy_tolerance_A": args.heavy_tol,
            "parent_distance_range_A": [
                args.parent_min,
                args.parent_max,
            ],
            "parent_change_max_A": args.parent_change_max,
            "min_HH_A": args.min_hh,
            "min_nonparent_Hheavy_A": args.min_nonparent_hheavy,
        },
        "input_group_invariance": group_report,
        "results": [asdict(r) for r in results],
        "overall": overall,
    }

    args.report.write_text(json.dumps(payload, indent=2) + "\n")

    print(f"\nOverall: {overall}")
    print(f"Report : {args.report}")

    if overall != "PASS":
        sys.exit(2)


if __name__ == "__main__":
    main()
