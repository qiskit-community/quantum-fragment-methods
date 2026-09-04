#!/usr/bin/env python3
"""
16b_optimize_hydrogens_gfnff.py

Proteasome Challenge
Stage 8g (revised): hydrogen-only optimization with GFN-FF.

Why GFN-FF?
-----------
xTB 6.7.1 has a known large-system (>~400 atoms) GFN1/GFN2 gradient crash.
Our structures contain 533-544 atoms, so GFN2 optimization is not reliable
with this xTB build.

GFN-FF uses a different code path and is used here ONLY to remove arbitrary
initial H placement while preserving every experimental/model heavy atom.

Important:
- all heavy atoms are exactly fixed through $fix
- RF optimizer is forced because older xTB versions can fail for
  GFN-FF + fixed atoms with the default optimizer
- default thread count is 1 for robustness on the older macOS build
- GFN-FF energies are NOT used as final microstate energetics

Example
-------
python3 scripts/16b_optimize_hydrogens_gfnff.py \
  --indir intermediate/hydrogenated_microstates \
  --validation validation/hydrogenated_geometry_validation.json \
  --outdir intermediate/hydrogen_optimized_gfnff \
  --report validation/hydrogen_optimization_gfnff.json
"""

import argparse
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np


EXPECTED = [
    ("6V8", "MS_A_THRN"),
    ("6V8", "MS_B_LYS33"),
    ("6V8", "MS_C_ASP17"),
    ("BO2", "MS_A_THRN"),
    ("BO2", "MS_B_LYS33"),
    ("BO2", "MS_C_ASP17"),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--indir", required=True)
    p.add_argument("--validation", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--report", required=True)
    p.add_argument("--xtb", default="xtb")
    p.add_argument("--charge", type=int, default=-1)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--opt-level", default="tight")
    p.add_argument("--heavy-tol", type=float, default=1e-5)
    return p.parse_args()


def parse_xyz(path):
    lines = Path(path).read_text().splitlines()
    n = int(lines[0].strip())
    atoms = []
    for line in lines[2:2+n]:
        f = line.split()
        atoms.append({
            "element": f[0],
            "xyz": np.array([float(f[1]), float(f[2]), float(f[3])]),
        })
    if len(atoms) != n:
        raise RuntimeError(f"{path}: XYZ atom-count mismatch")
    return atoms


def write_xyz(path, atoms, comment):
    lines = [str(len(atoms)), comment]
    for a in atoms:
        x, y, z = a["xyz"]
        lines.append(f'{a["element"]:2s} {x: .12f} {y: .12f} {z: .12f}')
    Path(path).write_text("\n".join(lines) + "\n")


def compress_indices(indices):
    vals = sorted(indices)
    ranges = []
    start = prev = vals[0]
    for x in vals[1:]:
        if x == prev + 1:
            prev = x
        else:
            ranges.append((start, prev))
            start = prev = x
    ranges.append((start, prev))
    return ",".join(str(a) if a == b else f"{a}-{b}" for a, b in ranges)


def parse_energy(text):
    pats = [
        r"TOTAL ENERGY\s+(-?\d+\.\d+(?:[Ee][+-]?\d+)?)",
        r"TOTAL ENERGY[^\-\d]*(-?\d+\.\d+(?:[Ee][+-]?\d+)?)",
    ]
    vals = []
    for p in pats:
        vals = re.findall(p, text)
        if vals:
            break
    return float(vals[-1]) if vals else None


def max_heavy_displacement(before, after):
    if len(before) != len(after):
        raise RuntimeError("Atom count changed")
    md = 0.0
    for i, (a, b) in enumerate(zip(before, after), 1):
        if a["element"].upper() != b["element"].upper():
            raise RuntimeError(f"Element identity changed at atom {i}")
        if a["element"].upper() != "H":
            md = max(md, float(np.linalg.norm(a["xyz"] - b["xyz"])))
    return md


def h_rms_displacement(before, after):
    vals = []
    for a, b in zip(before, after):
        if a["element"].upper() == "H":
            vals.append(float(np.linalg.norm(a["xyz"] - b["xyz"])) ** 2)
    return float(np.sqrt(np.mean(vals))) if vals else 0.0


def min_distances(atoms):
    H = [(i, a) for i, a in enumerate(atoms) if a["element"].upper() == "H"]
    heavy = [(i, a) for i, a in enumerate(atoms) if a["element"].upper() != "H"]

    min_hh = 999.0
    for ii in range(len(H)):
        _, a = H[ii]
        for jj in range(ii + 1, len(H)):
            _, b = H[jj]
            min_hh = min(min_hh, float(np.linalg.norm(a["xyz"] - b["xyz"])))

    # For each H, identify closest heavy as its parent, then find closest other heavy.
    min_nonparent = 999.0
    for _, h in H:
        dlist = sorted(
            (float(np.linalg.norm(h["xyz"] - a["xyz"])), i)
            for i, a in heavy
        )
        parent_i = dlist[0][1]
        for d, i in dlist[1:]:
            if i != parent_i:
                min_nonparent = min(min_nonparent, d)
                break

    return min_hh, min_nonparent


def main():
    args = parse_args()

    val = json.loads(Path(args.validation).read_text())
    if not val.get("overall_pass", False):
        raise RuntimeError("Stage-8f validation did not PASS")

    xtb = shutil.which(args.xtb)
    if xtb is None:
        raise RuntimeError("xTB not found on PATH")

    indir = Path(args.indir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(args.threads)
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env.setdefault("OMP_STACKSIZE", "1G")

    results = []

    for ligand, ms in EXPECTED:
        stem = f"{ligand}_{ms}"
        source = indir / f"{stem}.xyz"
        if not source.exists():
            raise RuntimeError(f"Missing {source}")

        atoms0 = parse_xyz(source)

        heavy_indices = [
            i for i, a in enumerate(atoms0, start=1)
            if a["element"].upper() != "H"
        ]
        fixed = compress_indices(heavy_indices)

        work = outdir / f"{stem}_work"
        work.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, work / "input.xyz")

        # RF optimizer avoids the known older GFN-FF/fixed-atom L-ANC issue.
        (work / "xtb.inp").write_text(
            "$opt\n"
            "  engine=rf\n"
            "$end\n"
            "$fix\n"
            f"  atoms: {fixed}\n"
            "$end\n"
        )

        cmd = [
            xtb,
            "input.xyz",
            "--gfnff",
            "--chrg", str(args.charge),
            "--opt", args.opt_level,
            "--input", "xtb.inp",
        ]

        proc = subprocess.run(
            cmd,
            cwd=work,
            env=env,
            capture_output=True,
            text=True,
        )

        log = work / "xtb.out"
        log.write_text(proc.stdout + "\n--- STDERR ---\n" + proc.stderr)

        if proc.returncode != 0:
            raise RuntimeError(
                f"GFN-FF failed for {stem}. See {log}"
            )

        optfile = work / "xtbopt.xyz"
        if not optfile.exists():
            raise RuntimeError(f"No xtbopt.xyz produced for {stem}")

        atoms1 = parse_xyz(optfile)

        maxheavy = max_heavy_displacement(atoms0, atoms1)
        if maxheavy > args.heavy_tol:
            raise RuntimeError(
                f"{stem}: heavy atoms moved {maxheavy:.6e} A "
                f"(tol {args.heavy_tol:.2e})"
            )

        hrms = h_rms_displacement(atoms0, atoms1)
        minhh, minhnp = min_distances(atoms1)
        energy = parse_energy(proc.stdout)

        final = outdir / f"{stem}_Hopt.xyz"
        write_xyz(
            final,
            atoms1,
            f"{ligand} {ms}; GFN-FF H-only optimized; charge={args.charge}",
        )

        results.append({
            "ligand": ligand,
            "microstate": ms,
            "charge": args.charge,
            "n_atoms": len(atoms1),
            "n_heavy_fixed": len(heavy_indices),
            "n_H_optimized": len(atoms1) - len(heavy_indices),
            "max_heavy_displacement_A": maxheavy,
            "H_RMS_displacement_A": hrms,
            "min_H_H_A_after": minhh,
            "min_nonparent_H_heavy_A_after": minhnp,
            "GFN_FF_total_energy": energy,
            "optimized_xyz": str(final),
            "log": str(log),
        })

    report = {
        "stage": "8g_revised_GFNFF_H_only_optimization",
        "reason_for_revision":
            "xTB 6.7.1 GFN1/GFN2 gradient crash for large systems; current complexes contain >500 atoms.",
        "method": "GFN-FF",
        "purpose": "hydrogen-coordinate relaxation only",
        "charge": args.charge,
        "threads": args.threads,
        "optimizer": "RF",
        "heavy_atoms_exactly_fixed": True,
        "results": results,
        "energy_policy":
            "GFN-FF energies are not accepted as the final catalytic microstate ranking.",
        "next_step":
            "Validate optimized geometries, then perform higher-level energetic comparison on a reduced catalytic proton-network model.",
    }

    Path(args.report).write_text(json.dumps(report, indent=2) + "\n")

    print(f"Wrote {args.report}\n")
    print("Stage-8g revised: GFN-FF H-only optimization")
    print(f"  xTB: {xtb}")
    print(f"  threads: {args.threads}")
    print(f"  optimizer: RF")
    print()

    for r in results:
        e = r["GFN_FF_total_energy"]
        etxt = "n/a" if e is None else f"{e:.10f}"
        print(
            f"  {r['ligand']:3s} {r['microstate']:12s} "
            f"E={etxt:16s} "
            f"H-RMSΔ={r['H_RMS_displacement_A']:.3f} A "
            f"heavy maxΔ={r['max_heavy_displacement_A']:.2e} A "
            f"min H-H={r['min_H_H_A_after']:.3f} A"
        )

    print()
    print("GFN-FF used for H geometry only; do not use these energies as final microstate ranking.")


if __name__ == "__main__":
    main()
