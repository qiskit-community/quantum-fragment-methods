#!/usr/bin/env python3
"""
16_optimize_hydrogens_xtb.py

Proteasome Challenge
Stage 8g: identical hydrogen-only GFN2-xTB optimization of the six
hydrogenated catalytic microstates.

All heavy atoms are EXACTLY fixed through xTB's $fix atoms: directive.
Only hydrogen coordinates are allowed to relax.

The resulting GFN2-xTB energies are recorded as a PRE-SCREEN only.
Do not treat them as the final protonation-state ranking without a higher-level
validation of the catalytic proton network.

Example
-------
python3 scripts/16_optimize_hydrogens_xtb.py \
  --indir intermediate/hydrogenated_microstates \
  --validation validation/hydrogenated_geometry_validation.json \
  --outdir intermediate/hydrogen_optimized_xtb \
  --report validation/hydrogen_optimization_xtb.json
"""

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path

import gemmi
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
    p.add_argument(
        "--opt-level",
        default="tight",
        choices=["crude", "sloppy", "loose", "normal", "tight", "verytight", "extreme"],
    )
    p.add_argument(
        "--acc",
        type=float,
        default=0.5,
        help="xTB SCC accuracy multiplier; lower is tighter",
    )
    p.add_argument(
        "--heavy-tol",
        type=float,
        default=1.0e-5,
        help="Maximum allowed heavy-atom displacement in Angstrom",
    )
    return p.parse_args()


def parse_xyz(path):
    lines = Path(path).read_text().splitlines()
    n = int(lines[0].strip())
    atoms = []
    for line in lines[2:2 + n]:
        parts = line.split()
        atoms.append(
            {
                "element": parts[0],
                "xyz": np.array(
                    [float(parts[1]), float(parts[2]), float(parts[3])],
                    dtype=float,
                ),
            }
        )
    if len(atoms) != n:
        raise RuntimeError(f"{path}: XYZ count mismatch")
    return atoms, lines[1] if len(lines) > 1 else ""


def write_xyz(path, atoms, comment):
    lines = [str(len(atoms)), comment]
    for a in atoms:
        x, y, z = a["xyz"]
        lines.append(f'{a["element"]:2s} {x: .12f} {y: .12f} {z: .12f}')
    Path(path).write_text("\n".join(lines) + "\n")


def compress_indices(indices):
    vals = sorted(set(indices))
    if not vals:
        return ""
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


def electron_count(atoms, charge):
    nuclear = 0
    for a in atoms:
        try:
            nuclear += gemmi.Element(a["element"]).atomic_number
        except Exception as exc:
            raise RuntimeError(f"Unknown element {a['element']}") from exc
    return nuclear - charge


def parse_total_energy(text):
    vals = re.findall(
        r"TOTAL ENERGY\s+(-?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)",
        text,
    )
    if not vals:
        vals = re.findall(
            r"TOTAL ENERGY[^\-\d]*(-?\d+\.\d+(?:[Ee][+-]?\d+)?)",
            text,
        )
    return float(vals[-1]) if vals else None


def max_heavy_displacement(before, after):
    if len(before) != len(after):
        raise RuntimeError("Atom count changed during xTB optimization")

    maxd = 0.0
    for i, (a, b) in enumerate(zip(before, after), start=1):
        if a["element"].upper() != b["element"].upper():
            raise RuntimeError(f"Element identity changed at atom {i}")
        if a["element"].upper() == "H":
            continue
        d = float(np.linalg.norm(a["xyz"] - b["xyz"]))
        maxd = max(maxd, d)
    return maxd


def hydrogen_rmsd(before, after):
    ds = []
    for a, b in zip(before, after):
        if a["element"].upper() == "H":
            ds.append(float(np.linalg.norm(a["xyz"] - b["xyz"])) ** 2)
    return float(np.sqrt(np.mean(ds))) if ds else 0.0


def main():
    args = parse_args()

    validation = json.loads(Path(args.validation).read_text())
    if not validation.get("overall_pass", False):
        raise RuntimeError(
            "Hydrogenated geometry validation did not PASS. "
            "Do not optimize failed structures."
        )

    xtb = shutil.which(args.xtb)
    if xtb is None:
        raise RuntimeError(
            "xTB executable not found on PATH. Check with `which xtb`."
        )

    indir = Path(args.indir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)

    results = []

    for ligand, microstate in EXPECTED:
        stem = f"{ligand}_{microstate}"
        input_xyz = indir / f"{stem}.xyz"
        if not input_xyz.exists():
            raise RuntimeError(f"Missing {input_xyz}")

        atoms0, _ = parse_xyz(input_xyz)

        heavy_indices = [
            i for i, atom in enumerate(atoms0, start=1)
            if atom["element"].upper() != "H"
        ]
        h_indices = [
            i for i, atom in enumerate(atoms0, start=1)
            if atom["element"].upper() == "H"
        ]

        fixed_spec = compress_indices(heavy_indices)
        nelec = electron_count(atoms0, args.charge)
        uhf = 0 if nelec % 2 == 0 else 1

        workdir = outdir / f"{stem}_work"
        workdir.mkdir(parents=True, exist_ok=True)

        local_xyz = workdir / "input.xyz"
        shutil.copy2(input_xyz, local_xyz)

        xcontrol = workdir / "xtb.inp"
        xcontrol.write_text(
            "$fix\n"
            f"  atoms: {fixed_spec}\n"
            "$end\n"
        )

        cmd = [
            xtb,
            "input.xyz",
            "--gfn", "2",
            "--chrg", str(args.charge),
            "--uhf", str(uhf),
            "--acc", str(args.acc),
            "--opt", args.opt_level,
            "--input", "xtb.inp",
        ]

        proc = subprocess.run(
            cmd,
            cwd=workdir,
            capture_output=True,
            text=True,
        )

        log_path = workdir / "xtb.out"
        log_path.write_text(proc.stdout + "\n--- STDERR ---\n" + proc.stderr)

        if proc.returncode != 0:
            raise RuntimeError(
                f"xTB failed for {stem}. See {log_path}"
            )

        xtbopt = workdir / "xtbopt.xyz"
        if not xtbopt.exists():
            raise RuntimeError(
                f"xTB returned success but {xtbopt} was not produced"
            )

        atoms1, _ = parse_xyz(xtbopt)

        max_heavy = max_heavy_displacement(atoms0, atoms1)
        if max_heavy > args.heavy_tol:
            raise RuntimeError(
                f"{stem}: fixed heavy atoms moved by {max_heavy:.6e} A "
                f"(tolerance {args.heavy_tol:.2e})"
            )

        h_rmsd = hydrogen_rmsd(atoms0, atoms1)
        energy = parse_total_energy(proc.stdout)

        final_xyz = outdir / f"{stem}_Hopt.xyz"
        write_xyz(
            final_xyz,
            atoms1,
            f"{ligand} {microstate}; GFN2-xTB H-only optimized; "
            f"charge={args.charge}; uhf={uhf}",
        )

        results.append(
            {
                "ligand": ligand,
                "microstate": microstate,
                "charge": args.charge,
                "electron_count": nelec,
                "uhf": uhf,
                "n_atoms": len(atoms0),
                "n_heavy_fixed": len(heavy_indices),
                "n_H_optimized": len(h_indices),
                "max_heavy_displacement_A": max_heavy,
                "H_RMS_displacement_A": h_rmsd,
                "GFN2_xTB_total_energy_Eh": energy,
                "optimized_xyz": str(final_xyz),
                "workdir": str(workdir),
                "log": str(log_path),
            }
        )

    hartree_to_kcal = 627.509474
    for ligand in ("6V8", "BO2"):
        subset = [r for r in results if r["ligand"] == ligand]
        energies = [
            r["GFN2_xTB_total_energy_Eh"]
            for r in subset
            if r["GFN2_xTB_total_energy_Eh"] is not None
        ]
        if len(energies) == len(subset):
            emin = min(energies)
            for r in subset:
                r["relative_GFN2_xTB_kcal_mol"] = (
                    r["GFN2_xTB_total_energy_Eh"] - emin
                ) * hartree_to_kcal
        else:
            for r in subset:
                r["relative_GFN2_xTB_kcal_mol"] = None

    report = {
        "stage": "8g_GFN2_xTB_hydrogen_only_optimization",
        "xtb_executable": xtb,
        "method": "GFN2-xTB",
        "charge": args.charge,
        "optimization_level": args.opt_level,
        "scc_accuracy": args.acc,
        "heavy_atoms_exactly_fixed_via_xcontrol": True,
        "results": results,
        "interpretation_policy": [
            "The primary purpose is to relax H coordinates while preserving experimental heavy atoms.",
            "GFN2-xTB microstate energies are a pre-screen, not the final protonation-state benchmark.",
            "Any microstate selection should be checked with a higher-level treatment of the catalytic proton network.",
            "Do not compare absolute energies between BO2 and 6V8 because the molecular compositions differ.",
            "Only relative energies among same-composition microstates of the same inhibitor are meaningful in this pre-screen.",
        ],
        "next_step":
            "Validate optimized H geometry and inspect the relative microstate pre-screen; "
            "then choose a higher-level validation strategy for the catalytic proton network.",
    }

    Path(args.report).write_text(json.dumps(report, indent=2) + "\n")

    print(f"Wrote {args.report}")
    print()
    print("Stage-8g GFN2-xTB H-only optimization")
    print(f"  xtb: {xtb}")
    print(f"  charge: {args.charge:+d}")
    print(f"  opt level: {args.opt_level}")
    print()

    for ligand in ("6V8", "BO2"):
        print(ligand)
        subset = [r for r in results if r["ligand"] == ligand]
        for r in subset:
            e = r["GFN2_xTB_total_energy_Eh"]
            etxt = "n/a" if e is None else f"{e:.10f} Eh"
            rel = r["relative_GFN2_xTB_kcal_mol"]
            rtxt = "n/a" if rel is None else f"{rel:.3f} kcal/mol"
            print(
                f"  {r['microstate']:12s} "
                f"E={etxt:18s} rel={rtxt:14s} "
                f"H-RMSΔ={r['H_RMS_displacement_A']:.3f} A "
                f"heavy maxΔ={r['max_heavy_displacement_A']:.2e} A"
            )
        print()

    print("IMPORTANT: use these energies only as a microstate pre-screen.")
    print("Heavy-atom fixing is verified after every optimization.")


if __name__ == "__main__":
    main()
