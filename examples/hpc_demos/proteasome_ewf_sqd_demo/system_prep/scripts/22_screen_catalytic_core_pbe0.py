#!/usr/bin/env python3
"""
22_screen_catalytic_core_pbe0.py

Run PBE0/def2-SVP single-point energies on the six compact catalytic-core
microstate models built by 21_build_catalytic_core_models.py.

Scientific role
---------------
This is a higher-level *ordering check* for A/B/C proton microstates.
It is not a final binding-energy calculation.

Method
------
- Restricted Kohn-Sham DFT
- PBE0 hybrid functional
- def2-SVP basis
- density fitting / RI-JK via PySCF
- fixed geometries
- charge = -1
- singlet

Why this level?
---------------
PBE0/def2-SVP is inexpensive enough for the compact core, substantially more
first-principles than GFN2-xTB, and avoids interpreting a semiempirical
charged proton-transfer result as final chemistry. Dispersion is omitted here
because A/B/C have identical composition and nearly identical fixed geometry;
this stage is specifically a proton-state ordering screen.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

HARTREE_TO_KCAL = 627.5094740631

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
        "--indir",
        type=Path,
        default=Path("intermediate/catalytic_core_dft"),
    )
    p.add_argument(
        "--report",
        type=Path,
        default=Path("validation/pbe0_def2svp_core_microstate_screen.json"),
    )
    p.add_argument(
        "--tsv",
        type=Path,
        default=Path("validation/pbe0_def2svp_core_microstate_screen.tsv"),
    )
    p.add_argument("--charge", type=int, default=-1)
    p.add_argument("--spin", type=int, default=0)
    p.add_argument("--basis", default="def2-svp")
    p.add_argument("--xc", default="pbe0")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--memory-mb", type=int, default=8000)
    p.add_argument("--conv-tol", type=float, default=1e-8)
    p.add_argument("--max-cycle", type=int, default=200)
    p.add_argument("--grid-level", type=int, default=3)
    p.add_argument("--case", choices=STEMS, default=None)
    return p.parse_args()


def read_xyz(path):
    lines = path.read_text().splitlines()
    n = int(lines[0])
    atoms = []
    for line in lines[2:2+n]:
        s = line.split()
        atoms.append(
            (s[0], (float(s[1]), float(s[2]), float(s[3])))
        )
    if len(atoms) != n:
        raise RuntimeError(f"{path}: expected {n} atoms, got {len(atoms)}")
    return atoms


def run_case(stem, args):
    try:
        from pyscf import gto, dft, lib
        import pyscf
    except ImportError as exc:
        return {
            "stem": stem,
            "status": "FAIL",
            "reason": (
                "PySCF not installed. Run: python3 -m pip install pyscf"
            ),
            "energy_Eh": None,
        }

    path = args.indir / f"{stem}_core.xyz"
    if not path.exists():
        return {
            "stem": stem,
            "status": "FAIL",
            "reason": f"missing input {path}",
            "energy_Eh": None,
        }

    atoms = read_xyz(path)
    lib.num_threads(args.threads)

    try:
        mol = gto.M(
            atom=atoms,
            unit="Angstrom",
            basis=args.basis,
            charge=args.charge,
            spin=args.spin,
            verbose=4,
            max_memory=args.memory_mb,
        )

        mf = dft.RKS(mol, xc=args.xc).density_fit()
        mf.grids.level = args.grid_level
        mf.conv_tol = args.conv_tol
        mf.max_cycle = args.max_cycle

        e = float(mf.kernel())

        converged = bool(mf.converged)
        if not converged:
            # One robust second attempt using Newton SCF from the current density.
            mf2 = mf.newton()
            mf2.conv_tol = args.conv_tol
            mf2.max_cycle = args.max_cycle
            e = float(mf2.kernel())
            converged = bool(mf2.converged)
            mf = mf2

        nelec = int(mol.nelectron)
        nao = int(mol.nao_nr())

        return {
            "stem": stem,
            "status": "PASS" if converged and math.isfinite(e) else "FAIL",
            "reason": None if converged else "SCF did not converge",
            "energy_Eh": e,
            "n_atoms": len(atoms),
            "n_electrons": nelec,
            "n_AO": nao,
            "scf_converged": converged,
            "pyscf_version": getattr(pyscf, "__version__", "unknown"),
        }

    except Exception as exc:
        return {
            "stem": stem,
            "status": "FAIL",
            "reason": f"{type(exc).__name__}: {exc}",
            "energy_Eh": None,
        }


def annotate(results):
    by = {r["stem"]: r for r in results}

    for ligand in ("6V8", "BO2"):
        group = [
            r for r in results
            if r["stem"].startswith(ligand + "_") and r["status"] == "PASS"
        ]
        if len(group) != 3:
            continue
        emin = min(r["energy_Eh"] for r in group)
        ea = by[f"{ligand}_MS_A_THRN"]["energy_Eh"]
        for r in group:
            r["delta_to_min_kcal_mol"] = (
                r["energy_Eh"] - emin
            ) * HARTREE_TO_KCAL
            r["delta_to_A_kcal_mol"] = (
                r["energy_Eh"] - ea
            ) * HARTREE_TO_KCAL

    differential = {}
    for ms in ("MS_A_THRN", "MS_B_LYS33", "MS_C_ASP17"):
        keys = [
            f"6V8_{ms}",
            f"BO2_{ms}",
            "6V8_MS_A_THRN",
            "BO2_MS_A_THRN",
        ]
        if not all(k in by and by[k]["status"] == "PASS" for k in keys):
            continue
        val = (
            (by[f"BO2_{ms}"]["energy_Eh"] - by["BO2_MS_A_THRN"]["energy_Eh"])
            - (by[f"6V8_{ms}"]["energy_Eh"] - by["6V8_MS_A_THRN"]["energy_Eh"])
        )
        differential[ms] = {
            "delta_delta_Eh": val,
            "delta_delta_kcal_mol": val * HARTREE_TO_KCAL,
        }
    return differential


def main():
    args = parse_args()
    stems = [args.case] if args.case else STEMS

    print("Stage 9b: PBE0/def2-SVP catalytic-core microstate screen")
    print(f"inputs : {args.indir}")
    print(f"xc     : {args.xc}")
    print(f"basis  : {args.basis}")
    print(f"charge : {args.charge}")
    print(f"spin   : {args.spin}")
    print(f"threads: {args.threads}")
    print()

    results = []
    for i, stem in enumerate(stems, 1):
        print(f"[{i}/{len(stems)}] {stem}")
        r = run_case(stem, args)
        results.append(r)
        if r["status"] == "PASS":
            print(
                f"  PASS E={r['energy_Eh']:.12f} Eh "
                f"nelec={r['n_electrons']} nao={r['n_AO']}"
            )
        else:
            print(f"  FAIL {r['reason']}")

    differential = {}
    if not args.case:
        differential = annotate(results)

    overall = "PASS" if all(r["status"] == "PASS" for r in results) else "FAIL"

    payload = {
        "stage": "9b_pbe0_def2svp_minimal_core_microstate_screen",
        "scientific_role": (
            "Higher-level catalytic-core A/B/C ordering check. "
            "Not a final binding energy."
        ),
        "method": {
            "xc": args.xc,
            "basis": args.basis,
            "density_fitting": True,
            "charge": args.charge,
            "spin": args.spin,
            "grid_level": args.grid_level,
            "conv_tol": args.conv_tol,
        },
        "results": results,
        "ligand_differential_microstate_shifts": differential,
        "overall": overall,
    }

    report = args.report
    tsv = args.tsv
    if args.case:
        report = report.with_name(report.stem + f"_{args.case}" + report.suffix)
        tsv = tsv.with_name(tsv.stem + f"_{args.case}" + tsv.suffix)

    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(payload, indent=2) + "\n")

    fields = [
        "stem", "status", "energy_Eh", "n_atoms", "n_electrons", "n_AO",
        "scf_converged", "delta_to_min_kcal_mol",
        "delta_to_A_kcal_mol", "reason",
    ]
    with tsv.open("w", newline="") as fh:
        w = csv.DictWriter(
            fh, fieldnames=fields, delimiter="\t", extrasaction="ignore"
        )
        w.writeheader()
        w.writerows(results)

    if not args.case and overall == "PASS":
        print()
        print("Relative PBE0/def2-SVP energies within each ligand:")
        for ligand in ("6V8", "BO2"):
            print(f"  {ligand}")
            for stem in STEMS:
                if not stem.startswith(ligand + "_"):
                    continue
                r = next(x for x in results if x["stem"] == stem)
                short = stem.split(ligand + "_", 1)[1]
                print(
                    f"    {short:12s} "
                    f"dE(min)={r['delta_to_min_kcal_mol']:9.3f} kcal/mol "
                    f"dE(A)={r['delta_to_A_kcal_mol']:9.3f} kcal/mol"
                )

        print()
        print("Ligand-differential shifts relative to A:")
        for ms, d in differential.items():
            print(
                f"  {ms:12s} "
                f"ddE={d['delta_delta_kcal_mol']:9.3f} kcal/mol"
            )

    print()
    print(f"Overall: {overall}")
    print(f"JSON: {report}")
    print(f"TSV : {tsv}")

    raise SystemExit(0 if overall == "PASS" else 2)


if __name__ == "__main__":
    main()
