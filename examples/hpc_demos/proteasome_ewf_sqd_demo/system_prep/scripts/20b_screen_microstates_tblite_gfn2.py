#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

try:
    import ase
    from ase.io import read
except ImportError as exc:
    raise SystemExit(
        "ASE is not installed. Install with:\n  python3 -m pip install ase"
    ) from exc

try:
    import tblite
    from tblite.ase import TBLite
except ImportError as exc:
    raise SystemExit(
        "tblite is not installed. Install with:\n  python3 -m pip install tblite"
    ) from exc

EV_TO_HARTREE = 1.0 / 27.211386245988
EV_TO_KCAL_MOL = 23.060547830619

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
    p.add_argument("--indir", type=Path,
                   default=Path("intermediate/hydrogen_optimized_ase_gfnff"))
    p.add_argument("--report", type=Path,
                   default=Path("validation/gfn2_tblite_microstate_screen.json"))
    p.add_argument("--tsv", type=Path,
                   default=Path("validation/gfn2_tblite_microstate_screen.tsv"))
    p.add_argument("--charge", type=int, default=-1)
    p.add_argument("--multiplicity", type=int, default=1)
    p.add_argument("--accuracy", type=float, default=1.0)
    p.add_argument("--max-iterations", type=int, default=500)
    p.add_argument("--electronic-temperature", type=float, default=300.0)
    p.add_argument("--case", choices=STEMS, default=None)
    p.add_argument("--verbosity", type=int, default=1, choices=[0, 1, 2])
    return p.parse_args()

def pkg_version(mod):
    return getattr(mod, "__version__", "unknown")

def run_case(stem, args):
    path = args.indir / f"{stem}.xyz"
    if not path.exists():
        return {"stem": stem, "status": "FAIL",
                "reason": f"missing input: {path}",
                "energy_eV": None, "energy_Eh": None}
    try:
        atoms = read(path)
        atoms.calc = TBLite(
            method="GFN2-xTB",
            charge=args.charge,
            multiplicity=args.multiplicity,
            accuracy=args.accuracy,
            max_iterations=args.max_iterations,
            electronic_temperature=args.electronic_temperature,
            verbosity=args.verbosity,
        )
        energy_ev = float(atoms.get_potential_energy())
        energy_eh = energy_ev * EV_TO_HARTREE
        charges = atoms.get_charges()
        qsum = float(charges.sum())
        if not math.isfinite(energy_ev):
            raise RuntimeError("non-finite total energy")
        return {
            "stem": stem,
            "status": "PASS",
            "reason": None,
            "n_atoms": len(atoms),
            "energy_eV": energy_ev,
            "energy_Eh": energy_eh,
            "mulliken_charge_sum_e": qsum,
        }
    except Exception as exc:
        return {
            "stem": stem,
            "status": "FAIL",
            "reason": f"{type(exc).__name__}: {exc}",
            "energy_eV": None,
            "energy_Eh": None,
        }

def annotate_relative(results):
    by_stem = {r["stem"]: r for r in results}
    for ligand in ("6V8", "BO2"):
        group = [r for r in results
                 if r["stem"].startswith(ligand + "_")
                 and r["status"] == "PASS"]
        if len(group) != 3:
            continue
        emin = min(r["energy_eV"] for r in group)
        ea = by_stem[f"{ligand}_MS_A_THRN"]["energy_eV"]
        for r in group:
            r["delta_to_min_eV"] = r["energy_eV"] - emin
            r["delta_to_min_kcal_mol"] = (
                r["energy_eV"] - emin) * EV_TO_KCAL_MOL
            r["delta_to_A_eV"] = r["energy_eV"] - ea
            r["delta_to_A_kcal_mol"] = (
                r["energy_eV"] - ea) * EV_TO_KCAL_MOL

    differential = {}
    for ms in ("MS_A_THRN", "MS_B_LYS33", "MS_C_ASP17"):
        needed = [
            f"6V8_{ms}", f"BO2_{ms}",
            "6V8_MS_A_THRN", "BO2_MS_A_THRN"
        ]
        if not all(s in by_stem and by_stem[s]["status"] == "PASS"
                   for s in needed):
            continue
        a = by_stem[f"6V8_{ms}"]["energy_eV"]
        b = by_stem[f"BO2_{ms}"]["energy_eV"]
        a0 = by_stem["6V8_MS_A_THRN"]["energy_eV"]
        b0 = by_stem["BO2_MS_A_THRN"]["energy_eV"]
        dde = (b - b0) - (a - a0)
        differential[ms] = {
            "delta_delta_eV": dde,
            "delta_delta_kcal_mol": dde * EV_TO_KCAL_MOL,
        }
    return differential

def main():
    args = parse_args()
    stems = [args.case] if args.case else STEMS

    print("Stage 9a: GFN2-xTB microstate prescreen via tblite")
    print(f"Python : {sys.version.split()[0]}")
    print(f"ASE    : {pkg_version(ase)}")
    print(f"tblite : {pkg_version(tblite)}")
    print(f"inputs : {args.indir}")
    print(f"charge : {args.charge}")
    print(f"mult   : {args.multiplicity}")
    print()

    results = []
    for i, stem in enumerate(stems, 1):
        print(f"[{i}/{len(stems)}] {stem}")
        r = run_case(stem, args)
        results.append(r)
        if r["status"] == "PASS":
            print(
                f"  PASS  E = {r['energy_Eh']:.12f} Eh  "
                f"({r['energy_eV']:.8f} eV)  "
                f"sum(q)={r['mulliken_charge_sum_e']:.6f}"
            )
        else:
            print(f"  FAIL  {r['reason']}")

    differential = {}
    if not args.case:
        differential = annotate_relative(results)

    overall = "PASS" if all(r["status"] == "PASS" for r in results) else "FAIL"

    payload = {
        "stage": "9a_whole_cluster_gfn2_tblite_microstate_prescreen",
        "scientific_role": (
            "Diagnostic whole-cluster prescreen of A/B/C catalytic "
            "protonation microstates using GFN2-xTB through tblite. "
            "These energies are not final binding energies or benchmark truth."
        ),
        "comparison_rule": (
            "Compare A/B/C only within the same ligand. BO2 and 6V8 have "
            "different compositions, so their absolute total energies must "
            "not be subtracted directly."
        ),
        "software": {
            "python": sys.version.split()[0],
            "ase": pkg_version(ase),
            "tblite": pkg_version(tblite),
        },
        "settings": {
            "method": "GFN2-xTB",
            "charge": args.charge,
            "multiplicity": args.multiplicity,
            "accuracy": args.accuracy,
            "max_iterations": args.max_iterations,
            "electronic_temperature_K": args.electronic_temperature,
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
        "stem", "status", "n_atoms", "energy_eV", "energy_Eh",
        "mulliken_charge_sum_e", "delta_to_min_eV",
        "delta_to_min_kcal_mol", "delta_to_A_eV",
        "delta_to_A_kcal_mol", "reason",
    ]
    with tsv.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    if not args.case and overall == "PASS":
        print()
        print("Relative microstate energies within each ligand:")
        for ligand in ("6V8", "BO2"):
            print(f"  {ligand}")
            for stem in STEMS:
                if not stem.startswith(ligand + "_"):
                    continue
                r = next(x for x in results if x["stem"] == stem)
                short = stem.split(ligand + "_", 1)[1]
                print(
                    f"    {short:12s} "
                    f"dE(min)={r['delta_to_min_kcal_mol']:9.4f} kcal/mol  "
                    f"dE(A)={r['delta_to_A_kcal_mol']:9.4f} kcal/mol"
                )
        print()
        print("Ligand-differential microstate shifts relative to MS_A:")
        for ms, d in differential.items():
            print(
                f"  {ms:12s} "
                f"ddE={d['delta_delta_kcal_mol']:9.4f} kcal/mol"
            )

    print()
    print(f"Overall: {overall}")
    print(f"JSON: {report}")
    print(f"TSV : {tsv}")

    raise SystemExit(0 if overall == "PASS" else 2)

if __name__ == "__main__":
    main()
