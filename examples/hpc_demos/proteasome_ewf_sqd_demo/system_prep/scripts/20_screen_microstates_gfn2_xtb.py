#!/usr/bin/env python3
"""
20_screen_microstates_gfn2_xtb.py

Proteasome Challenge
Stage 9a: inexpensive whole-cluster proton-microstate energy screen.

Purpose
-------
Run GFN2-xTB single-point calculations on the six chemically validated,
H-only-relaxed structures.  These energies are used ONLY as a prescreen for
A/B/C catalytic protonation states.  They are not final binding energies and
are not used as the challenge reference.

Important comparisons
---------------------
Only compare A/B/C within the SAME ligand, because BO2 and 6V8 have different
chemical compositions.

For each ligand:
    dE_ms = E_ms - min(E_A, E_B, E_C)

The script also reports a ligand-differential microstate shift:
    ddE_ms = (E_ms - E_A)_BO2 - (E_ms - E_A)_6V8
which can reveal whether a microstate is differentially stabilized by the
ligand.  This is diagnostic only.

Input
-----
intermediate/hydrogen_optimized_ase_gfnff/*.xyz

Output
------
intermediate/gfn2_microstate_screen/<case>/xtb.out
validation/gfn2_microstate_screen.json
validation/gfn2_microstate_screen.tsv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
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

ENERGY_PATTERNS = [
    re.compile(r"TOTAL ENERGY\s+(-?\d+\.\d+)\s+Eh", re.I),
    re.compile(r"::\s*total energy\s+(-?\d+\.\d+)\s+Eh", re.I),
    re.compile(r"\|\s*TOTAL ENERGY\s+(-?\d+\.\d+)\s+Eh", re.I),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--indir",
        type=Path,
        default=Path("intermediate/hydrogen_optimized_ase_gfnff"),
    )
    p.add_argument(
        "--workdir",
        type=Path,
        default=Path("intermediate/gfn2_microstate_screen"),
    )
    p.add_argument(
        "--report",
        type=Path,
        default=Path("validation/gfn2_microstate_screen.json"),
    )
    p.add_argument(
        "--tsv",
        type=Path,
        default=Path("validation/gfn2_microstate_screen.tsv"),
    )
    p.add_argument(
        "--xtb",
        type=Path,
        default=Path.home() / ".local" / "xtb-conda" / "bin" / "xtb",
    )
    p.add_argument("--charge", type=int, default=-1)
    p.add_argument("--uhf", type=int, default=0)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument(
        "--timeout",
        type=int,
        default=1800,
        help="Timeout per single point in seconds.",
    )
    p.add_argument(
        "--case",
        choices=STEMS,
        default=None,
        help="Run only one case (useful for smoke testing).",
    )
    return p.parse_args()


def parse_energy(text: str):
    hits = []
    for pat in ENERGY_PATTERNS:
        hits.extend(float(m.group(1)) for m in pat.finditer(text))
    return hits[-1] if hits else None


def run_case(stem: str, args):
    src = args.indir / f"{stem}.xyz"
    if not src.exists():
        return {
            "stem": stem,
            "status": "FAIL",
            "reason": f"missing input {src}",
            "energy_Eh": None,
        }

    case_dir = args.workdir / stem
    case_dir.mkdir(parents=True, exist_ok=True)

    inp = case_dir / "input.xyz"
    shutil.copy2(src, inp)

    out = case_dir / "xtb.out"
    err = case_dir / "xtb.err"

    cmd = [
        str(args.xtb),
        "input.xyz",
        "--gfn",
        "2",
        "--chrg",
        str(args.charge),
        "--uhf",
        str(args.uhf),
    ]

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(args.threads)
    env["MKL_NUM_THREADS"] = str(args.threads)
    env["OPENBLAS_NUM_THREADS"] = str(args.threads)

    try:
        proc = subprocess.run(
            cmd,
            cwd=case_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            timeout=args.timeout,
        )
    except subprocess.TimeoutExpired as exc:
        out.write_text(exc.stdout or "")
        err.write_text(exc.stderr or "")
        return {
            "stem": stem,
            "status": "FAIL",
            "reason": f"timeout after {args.timeout} s",
            "energy_Eh": None,
            "command": cmd,
        }

    out.write_text(proc.stdout)
    err.write_text(proc.stderr)

    text = proc.stdout + "\n" + proc.stderr
    energy = parse_energy(text)

    normal = "normal termination of xtb" in text.lower()
    ok = proc.returncode == 0 and energy is not None and normal

    return {
        "stem": stem,
        "status": "PASS" if ok else "FAIL",
        "returncode": proc.returncode,
        "normal_termination": normal,
        "energy_Eh": energy,
        "command": cmd,
        "stdout": str(out),
        "stderr": str(err),
        "reason": None if ok else "xTB did not yield a normal parsed single point",
    }


def annotate_relative(results):
    by_stem = {r["stem"]: r for r in results}

    for ligand in ("6V8", "BO2"):
        group = [
            r for r in results
            if r["stem"].startswith(ligand + "_") and r["energy_Eh"] is not None
        ]
        if len(group) != 3:
            continue

        emin = min(r["energy_Eh"] for r in group)
        ea = by_stem[f"{ligand}_MS_A_THRN"]["energy_Eh"]

        for r in group:
            r["delta_to_min_Eh"] = r["energy_Eh"] - emin
            r["delta_to_min_kcal_mol"] = (
                r["energy_Eh"] - emin
            ) * HARTREE_TO_KCAL
            r["delta_to_A_Eh"] = r["energy_Eh"] - ea
            r["delta_to_A_kcal_mol"] = (
                r["energy_Eh"] - ea
            ) * HARTREE_TO_KCAL

    differential = {}
    for ms in ("MS_A_THRN", "MS_B_LYS33", "MS_C_ASP17"):
        a = by_stem.get(f"6V8_{ms}")
        b = by_stem.get(f"BO2_{ms}")
        a_ref = by_stem.get("6V8_MS_A_THRN")
        b_ref = by_stem.get("BO2_MS_A_THRN")
        if not all(
            x is not None and x.get("energy_Eh") is not None
            for x in (a, b, a_ref, b_ref)
        ):
            continue

        val = (
            (b["energy_Eh"] - b_ref["energy_Eh"])
            - (a["energy_Eh"] - a_ref["energy_Eh"])
        )
        differential[ms] = {
            "delta_delta_Eh": val,
            "delta_delta_kcal_mol": val * HARTREE_TO_KCAL,
        }

    return differential


def main():
    args = parse_args()

    if not args.xtb.exists():
        raise SystemExit(
            f"xTB executable not found: {args.xtb}\n"
            "Pass --xtb /path/to/xtb if needed."
        )

    args.workdir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    stems = [args.case] if args.case else STEMS

    print("Stage 9a: whole-cluster GFN2-xTB microstate screen")
    print(f"xTB    : {args.xtb}")
    print(f"inputs : {args.indir}")
    print(f"charge : {args.charge}")
    print(f"threads: {args.threads}")
    print()

    results = []
    for i, stem in enumerate(stems, 1):
        print(f"[{i}/{len(stems)}] {stem}")
        r = run_case(stem, args)
        results.append(r)
        if r["status"] == "PASS":
            print(f"  PASS  E = {r['energy_Eh']:.12f} Eh")
        else:
            print(f"  FAIL  {r.get('reason')}")
            if r.get("stderr"):
                print(f"        see {r['stderr']}")

    # Relative energies require all six, so only annotate full-screen runs.
    differential = {}
    if not args.case:
        differential = annotate_relative(results)

    overall = "PASS" if all(r["status"] == "PASS" for r in results) else "FAIL"

    payload = {
        "stage": "9a_whole_cluster_gfn2_microstate_prescreen",
        "scientific_role": (
            "Inexpensive whole-cluster prescreen of catalytic protonation "
            "microstates. GFN2-xTB energies are diagnostic only and are not "
            "final binding energies or benchmark truth."
        ),
        "comparison_rule": (
            "Compare A/B/C only within the same ligand. Cross-ligand absolute "
            "GFN2 energies are not comparable because BO2 and 6V8 compositions differ."
        ),
        "charge": args.charge,
        "uhf": args.uhf,
        "threads": args.threads,
        "results": results,
        "ligand_differential_microstate_shifts": differential,
        "overall": overall,
    }

    # In one-case smoke-test mode, preserve a separate report.
    report = args.report
    tsv = args.tsv
    if args.case:
        report = report.with_name(report.stem + f"_{args.case}" + report.suffix)
        tsv = tsv.with_name(tsv.stem + f"_{args.case}" + tsv.suffix)

    report.write_text(json.dumps(payload, indent=2) + "\n")

    fields = [
        "stem",
        "status",
        "energy_Eh",
        "delta_to_min_Eh",
        "delta_to_min_kcal_mol",
        "delta_to_A_Eh",
        "delta_to_A_kcal_mol",
        "normal_termination",
        "returncode",
        "reason",
    ]
    with tsv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        w.writerows(results)

    if not args.case and overall == "PASS":
        print()
        print("Relative energies within each ligand:")
        for ligand in ("6V8", "BO2"):
            print(f"  {ligand}")
            for stem in STEMS:
                if not stem.startswith(ligand + "_"):
                    continue
                r = next(x for x in results if x["stem"] == stem)
                print(
                    f"    {stem.split(ligand + '_',1)[1]:12s} "
                    f"dE(min) = {r['delta_to_min_kcal_mol']:9.4f} kcal/mol  "
                    f"dE(A) = {r['delta_to_A_kcal_mol']:9.4f} kcal/mol"
                )

        print()
        print("Ligand-differential microstate shifts relative to MS_A:")
        for ms, d in differential.items():
            print(
                f"  {ms:12s} "
                f"ddE = {d['delta_delta_kcal_mol']:9.4f} kcal/mol"
            )

    print()
    print(f"Overall: {overall}")
    print(f"JSON: {report}")
    print(f"TSV : {tsv}")

    raise SystemExit(0 if overall == "PASS" else 2)


if __name__ == "__main__":
    main()
