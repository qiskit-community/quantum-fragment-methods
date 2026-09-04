#!/usr/bin/env python3
"""
10_build_capped_receptor_scaffold.py

Proteasome Challenge
Stage 8a: execute the molecular scalpel and build the first capped
heavy-atom receptor scaffold from canonical 5LF7 coordinates.

Frozen first-reference peptide spans:
  Y:1-3
  Y:17-22
  Y:45-50
  Y:129-131
  Y:167-170
  Z:124-126

Retained crystallographic waters:
  Y:437
  Z:406
  Y:462

Cap convention
--------------
Internal N-side cut:
    ACE-like cap using the immediately preceding residue's
    CA, C, O coordinates:
        previous CA -> ACE CH3 heavy-atom anchor
        previous C  -> ACE C
        previous O  -> ACE O

Internal C-side cut:
    NME-like cap using the immediately following residue's
    N, CA coordinates:
        next N  -> NME N
        next CA -> NME CH3 heavy-atom anchor

This preserves the experimental peptide-bond geometry at the cut and avoids
inventing cap heavy-atom orientations at this stage.

IMPORTANT
---------
This script intentionally builds a HEAVY-ATOM scaffold only.
It does NOT:
  - add cap hydrogens
  - protonate side chains
  - add water hydrogens
  - optimize geometry
  - assign final total charge
  - modify the catalytic Thr1 N-terminus

Catalytic Thr1 remains the native N-terminal residue and receives NO N-cap.

Outputs
-------
1. PDB heavy-atom receptor scaffold
2. JSON manifest documenting every retained residue/water/cap atom
3. TSV atom manifest for auditing

Example
-------
python3 scripts/10_build_capped_receptor_scaffold.py \
  raw/5LF7.cif \
  --out intermediate/capped_reference/receptor_heavy_scaffold.pdb \
  --manifest validation/capped_receptor_scaffold.json \
  --atoms validation/capped_receptor_atoms.tsv
"""

import argparse
import csv
import json
import math
from pathlib import Path

import gemmi


SEGMENTS = [
    ("Y", 1, 3),
    ("Y", 17, 22),
    ("Y", 45, 50),
    ("Y", 129, 131),
    ("Y", 167, 170),
    ("Z", 124, 126),
]

WATERS = [
    ("Y", "437"),
    ("Z", "406"),
    ("Y", "462"),
]

BACKBONE = {"N", "CA", "C", "O", "OXT"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("canonical_cif", help="Canonical 5LF7 mmCIF")
    p.add_argument("--out", required=True, help="Output heavy-atom PDB")
    p.add_argument("--manifest", required=True, help="Output JSON manifest")
    p.add_argument("--atoms", required=True, help="Output atom TSV")
    return p.parse_args()


def clean(x):
    if x is None:
        return None
    x = str(x).strip()
    if x in ("", ".", "?"):
        return None
    return x


def as_int(x):
    x = clean(x)
    if x is None:
        return None
    try:
        return int(x)
    except ValueError:
        return None


def load_structure(cif_path):
    doc = gemmi.cif.read_file(str(cif_path))
    block = doc.sole_block()

    tags = [
        "_atom_site.group_PDB",
        "_atom_site.type_symbol",
        "_atom_site.label_atom_id",
        "_atom_site.label_alt_id",
        "_atom_site.label_comp_id",
        "_atom_site.label_asym_id",
        "_atom_site.label_seq_id",
        "_atom_site.auth_asym_id",
        "_atom_site.auth_seq_id",
        "_atom_site.Cartn_x",
        "_atom_site.Cartn_y",
        "_atom_site.Cartn_z",
        "_atom_site.occupancy",
        "_atom_site.B_iso_or_equiv",
    ]

    table = block.find(tags)
    if not table:
        raise RuntimeError("Could not read required _atom_site fields")

    polymer = {}
    hetero = {}

    for row in table:
        group = clean(row[0])
        elem = clean(row[1])
        atom = clean(row[2])
        alt = clean(row[3])
        comp = clean(row[4])
        lchain = clean(row[5])
        lseq = as_int(row[6])
        achain = clean(row[7])
        aseq = clean(row[8])
        x, y, z = float(row[9]), float(row[10]), float(row[11])
        occ = float(row[12]) if clean(row[12]) is not None else 1.0
        b = float(row[13]) if clean(row[13]) is not None else 0.0

        if alt not in (None, "A"):
            continue
        if elem and elem.upper() == "H":
            continue

        rec_atom = {
            "name": atom,
            "element": elem,
            "xyz": (x, y, z),
            "occupancy": occ,
            "b": b,
        }

        if group == "ATOM" and lchain is not None and lseq is not None:
            key = (lchain, lseq)
            rec = polymer.setdefault(key, {
                "label_chain": lchain,
                "label_seq": lseq,
                "auth_chain": achain,
                "auth_seq": aseq,
                "comp_id": comp,
                "atoms": {},
            })
            rec["atoms"].setdefault(atom, rec_atom)

        elif group == "HETATM":
            key = (achain, aseq, comp)
            rec = hetero.setdefault(key, {
                "auth_chain": achain,
                "auth_seq": aseq,
                "label_chain": lchain,
                "label_seq": lseq,
                "comp_id": comp,
                "atoms": {},
            })
            rec["atoms"].setdefault(atom, rec_atom)

    return polymer, hetero


def dist(a, b):
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def pdb_atom_line(serial, atom_name, resname, chain, resseq,
                  xyz, element, record="ATOM", occupancy=1.0, b=0.0):
    """
    Basic fixed-column PDB writer sufficient for inspection/open-source tools.
    """
    x, y, z = xyz
    # PDB atom-name alignment: leading space for most non-4-char names.
    if len(atom_name) < 4:
        aname = f" {atom_name:<3s}"
    else:
        aname = atom_name[:4]
    return (
        f"{record:<6s}{serial:5d} {aname} {resname:>3s} {chain:1s}"
        f"{resseq:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}"
        f"{occupancy:6.2f}{b:6.2f}          {element:>2s}"
    )


def get_water(hetero, chain, seq):
    candidates = []
    for (achain, aseq, comp), rec in hetero.items():
        if achain == chain and str(aseq) == str(seq):
            candidates.append(rec)
    if not candidates:
        raise RuntimeError(f"Could not find retained water {chain}:{seq}")
    # Prefer standard water names.
    candidates.sort(key=lambda r: 0 if r["comp_id"] in ("HOH", "WAT", "H2O") else 1)
    return candidates[0]


def validate_segment(polymer, chain, start, end):
    for seq in range(start, end + 1):
        if (chain, seq) not in polymer:
            raise RuntimeError(f"Missing polymer residue {chain}:{seq}")


def main():
    args = parse_args()
    polymer, hetero = load_structure(Path(args.canonical_cif))

    for chain, start, end in SEGMENTS:
        validate_segment(polymer, chain, start, end)

    output_atoms = []
    cap_records = []
    retained_records = []

    # Stable artificial PDB chains for caps/waters are not necessary:
    # keep the parent chain for each cap to make provenance obvious.
    # Cap residue numbers are assigned negative/large internal manifest IDs,
    # while the PDB uses sequential harmless positive residue IDs per cap.
    cap_pdb_resseq = 900
    water_pdb_resseq = 950

    # ------------------------------------------------------------------
    # Retained protein atoms
    # ------------------------------------------------------------------
    for chain, start, end in SEGMENTS:
        for seq in range(start, end + 1):
            rec = polymer[(chain, seq)]
            retained_records.append({
                "label_chain": chain,
                "label_seq": seq,
                "auth_chain": rec["auth_chain"],
                "auth_seq": rec["auth_seq"],
                "comp_id": rec["comp_id"],
            })

            for atom_name, atom in rec["atoms"].items():
                output_atoms.append({
                    "record": "ATOM",
                    "atom_name": atom_name,
                    "resname": rec["comp_id"],
                    "chain": chain,
                    "resseq": seq,
                    "xyz": atom["xyz"],
                    "element": atom["element"],
                    "occupancy": atom["occupancy"],
                    "b": atom["b"],
                    "source": f"retained:{chain}:{seq}:{atom_name}",
                    "role": "protein",
                })

    # ------------------------------------------------------------------
    # Caps
    # ------------------------------------------------------------------
    for chain, start, end in SEGMENTS:
        # N cap except at natural catalytic N terminus Y:1.
        if start != 1:
            prev = polymer.get((chain, start - 1))
            if prev is None:
                raise RuntimeError(f"Missing previous residue for N cap {chain}:{start}")

            required = ("CA", "C", "O")
            for name in required:
                if name not in prev["atoms"]:
                    raise RuntimeError(
                        f"Missing {chain}:{start-1} {name} required for ACE cap"
                    )

            cap_pdb_resseq += 1
            mapping = [
                ("CH3", "C", "CA"),
                ("C",   "C", "C"),
                ("O",   "O", "O"),
            ]
            cap_atoms = []
            for new_name, elem, source_name in mapping:
                source = prev["atoms"][source_name]
                output_atoms.append({
                    "record": "HETATM",
                    "atom_name": new_name,
                    "resname": "ACE",
                    "chain": chain,
                    "resseq": cap_pdb_resseq,
                    "xyz": source["xyz"],
                    "element": elem,
                    "occupancy": 1.0,
                    "b": source["b"],
                    "source": f"ACE_from:{chain}:{start-1}:{source_name}",
                    "role": "N_cap",
                })
                cap_atoms.append({
                    "new_atom": new_name,
                    "source_residue": f"{chain}:{start-1}",
                    "source_atom": source_name,
                    "xyz": list(source["xyz"]),
                })

            # Verify preserved peptide C--N bond.
            cap_C = prev["atoms"]["C"]["xyz"]
            retained_N = polymer[(chain, start)]["atoms"]["N"]["xyz"]
            cn = dist(cap_C, retained_N)

            cap_records.append({
                "type": "ACE",
                "caps_N_terminus_of_segment": f"{chain}:{start}-{end}",
                "source_outside_residue":
                    f'{chain}:{start-1} {prev["comp_id"]}',
                "pdb_resseq": cap_pdb_resseq,
                "atoms": cap_atoms,
                "cap_C_to_retained_N_A": round(cn, 4),
            })

        # C-side NME cap for every current segment.
        nxt = polymer.get((chain, end + 1))
        if nxt is None:
            raise RuntimeError(f"Missing next residue for C cap {chain}:{end}")

        required = ("N", "CA")
        for name in required:
            if name not in nxt["atoms"]:
                raise RuntimeError(
                    f"Missing {chain}:{end+1} {name} required for NME cap"
                )

        cap_pdb_resseq += 1
        mapping = [
            ("N",   "N", "N"),
            ("CH3", "C", "CA"),
        ]
        cap_atoms = []
        for new_name, elem, source_name in mapping:
            source = nxt["atoms"][source_name]
            output_atoms.append({
                "record": "HETATM",
                "atom_name": new_name,
                "resname": "NME",
                "chain": chain,
                "resseq": cap_pdb_resseq,
                "xyz": source["xyz"],
                "element": elem,
                "occupancy": 1.0,
                "b": source["b"],
                "source": f"NME_from:{chain}:{end+1}:{source_name}",
                "role": "C_cap",
            })
            cap_atoms.append({
                "new_atom": new_name,
                "source_residue": f"{chain}:{end+1}",
                "source_atom": source_name,
                "xyz": list(source["xyz"]),
            })

        retained_C = polymer[(chain, end)]["atoms"]["C"]["xyz"]
        cap_N = nxt["atoms"]["N"]["xyz"]
        cn = dist(retained_C, cap_N)

        cap_records.append({
            "type": "NME",
            "caps_C_terminus_of_segment": f"{chain}:{start}-{end}",
            "source_outside_residue":
                f'{chain}:{end+1} {nxt["comp_id"]}',
            "pdb_resseq": cap_pdb_resseq,
            "atoms": cap_atoms,
            "retained_C_to_cap_N_A": round(cn, 4),
        })

    # ------------------------------------------------------------------
    # Retained waters (oxygen heavy atom only)
    # ------------------------------------------------------------------
    water_records = []
    for chain, seq in WATERS:
        w = get_water(hetero, chain, seq)
        # Usually water oxygen is atom O; retain all heavy atoms just in case.
        water_pdb_resseq += 1
        atom_recs = []
        for atom_name, atom in w["atoms"].items():
            if atom["element"] and atom["element"].upper() == "H":
                continue
            output_atoms.append({
                "record": "HETATM",
                "atom_name": "O" if atom["element"].upper() == "O" else atom_name,
                "resname": "HOH",
                "chain": chain,
                "resseq": water_pdb_resseq,
                "xyz": atom["xyz"],
                "element": atom["element"],
                "occupancy": 1.0,
                "b": atom["b"],
                "source": f"water:{chain}:{seq}:{atom_name}",
                "role": "water",
            })
            atom_recs.append({
                "source_atom": atom_name,
                "element": atom["element"],
                "xyz": list(atom["xyz"]),
            })

        water_records.append({
            "source_water": f"{chain}:{seq}",
            "source_comp_id": w["comp_id"],
            "pdb_resseq": water_pdb_resseq,
            "atoms": atom_recs,
        })

    # ------------------------------------------------------------------
    # Write files
    # ------------------------------------------------------------------
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
    Path(args.atoms).parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "REMARK 900 Proteasome Challenge capped receptor HEAVY-ATOM scaffold",
        "REMARK 900 No hydrogens/protonation/final charge assigned",
        "REMARK 900 Catalytic Thr1 natural N-terminus preserved",
    ]

    for serial, atom in enumerate(output_atoms, 1):
        lines.append(
            pdb_atom_line(
                serial=serial,
                atom_name=atom["atom_name"],
                resname=atom["resname"],
                chain=atom["chain"],
                resseq=atom["resseq"],
                xyz=atom["xyz"],
                element=atom["element"],
                record=atom["record"],
                occupancy=atom["occupancy"],
                b=atom["b"],
            )
        )
    lines.append("END")
    out_path.write_text("\n".join(lines) + "\n")

    manifest = {
        "stage": "8a_capped_heavy_atom_receptor_scaffold",
        "canonical_source": str(args.canonical_cif),
        "frozen_segments": [
            {"label_chain": c, "label_start": s, "label_end": e}
            for c, s, e in SEGMENTS
        ],
        "manual_boundary_decision": {
            "Y45_included": True,
            "reason":
                "Met45 lies at ~3.9-4.1 A from the two ligand poses and was retained in the generous reference despite only REVIEW status.",
            "Y51_Asp_excluded": True,
            "reason":
                "Asp51 side chain is geometrically peripheral to ligands, retained waters, and Thr1; kept as convergence candidate rather than reference residue.",
        },
        "retained_waters": water_records,
        "retained_residues": retained_records,
        "caps": cap_records,
        "counts": {
            "retained_residues": len(retained_records),
            "ACE_caps": sum(c["type"] == "ACE" for c in cap_records),
            "NME_caps": sum(c["type"] == "NME" for c in cap_records),
            "retained_waters": len(water_records),
            "heavy_atoms_total": len(output_atoms),
            "protein_heavy_atoms": sum(a["role"] == "protein" for a in output_atoms),
            "cap_heavy_atoms": sum(a["role"] in ("N_cap", "C_cap") for a in output_atoms),
            "water_heavy_atoms": sum(a["role"] == "water" for a in output_atoms),
        },
        "chemistry_policy": [
            "Thr1 remains the authentic natural N-terminus; no ACE cap is placed before it.",
            "ACE/NME heavy-atom geometry is inherited from the real neighboring peptide residues.",
            "No heavy-atom coordinate of a retained protein residue or water is changed.",
            "Cap hydrogens, protein hydrogens, water hydrogens, protonation states, and final charge are deferred.",
            "The cap scheme must be identical in the bortezomib and ixazomib complexes.",
        ],
        "next_step":
            "Inspect scaffold geometry, then assign protonation/formal charge and add hydrogens including cap and water hydrogens.",
    }
    Path(args.manifest).write_text(json.dumps(manifest, indent=2) + "\n")

    fields = [
        "serial", "role", "record", "atom_name", "resname", "chain",
        "resseq", "element", "x", "y", "z", "source"
    ]
    with open(args.atoms, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        w.writeheader()
        for serial, atom in enumerate(output_atoms, 1):
            x, y, z = atom["xyz"]
            w.writerow({
                "serial": serial,
                "role": atom["role"],
                "record": atom["record"],
                "atom_name": atom["atom_name"],
                "resname": atom["resname"],
                "chain": atom["chain"],
                "resseq": atom["resseq"],
                "element": atom["element"],
                "x": f"{x:.6f}",
                "y": f"{y:.6f}",
                "z": f"{z:.6f}",
                "source": atom["source"],
            })

    print(f"Wrote {args.out}")
    print(f"Wrote {args.manifest}")
    print(f"Wrote {args.atoms}")
    print()
    print("Stage-8a molecular scalpel execution")
    print("  frozen peptide spans:")
    for c, s, e in SEGMENTS:
        print(f"    {c}:{s}-{e}")
    print(f"  retained waters: {', '.join(f'{c}:{s}' for c, s in WATERS)}")
    print()
    print("  scaffold counts")
    for k, v in manifest["counts"].items():
        print(f"    {k}: {v}")
    print()
    print("  cap geometry checks")
    for cap in cap_records:
        if cap["type"] == "ACE":
            d = cap["cap_C_to_retained_N_A"]
            target = cap["caps_N_terminus_of_segment"]
            print(f"    ACE -> {target}: C-N = {d:.3f} A")
        else:
            d = cap["retained_C_to_cap_N_A"]
            target = cap["caps_C_terminus_of_segment"]
            print(f"    {target} -> NME: C-N = {d:.3f} A")

    print()
    print("  IMPORTANT: heavy-atom scaffold only.")
    print("  Do not run electronic-structure calculations on this file yet.")
    print("  Protonation, hydrogens, water orientation, and formal charge are next.")


if __name__ == "__main__":
    main()
