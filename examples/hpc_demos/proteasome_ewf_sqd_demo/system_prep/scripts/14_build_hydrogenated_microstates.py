#!/usr/bin/env python3
"""
14_build_hydrogenated_microstates.py

Proteasome Challenge
Stage 8e: build explicit hydrogenated BO2/6V8 catalytic microstates.

Strategy
--------
1. Merge the v2 capped receptor heavy scaffold with the appropriate ligand
   heavy atoms in the common 5LF7-Y receptor frame.
2. Add ordinary valence hydrogens with Open Babel (`obabel -h`).
3. Explicitly override hydrogen counts for all chemically controlled sites:
     - Thr1 N
     - Thr1 OG1
     - Lys33 NZ
     - Asp17 OD1/OD2
     - Arg19
     - Lys32
     - Asp167
     - Tyr169
     - Asp125
     - the two B-bound ligand oxygens
     - the three retained waters
4. Write PDB + XYZ for all 6 ligand/microstate combinations.
5. Audit H counts, total atom counts, and charge metadata.

IMPORTANT
---------
This creates INITIAL H coordinates only.
All receptor/ligand HEAVY ATOMS remain exactly fixed.
Hydrogen positions must subsequently be optimized with the same protocol for
all microstates.

Dependency
----------
Open Babel command line executable:
    obabel

Example
-------
python3 scripts/14_build_hydrogenated_microstates.py \
  raw/5LF3.cif raw/5LF7.cif \
  --scaffold intermediate/capped_reference/receptor_heavy_scaffold_v2.pdb \
  --manifest validation/capped_receptor_scaffold_v2.json \
  --microstates validation/catalytic_microstates.json \
  --boronate validation/boronate_coordination_audit.json \
  --outdir intermediate/hydrogenated_microstates \
  --report validation/hydrogenated_microstates.json
"""

import argparse
import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path

import gemmi
import numpy as np


# ---------------------------------------------------------------------------
# Basic geometry
# ---------------------------------------------------------------------------

def norm(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < 1e-10:
        raise ValueError("zero-length vector")
    return v / n


def orthogonal(v):
    v = norm(v)
    trial = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(v, trial)) > 0.85:
        trial = np.array([0.0, 1.0, 0.0])
    return norm(np.cross(v, trial))


def dist(a, b):
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


def h_vectors_one_heavy(center, heavy_neighbor, n_h, bond=1.01):
    """
    Approximate tetrahedral H placement around an atom with one heavy neighbor.
    Intended only as an initial geometry for later H-only optimization.
    """
    center = np.asarray(center)
    heavy_neighbor = np.asarray(heavy_neighbor)

    # Axis points away from the existing heavy-atom bond.
    axis = norm(center - heavy_neighbor)
    e1 = orthogonal(axis)
    e2 = norm(np.cross(axis, e1))

    axial = 1.0 / 3.0
    radial = math.sqrt(1.0 - axial * axial)

    if n_h == 1:
        phis = [0.0]
    elif n_h == 2:
        phis = [60.0, 300.0]
    elif n_h == 3:
        phis = [0.0, 120.0, 240.0]
    else:
        raise ValueError(f"Unsupported H count {n_h}")

    out = []
    for phi_deg in phis:
        phi = math.radians(phi_deg)
        v = axial * axis + radial * (math.cos(phi) * e1 + math.sin(phi) * e2)
        out.append(center + bond * norm(v))
    return out


def water_h_positions(O, away_from):
    """
    Initial water geometry: HOH = 104.5 deg, oriented generally away from the
    nearest local heavy atom. H-only optimization is expected to refine it.
    """
    O = np.asarray(O)
    axis = norm(O - np.asarray(away_from))
    e1 = orthogonal(axis)

    theta = math.radians(104.5 / 2.0)
    bond = 0.9572
    v1 = math.cos(theta) * axis + math.sin(theta) * e1
    v2 = math.cos(theta) * axis - math.sin(theta) * e1
    return [O + bond * norm(v1), O + bond * norm(v2)]


# ---------------------------------------------------------------------------
# PDB parsing/writing
# ---------------------------------------------------------------------------

def parse_pdb_atoms(path):
    atoms = []
    for line in Path(path).read_text().splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        atoms.append({
            "record": line[0:6].strip(),
            "serial": int(line[6:11]),
            "name": line[12:16].strip(),
            "resname": line[17:20].strip(),
            "chain": line[21:22].strip() or " ",
            "resseq": int(line[22:26]),
            "x": float(line[30:38]),
            "y": float(line[38:46]),
            "z": float(line[46:54]),
            "element": (line[76:78].strip() or line[12:14].strip()).upper(),
        })
    return atoms


def pdb_line(serial, atom):
    name = atom["name"]
    aname = f" {name:<3s}" if len(name) < 4 else name[:4]
    return (
        f'{atom["record"]:<6s}{serial:5d} {aname} '
        f'{atom["resname"]:>3s} {atom["chain"]:1s}{atom["resseq"]:4d}    '
        f'{atom["x"]:8.3f}{atom["y"]:8.3f}{atom["z"]:8.3f}'
        f'  1.00  0.00          {atom["element"]:>2s}'
    )


def write_pdb(path, atoms, remarks=None):
    lines = []
    for r in remarks or []:
        lines.append(f"REMARK 900 {r}")
    for i, atom in enumerate(atoms, 1):
        lines.append(pdb_line(i, atom))
    lines.append("END")
    Path(path).write_text("\n".join(lines) + "\n")


def write_xyz(path, atoms, comment):
    lines = [str(len(atoms)), comment]
    for a in atoms:
        lines.append(
            f'{a["element"]:2s} {a["x"]: .10f} {a["y"]: .10f} {a["z"]: .10f}'
        )
    Path(path).write_text("\n".join(lines) + "\n")


def xyz(atom):
    return np.array([atom["x"], atom["y"], atom["z"]], dtype=float)


def make_h(name, parent, position):
    return {
        "record": "HETATM" if parent["record"] == "HETATM" else "ATOM",
        "serial": 0,
        "name": name,
        "resname": parent["resname"],
        "chain": parent["chain"],
        "resseq": parent["resseq"],
        "x": float(position[0]),
        "y": float(position[1]),
        "z": float(position[2]),
        "element": "H",
    }


# ---------------------------------------------------------------------------
# mmCIF ligand handling
# ---------------------------------------------------------------------------

def clean(x):
    x = str(x).strip()
    return None if x in ("", ".", "?") else x


def asint(x):
    x = clean(x)
    try:
        return int(x) if x is not None else None
    except Exception:
        return None


def parse_cif(path):
    doc = gemmi.cif.read_file(str(path))
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
    ]
    table = block.find(tags)
    if not table:
        raise RuntimeError(f"Could not read atom_site from {path}")

    poly, het = {}, {}
    for row in table:
        group = clean(row[0])
        elem = clean(row[1])
        atom_name = clean(row[2])
        alt = clean(row[3])
        comp = clean(row[4])
        lchain = clean(row[5])
        lseq = asint(row[6])
        achain = clean(row[7])
        aseq = clean(row[8])
        pos = np.array([float(row[9]), float(row[10]), float(row[11])])

        if alt not in (None, "A"):
            continue

        if group == "ATOM" and lchain is not None and lseq is not None:
            key = (lchain, lseq)
            rec = poly.setdefault(
                key, {"comp": comp, "atoms": {}, "elem": {}}
            )
            rec["atoms"].setdefault(atom_name, pos)
            rec["elem"].setdefault(atom_name, elem)

        elif group == "HETATM":
            key = (achain, aseq, comp)
            rec = het.setdefault(
                key,
                {"comp": comp, "auth_chain": achain, "auth_seq": aseq,
                 "atoms": {}, "elem": {}}
            )
            rec["atoms"].setdefault(atom_name, pos)
            rec["elem"].setdefault(atom_name, elem)

    return poly, het


def align_5lf3_to_5lf7(p3, p7, chain="Y"):
    P, Q = [], []
    for key in sorted(set(p3) & set(p7)):
        if key[0] != chain or p3[key]["comp"] != p7[key]["comp"]:
            continue
        for atom in ("N", "CA", "C", "O"):
            if atom in p3[key]["atoms"] and atom in p7[key]["atoms"]:
                P.append(p3[key]["atoms"][atom])
                Q.append(p7[key]["atoms"][atom])

    P, Q = np.array(P), np.array(Q)
    cp, cq = P.mean(0), Q.mean(0)
    U, S, Vt = np.linalg.svd((P - cp).T @ (Q - cq))
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T

    def transform(x):
        return R @ (np.asarray(x) - cp) + cq

    return transform


def choose_ligand(het, comp, poly, chain="Y"):
    og1 = poly[(chain, 1)]["atoms"]["OG1"]
    cands = []
    for rec in het.values():
        if rec["comp"] != comp or "B26" not in rec["atoms"]:
            continue
        cands.append((dist(rec["atoms"]["B26"], og1), rec))
    if not cands:
        raise RuntimeError(f"No active-site {comp} found")
    return sorted(cands, key=lambda x: x[0])[0][1]


def ligand_pdb_atoms(rec, transform=None, resseq=800):
    out = []
    for name, pos in rec["atoms"].items():
        elem = (rec["elem"].get(name) or "").upper()
        if elem == "H":
            continue
        p = transform(pos) if transform is not None else pos
        out.append({
            "record": "HETATM",
            "serial": 0,
            "name": name,
            "resname": rec["comp"],
            "chain": "L",
            "resseq": resseq,
            "x": float(p[0]),
            "y": float(p[1]),
            "z": float(p[2]),
            "element": elem,
        })
    return out


# ---------------------------------------------------------------------------
# Atom selection and H editing
# ---------------------------------------------------------------------------

def select_one(atoms, chain, resseq, name, resname=None):
    hits = [
        a for a in atoms
        if a["chain"] == chain and a["resseq"] == resseq and a["name"] == name
        and (resname is None or a["resname"] == resname)
    ]
    if len(hits) != 1:
        raise RuntimeError(
            f"Expected one atom {chain}:{resseq} {resname or '*'} {name}; "
            f"found {len(hits)}"
        )
    return hits[0]


def attached_h_indices(atoms, parent, cutoff=1.30):
    p = xyz(parent)
    out = []
    for i, a in enumerate(atoms):
        if a["element"] != "H":
            continue
        if a["chain"] != parent["chain"] or a["resseq"] != parent["resseq"]:
            continue
        if dist(xyz(a), p) <= cutoff:
            out.append(i)
    return out


def bonded_heavy_neighbors(atoms, parent, cutoff=1.90):
    p = xyz(parent)
    cand = []
    for a in atoms:
        if a is parent or a["element"] == "H":
            continue
        d = dist(xyz(a), p)
        if d <= cutoff:
            cand.append((d, a))
    cand.sort(key=lambda x: x[0])
    return [a for d, a in cand]


def set_h_count(atoms, parent, target_count, prefix="H", preferred_direction=None):
    """
    Enforce number of H atoms geometrically attached to parent.

    Existing Open Babel H coordinates are kept when possible.
    Missing H atoms receive approximate starting coordinates.
    """
    idx = attached_h_indices(atoms, parent)

    # Remove extras, farthest first.
    if len(idx) > target_count:
        p = xyz(parent)
        ranked = sorted(
            idx, key=lambda i: dist(xyz(atoms[i]), p), reverse=True
        )
        remove = set(ranked[:len(idx) - target_count])
        atoms[:] = [a for i, a in enumerate(atoms) if i not in remove]
        idx = attached_h_indices(atoms, parent)

    if len(idx) == target_count:
        return

    need = target_count - len(idx)
    heavy = bonded_heavy_neighbors(atoms, parent)

    if preferred_direction is not None:
        positions = [
            xyz(parent) + 1.00 * norm(preferred_direction)
        ]
    elif heavy:
        # Generate a full idealized set; add as many missing coordinates as needed.
        positions = h_vectors_one_heavy(
            xyz(parent), xyz(heavy[0]), target_count, bond=1.01
        )
    else:
        axis = np.array([1.0, 0.0, 0.0])
        positions = [xyz(parent) + 1.0 * axis for _ in range(target_count)]

    existing_positions = [xyz(atoms[i]) for i in idx]
    candidates = sorted(
        positions,
        key=lambda p: min(
            [dist(p, q) for q in existing_positions] or [999.0]
        ),
        reverse=True,
    )

    for k in range(need):
        pos = candidates[k % len(candidates)]
        atoms.append(make_h(f"{prefix}{len(idx)+k+1}", parent, pos))


def remove_all_h(atoms, parent):
    idx = set(attached_h_indices(atoms, parent))
    atoms[:] = [a for i, a in enumerate(atoms) if i not in idx]


def nearest_heavy_other(atoms, atom, exclude_same_res=False):
    p = xyz(atom)
    cand = []
    for a in atoms:
        if a is atom or a["element"] == "H":
            continue
        if exclude_same_res and a["chain"] == atom["chain"] and a["resseq"] == atom["resseq"]:
            continue
        cand.append((dist(p, xyz(a)), a))
    cand.sort(key=lambda x: x[0])
    return cand[0][1]


# ---------------------------------------------------------------------------
# Open Babel
# ---------------------------------------------------------------------------

def run_obabel(obabel, input_pdb, output_pdb):
    cmd = [obabel, "-ipdb", str(input_pdb), "-opdb", "-O", str(output_pdb), "-h"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "Open Babel failed:\n"
            + " ".join(cmd) + "\n"
            + proc.stdout + "\n" + proc.stderr
        )


# ---------------------------------------------------------------------------
# Controlled chemistry
# ---------------------------------------------------------------------------

def enforce_ordinary_ionizable_groups(atoms):
    # Arg19 +1: NE-H1; terminal guanidinium nitrogens NH1/NH2 each H2
    for name, count in [("NE", 1), ("NH1", 2), ("NH2", 2)]:
        set_h_count(atoms, select_one(atoms, "Y", 19, name, "ARG"), count, prefix="H")

    # Lys32 +1
    set_h_count(atoms, select_one(atoms, "Y", 32, "NZ", "LYS"), 3, prefix="HZ")

    # Asp167 -1 and Asp125 -1
    for chain, resseq in [("Y", 167), ("Z", 125)]:
        for name in ("OD1", "OD2"):
            set_h_count(atoms, select_one(atoms, chain, resseq, name, "ASP"), 0)

    # Tyr169 neutral phenol
    set_h_count(atoms, select_one(atoms, "Y", 169, "OH", "TYR"), 1, prefix="HH")


def enforce_catalytic_state(atoms, state, ligand_name, oxygen_roles):
    thrN = select_one(atoms, "Y", 1, "N", "THR")
    thrOG = select_one(atoms, "Y", 1, "OG1", "THR")
    lysNZ = select_one(atoms, "Y", 33, "NZ", "LYS")
    aspOD1 = select_one(atoms, "Y", 17, "OD1", "ASP")
    aspOD2 = select_one(atoms, "Y", 17, "OD2", "ASP")

    # Thr1 OG1 is covalently bonded to B26 and has NO proton.
    set_h_count(atoms, thrOG, 0)

    if state["id"] == "MS_A_THRN":
        set_h_count(atoms, thrN, 3, prefix="HN")
        set_h_count(atoms, lysNZ, 2, prefix="HZ")
        set_h_count(atoms, aspOD1, 0)
        set_h_count(atoms, aspOD2, 0)

    elif state["id"] == "MS_B_LYS33":
        set_h_count(atoms, thrN, 2, prefix="HN")
        set_h_count(atoms, lysNZ, 3, prefix="HZ")
        set_h_count(atoms, aspOD1, 0)
        set_h_count(atoms, aspOD2, 0)

    elif state["id"] == "MS_C_ASP17":
        set_h_count(atoms, thrN, 2, prefix="HN")
        set_h_count(atoms, lysNZ, 2, prefix="HZ")

        # Protonate the Asp oxygen closest to Lys33 NZ.
        d1 = dist(xyz(aspOD1), xyz(lysNZ))
        d2 = dist(xyz(aspOD2), xyz(lysNZ))
        protonated = aspOD1 if d1 <= d2 else aspOD2
        deprotonated = aspOD2 if protonated is aspOD1 else aspOD1
        set_h_count(atoms, deprotonated, 0)

        # Initial O-H points toward neutral Lys33 as a plausible H-bond acceptor.
        remove_all_h(atoms, protonated)
        direction = xyz(lysNZ) - xyz(protonated)
        atoms.append(
            make_h(
                "HD",
                protonated,
                xyz(protonated) + 0.98 * norm(direction),
            )
        )
    else:
        raise RuntimeError(f"Unknown microstate {state['id']}")

    # Both B-bound ligand O atoms carry one H in the first-screen registry.
    B = select_one(atoms, "L", 800, "B26", ligand_name)

    role = oxygen_roles[ligand_name]
    for role_name, donor_target in [
        ("O_NTERM", select_one(atoms, "Y", 1, "N", "THR")),
        ("O_OXY", select_one(atoms, "Y", 47, "N", "GLY")),
    ]:
        oname = role[role_name]["atom"]
        O = select_one(atoms, "L", 800, oname, ligand_name)
        remove_all_h(atoms, O)

        # Place OH initially away from B and away from the nearby donor N,
        # leaving the oxygen lone-pair side exposed toward that N-H donor.
        direction = norm(xyz(O) - xyz(B)) + norm(xyz(O) - xyz(donor_target))
        if np.linalg.norm(direction) < 1e-6:
            direction = xyz(O) - xyz(B)
        Hpos = xyz(O) + 0.98 * norm(direction)
        atoms.append(make_h(f"H{oname}", O, Hpos))


def enforce_waters(atoms, manifest):
    for w in manifest["retained_waters"]:
        pdb_resseq = int(w["pdb_resseq"])
        chain = w["source_water"].split(":")[0]
        O = select_one(atoms, chain, pdb_resseq, "O", "HOH")
        remove_all_h(atoms, O)
        nearest = nearest_heavy_other(atoms, O, exclude_same_res=True)
        Hpos = water_h_positions(xyz(O), xyz(nearest))
        atoms.append(make_h("H1", O, Hpos[0]))
        atoms.append(make_h("H2", O, Hpos[1]))


def audit_special_counts(atoms, state, ligand_name, roles, manifest):
    checks = {}

    targets = [
        ("Thr1_N", select_one(atoms, "Y", 1, "N", "THR")),
        ("Thr1_OG1", select_one(atoms, "Y", 1, "OG1", "THR")),
        ("Lys33_NZ", select_one(atoms, "Y", 33, "NZ", "LYS")),
        ("Asp17_OD1", select_one(atoms, "Y", 17, "OD1", "ASP")),
        ("Asp17_OD2", select_one(atoms, "Y", 17, "OD2", "ASP")),
    ]
    for label, atom in targets:
        checks[label] = len(attached_h_indices(atoms, atom))

    for role_name in ("O_NTERM", "O_OXY"):
        oname = roles[ligand_name][role_name]["atom"]
        O = select_one(atoms, "L", 800, oname, ligand_name)
        checks[f"{ligand_name}_{role_name}_{oname}"] = len(attached_h_indices(atoms, O))

    for w in manifest["retained_waters"]:
        chain = w["source_water"].split(":")[0]
        O = select_one(atoms, chain, int(w["pdb_resseq"]), "O", "HOH")
        checks[f"water_{w['source_water']}"] = len(attached_h_indices(atoms, O))

    return checks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cif3")
    ap.add_argument("cif7")
    ap.add_argument("--scaffold", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--microstates", required=True)
    ap.add_argument("--boronate", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--obabel", default="obabel")
    a = ap.parse_args()

    obabel = shutil.which(a.obabel)
    if obabel is None:
        raise RuntimeError(
            "Open Babel executable 'obabel' was not found on PATH. "
            "Install Open Babel first (for example with Homebrew: brew install open-babel)."
        )

    manifest = json.loads(Path(a.manifest).read_text())
    registry = json.loads(Path(a.microstates).read_text())
    boronate = json.loads(Path(a.boronate).read_text())

    if registry["comparison_invariants"]["same_total_charge"] != -1:
        raise RuntimeError("Expected first-screen total charge -1")

    p3, h3 = parse_cif(a.cif3)
    p7, h7 = parse_cif(a.cif7)
    transform = align_5lf3_to_5lf7(p3, p7, "Y")

    ligands = {
        "6V8": ligand_pdb_atoms(choose_ligand(h7, "6V8", p7, "Y")),
        "BO2": ligand_pdb_atoms(
            choose_ligand(h3, "BO2", p3, "Y"),
            transform=transform,
        ),
    }

    receptor_heavy = parse_pdb_atoms(a.scaffold)
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    Path(a.report).parent.mkdir(parents=True, exist_ok=True)

    results = []

    for ligand_name, ligand_heavy in ligands.items():
        for state in registry["first_screen_microstates"]:
            state_id = state["id"]
            stem = f"{ligand_name}_{state_id}"

            heavy_combined = [dict(x) for x in receptor_heavy] + [dict(x) for x in ligand_heavy]

            with tempfile.TemporaryDirectory() as td:
                td = Path(td)
                heavy_pdb = td / f"{stem}_heavy.pdb"
                obabel_pdb = td / f"{stem}_obabelH.pdb"

                write_pdb(
                    heavy_pdb,
                    heavy_combined,
                    remarks=[
                        f"{ligand_name} {state_id} heavy-atom starting complex",
                        "Thr1 OG1--B26 covalent geometry inherited from experiment",
                    ],
                )
                run_obabel(obabel, heavy_pdb, obabel_pdb)
                atoms = parse_pdb_atoms(obabel_pdb)

            # Enforce controlled chemistry.
            enforce_ordinary_ionizable_groups(atoms)
            enforce_catalytic_state(
                atoms,
                state,
                ligand_name,
                registry["boronate_oxygen_functional_roles"],
            )
            enforce_waters(atoms, manifest)

            # Stable ordering: preserve Open Babel order, appended explicit H afterward.
            pdb_out = outdir / f"{stem}.pdb"
            xyz_out = outdir / f"{stem}.xyz"

            write_pdb(
                pdb_out,
                atoms,
                remarks=[
                    f"{ligand_name} {state_id}",
                    "INITIAL HYDROGENATED MICROSTATE; heavy atoms fixed from reference",
                    "Hydrogen-only optimization required before energetic comparison",
                    "Expected total charge = -1",
                ],
            )
            write_xyz(
                xyz_out,
                atoms,
                f"{ligand_name} {state_id}; charge=-1; initial H geometry",
            )

            nH = sum(x["element"] == "H" for x in atoms)
            nheavy = len(atoms) - nH
            special = audit_special_counts(
                atoms,
                state,
                ligand_name,
                registry["boronate_oxygen_functional_roles"],
                manifest,
            )

            results.append({
                "ligand": ligand_name,
                "microstate": state_id,
                "total_charge": -1,
                "n_atoms": len(atoms),
                "n_heavy": nheavy,
                "n_H": nH,
                "pdb": str(pdb_out),
                "xyz": str(xyz_out),
                "special_H_counts": special,
            })

    # Enforce identical composition across microstates of a given ligand.
    for ligand_name in ligands:
        subset = [r for r in results if r["ligand"] == ligand_name]
        atom_counts = {r["n_atoms"] for r in subset}
        h_counts = {r["n_H"] for r in subset}
        heavy_counts = {r["n_heavy"] for r in subset}
        if len(atom_counts) != 1 or len(h_counts) != 1 or len(heavy_counts) != 1:
            raise RuntimeError(
                f"{ligand_name} microstates do not have identical composition: "
                f"atoms={atom_counts}, H={h_counts}, heavy={heavy_counts}"
            )

    report = {
        "stage": "8e_initial_hydrogenated_microstates",
        "openbabel": obabel,
        "method": [
            "merge capped receptor heavy scaffold with experimental ligand heavy atoms",
            "Open Babel adds ordinary valence hydrogens",
            "explicit script overrides catalytic/ionizable/water hydrogen counts",
            "all heavy atoms remain fixed",
        ],
        "controlled_sites": [
            "Thr1 N",
            "Thr1 OG1",
            "Asp17 OD1/OD2",
            "Arg19",
            "Lys32",
            "Lys33",
            "Asp167",
            "Tyr169",
            "Asp125",
            "both B-bound ligand oxygens",
            "three retained waters",
        ],
        "results": results,
        "validation": {
            "same_composition_across_microstates_per_ligand": True,
            "same_total_charge_across_all_first_screen_states": -1,
        },
        "important_warning":
            "These are starting hydrogen coordinates only. Do not compare quantum energies before performing the same hydrogen-only optimization protocol on every structure.",
        "next_step":
            "Run geometry sanity checks, then perform hydrogen-only constrained optimization with all heavy atoms frozen.",
    }
    Path(a.report).write_text(json.dumps(report, indent=2) + "\n")

    print(f"Wrote {a.report}")
    print()
    print("Stage-8e explicit hydrogenated microstates")
    for r in results:
        print(
            f"  {r['ligand']:3s} {r['microstate']:12s} "
            f"atoms={r['n_atoms']:3d} heavy={r['n_heavy']:3d} "
            f"H={r['n_H']:3d} q={r['total_charge']:+d}"
        )
        print(f"      {r['pdb']}")
        print(f"      {r['xyz']}")

    print()
    print("Composition checks passed within each ligand.")
    print("IMPORTANT: H positions are initial guesses; H-only optimization is next.")


if __name__ == "__main__":
    main()
