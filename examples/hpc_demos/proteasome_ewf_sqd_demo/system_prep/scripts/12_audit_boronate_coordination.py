#!/usr/bin/env python3
"""
12_audit_boronate_coordination.py

Proteasome Challenge
Stage 8c: inspect the local B26 coordination sphere for both inhibitors.

Diagnostic only:
  - no hydrogen atoms added
  - no protonation state assigned
  - no coordinates modified

For native 6V8 in canonical 5LF7-Y and BO2 transformed from 5LF3-Y into the
same 5LF7-Y frame, report:
  * all ligand heavy atoms within a bonding-scale cutoff of B26
  * Thr1 OG1--B26 distance
  * all four-neighbor B-centered angles when available
  * for ligand O atoms bonded to B26, distances to:
      Thr1 N
      Lys33 NZ
      Gly47 N
      retained water oxygens
  * closest protein polar atom to each B-bound oxygen
"""

import argparse, csv, json, math
from pathlib import Path
import gemmi
import numpy as np

POLAR = {"N", "O", "S"}


def clean(x):
    x = str(x).strip()
    return None if x in ("", ".", "?") else x


def asint(x):
    x = clean(x)
    try:
        return int(x) if x is not None else None
    except Exception:
        return None


def parse(path):
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
    t = block.find(tags)
    if not t:
        raise RuntimeError(f"Could not read atom_site from {path}")

    poly, het = {}, {}
    for r in t:
        g = clean(r[0])
        elem = clean(r[1])
        atom = clean(r[2])
        alt = clean(r[3])
        comp = clean(r[4])
        lchain = clean(r[5])
        lseq = asint(r[6])
        achain = clean(r[7])
        aseq = clean(r[8])
        xyz = np.array([float(r[9]), float(r[10]), float(r[11])])

        if alt not in (None, "A"):
            continue

        if g == "ATOM" and lchain is not None and lseq is not None:
            key = (lchain, lseq)
            rec = poly.setdefault(
                key,
                {"comp": comp, "auth": aseq, "atoms": {}, "elem": {}}
            )
            rec["atoms"].setdefault(atom, xyz)
            rec["elem"].setdefault(atom, elem)

        elif g == "HETATM":
            key = (achain, aseq, comp)
            rec = het.setdefault(
                key,
                {"comp": comp, "auth_chain": achain, "auth_seq": aseq,
                 "atoms": {}, "elem": {}}
            )
            rec["atoms"].setdefault(atom, xyz)
            rec["elem"].setdefault(atom, elem)

    return poly, het


def heavy_atom_items(rec):
    return [
        (name, rec["elem"].get(name), xyz)
        for name, xyz in rec["atoms"].items()
        if (rec["elem"].get(name) or "").upper() != "H"
    ]


def align(p3, p7, chain):
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
    U, S, Vt = np.linalg.svd((P-cp).T @ (Q-cq))
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T

    def transform(x):
        return R @ (np.asarray(x) - cp) + cq

    fit = np.vstack([transform(x) for x in P])
    rms = float(np.sqrt(np.mean(np.sum((fit-Q)**2, axis=1))))
    return transform, rms, len(P)


def choose_ligand(het, comp, poly, chain):
    thr1 = poly[(chain, 1)]
    og1 = thr1["atoms"]["OG1"]
    cands = []
    for rec in het.values():
        if rec["comp"] != comp or "B26" not in rec["atoms"]:
            continue
        d = float(np.linalg.norm(rec["atoms"]["B26"] - og1))
        cands.append((d, rec))
    if not cands:
        raise RuntimeError(f"No active-site {comp} with B26 found")
    return sorted(cands, key=lambda x: x[0])[0][1]


def angle(a, center, b):
    v1 = np.asarray(a) - np.asarray(center)
    v2 = np.asarray(b) - np.asarray(center)
    c = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    c = max(-1.0, min(1.0, float(c)))
    return math.degrees(math.acos(c))


def locate_waters(het7, manifest):
    out = {}
    for w in manifest["retained_waters"]:
        chain, seq = w["source_water"].split(":")
        rec = next(
            rec for (c, s, comp), rec in het7.items()
            if c == chain and str(s) == seq
        )
        oxy = [
            xyz for name, elem, xyz in heavy_atom_items(rec)
            if (elem or "").upper() == "O"
        ]
        if not oxy:
            raise RuntimeError(f"No O atom found for retained water {chain}:{seq}")
        out[f"{chain}:{seq}"] = oxy[0]
    return out


def transform_ligand(rec, transform):
    return {
        "comp": rec["comp"],
        "atoms": {k: transform(v) for k, v in rec["atoms"].items()},
        "elem": dict(rec["elem"]),
    }


def closest_retained_polar(poly7, kept, point):
    hits = []
    for key in kept:
        rec = poly7[key]
        for atom, xyz in rec["atoms"].items():
            elem = (rec["elem"].get(atom) or "").upper()
            if elem not in POLAR:
                continue
            d = float(np.linalg.norm(xyz - point))
            hits.append((d, key, rec["comp"], atom, elem))
    hits.sort(key=lambda x: x[0])
    return hits


def audit_one(name, lig, poly7, kept, waters, chain, bond_cutoff):
    B = lig["atoms"]["B26"]

    neighbors = []
    for atom, elem, xyz in heavy_atom_items(lig):
        if atom == "B26":
            continue
        d = float(np.linalg.norm(xyz - B))
        if d <= bond_cutoff:
            neighbors.append({
                "source": "ligand",
                "atom": atom,
                "element": elem,
                "distance_A": d,
                "xyz": xyz,
            })

    thr1 = poly7[(chain, 1)]
    thr_d = float(np.linalg.norm(thr1["atoms"]["OG1"] - B))
    if thr_d <= bond_cutoff:
        neighbors.append({
            "source": "protein",
            "atom": "Thr1_OG1",
            "element": "O",
            "distance_A": thr_d,
            "xyz": thr1["atoms"]["OG1"],
        })

    neighbors.sort(key=lambda x: x["distance_A"])

    angles = []
    for i in range(len(neighbors)):
        for j in range(i + 1, len(neighbors)):
            angles.append({
                "atom1": neighbors[i]["atom"],
                "atom2": neighbors[j]["atom"],
                "angle_deg": angle(neighbors[i]["xyz"], B, neighbors[j]["xyz"]),
            })

    targets = {
        "Thr1_N": poly7[(chain, 1)]["atoms"].get("N"),
        "Lys33_NZ": poly7[(chain, 33)]["atoms"].get("NZ"),
        "Gly47_N": poly7[(chain, 47)]["atoms"].get("N"),
    }

    Orows = []
    for n in neighbors:
        if n["source"] != "ligand" or (n["element"] or "").upper() != "O":
            continue

        point = n["xyz"]
        row = {
            "inhibitor": name,
            "oxygen_atom": n["atom"],
            "B26_O_A": n["distance_A"],
        }

        for label, xyz in targets.items():
            row[f"{label}_A"] = None if xyz is None else float(np.linalg.norm(point - xyz))

        for wid, xyz in waters.items():
            row[f"water_{wid}_A"] = float(np.linalg.norm(point - xyz))

        hits = closest_retained_polar(poly7, kept, point)
        row["closest_retained_polar"] = [
            {
                "residue": f"{key[0]}:{key[1]} {comp}",
                "atom": atom,
                "distance_A": d,
            }
            for d, key, comp, atom, elem in hits[:8]
        ]
        Orows.append(row)

    return {
        "B26_neighbors": [
            {k: (round(v, 4) if k == "distance_A" else v)
             for k, v in n.items() if k != "xyz"}
            for n in neighbors
        ],
        "B_centered_angles": [
            {"atom1": a["atom1"], "atom2": a["atom2"],
             "angle_deg": round(a["angle_deg"], 2)}
            for a in angles
        ],
        "boronate_oxygen_environment": Orows,
        "Thr1_OG1_B26_A": thr_d,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cif3")
    ap.add_argument("cif7")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tsv", required=True)
    ap.add_argument("--chain", default="Y")
    ap.add_argument("--bond-cutoff", type=float, default=1.90)
    a = ap.parse_args()

    man = json.loads(Path(a.manifest).read_text())
    p3, h3 = parse(a.cif3)
    p7, h7 = parse(a.cif7)

    transform, rms, nfit = align(p3, p7, a.chain)

    bo2_native = choose_ligand(h3, "BO2", p3, a.chain)
    v8 = choose_ligand(h7, "6V8", p7, a.chain)
    bo2 = transform_ligand(bo2_native, transform)

    kept = [
        (r["label_chain"], int(r["label_seq"]))
        for r in man["retained_residues"]
    ]
    waters = locate_waters(h7, man)

    audits = {
        "6V8": audit_one("6V8", v8, p7, kept, waters, a.chain, a.bond_cutoff),
        "BO2": audit_one("BO2", bo2, p7, kept, waters, a.chain, a.bond_cutoff),
    }

    report = {
        "stage": "8c_boronate_coordination_audit",
        "alignment_rmsd_A": round(rms, 4),
        "alignment_atoms": nfit,
        "bonding_scale_cutoff_A": a.bond_cutoff,
        "retained_waters": list(waters.keys()),
        "inhibitors": audits,
        "interpretation_policy": [
            "Heavy-atom geometry can define likely coordination and H-bond geometry, but not proton locations.",
            "Do not infer O-H protonation solely from B-O or donor-acceptor distances.",
            "Use the same atom/proton count when comparing alternative proton-transfer microstates.",
        ],
        "next_step":
            "Define a small set of same-composition Thr1/Lys33/boronate proton-transfer microstates and place hydrogens explicitly.",
    }
    Path(a.out).write_text(json.dumps(report, indent=2, default=lambda x: float(x)) + "\n")

    rows = []
    for inhibitor, aud in audits.items():
        for r in aud["boronate_oxygen_environment"]:
            flat = {
                "inhibitor": inhibitor,
                "oxygen_atom": r["oxygen_atom"],
                "B26_O_A": r["B26_O_A"],
                "Thr1_N_A": r.get("Thr1_N_A"),
                "Lys33_NZ_A": r.get("Lys33_NZ_A"),
                "Gly47_N_A": r.get("Gly47_N_A"),
            }
            for wid in waters:
                flat[f"water_{wid}_A"] = r.get(f"water_{wid}_A")
            flat["closest_retained_polar"] = ";".join(
                f'{x["residue"]}:{x["atom"]}@{x["distance_A"]:.3f}'
                for x in r["closest_retained_polar"][:5]
            )
            rows.append(flat)

    fields = [
        "inhibitor", "oxygen_atom", "B26_O_A",
        "Thr1_N_A", "Lys33_NZ_A", "Gly47_N_A",
    ] + [f"water_{wid}_A" for wid in waters] + ["closest_retained_polar"]

    with open(a.tsv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {a.out}")
    print(f"Wrote {a.tsv}")
    print()
    print("Stage-8c boronate coordination audit")
    print(f"  alignment RMSD: {rms:.3f} A over {nfit} backbone atoms")
    print()

    for inhibitor, aud in audits.items():
        print(f"{inhibitor}")
        print(f"  Thr1 OG1--B26: {aud['Thr1_OG1_B26_A']:.3f} A")
        print("  B26 bonding-scale neighbors:")
        for n in aud["B26_neighbors"]:
            print(
                f"    {n['source']:7s} {n['atom']:10s} "
                f"{n['element']:>2s}  {n['distance_A']:.3f} A"
            )
        print("  B-bound ligand oxygen environment:")
        for r in aud["boronate_oxygen_environment"]:
            vals = (
                f"Thr1N={r['Thr1_N_A']:.3f} "
                f"Lys33NZ={r['Lys33_NZ_A']:.3f} "
                f"Gly47N={r['Gly47_N_A']:.3f}"
            )
            print(f"    {r['oxygen_atom']:6s} B-O={r['B26_O_A']:.3f}  {vals}")
        print("  B-centered angles:")
        for x in aud["B_centered_angles"]:
            print(
                f"    {x['atom1']}--B26--{x['atom2']}: "
                f"{x['angle_deg']:.1f} deg"
            )
        print()

    print("IMPORTANT: this still does not assign any proton.")
    print("The next decision is a same-composition proton-transfer microstate set.")


if __name__ == "__main__":
    main()
