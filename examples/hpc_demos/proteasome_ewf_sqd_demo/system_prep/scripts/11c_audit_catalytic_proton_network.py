#!/usr/bin/env python3
"""Stage 8b-rev2: audit the catalytic proton network before adding H atoms.

Diagnostic only:
- no hydrogens added
- no protonation states assigned
- no water orientations changed
- no coordinates modified

Key correction relative to 11b:
- Thr1 and Lys33 are BOTH treated as SPECIAL catalytic sites.
- The reported ordinary fixed-charge sum excludes Thr1, Lys33, and ligand.
- Explicit Thr1/Asp17/Lys33 heavy-atom distances are reported.

Example:
python3 scripts/11c_audit_catalytic_proton_network.py \
  raw/5LF3.cif raw/5LF7.cif \
  --manifest validation/capped_receptor_scaffold_v2.json \
  --out validation/protonation_audit_v3.json \
  --tsv validation/protonation_audit_v3.tsv
"""

import argparse, csv, json
from pathlib import Path
import gemmi, numpy as np

ION = {
    "ASP": ("deprotonated", -1),
    "GLU": ("deprotonated", -1),
    "LYS": ("protonated", +1),
    "ARG": ("protonated", +1),
    "HIS": ("manual_review", None),
    "CYS": ("neutral_unless_evidence", 0),
    "TYR": ("neutral_unless_evidence", 0),
}
FOCUS = {
    "ASP": ["OD1", "OD2"],
    "GLU": ["OE1", "OE2"],
    "LYS": ["NZ"],
    "ARG": ["NE", "NH1", "NH2"],
    "HIS": ["ND1", "NE2"],
    "CYS": ["SG"],
    "TYR": ["OH"],
}
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
        g, e, atom, alt, comp, lchain, lseq, achain, aseq = [
            clean(r[i]) for i in range(9)
        ]
        if alt not in (None, "A"):
            continue
        xyz = np.array([float(r[9]), float(r[10]), float(r[11])])

        if g == "ATOM" and lchain is not None and asint(lseq) is not None:
            key = (lchain, asint(lseq))
            rec = poly.setdefault(
                key, {"comp": comp, "auth": aseq, "atoms": {}, "elem": {}}
            )
            rec["atoms"].setdefault(atom, xyz)
            rec["elem"].setdefault(atom, e)

        elif g == "HETATM":
            key = (achain, aseq, comp)
            rec = het.setdefault(key, {"comp": comp, "atoms": {}, "elem": {}})
            rec["atoms"].setdefault(atom, xyz)
            rec["elem"].setdefault(atom, e)

    return block, poly, het


def heavy(rec):
    return [
        xyz
        for name, xyz in rec["atoms"].items()
        if (rec["elem"].get(name) or "").upper() != "H"
    ]


def md(a, b):
    if not a or not b:
        return None
    A, B = np.vstack(a), np.vstack(b)
    d2 = ((A[:, None, :] - B[None, :, :]) ** 2).sum(2)
    return float(np.sqrt(d2.min()))


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
    U, S, Vt = np.linalg.svd((P - cp).T @ (Q - cq))
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T

    def transform(x):
        return R @ (np.asarray(x) - cp) + cq

    fit = np.vstack([transform(x) for x in P])
    rms = float(np.sqrt(np.mean(np.sum((fit - Q) ** 2, axis=1))))
    return transform, rms, len(P)


def choose_ligand(het, comp, poly, chain):
    og = poly[(chain, 1)]["atoms"]["OG1"]
    cands = []
    for rec in het.values():
        if rec["comp"] != comp:
            continue
        d = (
            np.linalg.norm(rec["atoms"]["B26"] - og)
            if "B26" in rec["atoms"]
            else md(heavy(rec), [og])
        )
        cands.append((d, rec))
    if not cands:
        raise RuntimeError(f"No {comp} found")
    return sorted(cands, key=lambda x: x[0])[0][1]


def comp_charge(block, comp):
    t = block.find(
        ["_chem_comp_atom.comp_id", "_chem_comp_atom.atom_id", "_chem_comp_atom.charge"]
    )
    if not t:
        return None
    vals = []
    for r in t:
        if clean(r[0]) != comp:
            continue
        q = clean(r[2])
        try:
            vals.append(int(q))
        except Exception:
            return None
    return sum(vals) if vals else None


def adist(rec1, atom1, rec2, atom2):
    if atom1 not in rec1["atoms"] or atom2 not in rec2["atoms"]:
        return None
    return float(np.linalg.norm(rec1["atoms"][atom1] - rec2["atoms"][atom2]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cif3")
    ap.add_argument("cif7")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tsv", required=True)
    ap.add_argument("--chain", default="Y")
    ap.add_argument("--cutoff", type=float, default=4.0)
    a = ap.parse_args()

    man = json.loads(Path(a.manifest).read_text())
    b3, p3, h3 = parse(a.cif3)
    b7, p7, h7 = parse(a.cif7)

    transform, rms, nfit = align(p3, p7, a.chain)
    bo2 = choose_ligand(h3, "BO2", p3, a.chain)
    v8 = choose_ligand(h7, "6V8", p7, a.chain)
    Lb = [transform(x) for x in heavy(bo2)]
    Li = heavy(v8)

    kept = [
        (r["label_chain"], int(r["label_seq"]))
        for r in man["retained_residues"]
    ]

    waters = {}
    for w in man["retained_waters"]:
        c, s = w["source_water"].split(":")
        rec = next(
            v for (cc, ss, cp), v in h7.items()
            if cc == c and str(ss) == s
        )
        waters[w["source_water"]] = heavy(rec)

    rows = []

    for key in kept:
        rec = p7[key]
        special_thr1 = (key == (a.chain, 1))
        special_lys33 = (key == (a.chain, 33) and rec["comp"] == "LYS")

        if not (special_thr1 or special_lys33) and rec["comp"] not in ION:
            continue

        if special_thr1:
            names = ["N", "OG1"]
            state, q = "SPECIAL_Thr1_boronate_coupled", None
        elif special_lys33:
            names = ["NZ"]
            state, q = "SPECIAL_Lys33_catalytic_proton_network", None
        else:
            names = FOCUS[rec["comp"]]
            state, q = ION[rec["comp"]]

        fc = [rec["atoms"][n] for n in names if n in rec["atoms"]]

        polar = []
        for other_key in kept:
            if other_key == key:
                continue
            rr = p7[other_key]
            for atom_name, xyz in rr["atoms"].items():
                if (rr["elem"].get(atom_name) or "").upper() in POLAR:
                    d = min(np.linalg.norm(xyz - y) for y in fc)
                    if d <= a.cutoff:
                        polar.append(
                            (d, f"{other_key[0]}:{other_key[1]} {rr['comp']} {atom_name}")
                        )
        polar.sort()

        wd = {wid: md(fc, wc) for wid, wc in waters.items()}

        rows.append({
            "residue": f"{key[0]}:{key[1]} {rec['comp']}",
            "special_thr1": special_thr1,
            "special_lys33": special_lys33,
            "hypothesis_only": state,
            "default_charge": q,
            "focus_atoms": ",".join(names),
            "to_6V8_A": md(fc, Li),
            "to_BO2_A": md(fc, Lb),
            "nearest_water_A": min(wd.values()),
            "nearby_polar": "; ".join(
                f"{partner}@{dist:.3f}" for dist, partner in polar[:8]
            ),
        })

    ordinary = sum(
        r["default_charge"]
        for r in rows
        if r["default_charge"] is not None
    )

    thr1 = p7[(a.chain, 1)]
    asp17 = p7[(a.chain, 17)]
    lys33 = p7[(a.chain, 33)]

    catalytic_geometry = {
        "Lys33_NZ_to_Thr1_OG1_A": adist(lys33, "NZ", thr1, "OG1"),
        "Lys33_NZ_to_Thr1_N_A": adist(lys33, "NZ", thr1, "N"),
        "Lys33_NZ_to_Asp17_OD1_A": adist(lys33, "NZ", asp17, "OD1"),
        "Lys33_NZ_to_Asp17_OD2_A": adist(lys33, "NZ", asp17, "OD2"),
        "Thr1_N_to_Asp17_OD1_A": adist(thr1, "N", asp17, "OD1"),
        "Thr1_N_to_Asp17_OD2_A": adist(thr1, "N", asp17, "OD2"),
    }

    report = {
        "stage": "8b_rev2_protonation_audit",
        "alignment_rmsd_A": round(rms, 4),
        "alignment_atoms": nfit,
        "ordinary_fixed_charge_sum_excluding_Thr1_Lys33_and_ligand": ordinary,
        "ionizable_groups": rows,
        "catalytic_proton_network_geometry": catalytic_geometry,
        "caps": {"ACE": "neutral", "NME": "neutral"},
        "ligand_dictionary_charge": {
            "BO2": comp_charge(b3, "BO2"),
            "6V8": comp_charge(b7, "6V8"),
        },
        "warnings": [
            "Dictionary ligand charge is not automatically the covalent-adduct charge.",
            "Thr1, Lys33, Asp17, and the boronate adduct form a coupled proton-location problem.",
            "Lys33 is deliberately excluded from the generic +1 lysine default.",
            "No protonation state or hydrogen position is assigned by this script.",
        ],
        "next_step": (
            "Define same-composition proton-transfer microstates for the "
            "Thr1/Asp17/Lys33/boronate network, then place hydrogens and compare them."
        ),
    }

    Path(a.out).write_text(json.dumps(report, indent=2) + "\n")

    with open(a.tsv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=rows[0].keys(), delimiter="\t")
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {a.out}")
    print(f"Wrote {a.tsv}")
    print()
    print("Stage-8b-rev2 catalytic proton-network audit")
    print(f"  alignment RMSD: {rms:.3f} A over {nfit} backbone atoms")
    print(
        "  ordinary fixed-charge sum "
        f"(excluding Thr1 + Lys33 + ligand): {ordinary:+d}"
    )
    print()

    for r in rows:
        q = "SPECIAL" if r["default_charge"] is None else f"{r['default_charge']:+d}"
        print(
            f"  {r['residue']:12s} q={q:7s} "
            f"6V8={r['to_6V8_A']:.3f} "
            f"BO2={r['to_BO2_A']:.3f} "
            f"water={r['nearest_water_A']:.3f}"
        )

    print()
    print("Catalytic proton-network heavy-atom geometry")
    for key, value in catalytic_geometry.items():
        txt = "n/a" if value is None else f"{value:.3f} A"
        print(f"  {key}: {txt}")

    print()
    print("Ligand dictionary charge (metadata only; not final adduct charge)")
    for key, value in report["ligand_dictionary_charge"].items():
        print(f"  {key}: {value}")

    print()
    print("IMPORTANT: no final protonation state has been assigned.")
    print(
        "Thr1 + Lys33 + Asp17 + boronate proton placement "
        "is the next explicit modeling decision."
    )


if __name__ == "__main__":
    main()
