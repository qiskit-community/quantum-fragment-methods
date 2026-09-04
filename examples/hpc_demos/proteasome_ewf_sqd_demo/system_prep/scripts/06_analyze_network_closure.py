"""
06_analyze_network_closure.py

Proteasome Challenge: chemistry-first active-site network closure.

This stage does NOT cut coordinates, add caps, assign protonation,
add hydrogens, or optimize anything.

It constructs two bookkeeping models from the Stage-6b chemistry report:

1) functional_network_core
   - catalytic residue
   - strong initial direct-contact residues
   - protein residues supporting the high-confidence conserved water bridges

2) peptide_coherent_reference
   - functional-network core
   - one polymer residue of context on either side
   - fills small sequence gaps so nearby core residues are represented as
     chemically coherent peptide spans

The second model is deliberately generous. It is a reference topology for
subsequent pruning/convergence, not the final quantum cluster.

Usage:
  python3 scripts/06_analyze_network_closure.py \
    raw/5LF7.cif \
    --chemistry validation/active_site_chemistry_report.json \
    --out validation/network_closure_report.json \
    --residues validation/network_closure_residues.tsv \
    --segments validation/network_closure_segments.tsv
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import gemmi


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("canonical_cif", help="Canonical receptor mmCIF (5LF7)")
    p.add_argument("--chemistry", required=True,
                   help="Stage-6b active_site_chemistry_report.json")
    p.add_argument("--out", required=True,
                   help="Output JSON report")
    p.add_argument("--residues", required=True,
                   help="Output residue TSV")
    p.add_argument("--segments", required=True,
                   help="Output peptide-segment TSV")
    p.add_argument("--padding", type=int, default=1,
                   help="Sequence-neighbor padding for peptide reference (default 1)")
    p.add_argument("--max-gap-fill", type=int, default=1,
                   help="Fill at most this many missing sequence residues between core residues")
    return p.parse_args()


def clean(v):
    if v is None:
        return None
    v = str(v).strip()
    if v in ("", ".", "?"):
        return None
    return v


def int_or_none(v):
    v = clean(v)
    if v is None:
        return None
    try:
        return int(v)
    except ValueError:
        return None


def load_atom_site(cif_path):
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
        "_atom_site.occupancy",
    ]
    table = block.find(tags)
    if not table:
        raise RuntimeError("Could not find required _atom_site columns")

    residues = {}
    auth_lookup = {}

    for row in table:
        group_pdb = clean(row[0])
        elem = clean(row[1])
        atom_name = clean(row[2])
        alt = clean(row[3])
        comp = clean(row[4])
        label_chain = clean(row[5])
        label_seq = int_or_none(row[6])
        auth_chain = clean(row[7])
        auth_seq_raw = clean(row[8])
        occ = clean(row[9])

        # Polymer residues only for residue map.
        if group_pdb != "ATOM" or label_seq is None:
            continue

        key = (label_chain, label_seq)
        if key not in residues:
            residues[key] = {
                "label_chain": label_chain,
                "label_seq": label_seq,
                "auth_chain": auth_chain,
                "auth_seq": auth_seq_raw,
                "comp_id": comp,
                "atom_names": set(),
                "heavy_atom_names": set(),
            }

        # Avoid double-counting alternate conformers by unique atom name.
        if atom_name:
            residues[key]["atom_names"].add(atom_name)
            if elem and elem.upper() != "H":
                residues[key]["heavy_atom_names"].add(atom_name)

        if auth_chain and auth_seq_raw:
            auth_lookup[(auth_chain, auth_seq_raw)] = key

    for r in residues.values():
        r["atom_count_deposited"] = len(r.pop("atom_names"))
        r["heavy_atoms_deposited"] = len(r.pop("heavy_atom_names"))

    return residues, auth_lookup


def keystr(key):
    return f"{key[0]}:{key[1]}"


def merge_ranges(values):
    if not values:
        return []
    vals = sorted(set(values))
    out = []
    start = prev = vals[0]
    for x in vals[1:]:
        if x == prev + 1:
            prev = x
        else:
            out.append((start, prev))
            start = prev = x
    out.append((start, prev))
    return out


def main():
    args = parse_args()
    chemistry = json.loads(Path(args.chemistry).read_text())
    residue_map, auth_lookup = load_atom_site(Path(args.canonical_cif))

    evidence = defaultdict(lambda: {
        "catalytic": False,
        "direct_strong": False,
        "direct_signal": None,
        "water_support": [],
    })

    # ------------------------------------------------------------------
    # Direct chemical seeds
    # ------------------------------------------------------------------
    for r in chemistry["residue_summaries"]:
        signal = r["initial_retention_signal"]
        if signal not in ("mandatory_catalytic", "strong_initial_retain"):
            continue
        key = (r["label_chain"], int(r["label_seq"]))
        evidence[key]["catalytic"] = bool(r.get("catalytic_thr"))
        evidence[key]["direct_strong"] = True
        evidence[key]["direct_signal"] = signal

    # ------------------------------------------------------------------
    # High-confidence conserved water-bridge supports
    # ------------------------------------------------------------------
    retained_waters = []
    for w in chemistry["water_candidates"]:
        if w.get("priority") != "high_conserved_bridge_both":
            continue

        water_id = f'{w["5LF7_auth_chain"]}:{w["5LF7_auth_seq"]}'
        retained_waters.append({
            "water": water_id,
            "auth_chain": w["5LF7_auth_chain"],
            "auth_seq": w["5LF7_auth_seq"],
            "min_to_6V8_A": w["min_to_6V8_A"],
            "min_to_BO2_A": w["min_to_BO2_A"],
            "nearest_5LF3_water_distance_A":
                w["nearest_5LF3_water"]["nearest_5LF3_water_distance_A"],
            "nearest_5LF3_water":
                f'{w["nearest_5LF3_water"]["5LF3_auth_chain"]}:'
                f'{w["nearest_5LF3_water"]["5LF3_auth_seq"]}',
            "protein_supports": [],
        })

        for p in w.get("protein_heteroatoms_within_3.5A", []):
            key = (p["label_chain"], int(p["label_seq"]))
            evidence[key]["water_support"].append({
                "water": water_id,
                "atom": p["atom"],
                "element": p["element"],
                "distance_A": p["distance_A"],
            })
            retained_waters[-1]["protein_supports"].append({
                "residue": keystr(key),
                "comp_id": p["comp_id"],
                "atom": p["atom"],
                "distance_A": p["distance_A"],
            })

    functional_core = set(evidence.keys())

    # Verify all network residues are found in canonical mmCIF.
    missing = sorted(k for k in functional_core if k not in residue_map)
    if missing:
        raise RuntimeError(
            "Network residues missing from canonical mmCIF: "
            + ", ".join(map(keystr, missing))
        )

    # ------------------------------------------------------------------
    # Optional convergence candidates: all screened non-core residues.
    # These are not automatically promoted into the core.
    # ------------------------------------------------------------------
    optional = []
    for r in chemistry["residue_summaries"]:
        key = (r["label_chain"], int(r["label_seq"]))
        if key in functional_core:
            continue
        optional.append({
            "residue": keystr(key),
            "comp_id": r["comp_id"],
            "signal": r["initial_retention_signal"],
            "min_to_6V8_A": r["min_to_6V8_A"],
            "min_to_BO2_A": r["min_to_BO2_A"],
            "chemistry": r["residue_chemistry"],
        })

    # ------------------------------------------------------------------
    # Peptide-coherent reference
    #
    # Start from core residues, add +/- padding on same polymer chain, then
    # fill very small gaps (e.g. core residues 47 and 49 imply residue 48).
    # ------------------------------------------------------------------
    available_by_chain = defaultdict(set)
    for chain, seq in residue_map:
        available_by_chain[chain].add(seq)

    peptide_reference = set(functional_core)
    context_reason = defaultdict(list)

    for chain, seq in sorted(functional_core):
        for delta in range(-args.padding, args.padding + 1):
            cand = (chain, seq + delta)
            if cand in residue_map:
                peptide_reference.add(cand)
                if cand not in functional_core:
                    context_reason[cand].append(
                        f"sequence_context_for_{chain}:{seq}"
                    )

    # Fill small gaps inside each chain if bounded by selected residues.
    changed = True
    while changed:
        changed = False
        for chain in sorted({k[0] for k in peptide_reference}):
            seqs = sorted(s for c, s in peptide_reference if c == chain)
            for a, b in zip(seqs, seqs[1:]):
                gap = b - a - 1
                if 0 < gap <= args.max_gap_fill:
                    for s in range(a + 1, b):
                        cand = (chain, s)
                        if cand in residue_map and cand not in peptide_reference:
                            peptide_reference.add(cand)
                            context_reason[cand].append(
                                f"gap_fill_between_{chain}:{a}_and_{chain}:{b}"
                            )
                            changed = True

    # ------------------------------------------------------------------
    # Summaries
    # ------------------------------------------------------------------
    def heavy_count(keys):
        return sum(residue_map[k]["heavy_atoms_deposited"] for k in keys)

    functional_heavy = heavy_count(functional_core)
    peptide_heavy = heavy_count(peptide_reference)

    # Deposited ligand heavy atom counts from chemistry report.
    # 6V8: 23, BO2: 28 in the current structures.
    ligand_heavy = {"6V8": 23, "BO2": 28}
    water_heavy = len(retained_waters)  # O atoms only in deposited waters

    # Build segment summaries.
    segments = []
    for chain in sorted({k[0] for k in peptide_reference}):
        seqs = sorted(s for c, s in peptide_reference if c == chain)
        for start, end in merge_ranges(seqs):
            keys = [(chain, s) for s in range(start, end + 1)
                    if (chain, s) in residue_map]
            names = [residue_map[k]["comp_id"] for k in keys]
            segments.append({
                "label_chain": chain,
                "label_start": start,
                "label_end": end,
                "n_residues": len(keys),
                "residue_names": names,
                "heavy_atoms_deposited": heavy_count(keys),
                "core_residues_in_segment": [
                    keystr(k) for k in keys if k in functional_core
                ],
            })

    residue_rows = []
    for key in sorted(peptide_reference):
        r = residue_map[key]
        ev = evidence.get(key)
        status = "functional_core" if key in functional_core else "peptide_context"
        reasons = []
        if key in functional_core:
            if ev["catalytic"]:
                reasons.append("catalytic")
            if ev["direct_strong"]:
                reasons.append(ev["direct_signal"])
            if ev["water_support"]:
                for ws in ev["water_support"]:
                    reasons.append(
                        f'water_support_{ws["water"]}_{ws["atom"]}_{ws["distance_A"]:.3f}A'
                    )
        else:
            reasons.extend(context_reason[key])

        residue_rows.append({
            "status": status,
            "label_chain": key[0],
            "label_seq": key[1],
            "auth_chain": r["auth_chain"],
            "auth_seq": r["auth_seq"],
            "comp_id": r["comp_id"],
            "heavy_atoms_deposited": r["heavy_atoms_deposited"],
            "reason": ";".join(reasons),
        })

    report = {
        "purpose":
            "Chemistry-first network closure before cutting/capping/protonation",
        "canonical_receptor": chemistry["canonical_receptor"],
        "model_definitions": {
            "functional_network_core": [
                "catalytic residue",
                "strong initial direct-contact residues",
                "protein residues supporting all high_conserved_bridge_both waters",
            ],
            "peptide_coherent_reference": [
                "functional_network_core",
                f"+/- {args.padding} polymer residue sequence context",
                f"fill sequence gaps of at most {args.max_gap_fill} residue",
                "deliberately generous reference topology; not final cluster",
            ],
        },
        "functional_network_core": {
            "residues": [keystr(k) for k in sorted(functional_core)],
            "n_residues": len(functional_core),
            "protein_heavy_atoms_deposited": functional_heavy,
            "retained_water_oxygens": water_heavy,
            "heavy_atom_totals_with_ligand": {
                "6V8": functional_heavy + water_heavy + ligand_heavy["6V8"],
                "BO2": functional_heavy + water_heavy + ligand_heavy["BO2"],
            },
        },
        "peptide_coherent_reference": {
            "residues": [keystr(k) for k in sorted(peptide_reference)],
            "n_residues": len(peptide_reference),
            "protein_heavy_atoms_deposited": peptide_heavy,
            "retained_water_oxygens": water_heavy,
            "heavy_atom_totals_with_ligand": {
                "6V8": peptide_heavy + water_heavy + ligand_heavy["6V8"],
                "BO2": peptide_heavy + water_heavy + ligand_heavy["BO2"],
            },
            "segments": segments,
        },
        "retained_water_candidates": retained_waters,
        "optional_convergence_candidates_not_in_core": optional,
        "important_caveats": [
            "Counts are deposited heavy atoms only; hydrogens and caps are not included.",
            "Crystallographic waters currently contribute oxygen coordinates only; H positions are unresolved.",
            "Water bridge evidence is heavy-atom geometry, not a finalized hydrogen-bond assignment.",
            "Peptide context is a topology proposal, not evidence that every added context residue must survive in the final model.",
            "No protonation state or formal cluster charge is assigned here.",
            "No coordinates are modified in this stage.",
        ],
        "next_decision": [
            "Inspect the functional-network core and peptide-coherent reference geometrically.",
            "Choose chemically sensible cut locations and cap types.",
            "Then build a generous first explicit cluster and assign protonation/hydrogens.",
            "Use smaller variants only as convergence/pruning tests against that reference.",
        ],
    }

    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")

    with open(args.residues, "w", newline="") as f:
        fields = [
            "status", "label_chain", "label_seq", "auth_chain", "auth_seq",
            "comp_id", "heavy_atoms_deposited", "reason"
        ]
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        w.writeheader()
        w.writerows(residue_rows)

    with open(args.segments, "w", newline="") as f:
        fields = [
            "label_chain", "label_start", "label_end", "n_residues",
            "residue_names", "heavy_atoms_deposited", "core_residues_in_segment"
        ]
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        w.writeheader()
        for row in segments:
            out = dict(row)
            out["residue_names"] = ",".join(out["residue_names"])
            out["core_residues_in_segment"] = ",".join(out["core_residues_in_segment"])
            w.writerow(out)

    print(f"Wrote {args.out}")
    print(f"Wrote {args.residues}")
    print(f"Wrote {args.segments}")
    print()
    print("Network-closure summary")
    print(f"  functional core: {len(functional_core)} residues, "
          f"{functional_heavy} protein heavy atoms + {water_heavy} water O")
    print(f"  peptide reference: {len(peptide_reference)} residues, "
          f"{peptide_heavy} protein heavy atoms + {water_heavy} water O")
    print()
    print("Functional-network core residues")
    for key in sorted(functional_core):
        r = residue_map[key]
        ev = evidence[key]
        reason = []
        if ev["catalytic"]:
            reason.append("catalytic")
        if ev["direct_strong"]:
            reason.append(ev["direct_signal"])
        if ev["water_support"]:
            reason.append(
                "supports " + ",".join(sorted({x["water"] for x in ev["water_support"]}))
            )
        print(f"  {keystr(key):8s} {r['comp_id']:3s}  "
              f"heavy={r['heavy_atoms_deposited']:2d}  {'; '.join(reason)}")
    print()
    print("Peptide-coherent reference spans")
    for s in segments:
        names = "-".join(s["residue_names"])
        print(f"  {s['label_chain']}:{s['label_start']}-{s['label_end']}  "
              f"{s['n_residues']:2d} residues  heavy={s['heavy_atoms_deposited']:3d}  "
              f"{names}")
    print()
    print("Heavy-atom totals including 3 water oxygens and ligand")
    print("  functional core + 6V8:",
          report["functional_network_core"]["heavy_atom_totals_with_ligand"]["6V8"])
    print("  functional core + BO2:",
          report["functional_network_core"]["heavy_atom_totals_with_ligand"]["BO2"])
    print("  peptide reference + 6V8:",
          report["peptide_coherent_reference"]["heavy_atom_totals_with_ligand"]["6V8"])
    print("  peptide reference + BO2:",
          report["peptide_coherent_reference"]["heavy_atom_totals_with_ligand"]["BO2"])


if __name__ == "__main__":
    main()
