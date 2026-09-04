# System Preparation — 20S Proteasome Inhibitor Binding Models

This directory contains the full 27-step pipeline that produces the four
production XYZ files used by the EWF+SQD workflow.  The pipeline was
migrated from `proteasome_challenge/` into this repo so that geometry
preparation and quantum simulation live together.

---

## Raw inputs

| File | Description |
|---|---|
| `raw/5LF3.cif` | PDB 5LF3 — bortezomib / 20S proteasome, 2.1 Å |
| `raw/5LF7.cif` | PDB 5LF7 — ixazomib / 20S proteasome, 2.0 Å |

---

## Pipeline overview

```
01_inventory.py                 Structural audit of deposited mmCIF files
02_align_active_sites.py        Align β5 active sites of 5LF3 and 5LF7
03_choose_canonical_receptor.py Select 5LF7-Y as canonical receptor frame
04_define_common_cluster.py     Define 5 Å union active-site cluster
05_map_active_site_chemistry.py Map hydrogen-bonding network
06_analyze_network_closure.py   Chemistry-first network closure analysis
07_inspect_cut_boundaries.py    Inspect covalent cut boundaries
08_inspect_boundary_environment.py  Inspect cut-point chemical environments
09_finalize_segment_boundaries.py   Finalize segment boundaries
10_build_capped_receptor_scaffold.py  Build capped heavy-atom receptor (v1)
10b_build_capped_receptor_with_lys33.py  Capped receptor including Lys33 (v2)
11_audit_protonation_states.py  Audit protonation states
11b_audit_protonation_with_lys33.py  Audit with Lys33 included
11c_audit_catalytic_proton_network.py  Full catalytic proton network audit
12_audit_boronate_coordination.py   Audit Thr1-O-B tetrahedral adduct
13_define_catalytic_microstates.py  Define three proton microstates (A/B/C)
14_build_hydrogenated_microstates.py  Add hydrogens (Open Babel + overrides)
15_validate_hydrogenated_microstates.py  Validate H counts and charges
16_optimize_hydrogens_xtb.py    H-only relaxation with xTB/GFN-FF (v1)
16b_optimize_hydrogens_gfnff.py H-only relaxation with GFNFF (v2)
17_optimize_hydrogens_ase_gfnff.py  H-only relaxation with ASE + GFN-FF ✓
18_repair_special_hydrogen_geometries.py  Repair special H geometries
19_audit_optimized_microstate_chemistry.py  Chemical audit post-optimization
20_screen_microstates_gfn2_xtb.py   GFN2 proton-microstate energy screen
20b_screen_microstates_tblite_gfn2.py  GFN2 screen via tblite
21_build_catalytic_core_models.py   Build compact DFT core models
22_screen_catalytic_core_pbe0.py    PBE0/def2-SVP core microstate screen
23_freeze_production_models.py  ★ Freeze final four production XYZ files
24_validate_production_models.py    Independent final audit
25_inspect_short_receptor_ligand_contacts.py  Contact geometry audit
26_relax_production_ligand_hydrogens.py  Final ligand H refinement
27_audit_short_hbond_geometry.py    H-bond geometry sanity check
```

The **final production XYZ files** produced by steps 23 and 26 are copied
to the parent demo directory:

```
../ixazomib_complex.xyz    (step 23)
../ixazomib_ligand.xyz     (step 23)
../bortezomib_complex.xyz  (step 23)
../bortezomib_ligand.xyz   (step 23)
```

---

## Proton microstate selection

Three proton microstates were screened for the catalytic Thr1/Lys33/Asp17
triad:

| Microstate | Thr1 N-term | Lys33 | Asp17 |
|---|---|---|---|
| MS_A_THRN  | NH₃⁺ | NH₂ | COO⁻ |
| MS_B_LYS33 | NH₂  | NH₃⁺ | COO⁻ |
| MS_C_ASP17 | NH₂  | NH₂  | COOH |

**MS_B_LYS33** was selected as the production microstate based on:

- GFN2/whole-cluster: ixazomib ΔE(B−A) = −7.2 kcal/mol
- PBE0/def2-SVP catalytic core: ixazomib ΔE(B−A) = −25.2 kcal/mol,
  bortezomib ΔE(B−A) = −19.3 kcal/mol

Both inhibitors strongly favour MS_B.

---

## Charge bookkeeping

| System | Charge | Rationale |
|---|---|---|
| Complex (receptor + ligand) | −1 | Tetrahedral boronate adduct: Thr1 O⁻ donates to B |
| Isolated ligand | 0 | B returns to three-coordinate R-B(OH)₂ after bond removal |
| Implied cut receptor | −1 | Thr1 alkoxide; cancels exactly in ΔΔE_bind |

The receptor contribution cancels in the **matched-receptor** ΔΔE_bind
observable, so no standalone receptor calculation is required.

---

## Dependencies

```
gemmi>=0.7.0
numpy>=1.26
# Step 14: obabel (Open Babel CLI)
# Steps 16/16b/17: xtb, gfnff[ase]
# Step 20/20b: xtb, tblite
# Step 22: PySCF or ORCA for PBE0/def2-SVP
```

Install Python dependencies:
```bash
pip install -r requirements.txt
```

---

## Reproducibility

All intermediate outputs are preserved in `intermediate/`.  Validation
reports (JSON and TSV) are in `validation/`.  The pipeline is deterministic
given the same `obabel`, `xtb`, and `gfnff` versions.

To regenerate the production XYZ files from scratch:

```bash
cd <repo-root>/examples/hpc_demos/proteasome_ewf_sqd_demo/system_prep
pip install -r requirements.txt   # gemmi, numpy
# also requires: obabel, xtb, gfnff[ase]

python scripts/01_inventory.py raw/5LF3.cif raw/5LF7.cif \
    --out validation/structure_inventory.json

# ... run scripts 02–22 in order ...

python scripts/23_freeze_production_models.py \
    --optimized   intermediate/hydrogen_optimized_ase_gfnff \
    --metadata    intermediate/hydrogenated_microstates_repaired_v2 \
    --outdir      production_models \
    --manifest    validation/production_model_manifest.json

python scripts/26_relax_production_ligand_hydrogens.py

python scripts/24_validate_production_models.py \
    --indir  production_models_final \
    --report validation/final_production_model_audit.json

# Copy validated models to the demo root:
cp production_models_final/*.xyz ../
```
