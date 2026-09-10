# HPC EWF+SQD Demo — 20S Proteasome Inhibitor Binding

A 5-stage Slurm workflow computing the EWF+SQD binding energies of
**ixazomib** and **bortezomib** against the 20S proteasome catalytic site.

The four input XYZ structures are covalent boronate-adduct complexes of the
human 20S proteasome β5 subunit active site, prepared from PDB entries
**5LF3** (bortezomib) and **5LF7** (ixazomib).  All system preparation
scripts live under [`system_prep/`](system_prep/README.md).

---

## Scientific background

### Observable

The matched-receptor construction gives a direct, receptor-cancellation
binding energy difference:

```
ΔΔE_bind = ΔE_bind(ixazomib) − ΔE_bind(bortezomib)
```

Because the common receptor fragment is **exactly identical** (same atom
order, same coordinates) in both complexes, it cancels without requiring
a separate receptor calculation:

```
ΔΔE_bind = E(ixa_complex) − E(ixa_ligand)
          − E(bo2_complex) + E(bo2_ligand)
```

### Systems

| System | Atoms | Charge | XYZ file |
|---|---|---|---|
| Ixazomib complex  | 533 | −1 | `ixazomib_complex.xyz`  |
| Bortezomib complex | 544 | −1 | `bortezomib_complex.xyz` |
| Ixazomib ligand   |  42 |  0 | `ixazomib_ligand.xyz`   |
| Bortezomib ligand |  53 |  0 | `bortezomib_ligand.xyz`  |

The **−1** complex charge arises from the tetrahedral Thr1-O-B boronate
adduct (MS_B_LYS33 proton microstate).  The extracted ligands are neutral
three-coordinate boronic acid references.

---

## Workflow

### Classical reference mode (default)

```
Step 1 (GPU)         Step 2 (GPU)           Step 3 (GPU)            Step 4 (CPU)
──────────────       ──────────────────────  ────────────────────    ─────────────────────
01_meanfield.py  →   02_fragments.py     →   03_solve.py         →   04_reconstruct.py
  HF mean-field        Vayesta EWF embed       FCI (small frags)       Partitioned cumulant
  (charge-aware)       IAO fragmentation       CCSD (large frags)      energy reconstruction
  Save mf_data.pkl     Save embedding_data     Save solver_results     Save summary.json

                           ┌──── ixazomib_complex   (1→2→3→4) ────┐
                           │     ixazomib_ligand     (1→2→3→4) ────│
All four systems           │     bortezomib_complex  (1→2→3→4) ────│──→ 05_binding_energy.py
run in parallel ───────────┘     bortezomib_ligand   (1→2→3→4) ────┘     ΔΔE_bind output
```

### QPU mode (`--qpu`)

The QPU workflow splits step 3 into three sub-steps so that expensive HPC
nodes are not held idle while waiting for the IBM Quantum queue:

```
Step 3a (GPU, ~1d)          Step 3b (GPU, ~1h)       Step 3c (GPU, ~4h)
───────────────────         ─────────────────────    ─────────────────────────
03_solve_ccsd.py        →   03_solve_sqd_submit.py → 03_solve_sqd_collect.py
  FCI (small frags)           Build LUCJ circuits      Retrieve QPU counts
  CCSD (large frags)          Submit to IBM Quantum     Run SBD postprocessing
  Save ccsd_results.pkl       Save job_manifest.json    Merge → solver_results.pkl
                              EXIT (HPC node released)

                          ↑ MANUAL trigger after 3a
                          Monitor with 03_solve_sqd_monitor.sh (login node)
                          ↑ MANUAL trigger after all QPU jobs DONE
```

`run_workflow.sh --qpu` submits steps 1, 2, 3a, and 4 automatically.
Steps 3b and 3c are printed as ready-to-paste `sbatch` commands.

---

## Files

| File | Description |
|---|---|
| `config_ixazomib_complex_sto-3g.yaml`  | Config for ixazomib complex (charge −1) |
| `config_ixazomib_ligand_sto-3g.yaml`   | Config for ixazomib ligand (charge 0) |
| `config_bortezomib_complex_sto-3g.yaml` | Config for bortezomib complex (charge −1) |
| `config_bortezomib_ligand_sto-3g.yaml`  | Config for bortezomib ligand (charge 0) |
| `ixazomib_complex.xyz`   | 533-atom covalent complex |
| `bortezomib_complex.xyz` | 544-atom covalent complex |
| `ixazomib_ligand.xyz`    | 42-atom extracted ligand |
| `bortezomib_ligand.xyz`  | 53-atom extracted ligand |
| `01_meanfield.py`  | Hartree-Fock via `QFWorkflow` (charge/spin aware) |
| `02_fragments.py`  | EWF fragmentation via Vayesta |
| `03_solve.py`      | Monolithic FCI/CCSD adaptive solver (classical mode) |
| `03_solve_ccsd.py` | Step 3a — CCSD reference for all SQD-eligible fragments |
| `03_solve_sqd_submit.py` | Step 3b — Submit QPU jobs and exit (fire-and-forget) |
| `03_solve_sqd_monitor.sh` | Monitor QPU job status from login node (no HPC allocation) |
| `03_solve_sqd_collect.py` | Step 3c — Retrieve counts + SBD postprocessing + merge |
| `04_reconstruct.py` | Partitioned cumulant energy reconstruction |
| `05_binding_energy.py` | ΔΔE_bind assembly from four summaries |
| `01_meanfield.slurm` – `05_binding_energy.slurm` | Slurm job scripts |
| `03_solve_ccsd.slurm` | Slurm script for step 3a (QPU mode) |
| `03_solve_sqd_submit.slurm` | Slurm script for step 3b (QPU mode) |
| `03_solve_sqd_collect.slurm` | Slurm script for step 3c (QPU mode) |
| `run_workflow.sh`  | Submit all systems; `--qpu` flag enables QPU split workflow |
| `system_prep/`     | Full system preparation pipeline (scripts 01–27) |

---

## Setup

Same two one-time steps as the alanine demo.

> **Checklist**
> - [ ] `cp .hpc_config.template .hpc_config` and fill in 5 values
> - [ ] Create `.qrmi_config` on the shared filesystem with your IBM Quantum API key

See the [alanine demo README](../alanine_ewf_sqd_demo/README.md) for detailed
setup instructions — the configuration files are identical.

---

## Configuration

All tunable parameters live in the four YAML config files.  The charge and
spin values are part of the `workflow:` block:

```yaml
workflow:
  xyz_file: ixazomib_complex.xyz
  basis: sto-3g
  charge: -1      # ← -1 for complexes, 0 for ligands
  spin: 0

embedder:
  ewf:
    bath_type: mp2
    truncation: 1.0e-5
    fragmentation: iao

solver_selection:
  strategy: adaptive
  orbital_threshold: 15   # n_orb < 15 → FCI; otherwise → SQD
```

---

## Quick Start

### 1. Build the container (from repo root)

```bash
podman build --platform=linux/amd64 --tag qfm-hpc-py312:v3.0 --file Containerfile .
```

### 2. Transfer to cluster and convert to sqsh

See [alanine demo README](../alanine_ewf_sqd_demo/README.md).

### 3. Sync repo and submit

```bash
source .hpc_config
rsync -avP --partial --exclude='*.tar' --exclude='*.sqsh' --exclude='__pycache__' \
  . $HPC_USERNAME@$HPC_LOGIN_HOST:$HPC_PROJECT/quantum-fragment-methods/

cd $HPC_PROJECT/quantum-fragment-methods/examples/hpc_demos/proteasome_ewf_sqd_demo

# Classical CCSD reference (no QPU required):
bash run_workflow.sh

# QPU workflow — submits 3a automatically, prints manual 3b/3c commands:
bash run_workflow.sh --qpu
```

`run_workflow.sh` submits all four systems in parallel (each as a multi-job
chain) and a final binding energy job that waits for all four reconstructions.

---

## Expected Outputs

After a successful run:

```
data/
  ixazomib_complex_sto-3g/
    mf_data.pkl              # HF mean-field  (step 1)
    embedding_data.pkl       # EWF fragments  (step 2)
    ewf_dumpfile.h5          # Vayesta HDF5 cluster Hamiltonians
    solver_results.pkl       # Fragment solutions  (step 3)
    rdms/                    # Per-fragment RDM1/RDM2/NOONs
    timing.json
  ixazomib_ligand_sto-3g/    # same structure
  bortezomib_complex_sto-3g/ # same structure
  bortezomib_ligand_sto-3g/  # same structure

results/
  ixazomib_complex_sto-3g/summary.json
  ixazomib_ligand_sto-3g/summary.json
  bortezomib_complex_sto-3g/summary.json
  bortezomib_ligand_sto-3g/summary.json
  binding_energies.json      # Final ΔΔE_bind  (step 5)
```

### `binding_energies.json` structure

```json
{
  "method": "EWF+SQD",
  "basis": "sto-3g",
  "delta_delta_E_Ha": ...,
  "delta_delta_E_kcal_mol": ...,
  ...
}
```

---

## Checkpoint-aware resumption

### Classical mode (`03_solve.py`)

Step 3 is checkpoint-aware.  If the Slurm wall time expires mid-run, resubmit:

```bash
python 03_solve.py --config config_ixazomib_complex_sto-3g.yaml --wait
# discard checkpoints and restart:
python 03_solve.py --config config_ixazomib_complex_sto-3g.yaml --force-resubmit
```

### QPU mode (3a/3b/3c)

Each step checkpoints independently:

- **3a** (`03_solve_ccsd.py`): writes `ccsd_results.pkl` incrementally
  (one fragment at a time).  Re-running resumes from the last completed
  fragment; pass `--force-rerun` to start from scratch.

- **3b** (`03_solve_sqd_submit.py`): writes `job_manifest.json` and per-fragment
  `job_id.txt`.  Re-running skips already-submitted fragments.  Use
  `--force-resubmit` to resubmit all.

- **3c** (`03_solve_sqd_collect.py`): writes per-fragment `sbd_result.pkl`
  (final) and `sbd_checkpoint.pkl` (per-iteration).  Re-running loads
  cached results and resumes interrupted SBD iterations.

Monitor QPU jobs from the login node (no allocation needed):

```bash
bash 03_solve_sqd_monitor.sh ixazomib_complex_sto-3g --watch 300
bash 03_solve_sqd_monitor.sh bortezomib_complex_sto-3g --watch 300
```

---

## System Preparation

The four input XYZ files were produced by a 27-step system preparation
pipeline.  All source scripts, raw PDB/CIF data, intermediate geometries,
and validation reports are preserved in [`system_prep/`](system_prep/).

See [`system_prep/README.md`](system_prep/README.md) for full documentation.

---

## References

- PDB 5LF3 — Bortezomib / human 20S proteasome, 2.1 Å
- PDB 5LF7 — Ixazomib / human 20S proteasome, 2.0 Å
- Alanine EWF+SQD reference workflow: `examples/hpc_demos/alanine_ewf_sqd_demo/`
- HPC setup: `docs/IBM-SCC-BUILD-GUIDE.md`
