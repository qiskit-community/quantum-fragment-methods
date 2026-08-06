# HPC EWF SQD Demo Plan

## Top-Level Overview

Reproduce the `tutorial.ipynb` notebook from `local_ewf_sqd_demo` as a production-ready
4-stage HPC Slurm workflow in `examples/hpc_ewf_sqd_demo/`. The target system is
alanine (13 atoms) from `system.xyz`. The workflow mirrors the pipeline already
established in `hpc_sqd_demo` but replaces the monolithic N2 SQD script with the
full EWF fragmentation + adaptive solver path from the tutorial.

**The 4 stages map exactly to the existing stub slurm scripts:**

```
01_meanfield.slurm  → 01_meanfield.py    (Hartree-Fock via QFWorkflow.run_mean_field)
02_fragments.slurm  → 02_fragments.py    (EWF embedding via QFWorkflow.create_fragments)
03_solve.slurm      → 03_solve.py        (Adaptive FCI/SQD via QFWorkflow.solve_fragments)
04_reconstruct.slurm → 04_reconstruct.py (Energy reconstruction via QFWorkflow.reconstruct_energy)
```

**Key design decisions:**
- **All molecular and algorithmic parameters are read from the YAML config** — no hardcoded
  values in Python scripts. This includes: basis set, XYZ file path, bath type, truncation,
  fragmentation scheme, orbital threshold, SQD iterations, shots, and SBD exe path.
- Solver selection is configurable in YAML: `solver_selection.strategy` (`adaptive` or `sqd_all`)
  with an `orbital_threshold` controlling the FCI/SQD boundary.
- QRMI credential injection follows the exact pattern from `hpc_sqd_demo/run_workflow.sh`.
- Intermediate results are persisted as pickle files between stages (same pattern as `hpc_sqd_demo`).
- A single `run_workflow.sh` chains all 4 jobs with `--dependency=afterok`.
- Config file is `config_alanine_sto-3g.yaml`; credentials stay in `.hpc_config` / `.qrmi_config`.
- Every Python driver takes `--config` as its primary argument; all other CLI flags are
  overrides or path redirects only (`--output-dir`, `--data-dir`, `--xyz`).

---

## Sub-Tasks

---

### Sub-Task 1 — Config YAML

**Intent:** Create the config file that drives all 4 stages. Solver selection strategy and
orbital threshold are defined here, keeping all tunable parameters out of the Python scripts.

**Expected Outcomes:**
- `examples/hpc_ewf_sqd_demo/config_alanine_sto-3g.yaml` exists with sections for
  `workflow`, `embedder`, `solver_selection`, `qpu`, and `sqd`.
- The `solver_selection` section has `strategy: adaptive`, `orbital_threshold: 15`,
  and `fci_max_orbitals: 20` (hard upper-limit guard).
- `sqd.classical_backend` defaults to `hpc` (uses C++ SBD binary baked into the container).
- `sqd.sbd.exe_path` points to `/opt/executable/diag` (container v3 path).

**Todo List:**
1. Create `config_alanine_sto-3g.yaml` modelled on `hpc_sqd_demo/config_N2_sto-3g.yaml`
   and `local_ewf_sqd_demo/config_N2_sto-3g_demo.yaml`.
2. Add `workflow.basis: sto-3g` and `workflow.xyz_file: system.xyz`.
3. Add `embedder.ewf` block: `bath_type: mp2`, `truncation: 1.0e-5`, `fragmentation: iao`.
4. Add `solver_selection` block: `strategy: adaptive`, `orbital_threshold: 15`.
5. Add `qpu` block with `provider: qrmi` and `backend_name: ibm_pittsburgh`
   (matching `QRMI_JOB_QPU_RESOURCES` convention).
6. Add `sqd` block with the same algorithm parameters as `hpc_sqd_demo` config,
   `classical_backend: hpc`, and `sbd.exe_path: /opt/executable/diag`.

**Relevant Context:**
- `examples/hpc_sqd_demo/config_N2_sto-3g.yaml` — reference for qpu/sqd sections.
- `examples/local_ewf_sqd_demo/config_N2_sto-3g_demo.yaml` — reference for embedder section.

**Status:** [x] done

---

### Sub-Task 2 — Python Driver: `01_meanfield.py`

**Intent:** Run Hartree-Fock on the alanine geometry and save the mean-field object
so that `02_fragments.py` can resume without re-running SCF.

**Expected Outcomes:**
- `examples/hpc_ewf_sqd_demo/01_meanfield.py` exists.
- Reads `system.xyz` path from config (or CLI `--xyz` override).
- Constructs a `QFWorkflow` with the embedder from config and calls `run_mean_field()`.
- Saves `mf_data.pkl` (MO coefficients, energies, mol info) to `--output-dir`.
- Prints HF energy on stdout.

**Todo List:**
1. Create `01_meanfield.py` with argparse: `--config` (required), `--xyz` (optional override),
   `--output-dir` (default: `data/`).
2. Load YAML and read `workflow.basis` and `workflow.xyz_file`; if `--xyz` CLI flag is
   provided it overrides `workflow.xyz_file`. Resolve the XYZ path relative to the
   script's directory so it works from any working directory.
3. Construct `QFWorkflow(geometry=xyz_path, basis=basis)` where both values come from
   config (or CLI override) — no hardcoded strings.
4. Call `workflow.run_mean_field()` and collect the returned `mf` object.
5. Serialize the `mf` object (mo_coeff, mo_occ, mo_energy, e_tot) and the mol geometry
   string to `mf_data.pkl` — same pickle pattern used in `hpc_sqd_demo/01_setup_molecule.py`.
   Also persist `basis` and `xyz_path` in the pickle so downstream steps don't need to
   re-read the config for mol reconstruction.
6. Print summary: HF energy, basis set used, number of atoms, number of orbitals.

**Relevant Context:**
- `quantum_fragment_methods/workflow.py` — `QFWorkflow.run_mean_field()` does the HF.
- `examples/hpc_sqd_demo/01_setup_molecule.py` — reference for pickle save pattern.
- `examples/hpc_ewf_sqd_demo/system.xyz` — 13-atom alanine geometry.

**Status:** [x] done

---

### Sub-Task 3 — Python Driver: `02_fragments.py`

**Intent:** Load the mean-field result from step 1 and run EWF fragmentation, saving
the embedding result (HDF5 dumpfile path + fragment metadata) for step 3.

**Expected Outcomes:**
- `examples/hpc_ewf_sqd_demo/02_fragments.py` exists.
- Loads `mf_data.pkl` and reconstructs the PySCF `mf` object (same pattern as `02_run_sqd.py`).
- Creates `EWF(bath_type, truncation)` from config and calls `create_fragments(mf, fragmentation=...)`.
- Saves `embedding_data.pkl` containing: dumpfile path, fragment metadata (n_orbitals,
  n_electrons, fragment_id for each fragment), and mean-field energy.
- Prints fragment count and orbital counts per fragment.

**Todo List:**
1. Create `02_fragments.py` with argparse: `--config` (required), `--data-dir` (default: `data/`),
   `--output-dir` (default: `data/`).
2. Load YAML and read `embedder.ewf.bath_type`, `embedder.ewf.truncation`,
   `embedder.ewf.fragmentation` — all sourced from config, none hardcoded.
3. Load `mf_data.pkl` from `--data-dir`; use the `basis` and `xyz_path` persisted in
   the pickle to reconstruct the PySCF `mol` and restore `mf` (mo_coeff, mo_occ, mo_energy, e_tot).
4. Instantiate `EWF(bath_type=..., truncation=...)` with config values and call
   `create_fragments(mf, fragmentation=fragmentation)`.
5. Serialize embedding metadata to `embedding_data.pkl`: dumpfile path, mean-field energy,
   and per-fragment dict of `fragment_id → {n_orbitals, n_electrons}`.
6. Print per-fragment summary table: fragment id, n_orbitals, n_electrons, bath_type used.

**Relevant Context:**
- `quantum_fragment_methods/application/embedding/ewf.py` — `EWF.create_fragments()` / `kernel()`.
- `quantum_fragment_methods/application/embedding/base.py` — `EmbeddingResult`, `Fragment`.
- `examples/hpc_sqd_demo/02_run_sqd.py` — reference for restoring mf from pickle.

**Status:** [x] done

---

### Sub-Task 4 — Python Driver: `03_solve.py`

**Intent:** Load fragment data from step 2 and solve each fragment with the
configured solver (FCI for small, SQD for larger — based on `orbital_threshold`),
saving per-fragment solver results.

**Expected Outcomes:**
- `examples/hpc_ewf_sqd_demo/03_solve.py` exists.
- Reads `solver_selection.strategy` and `orbital_threshold` from config.
- When `strategy: adaptive`, assigns `FCI` if `fragment.n_orbitals < orbital_threshold`,
  else `SQDSolver` with QRMI backend.
- When `strategy: sqd_all`, all fragments use `SQDSolver`.
- SQD solver is checkpoint-aware (re-run resumes from `job_id.txt` / `counts.npy`).
- Saves `solver_results.pkl` (dict of fragment_id → SolverResult) to `--output-dir`.
- Prints per-fragment solver assignment and correlation energy.

**Todo List:**
1. Create `03_solve.py` with argparse: `--config`, `--data-dir`, `--output-dir`,
   `--force-resubmit`, `--wait`, `--max-wait-time`.
2. Load `mf_data.pkl` and `embedding_data.pkl`.
3. Reconstruct the full `EmbeddingResult` object (fragments dict + metadata with dumpfile path).
4. Initialize `QRMIBackend` from qpu config (credentials from env vars, same as `02_run_sqd.py`).
5. Read `solver_selection` config; implement the `adaptive` / `sqd_all` branching.
6. Build `QFWorkflow` with the reconstructed embedder + embedding result, add solver rules,
   then call `workflow.solve_fragments()` directly (skip `run_mean_field` and `create_fragments`).
7. Persist `solver_results.pkl` to `--output-dir`.
8. Print solver assignment summary and per-fragment correlation energy.

**Relevant Context:**
- `quantum_fragment_methods/workflow.py` — `QFWorkflow.solve_fragments()`, `_assign_solvers()`.
- `quantum_fragment_methods/application/solvers/quantum_zoo/sqd.py` — `SQDSolver.solve_from_integrals()`.
- `quantum_fragment_methods/application/solvers/classical_zoo/fci.py` — `FCI` solver.
- `quantum_fragment_methods/qpu/qrmi.py` — `QRMIBackend`.
- `examples/hpc_sqd_demo/02_run_sqd.py` — reference for QRMI initialization.

**Status:** [x] done

---

### Sub-Task 5 — Python Driver: `04_reconstruct.py`

**Intent:** Load all fragment results from step 3 and reconstruct the total energy
using the EWF partitioned cumulant approach, saving a JSON summary.

**Expected Outcomes:**
- `examples/hpc_ewf_sqd_demo/04_reconstruct.py` exists.
- Loads `mf_data.pkl`, `embedding_data.pkl`, and `solver_results.pkl`.
- Reconstructs the `EWF` embedder and `EmbeddingResult` objects needed by `reconstruct_energy`.
- Calls `ewf_embedder.reconstruct_energy(fragment_results, embedding_result)`.
- Saves `results/summary.json` with: HF energy, total EWF+SQD energy, correlation
  energy, per-fragment energies.
- Prints final energy table.

**Todo List:**
1. Create `04_reconstruct.py` with argparse: `--config`, `--data-dir`, `--results-dir`.
2. Load all three pickle files.
3. Reconstruct `EWF` embedder instance and `EmbeddingResult` (with dumpfile path in metadata).
4. Reconstruct `QFWorkflow` and attach the loaded `embedding_result` and `mf`.
5. Call `workflow.reconstruct_energy(fragment_results)`.
6. Compute correlation energy (`total - hf`).
7. Write `results/summary.json` with all key values.
8. Print formatted energy table.

**Relevant Context:**
- `quantum_fragment_methods/application/embedding/ewf.py` — `EWF.reconstruct_energy()`.
- `quantum_fragment_methods/workflow.py` — `QFWorkflow.reconstruct_energy()`.
- `examples/hpc_sqd_demo/02_run_sqd.py` — reference for JSON summary pattern.

**Status:** [x] done

---

### Sub-Task 6 — Slurm Scripts (fill in stubs)

**Intent:** Replace the TODO-stub slurm scripts with working jobs that invoke the
Python drivers with correct container mounts and environment variable injection.
Follow the exact Pyxis/enroot patterns from `hpc_sqd_demo`.

**Expected Outcomes:**
- All 4 slurm scripts call their corresponding Python driver via
  `python examples/hpc_ewf_sqd_demo/<N>_<name>.py`.
- Container image, mount, and GPU env flags are NOT hardcoded — they are passed by
  `run_workflow.sh` at submission time via CLI args (matching `hpc_sqd_demo` pattern).
- QRMI credential env vars are injected via `--container-env` in `run_workflow.sh`
  (only `03_solve.slurm` needs them; others are CPU/GPU classical jobs).
- `logs/` directory is created before submission (handled by `run_workflow.sh`).
- Wall times are appropriate: 30min for steps 1,2,4; 4h for step 3 (QPU queue).

**Todo List:**
1. Fill in `01_meanfield.slurm`: call `01_meanfield.py --config ... --output-dir data/`.
2. Fill in `02_fragments.slurm`: call `02_fragments.py --config ... --data-dir data/ --output-dir data/`.
3. Fill in `03_solve.slurm`: call `03_solve.py --config ... --data-dir data/ --output-dir data/ --wait --max-wait-time 14400`.
   Extend wall time to 4 hours to accommodate QPU queue.
4. Fill in `04_reconstruct.slurm`: call `04_reconstruct.py --config ... --data-dir data/ --results-dir results/`.
5. Ensure all scripts use `--container-workdir=/workspace` and path relative to `/workspace`.

**Relevant Context:**
- `examples/hpc_sqd_demo/01_meanfield.slurm`, `02_sqd_solve.slurm` — reference for slurm header patterns.
- `examples/hpc_sqd_demo/run_workflow.sh` — shows how CLI flags are injected at sbatch time.

**Status:** [x] done

---

### Sub-Task 7 — `run_workflow.sh`

**Intent:** Create the submission orchestrator that sources `.hpc_config`, injects
QRMI credentials, creates `logs/`, `data/`, and `results/` directories, then chains
all 4 jobs with `--dependency=afterok`.

**Expected Outcomes:**
- `examples/hpc_ewf_sqd_demo/run_workflow.sh` exists and is executable.
- Reads `.hpc_config` from repo root (two `..` levels up, same logic as `hpc_sqd_demo`).
- Validates `CONTAINER_IMAGE`, `SLURM_PARTITION`, `QRMI_CREDS`.
- QRMI env vars are sourced and injected only for `03_solve.slurm`.
- Jobs are chained: job2 depends on job1, job3 on job2, job4 on job3.
- Prints job IDs and log paths after submission (matching `hpc_sqd_demo` style).

**Todo List:**
1. Create `run_workflow.sh` modelled on `hpc_sqd_demo/run_workflow.sh`.
2. Add `mkdir -p logs data results` before the first `sbatch`.
3. Submit jobs 1–2 without QRMI env flags (classical stages).
4. Submit job 3 with QRMI env flags (QPU stage).
5. Submit job 4 without QRMI env flags (classical reconstruction).
6. Print the 4-job chain summary with log file paths.

**Relevant Context:**
- `examples/hpc_sqd_demo/run_workflow.sh` — direct template.
- `.hpc_config.template` — shows what variables are available.

**Status:** [x] done

---

### Sub-Task 8 — Update README

**Intent:** Replace the TODO-placeholder sections in the existing README with
complete quickstart, configuration, monitoring, and expected-outputs documentation.

**Expected Outcomes:**
- `examples/hpc_ewf_sqd_demo/README.md` has a complete Quick Start section.
- Documents all 5 config variables in `.hpc_config` (same table style as `hpc_sqd_demo`).
- Documents the `config_alanine_sto-3g.yaml` solver_selection block.
- Has a "Monitoring jobs" section matching `hpc_sqd_demo` style.
- Has an "Expected results" placeholder table for alanine HF / EWF+SQD energies.
- References are updated to point to `hpc_sqd_demo/README.md` for HPC setup.

**Todo List:**
1. Fill in Quick Start with the 5-step pattern (build container → save/transfer → convert → sync → submit).
2. Replace Configuration TODO with table of `.hpc_config` variables.
3. Document `solver_selection` YAML block and how to change the threshold.
4. Add "Running the Workflow" section with `bash run_workflow.sh`.
5. Add "Monitoring jobs" section with `squeue`, `tail -f logs/` commands.
6. Add "Expected Outputs" section listing `data/mf_data.pkl`, `data/embedding_data.pkl`,
   `data/solver_results.pkl`, `results/summary.json`.
7. Remove all `# TODO` placeholders.

**Relevant Context:**
- `examples/hpc_sqd_demo/README.md` — gold standard for documentation style.
- `examples/hpc_ewf_sqd_demo/README.md` — file to update (stubs in place).

**Status:** [x] done
