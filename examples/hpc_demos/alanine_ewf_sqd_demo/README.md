# HPC EWF+SQD Demo — Alanine Workflow

A 4-stage Slurm workflow reproducing the EWF+SQD tutorial notebook at
`examples/notebook_demos/ewf_sqd_demo/ewf_sqd.ipynb` on an HPC system. The system is alanine (13 atoms,
`system.xyz`) using Embedded Wave Function (EWF) theory with adaptive
FCI/SQD fragment solvers.

## How it works

```
Step 1 (GPU)             Step 2 (GPU)              Step 3 (GPU + QPU)         Step 4 (CPU)
────────────────         ────────────────────────   ────────────────────────   ─────────────────────
01_meanfield.py      →   02_fragments.py        →   03_solve.py            →   04_reconstruct.py
  - Read system.xyz        - Load mf_data.pkl         - Load embedding_data        - Load solver_results
  - Hartree-Fock (GPU)     - Vayesta EWF embed        - Assign FCI / SQD           - Rebuild Vayesta EWF
  - Save mf_data.pkl       - IAO fragmentation        - FCI for small frags        - Partitioned cumulant
                           - Save embedding_data       - SQD+QPU for larger         - Save summary.json
                             + HDF5 dumpfile           - Save solver_results
                                                       - Save circuits (QPY)
```

All molecular and algorithmic parameters are set in `config_alanine_sto-3g.yaml`.
No credentials or hardcoded values appear in the Python scripts.

## Files

| File | Description |
|---|---|
| `config_alanine_sto-3g.yaml` | Single config file for all 4 stages |
| `system.xyz` | Alanine geometry (13 atoms) |
| `01_meanfield.py` | Hartree-Fock via `QFWorkflow.run_mean_field()` |
| `02_fragments.py` | EWF fragmentation via Vayesta |
| `03_solve.py` | Adaptive FCI/SQD solver with QPU submission |
| `04_reconstruct.py` | Total energy via partitioned cumulant reconstruction |
| `01_meanfield.slurm` | Slurm script for step 1 (30 min, 1 GPU) |
| `02_fragments.slurm` | Slurm script for step 2 (30 min, 1 GPU) |
| `03_solve.slurm` | Slurm script for step 3 (4 h wall time, 1 GPU + QPU) |
| `04_reconstruct.slurm` | Slurm script for step 4 (30 min, CPU only) |
| `run_workflow.sh` | Submit all 4 jobs with `--dependency=afterok` chain |

---

## Setup

Two one-time configuration steps before the first run.

> **Checklist**
> - [ ] `cp .hpc_config.template .hpc_config` and fill in 5 values
> - [ ] Create `.qrmi_config` on the shared filesystem with your IBM Quantum API key

### Step 1 — Cluster config file

Copy the template to the repo root and fill in your cluster values:

```bash
# From the repo root on your local machine
cp .hpc_config.template .hpc_config
```

Open `.hpc_config` and set these 5 values:

| Variable | What to set |
|---|---|
| `HPC_PROJECT` | Your project directory on the shared filesystem (GPFS / Lustre) |
| `CONTAINER_IMAGE` | Full path to `qfm-hpc-py312-v3.sqsh` on the shared filesystem |
| `SLURM_PARTITION` | The GPU partition on your cluster (e.g. `gpu-mid`, `gpu`, `gpu-a100`) |
| `SLURM_ACCOUNT` | Your Slurm account/project code (leave blank if not required) |
| `QRMI_CREDS` | Full path to your QRMI credentials file on the shared filesystem |

`.hpc_config` is gitignored — it will never be committed.

### Step 2 — QRMI credentials file

```bash
cat > $HPC_PROJECT/.qrmi_config << 'EOF'
export QRMI_JOB_QPU_RESOURCES=ibm_kingston
export QRMI_JOB_QPU_TYPES=qiskit-runtime-service
export ibm_kingston_QRMI_IBM_QRS_ENDPOINT=https://quantum.cloud.ibm.com/api/v1
export ibm_kingston_QRMI_IBM_QRS_IAM_ENDPOINT=https://iam.cloud.ibm.com
export ibm_kingston_QRMI_IBM_QRS_IAM_APIKEY=your_apikey        # ← your key here
export ibm_kingston_QRMI_IBM_QRS_SERVICE_CRN=your_crn          # ← your CRN here
EOF
chmod 600 $HPC_PROJECT/.qrmi_config
```

> **Note:** the prefix before `_QRMI_` must match `QRMI_JOB_QPU_RESOURCES` and
> `qpu.backend_name` in `config_alanine_sto-3g.yaml`. Update all three if you
> switch backends.

---

## Configuration

All tunable parameters live in `config_alanine_sto-3g.yaml`. The most commonly
changed values are:

```yaml
workflow:
  xyz_file: system.xyz    # ← molecule geometry
  basis: sto-3g           # ← change basis set here (affects all 4 stages)

embedder:
  ewf:
    bath_type: mp2        # mp2 | dmet | full
    truncation: 1.0e-5
    fragmentation: iao    # iao | atomic

solver_selection:
  strategy: adaptive      # adaptive (FCI+SQD) | sqd_all (SQD for every fragment)
  orbital_threshold: 15   # fragments with n_orbitals < 15 use FCI; others use SQD

sqd:
  circuit_save_dir: circuits  # QPY circuits saved here (null to disable)
  iterations: 3
  samples_per_batch: 200
```

---

## Quick Start

### 1. Build the container image (local machine)

```bash
cd <repo-root>
podman build \
  --platform=linux/amd64 \
  --tag qfm-hpc-py312:v3.0 \
  --file Containerfile \
  . 2>&1 | tee build_v3.log
```

### 2. Save and transfer to the shared filesystem

```bash
podman save qfm-hpc-py312:v3.0 -o qfm-hpc-py312-v3.tar

source .hpc_config
rsync -avP --partial qfm-hpc-py312-v3.tar \
  $HPC_USERNAME@$HPC_LOGIN_HOST:$HPC_PROJECT/
```

### 3. Convert tar → sqsh on a compute node

```bash
srun --partition=$SLURM_PARTITION --time=01:00:00 --cpus-per-task=4 --mem=64G --pty bash

# IBM SCC:
/gpfs/scc6000/setup/scripts/docker-tar-to-sqsh.sh \
  $HPC_PROJECT/qfm-hpc-py312-v3.tar \
  $HPC_PROJECT/qfm-hpc-py312-v3.sqsh
exit
```

### 4. Sync the repo to the cluster

```bash
source .hpc_config
rsync -avP --partial \
  --exclude='*.tar' --exclude='*.sqsh' \
  --exclude='__pycache__' --exclude='.git' \
  . $HPC_USERNAME@$HPC_LOGIN_HOST:$HPC_PROJECT/quantum-fragment-methods/
```

### 5. Submit the workflow

```bash
cd $HPC_PROJECT/quantum-fragment-methods/examples/hpc_demos/alanine_ewf_sqd_demo
bash run_workflow.sh
```

`run_workflow.sh` sources `.hpc_config`, injects QRMI credentials only for
step 3 (the QPU stage), creates `logs/`, `data/`, and `results/` directories,
and chains all 4 jobs.

---

## Running the Workflow

```bash
bash run_workflow.sh
```

Example output:

```
Submitted step 1 (meanfield):    job 81400
Submitted step 2 (fragments):    job 81401 (depends on 81400)
Submitted step 3 (solve/QPU):    job 81402 (depends on 81401)
Submitted step 4 (reconstruct):  job 81403 (depends on 81402)

Workflow submitted: 81400 → 81401 → 81402 → 81403

Logs:
  logs/01_meanfield_81400.out
  logs/02_fragments_81401.out
  logs/03_solve_81402.out
  logs/04_reconstruct_81403.out

Results: examples/hpc_demos/alanine_ewf_sqd_demo/results/summary.json
```

---

## Monitoring Jobs

```bash
# Watch queue status
watch -n 10 squeue -u $USER

# Tail individual step logs
tail -f logs/01_meanfield_<JOBID>.out
tail -f logs/02_fragments_<JOBID>.out
tail -f logs/03_solve_<JOBID>.out
tail -f logs/04_reconstruct_<JOBID>.out

# Check for errors
tail -f logs/03_solve_<JOBID>.err
```

---

## Checkpoint-aware QPU workflow

Step 3 (`03_solve.py`) is checkpoint-aware. If the QPU job is still queued when
the Slurm wall time expires, resubmit step 3 — it resumes from the saved
`job_id.txt` and `counts.npy` without resubmitting to the QPU.

```bash
# Resubmit step 3 only (reuses existing QPU counts)
source .hpc_config
source $QRMI_CREDS
cd $HPC_PROJECT/quantum-fragment-methods/examples/hpc_demos/alanine_ewf_sqd_demo

sbatch \
  --partition="$SLURM_PARTITION" \
  --container-image="$CONTAINER_IMAGE" \
  --container-mounts="$HPC_PROJECT/quantum-fragment-methods:/workspace" \
  --container-workdir=/workspace \
  --container-env=NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  --container-env=QRMI_JOB_QPU_RESOURCES \
  --container-env=QRMI_JOB_QPU_TYPES \
  $(env | grep _QRMI_ | cut -d= -f1 | sed 's/^/--container-env=/') \
  03_solve.slurm

# Then resubmit step 4 with dependency:
sbatch --dependency=afterok:<JOB3_ID> \
  --partition="$SLURM_PARTITION" \
  --container-image="$CONTAINER_IMAGE" \
  --container-mounts="$HPC_PROJECT/quantum-fragment-methods:/workspace" \
  --container-workdir=/workspace \
  04_reconstruct.slurm
```

To discard all checkpoints and start fresh:

```bash
python 03_solve.py --config config_alanine_sto-3g.yaml --force-resubmit
```

---

## Expected Outputs

After a successful run:

```
data/
  mf_data.pkl             # HF mean-field data (step 1)
  embedding_data.pkl      # EWF fragment metadata + dumpfile path (step 2)
  *.h5                    # Vayesta HDF5 dumpfile with cluster Hamiltonians (step 2)
  solver_results.pkl      # Per-fragment SolverResult objects (step 3)

results/
  fragment_<N>/
    job_id.txt            # QPU job ID (SQD fragments only)
    counts.npy            # Measurement counts from QPU
    circuits/
      circuit_abstract.qpy   # Pre-transpile LUCJ circuit (QPY format)
      circuit_isa.qpy        # Post-transpile ISA circuit (QPY format)
    sqd_diagonalizer/     # SBD post-processing workspace
  summary.json            # Final energy summary
```

`results/summary.json` structure:

```json
{
  "molecule": "alanine",
  "basis": "sto-3g",
  "bath_type": "mp2",
  "fragmentation": "iao",
  "n_fragments": 13,
  "hf_energy": -...,
  "correlation_energy": -...,
  "total_energy": -...,
  "fragments": { "0": {...}, "1": {...}, ... }
}
```

---

## Troubleshooting

**Step 1 fails immediately**
- Check that `system.xyz` exists alongside `config_alanine_sto-3g.yaml`.
- Confirm `workflow.basis` is a valid PySCF basis name.

**Step 2 fails with Vayesta ImportError**
- Vayesta must be installed in the container. Verify with `python -c "import vayesta"`.

**Step 3 fails with `No QRMI quantum resources available`**
- QRMI credentials are not in the container environment. Verify `run_workflow.sh` sourced
  `$QRMI_CREDS` successfully and passed `--container-env=QRMI_JOB_QPU_RESOURCES`.

**Step 3 QPU job is still QUEUED when wall time expires**
- Normal for busy queues. Resubmit step 3 as shown in the checkpoint section above.
  The script resumes from `job_id.txt` and skips resubmission.

**Step 4 fails with `dumpfile not found`**
- The HDF5 dumpfile created in step 2 must still be on disk. Check `data/*.h5`.
  If deleted, re-run step 2 to regenerate it.

---

## References

- Tutorial notebook: `examples/notebook_demos/ewf_sqd_demo/ewf_sqd.ipynb`
- HPC setup instructions: `docs/IBM-SCC-BUILD-GUIDE.md`
- N2 SQD reference workflow: `examples/hpc_demos/N2_hpc_sqd_demo/README.md`
