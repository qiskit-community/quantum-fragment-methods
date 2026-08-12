# HPC SQD Demo — N2 SQD Workflow

A 2-step workflow demonstrating Sample-based Quantum Diagonalization (SQD) on the
IBM SCC H200 cluster using [QRMI](https://github.com/qiskit-community/qrmi) as the
quantum backend. The system is N2 in the STO-3G basis — a small, tractable example
for verifying the end-to-end pipeline.

## How it works

```
Step 1 (CPU/GPU)          Step 2 (GPU + QPU)
─────────────────         ─────────────────────────────────────────
01_setup_molecule.py  →   02_run_sqd.py
  - Define N2               - Load molecule data
  - Hartree-Fock (GPU)      - Initialize QRMI backend (env vars)
  - Hamiltonian integrals   - Build LUCJ circuit from CCSD amplitudes
  - Save pickle             - Submit to IBM Quantum via QRMI
                            - Retrieve counts (checkpoint-aware)
                            - SBD post-processing (MPI diag)
                            - Compare with FCI
```

QRMI reads QPU credentials from environment variables — no secrets in config files.
`run_workflow.sh` sources `.qrmi_config` from GPFS on the host and injects each
variable into the container via `--container-env`.

## Files

| File | Description |
|---|---|
| `01_setup_molecule.py` | N2 setup, Hartree-Fock (GPU), Hamiltonian integrals → `molecule_data.pkl` |
| `02_run_sqd.py` | SQD solver via `QRMIBackend` |
| `01_meanfield.slurm` | Slurm script for step 1 (`gpu-mid` partition) |
| `02_sqd_solve.slurm` | Slurm script for step 2 (`gpu-mid` partition, 37h wall time) |
| `run_workflow.sh` | Submit both jobs with `--dependency=afterok` chain |
| `config_N2_sto-3g.yaml` | SQD and QPU configuration (no credentials) |

---

## Setup

Before running, complete these two one-time steps. Everything else — partition
selection, credential injection, job chaining — is handled automatically by
`run_workflow.sh`.

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

Example (IBM SCC):

```bash
export HPC_PROJECT=/gpfs/scc6000/proj/your_account/your_username
export CONTAINER_IMAGE=$HPC_PROJECT/qfm-hpc-py312-v3.sqsh
export SLURM_PARTITION=gpu-mid
export SLURM_ACCOUNT=your_account
export QRMI_CREDS=$HPC_PROJECT/.qrmi_config
```

`.hpc_config` is gitignored — it will never be committed to the repo.

### Step 2 — QRMI credentials file

Create a credentials file on the shared filesystem with your IBM Quantum API key.
**Never commit this file or put it inside the repo directory.**

```bash
# Run this on the cluster (or copy the file there via scp/rsync)
cat > $HPC_PROJECT/.qrmi_config << 'EOF'
export QRMI_JOB_QPU_RESOURCES=ibm_pittsburgh
export QRMI_JOB_QPU_TYPES=qiskit-runtime-service
export ibm_pittsburgh_QRMI_IBM_QRS_ENDPOINT=https://quantum.cloud.ibm.com/api/v1
export ibm_pittsburgh_QRMI_IBM_QRS_IAM_ENDPOINT=https://iam.cloud.ibm.com
export ibm_pittsburgh_QRMI_IBM_QRS_IAM_APIKEY=your_apikey        # ← your key here
export ibm_pittsburgh_QRMI_IBM_QRS_SERVICE_CRN=your_crn          # ← your CRN here
EOF
chmod 600 $HPC_PROJECT/.qrmi_config
```

> **Note on variable naming:** the prefix before `_QRMI_` must exactly match the
> value of `QRMI_JOB_QPU_RESOURCES`. If you use a different backend (e.g.
> `ibm_torino`), rename all `ibm_pittsburgh_QRMI_*` variables to `ibm_torino_QRMI_*`
> and update `QRMI_JOB_QPU_RESOURCES` to match. Also update `backend_name` in
> `config_N2_sto-3g.yaml`.

---

## Quickstart

### 1. Build the container image (on your local machine)

```bash
cd <repo-root>

podman build \
  --platform=linux/amd64 \
  --tag qfm-hpc-py312:v3.0 \
  --file Containerfile \
  . 2>&1 | tee build_v3.log
```

This takes 20–40 minutes. The image includes CUDA 12.8, GPU4PySCF, QRMI, Fulqrum
(`qiskit-addon-sqd-hpc`), and the SBD `diag` binary compiled for x86-64 — all baked
in at `/opt/` so the `/workspace` bind-mount cannot shadow them.

### 2. Save and transfer to the shared filesystem

```bash
# Save as tar
podman save qfm-hpc-py312:v3.0 -o qfm-hpc-py312-v3.tar

# Transfer (rsync resumes if interrupted)
source .hpc_config
rsync -avP --partial qfm-hpc-py312-v3.tar \
  $HPC_USERNAME@$HPC_LOGIN_HOST:$HPC_PROJECT/
```

### 3. Convert tar → sqsh on a compute node

Enroot is only available on compute nodes, not on login nodes. Check your cluster
docs for the conversion tool — on IBM SCC it is:

```bash
# Start an interactive session (no container needed)
srun --partition=$SLURM_PARTITION --time=01:00:00 --cpus-per-task=4 --mem=32G --pty bash

# IBM SCC conversion script (adjust path for your cluster)
/gpfs/scc6000/setup/scripts/docker-tar-to-sqsh.sh \
  $HPC_PROJECT/qfm-hpc-py312-v3.tar \
  $HPC_PROJECT/qfm-hpc-py312-v3.sqsh

exit
```

Alternatively, if your cluster has `enroot` available:
```bash
enroot import --output $HPC_PROJECT/qfm-hpc-py312-v3.sqsh \
  dockerarchive://$HPC_PROJECT/qfm-hpc-py312-v3.tar
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
cd $HPC_PROJECT/quantum-fragment-methods/examples/hpc_demos/N2_hpc_sqd_demo
bash run_workflow.sh
```

`run_workflow.sh` automatically sources `.hpc_config`, injects QRMI credentials,
and chains the two Slurm jobs. Step 2 runs only after step 1 succeeds.

---

## Monitoring jobs

```bash
# Watch queue status (refreshes every 10 seconds)
watch -n 10 squeue -u $USER

# Tail step 1 log (mean-field)
tail -f logs/01_meanfield_<JOBID>.out

# Tail step 2 log (SQD) — only appears after step 1 completes
tail -f logs/02_sqd_<JOBID>.out

# Check errors
tail -f logs/01_meanfield_<JOBID>.err
tail -f logs/02_sqd_<JOBID>.err
```

`run_workflow.sh` prints the exact log paths after submission:

```
Submitted step 1 (meanfield): job 81400
Submitted step 2 (SQD solve): job 81401 (depends on 81400)

Monitor with:  squeue -u $USER
Logs:          logs/01_meanfield_81400.out
               logs/02_sqd_81401.out
```

---

## Checkpoint-aware workflow

Step 2 is checkpoint-aware. If the QPU job is still queued when the Slurm wall time
expires, resubmit `02_sqd_solve.slurm` — it resumes from the saved `job_id.txt`
and `counts.npy` without resubmitting to the QPU.

```bash
# Resubmit step 2 only (reuses existing QPU counts)
source .hpc_config
source $QRMI_CREDS
cd $HPC_PROJECT/quantum-fragment-methods/examples/hpc_demos/N2_hpc_sqd_demo
bash run_workflow.sh   # re-chains both steps safely, or sbatch directly:

# --or-- submit step 2 alone:
sbatch \
  --partition="$SLURM_PARTITION" \
  --container-image="$CONTAINER_IMAGE" \
  --container-mounts="$HPC_PROJECT/quantum-fragment-methods:/workspace" \
  --container-workdir=/workspace \
  --container-env=NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  --container-env=QRMI_JOB_QPU_RESOURCES \
  --container-env=QRMI_JOB_QPU_TYPES \
  $(env | grep _QRMI_ | cut -d= -f1 | sed 's/^/--container-env=/') \
  02_sqd_solve.slurm
```

Use `--force-resubmit` to discard all checkpoints and start fresh:

```bash
python 02_run_sqd.py --config config_N2_sto-3g.yaml --force-resubmit
```

---

## Expected results

Verified run on IBM SCC H200, `ibm_pittsburgh` QPU.

| Method | Total Energy (Ha) | Error vs FCI |
|---|---|---|
| Hartree-Fock | −107.41953245 | +81.40 kcal/mol |
| FCI | −107.54930096 | — |
| **SQD** | **−107.54928061** | **0.0128 kcal/mol** |

SQD error is well below the 1 kcal/mol chemical accuracy threshold with only
1000 shots and 3 SBD iterations.

Results are saved to `results/sqd_results/summary.json`.

---

## Container contents (v3.0)

| Component | Version / Notes |
|---|---|
| Base | Ubuntu 22.04 x86_64 |
| CUDA | 12.8 + `libnvidia-compute-590` |
| Python | 3.12 (conda-forge) |
| PySCF | 2.9.0 |
| GPU4PySCF | cuda12x wheel |
| Qiskit | 2.3.0 |
| qiskit-ibm-runtime | 0.45.1 |
| qiskit-addon-sqd | 0.12.1 |
| QRMI | `qrmi[ibm]` (PyPI) |
| Fulqrum + sqd-hpc | `/opt/fulqrum` (git submodule) |
| SBD `diag` | `/opt/executable/diag` (compiled linux-cpu preset) |
| Vayesta | `/opt/Vayesta` |
| block2 | preview wheel |
