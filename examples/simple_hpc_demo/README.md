# Simple HPC Demo — N2 SQD Workflow

A 2-step workflow demonstrating Sample-based Quantum Diagonalization (SQD) on the
IBM SCC H200 cluster using [QRMI](https://github.com/qiskit-community/qrmi) as the
quantum backend. The system is N2 in the STO-3G basis — a small, tractable example
suitable for verifying the end-to-end pipeline.

## How it works

```
Step 1 (CPU/GPU)          Step 2 (GPU + QPU)
─────────────────         ─────────────────────────────────────────
01_setup_molecule.py  →   02_run_sqd.py
  - Define N2               - Load molecule data
  - Hartree-Fock            - Initialize QRMI backend (env vars)
  - Hamiltonian             - Build LUCJ circuit from CCSD amplitudes
  - Save pickle             - Submit to IBM Quantum via QRMI
                            - Retrieve counts (checkpoint-aware)
                            - SBD post-processing
                            - Compare with FCI
```

QRMI reads QPU credentials from environment variables — no secrets in config files.
On SCC, the Slurm SPANK plugin injects these automatically when a QPU resource is
allocated. See [QRMI credentials](#qrmi-credentials) for manual setup.

## Files

| File | Description |
|---|---|
| `01_setup_molecule.py` | N2 setup, Hartree-Fock, Hamiltonian integrals → `molecule_data.pkl` |
| `02_run_sqd.py` | SQD solver via `QRMIBackend` |
| `01_meanfield.slurm` | Slurm script for step 1 (`gpu-mid` partition) |
| `02_sqd_solve.slurm` | Slurm script for step 2 (`gpu-mid` partition) |
| `run_workflow.sh` | Submit both jobs with `--dependency=afterok` chain |
| `config_N2_sto-3g.yaml` | SQD and QPU configuration (no credentials) |

## Prerequisites

1. Container deployed: `$HOME/qfm-hpc-py312.sqsh`
   (see [`docs/IBM-SCC-BUILD-GUIDE.md`](../../docs/IBM-SCC-BUILD-GUIDE.md))
2. SBD compiled and persisted to GPFS (see [Compile SBD](#compile-sbd) below)
3. QRMI credentials file created on GPFS (see [QRMI Credentials](#qrmi-credentials) below)
4. Project synced to GPFS (see [Sync to GPFS](#sync-to-gpfs) below)

## Compile SBD

SBD must be compiled natively on the cluster. Run once inside a writable container session:

```bash
srun --partition=cpu-mid --time=01:00:00 --container-image=$HOME/qfm-hpc-py312.sqsh --container-writable --pty bash

source /opt/conda/etc/profile.d/conda.sh && conda activate qfrag-env

cd /workspace
ABS_PATH=$(pwd)
cd sbd/include && find . -type f -name "*.h" -exec sed -i "s|sbd/|${ABS_PATH}/sbd/include/sbd/|g" {} + && cd ../..
cp -r sbd/apps/chemistry_tpb_selected_basis_diagonalization Compilation_and_Test
cd Compilation_and_Test
sed -i 's|#include "sbd/|#include "/workspace/sbd/include/sbd/|g' main.cc
cat > Configuration << 'EOF'
SBD_PATH=../
CCCOM=mpicxx
CCFLAGS= -std=c++17 -fopenmp -O3
SYSLIB= -lopenblas -lgomp
EOF
make
cd /workspace && mkdir -p executable && cp Compilation_and_Test/diag executable/

# Persist to GPFS before exiting (survives container exit)
mkdir -p /gpfs/scc6000/proj/your_account/your_username/executable
cp /workspace/executable/diag /gpfs/scc6000/proj/your_account/your_username/executable/diag
exit
```

## QRMI Credentials

On SCC with the Slurm SPANK plugin, QPU credentials are injected automatically
when you request a QPU resource (`--gres=qpu:ibm_torino`). No credentials are
stored in config files or committed to the repo.

Without SPANK, create a credentials file on GPFS (**never commit this file**):

```bash
cat > /gpfs/scc6000/proj/your_account/your_username/.qrmi_config << 'EOF'
export QRMI_JOB_QPU_RESOURCES=ibm_torino
export QRMI_JOB_QPU_TYPES=qiskit-runtime-service
export ibm_torino_QRMI_IBM_QRS_ENDPOINT=https://quantum.cloud.ibm.com/api/v1
export ibm_torino_QRMI_IBM_QRS_IAM_ENDPOINT=https://iam.cloud.ibm.com
export ibm_torino_QRMI_IBM_QRS_IAM_APIKEY=your_apikey
export ibm_torino_QRMI_IBM_QRS_SERVICE_CRN=your_crn
EOF
chmod 600 /gpfs/scc6000/proj/your_account/your_username/.qrmi_config
```

Then add to `02_sqd_solve.slurm` after `conda activate qfrag-env`:
```bash
source /gpfs/scc6000/proj/your_account/your_username/.qrmi_config
```

## Sync to GPFS

From your local machine, sync the project to GPFS:

```bash
source .hpc_config
rsync -avP --partial --exclude='*.tar' --exclude='*.sqsh' --exclude='__pycache__' --exclude='.git' . $HPC_USERNAME@$HPC_LOGIN_HOST:$HPC_PROJECT/quantum-fragment-methods/
```

## Usage

```bash
# From the SCC login node:
cd /gpfs/scc6000/proj/your_account/your_username/quantum-fragment-methods/examples/simple_hpc_demo

bash run_workflow.sh
```

This submits both jobs and chains them — step 2 runs only after step 1 succeeds.

## Checkpoint-aware workflow

Step 2 is checkpoint-aware. If the QPU job is still queued when the Slurm
job times out, simply resubmit `02_sqd_solve.slurm` — it will resume from
the saved `job_id.txt` without resubmitting to the QPU.

Use `--force-resubmit` to discard checkpoints and start fresh:

```bash
python 02_run_sqd.py --config config_N2_sto-3g.yaml --force-resubmit
```

## Expected results

| Method | Energy (Ha) |
|---|---|
| Hartree-Fock | -107.41953245 |
| FCI | -107.54930096 |
| SQD | -107.54930034 |
| SQD vs FCI error | 0.62 μHa (0.0004 kcal/mol) |

Results saved to `results/summary.json`.
