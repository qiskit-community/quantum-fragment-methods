# HPC Deployment Guide

Deploy quantum fragment methods on an HPC cluster using containers.
The guide is written for **IBM SCC H200** but the steps are general —
cluster-specific values are called out in notes wherever they differ.

## Prerequisites

- SSH access to the cluster login node
- Slurm account with GPU and CPU partition access
- Podman installed locally (for building the container image)
- `.hpc_config` configured (see Step 1)

---

## Step 1: Configure credentials

```bash
cp .hpc_config.template .hpc_config
# Edit .hpc_config with your cluster details
```

Full `.hpc_config` example for IBM SCC H200
(note `export` — required so variables propagate to subprocesses):

```bash
# HPC Login
export HPC_USERNAME=your_username
export HPC_LOGIN_HOST=scc-login1.pok.ibm.com   # IBM SCC; change for other clusters

# Storage locations
export HPC_HOME=/u/your_username
export HPC_PROJECT=/gpfs/scc6000/proj/your_account/your_username
export HPC_PROJECT_DIR=/gpfs/scc6000/proj/your_account/your_username

# Slurm
export SLURM_PARTITION=gpu-mid          # IBM SCC GPU partition
export SLURM_ACCOUNT=your_account

# Jupyter (optional — for remote notebook sessions)
export JUPYTER_PORT=8899
export CONDA_ENV_NAME=qfrag-env

# Monitoring (IBM SCC only)
export GRAFANA_URL=https://scc-grafana.pok.ibm.com:3000/login
export JOB_MONITOR_URL=https://scc-grafana.pok.ibm.com:3000/d/scc-h200-job-monitor
export DOCS_URL=https://pages.github.ibm.com/ETE-DC/slurmh200cluster/
```

`source .hpc_config` before any local `scp`/`rsync`/`ssh` command.
These variables are **not** set on the remote cluster — use literal paths there.

---

## Step 2: Build container locally

```bash
cd quantum-fragment-methods

# Clean previous builds
podman stop qfm-jupyter-py312 2>/dev/null || true
podman rm   qfm-jupyter-py312 2>/dev/null || true
podman rmi  localhost/qfm-hpc:py312 2>/dev/null || true
podman image prune -f

# Build for x86_64 (HPC architecture — required even on Apple Silicon)
podman build --platform=linux/amd64 -t qfm-hpc:py312 .

# Export to tar
podman save qfm-hpc:py312 -o qfm-hpc-py312.tar
```

Build time: ~30–45 min on Apple Silicon Mac, ~5–10 min on Linux x86_64.
Output file: `qfm-hpc-py312.tar` (~15 GB uncompressed).

---

## Step 3: Transfer to HPC

```bash
source .hpc_config

# rsync preferred over scp — --partial allows resuming interrupted transfers
rsync -avP --partial qfm-hpc-py312.tar \
  $HPC_USERNAME@$HPC_LOGIN_HOST:$HPC_PROJECT/
```

Transfer time: 5–20 min depending on network speed.

---

## Step 4: Convert to Enroot `.sqsh` format

Enroot is only available on compute nodes, not login nodes.
Run the conversion inside an `srun` session.

> **Memory requirement:** a ~15 GB tar needs `--mem=64G` —
> 32G will OOM-kill `mksquashfs` mid-conversion.

```bash
# SSH to cluster
ssh $HPC_USERNAME@$HPC_LOGIN_HOST

# Start interactive session on a CPU node
# IBM SCC: partition=cpu-mid  |  generic cluster: partition=cpu (or your CPU partition)
srun --partition=cpu-mid --time=01:00:00 --cpus-per-task=8 --mem=64G --pty bash

# Convert (use literal GPFS path — $HPC_PROJECT is not set inside srun)
# IBM SCC — use the cluster-provided conversion script:
/gpfs/scc6000/setup/scripts/docker-tar-to-sqsh.sh \
  /gpfs/scc6000/proj/your_account/your_username/qfm-hpc-py312.tar \
  $HOME/qfm-hpc-py312.sqsh

# Generic cluster — use enroot directly:
# enroot import -o $HOME/qfm-hpc-py312.sqsh \
#   dockerd:///gpfs/.../qfm-hpc-py312.tar

# Verify
ls -lh $HOME/qfm-hpc-py312.sqsh

exit

# Optional: remove tar to free shared filesystem space
rm /gpfs/scc6000/proj/your_account/your_username/qfm-hpc-py312.tar
```

Conversion time: ~10–15 min. Compresses ~15 GB → ~6–7 GB.

If the job is killed mid-conversion (OOM or wall-time), a partial file
remains. Remove it before retrying:

```bash
rm $HOME/qfm-hpc-py312.sqsh
```

---

## Step 5: Verify container

The container's default Python is the base conda environment — **not**
`qfrag-env` where QFM is installed. Always activate `qfrag-env` explicitly.

```bash
# Test QFM package import
# IBM SCC: --partition=gpu-mid  |  generic: --partition=gpu (or your GPU partition)
srun --partition=gpu-mid --gres=gpu:1 \
  --container-image=$HOME/qfm-hpc-py312.sqsh \
  bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate qfrag-env && \
    python3 -c \"import sys; print(sys.version); \
                 import quantum_fragment_methods; print('QFM loaded')\""

# Test GPU allocation
srun --partition=gpu-mid --gres=gpu:1 \
  --container-image=$HOME/qfm-hpc-py312.sqsh \
  bash -c 'echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"'

# Test shared filesystem mount
srun --partition=gpu-mid --gres=gpu:1 \
  --container-image=$HOME/qfm-hpc-py312.sqsh \
  --container-mounts=/gpfs/scc6000/proj/your_account/your_username:/data \
  ls -la /data
```

---

## Step 6: Compile PyCI and SBD (optional)

PyCI and SBD require native compilation for full performance.
They are cloned into the container but not pre-compiled.

> **Important:** `--container-writable` writes into the container's
> ephemeral layer. Compiled binaries are **lost when the job ends** —
> copy them to the shared filesystem (GPFS) before exiting.

```bash
# Start interactive session with writable container
srun --partition=cpu-mid --time=01:00:00 \
  --container-image=$HOME/qfm-hpc-py312.sqsh \
  --container-writable \
  --pty bash

# Activate conda environment
source /opt/conda/etc/profile.d/conda.sh
conda activate qfrag-env

# --- Compile PyCI ---
cd /workspace/pyci
make
pip install .
python -c "import pyci; print('PyCI installed')"

# --- Compile SBD ---
cd /workspace
ABS_PATH=$(pwd)

# Fix header paths in include directory
cd sbd/include
find . -type f -name "*.h" \
  -exec sed -i "s|sbd/|${ABS_PATH}/sbd/include/sbd/|g" {} +
cd ../..

# Create compilation directory
cp -r sbd/apps/chemistry_tpb_selected_basis_diagonalization Compilation_and_Test
cd Compilation_and_Test

# Fix include paths in source files
sed -i 's|#include "sbd/|#include "/workspace/sbd/include/sbd/|g' main.cc

# Create Configuration file
cat > Configuration << 'EOF'
SBD_PATH=../
CCCOM=mpicxx
CCFLAGS= -std=c++17 -fopenmp -O3
SYSLIB= -lopenblas -lgomp
EOF

# Compile
make

# Install executable inside container
cd /workspace
mkdir -p executable
cp Compilation_and_Test/diag executable/
ls -lh /workspace/executable/diag

# Persist compiled binary to GPFS before exiting
cp /workspace/executable/diag \
  /gpfs/scc6000/proj/your_account/your_username/diag

exit
```

---

## Using the container in jobs

### Interactive job

```bash
# IBM SCC
srun --partition=gpu-mid --gres=gpu:1 \
  --container-image=$HOME/qfm-hpc-py312.sqsh \
  --container-mounts=/gpfs/scc6000/proj/your_account/your_username:/data \
  --pty bash
```

Once inside, activate the conda environment:

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate qfrag-env
```

### Batch job

```bash
#!/bin/bash
#SBATCH --job-name=qfm-job
#SBATCH --partition=gpu-mid
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --container-image=/path/to/qfm-hpc-py312.sqsh
#SBATCH --container-mounts=/gpfs/scc6000/proj/your_account/your_username:/data

bash -c "source /opt/conda/etc/profile.d/conda.sh && \
         conda activate qfrag-env && \
         python /data/my_script.py"
```

For the full 4-step EWF+SQD pipeline see
`examples/hpc_demos/alanine_ewf_sqd_demo/run_workflow.sh` —
it handles all `sbatch` flags, QRMI credential injection, and job chaining.

---

## IBM SCC quick reference

| Resource | Value |
|---|---|
| Login node | `scc-login1.pok.ibm.com` |
| GPU partition | `gpu-mid` |
| CPU partition | `cpu-mid` |
| Project GPFS | `/gpfs/scc6000/proj/your_account/your_username` |
| Home directory | `/u/your_username` |
| Conda env | `qfrag-env` |
| Conversion script | `/gpfs/scc6000/setup/scripts/docker-tar-to-sqsh.sh` |
| Grafana monitor | https://scc-grafana.pok.ibm.com:3000/login |
| Cluster docs | https://pages.github.ibm.com/ETE-DC/slurmh200cluster/ |

---

## References

- [Enroot documentation](https://github.com/NVIDIA/enroot)
- [Pyxis Slurm plugin](https://github.com/NVIDIA/pyxis)
