# IBM SCC H200 Deployment Guide

Deploying quantum fragment methods on the IBM SCC H200 cluster using containers.

## Prerequisites

- SSH access to `scc-login1.pok.ibm.com`
- Slurm account with `gpu-mid` or `cpu-mid` partition access
- Podman installed locally (for building container on Mac/Linux)
- `.hpc_config` configured (see Step 1)

## Step 1: Configure Credentials

```bash
cp .hpc_config.template .hpc_config
# Edit .hpc_config with your SCC details
```

Example `.hpc_config` (`export` is required so variables propagate to subprocesses):

```bash
# HPC Login Information (IBM SCC Cluster)
# PRIVATE - DO NOT COMMIT TO GIT
export HPC_USERNAME=your_username
export HPC_LOGIN_HOST=scc-login1.pok.ibm.com

# Storage Locations
export HPC_HOME=/u/your_username
export HPC_PROJECT=/gpfs/scc6000/proj/your_account/your_username
export HPC_PROJECT_DIR=/gpfs/scc6000/proj/your_account/your_username

# Slurm Configuration
export SLURM_PARTITION=gpu-mid
export SLURM_ACCOUNT=your_account

# Jupyter Configuration
export JUPYTER_PORT=8899

# Conda Environment
export CONDA_ENV_NAME=qfrag-env

# Monitoring URLs
export GRAFANA_URL=https://scc-grafana.pok.ibm.com:3000/login
export JOB_MONITOR_URL=https://scc-grafana.pok.ibm.com:3000/d/scc-h200-job-monitor
export DOCS_URL=https://pages.github.ibm.com/ETE-DC/slurmh200cluster/
```

Always `source .hpc_config` before running local `scp`/`rsync`/`ssh` commands.
Variables are not available on the remote cluster — use literal paths for on-cluster commands.

## Step 2: Build Container Locally

```bash
cd quantum-fragment-methods

# Clean previous builds
podman stop qfm-jupyter-py312 2>/dev/null || true
podman rm qfm-jupyter-py312 2>/dev/null || true
podman rmi localhost/qfm-hpc:py312 2>/dev/null || true
podman image prune -f

# Build for x86_64 (SCC architecture)
podman build --platform=linux/amd64 -t qfm-hpc:py312 .

# Export to uncompressed tar
podman save qfm-hpc:py312 -o qfm-hpc-py312.tar
```

Build time: ~30-45 min on Mac (Apple Silicon). Output: `qfm-hpc-py312.tar` (~15 GB).

## Step 3: Transfer to SCC

```bash
source .hpc_config

rsync -avP --partial qfm-hpc-py312.tar $HPC_USERNAME@$HPC_LOGIN_HOST:$HPC_PROJECT/
```

`rsync` is preferred over `scp` — `--partial` allows resuming interrupted transfers.

## Step 4: Convert to Enroot .sqsh Format

Enroot is only available on compute nodes, not login nodes. For a ~15 GB image,
`--mem=64G` is required — 32G will OOM-kill `mksquashfs`.

```bash
# SSH to SCC
ssh $HPC_USERNAME@$HPC_LOGIN_HOST

# Start interactive session on CPU compute node
# Note: partition is cpu-mid (not cpu); 64G memory required for a ~15 GB image
srun --partition=cpu-mid --time=01:00:00 --cpus-per-task=8 --mem=64G --pty bash

# Convert tar to sqsh (use literal paths — $HPC_PROJECT is not set on the cluster)
/gpfs/scc6000/setup/scripts/docker-tar-to-sqsh.sh \
  /gpfs/scc6000/proj/your_account/your_username/qfm-hpc-py312.tar \
  $HOME/qfm-hpc-py312.sqsh

# Verify
ls -lh $HOME/qfm-hpc-py312.sqsh

exit

# Optional: remove tar to free GPFS space
rm /gpfs/scc6000/proj/your_account/your_username/qfm-hpc-py312.tar
```

Conversion time: ~10-15 min. Compresses ~15 GB → ~6.7 GB.

If the job is killed mid-conversion (e.g. OOM), a partial file will remain.
Remove it before retrying:

```bash
rm $HOME/qfm-hpc-py312.sqsh
```

**Note:** `enroot import` is not available on login nodes. If used as an alternative
to the conversion script, it must also be run inside an `srun` session on a compute node.

## Step 5: Verify Container

The container's default Python is base conda — **not** the `qfrag-env` environment
where QFM is installed. Always activate `qfrag-env` explicitly.

```bash
# Test Python and QFM package
srun --partition=gpu-mid --gres=gpu:1 \
  --container-image=$HOME/qfm-hpc-py312.sqsh \
  bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate qfrag-env && \
    python3 -c \"import sys; print(sys.version); import quantum_fragment_methods; print('QFM loaded')\""

# Test GPU allocation
srun --partition=gpu-mid --gres=gpu:1 \
  --container-image=$HOME/qfm-hpc-py312.sqsh \
  bash -c 'echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"'

# Test GPFS mount
srun --partition=gpu-mid --gres=gpu:1 \
  --container-image=$HOME/qfm-hpc-py312.sqsh \
  --container-mounts=/gpfs/scc6000/proj/your_account/your_username:/data \
  ls -la /data
```

## Step 6: Compile PyCI and SBD (Optional)

PyCI and SBD require native HPC compilation for optimal performance.

**Note:** `--container-writable` writes into the container's ephemeral layer.
Compiled binaries are lost when the job ends — copy them to GPFS before exiting.

```bash
# Start interactive session with writable container
srun --partition=cpu-mid --time=01:00:00 \
  --container-image=$HOME/qfm-hpc-py312.sqsh \
  --container-writable \
  --pty bash

# Activate conda environment
source /opt/conda/etc/profile.d/conda.sh
conda activate qfrag-env

# Compile PyCI
cd /workspace/pyci
make
pip install .
python -c "import pyci; print('PyCI installed')"

# Compile SBD
cd /workspace
ABS_PATH=$(pwd)

# Fix header paths in include directory
cd sbd/include
find . -type f -name "*.h" -exec sed -i "s|sbd/|${ABS_PATH}/sbd/include/sbd/|g" {} +
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

# Install executable
cd /workspace
mkdir -p executable
cp Compilation_and_Test/diag executable/

# Verify
ls -lh /workspace/executable/diag

# Persist to GPFS before exiting
cp /workspace/executable/diag /gpfs/scc6000/proj/your_account/your_username/diag

exit
```

## Using Container in Jobs

### Interactive Job

```bash
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

### Batch Job

```bash
#!/bin/bash
#SBATCH --job-name=qfm-job
#SBATCH --partition=gpu-mid
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --container-image=$HOME/qfm-hpc-py312.sqsh
#SBATCH --container-mounts=/gpfs/scc6000/proj/your_account/your_username:/data

bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate qfrag-env && python /data/my_script.py"
```

## Cluster Quick Reference

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

## References

- Enroot Documentation: https://github.com/NVIDIA/enroot
- Pyxis (Slurm plugin): https://github.com/NVIDIA/pyxis
