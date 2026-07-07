# HPC Deployment Guide

Deploying quantum fragment methods on HPC clusters using containers.

## Prerequisites

- SSH access to HPC cluster
- Slurm account with partition access
- Podman installed locally (for building container)
- Enroot container runtime on HPC cluster

## Step 1: Configure Credentials

```bash
cp .hpc_config.template .hpc_config
# Edit .hpc_config with your HPC details
```

Example `.hpc_config`:
```bash
HPC_USERNAME=your_username
HPC_LOGIN_HOST=login.hpc.example.edu
HPC_PROJECT_DIR=/path/to/your/project
SLURM_PARTITION=gpu
```

## Step 2: Build Container Locally

```bash
cd quantum-fragment-methods

# Clean previous builds
podman stop qfm-jupyter-py312 2>/dev/null || true
podman rm qfm-jupyter-py312 2>/dev/null || true
podman rmi localhost/qfm-hpc:py312 2>/dev/null || true
podman image prune -f

# Build for x86_64 (HPC architecture)
podman build --platform=linux/amd64 -t qfm-hpc:py312 .

# Export to uncompressed tar
podman save qfm-hpc:py312 -o qfm-hpc-py312.tar
```

Build time: 15-30 min on Mac, 5-10 min on Linux x86_64

## Step 3: Transfer to HPC

```bash
source .hpc_config

# Transfer (5-20 minutes for 6 GB)
scp qfm-hpc-py312.tar $HPC_USERNAME@$HPC_LOGIN_HOST:$HPC_PROJECT_DIR/

# Or use rsync for resumable transfer
rsync -avP --partial qfm-hpc-py312.tar $HPC_USERNAME@$HPC_LOGIN_HOST:$HPC_PROJECT_DIR/
```

## Step 4: Convert to Enroot .sqsh Format

**Important:** Run on CPU compute node (has adequate /tmp space).

```bash
# SSH to HPC
ssh $HPC_USERNAME@$HPC_LOGIN_HOST

# Start interactive session on CPU node
srun --partition=cpu --time=01:00:00 --cpus-per-task=4 --mem=32G --pty bash

# Convert using your cluster's conversion script
# Example script path (adjust for your cluster):
/path/to/docker-tar-to-sqsh.sh \
  $HPC_PROJECT_DIR/qfm-hpc-py312.tar \
  $HOME/qfm-hpc-py312.sqsh

# Verify
ls -lh $HOME/qfm-hpc-py312.sqsh

# Exit interactive session
exit

# Optional: Remove tar file
rm $HPC_PROJECT_DIR/qfm-hpc-py312.tar
```

Conversion time: 5-10 minutes. Compresses 6 GB → 2-3 GB.

**Note:** If your cluster doesn't have a conversion script, use:
```bash
enroot import -o $HOME/qfm-hpc-py312.sqsh dockerd://$HPC_PROJECT_DIR/qfm-hpc-py312.tar
```

## Step 5: Verify Container

```bash
# Test Python and package
srun --partition=gpu --gres=gpu:1 \
  --container-image=$HOME/qfm-hpc-py312.sqsh \
  python3 -c "import sys; print(sys.version); import quantum_fragment_methods; print('QFM loaded')"

# Test GPU allocation
srun --partition=gpu --gres=gpu:1 \
  --container-image=$HOME/qfm-hpc-py312.sqsh \
  bash -c 'echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"'

# Test data mount (replace with your actual project path)
srun --partition=gpu --gres=gpu:1 \
  --container-image=$HOME/qfm-hpc-py312.sqsh \
  --container-mounts=/path/to/your/project:/data \
  ls -la /data
```

## Step 6: Compile PyCI and SBD (Optional)

PyCI and SBD require native HPC compilation for optimal performance.

```bash
# Start interactive session with writable container
srun --partition=cpu --time=01:00:00 \
  --container-image=$HOME/qfm-hpc-py312.sqsh \
  --container-writable \
  --pty bash

# Inside container - activate conda environment
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

# Exit
exit
```

## Using Container in Jobs

### Interactive Job

```bash
# Replace /path/to/your/project with your actual project path
srun --partition=gpu --gres=gpu:1 \
  --container-image=$HOME/qfm-hpc-py312.sqsh \
  --container-mounts=/path/to/your/project:/data \
  --pty bash
```

### Batch Job

```bash
#!/bin/bash
#SBATCH --job-name=qfm-job
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --container-image=$HOME/qfm-hpc-py312.sqsh
#SBATCH --container-mounts=/path/to/your/project:/data

python /data/my_script.py
```

## References

- Enroot Documentation: https://github.com/NVIDIA/enroot
- Pyxis (Slurm plugin): https://github.com/NVIDIA/pyxis