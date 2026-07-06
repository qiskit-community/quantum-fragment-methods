# Connecting to HPC with Jupyter Frontend

Step-by-step guide for connecting VS Code (or VS Code-based IDEs) to HPC cluster and running Jupyter notebooks on GPU compute nodes.

## Prerequisites

### Required Access
- SSH access to HPC cluster login node
- Slurm account with GPU partition access
- Valid credentials (username, password/key)

### Local Tools (VS Code)
- VS Code or VS Code-based IDE with Remote-SSH extension
- SSH client configured
- Terminal access

### HPC Cluster Information Needed
- Login node hostname (e.g., `scc-login1.bu.edu`)
- Available GPU partitions (e.g., `gpu-mid`, `gpu-short`)
- GPU types available (e.g., H200, A100)
- Account/QoS settings (if required)

### Verify Local SSH
```bash
# Test SSH connection
ssh <username>@<hpc-login-host>

# Example:
ssh your_username@your-cluster.example.com
```

If connection succeeds, you have the required access.

## Step 1: Configure HPC Credentials

Create your private HPC configuration file:

```bash
# In project root directory
cp .hpc_config.template .hpc_config
```

Edit `.hpc_config` with your credentials:

```bash
# HPC Login Information
HPC_USERNAME=your_username
HPC_LOGIN_HOST=your-cluster.example.com

# Slurm Configuration
SLURM_PARTITION=gpu-mid
SLURM_ACCOUNT=your_account  # If required
SLURM_QOS=normal            # If required

# Jupyter Configuration
JUPYTER_PORT=8899

# Conda Environment
CONDA_ENV_NAME=qfm-hpc
```

**Important:** `.hpc_config` is gitignored and will never be committed. Keep your credentials private.

## Step 2: Connect VS Code to HPC Login Node

### Open VS Code Terminal

1. Open VS Code (or your VS Code-based IDE)
2. Open integrated terminal:
   - Use menu: Terminal → New Terminal
   - Or keyboard shortcut (if configured in your IDE)

### SSH to Login Node

```bash
# Load your config (optional - for reference)
source .hpc_config

# Connect to HPC
ssh $HPC_USERNAME@$HPC_LOGIN_HOST

# Or directly:
ssh your_username@your-cluster.example.com
```

### Verify Connection

Once connected, verify you're on the login node:

```bash
# Check hostname
hostname

# Check Slurm availability
sinfo

# Check your account
sacctmgr show user $USER
```

You should see cluster information and available partitions.

**Note:** Do NOT run heavy computations on login nodes. They are for job submission and light tasks only.

## Step 2.5: IBM SCC Container System

**IBM SCC uses Enroot + Pyxis** - NVIDIA's HPC container stack integrated directly into Slurm.

### Key Features

✅ **No installation needed** - Enroot is pre-installed on all compute nodes
✅ **GPU passthrough** - Automatic NVIDIA GPU access in containers
✅ **InfiniBand support** - High-speed networking for multi-node jobs
✅ **GPFS mounts** - Direct access to cluster storage
✅ **Slurm integration** - Use `--container-image=` flag with `srun`/`sbatch`

### Quick Test

```bash
# Test GPU access in container (from login node)
srun --partition=gpu-debug --gres=gpu:1 \
  --container-image=nvcr.io#nvidia/cuda:12.6.0-base-ubuntu22.04 \
  nvidia-smi
```

**Note:** Enroot is only on compute nodes, not login nodes. Always use `srun` or `sbatch` to run containers.

### Documentation

Full IBM SCC container guide: https://pages.github.ibm.com/ETE-DC/slurmh200cluster/software/containers/

### Disconnect from HPC

```bash
# Type exit or press Ctrl+D
exit
```

## Step 3: Set Up Environment on HPC

**Important:** Do this ON THE LOGIN NODE (after SSH connection).

### Understanding Storage

Check your cluster's storage locations (varies by cluster):
- Home directory: Usually `~` or `/home/$USER` - Persistent, backed up
- Project space: Shared team storage (if available)
- Scratch: Temporary, per-job storage

Recommendation: Clone to home directory for persistence.

### Clone Project Repository

```bash
# Navigate to your home directory
cd ~

# Clone the repository
git clone https://github.com/qiskit-community/quantum-fragment-methods.git
cd quantum-fragment-methods
```

### Choose Installation Method

Based on Step 2.5 results, choose the appropriate method:

---

### Option A: Container Runtime (Recommended)

**Prerequisites:** Apptainer, Singularity, Podman, or Enroot available

**Recommended Workflow:** Build container locally on your Mac/laptop, then transfer to HPC.

#### Why Build Locally?

1. **Faster development:** Build on your local machine while HPC is busy
2. **No HPC resources:** Doesn't consume HPC compute time
3. **Easier debugging:** Full control over build environment
4. **Works during maintenance:** Can prepare while HPC is offline

#### Step A1: Build Container Locally (Mac/Linux)

**On your local machine:**

```bash
cd quantum-fragment-methods

# Clean any old builds (ensures reproducibility)
podman stop qfm-jupyter-py312 2>/dev/null || true
podman rm qfm-jupyter-py312 2>/dev/null || true
podman rmi localhost/qfm-hpc:py312 2>/dev/null || true
podman image prune -f

# Build container for x86_64 (HPC architecture)
podman build --platform=linux/amd64 -t qfm-hpc:py312 .
```

**Build time:** 15-30 minutes on Mac (due to x86_64 emulation), 5-10 minutes on Linux x86_64

**What gets installed:**
- ✅ Python 3.12 + conda environment
- ✅ PySCF, Qiskit, ffsim, QRMI, Fulqrum
- ✅ Vayesta, block2
- ✅ JupyterLab, matplotlib, pandas
- ⏸️ PyCI (cloned, needs manual compilation)
- ⏸️ SBD (cloned, needs manual compilation)

#### Step A2: Export Container

```bash
# Export to tarball
podman save qfm-hpc:py312 -o qfm-hpc-py312.tar

# Compress (reduces size by ~50%)
gzip qfm-hpc-py312.tar

# Check size
ls -lh qfm-hpc-py312.tar.gz
```

**Expected size:** 1.5-2.5 GB compressed

#### Step A3: Transfer to HPC

**Using your HPC credentials from `.hpc_config`:**

```bash
# Load config
source .hpc_config

# Transfer container
scp qfm-hpc-py312.tar.gz $HPC_USERNAME@$HPC_LOGIN_HOST:/gpfs/scc6000/proj/toffoli/$HPC_USERNAME/

# Or specify path directly:
scp qfm-hpc-py312.tar.gz your_username@your-cluster.example.com:~/
```

**Transfer time:** 5-15 minutes depending on network speed

#### Step A4: Load Container on HPC

**SSH to HPC and load the container:**

```bash
# SSH to HPC
ssh $HPC_USERNAME@$HPC_LOGIN_HOST

# Navigate to where you transferred the file
cd /gpfs/scc6000/proj/toffoli/$HPC_USERNAME/  # Or: cd ~/

# Load container with Enroot (IBM SCC uses Enroot+Pyxis)
enroot import -o qfm-hpc-py312.sqsh docker-archive://qfm-hpc-py312.tar.gz

# Or with Apptainer/Singularity
apptainer build qfm-hpc-py312.sif docker-archive://qfm-hpc-py312.tar.gz
# Or: singularity build qfm-hpc-py312.sif docker-archive://qfm-hpc-py312.tar.gz
```

**Note:** The `.sqsh` format (Enroot) or `.sif` format (Apptainer/Singularity) is optimized for HPC use.

#### Step A5: Compile PyCI and SBD on HPC

**Important:** PyCI and SBD must be compiled on the HPC for optimal performance.

**Start interactive session with container:**

```bash
# For Enroot (IBM SCC)
srun --partition=cpu-debug --container-image=./qfm-hpc-py312.sqsh --pty bash

# For Apptainer
srun --partition=cpu-debug apptainer shell qfm-hpc-py312.sif

# For Singularity
srun --partition=cpu-debug singularity shell qfm-hpc-py312.sif
```

**Inside the container, compile PyCI:**

```bash
# Activate conda environment
source /opt/conda/etc/profile.d/conda.sh
conda activate qfrag-env

# Compile PyCI
cd /workspace/pyci
make
pip install .

# Verify
python -c "import pyci; print('PyCI installed!')"
```

**Inside the container, compile SBD:**

```bash
# SBD requires complex setup - see detailed instructions
cd /workspace/sbd

# Follow steps from docs/installation.md lines 75-126:
# 1. Fix header paths in source files
# 2. Create Configuration file
# 3. Compile with make
# 4. Install with pip

# Verify
python -c "import sbd; print('SBD installed!')"
```

**Why compile on HPC?**
- ✅ Native x86_64 (no emulation overhead)
- ✅ Faster compilation (2-5 min vs 15-30 min on Mac)
- ✅ Optimized for target hardware
- ✅ Avoids platform-specific issues

#### Test Container

```bash
# Test with Enroot
srun --container-image=./qfm-hpc-py312.sqsh python3 -c "import quantum_fragment_methods; print('Success!')"

# Test with Apptainer
apptainer exec qfm-hpc-py312.sif python3 -c "import quantum_fragment_methods; print('Success!')"

# Test with Singularity
singularity exec qfm-hpc-py312.sif python3 -c "import quantum_fragment_methods; print('Success!')"
```

**Advantages:**
- ✅ Complete isolation and reproducibility
- ✅ All dependencies bundled (Python, C++, libraries)
- ✅ Easy to share and deploy
- ✅ GPU passthrough supported
- ✅ Build once, run anywhere
- ✅ No dependency conflicts

**Note:** Container build only needs to be done once. Updates can be made by rebuilding locally and re-transferring.

#### Using Container in Slurm Jobs

**Interactive job:**
```bash
srun --partition=gpu-mid --gres=gpu:1 \
  --container-image=/gpfs/scc6000/proj/toffoli/$HPC_USERNAME/qfm-hpc-py312.sqsh \
  --container-mounts=/gpfs/scc6000/proj/toffoli/$HPC_USERNAME:/data \
  --pty bash
```

**Batch job template:**
```bash
#!/bin/bash
#SBATCH --job-name=qfm-job
#SBATCH --partition=gpu-mid
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --container-image=/gpfs/scc6000/proj/toffoli/$HPC_USERNAME/qfm-hpc-py312.sqsh
#SBATCH --container-mounts=/gpfs/scc6000/proj/toffoli/$HPC_USERNAME:/data

python /data/my_script.py
```

**Key Enroot+Pyxis flags:**
- `--container-image=` - Path to `.sqsh` file or registry URI
- `--container-mounts=` - Bind GPFS directories (comma-separated)
- `--container-writable` - Make container filesystem writable
- `--container-workdir=` - Set working directory inside container

**Full IBM SCC container documentation:**
https://pages.github.ibm.com/ETE-DC/slurmh200cluster/software/containers/

---

### Option B: Python Virtual Environment (Fallback)

**Prerequisites:** Python 3.9+, GCC/G++, pip

#### Install pip (if needed)

```bash
# Check if pip is available
which pip3 || python3 -m pip --version

# If pip is missing, install it
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python3 get-pip.py --user
rm get-pip.py

# Verify
python3 -m pip --version
```

#### Install cmake (if needed)

```bash
# Check if cmake is available
which cmake || python3 -m pip install --user cmake

# Verify
cmake --version
```

#### Create Virtual Environment

```bash
# Create environment
python3 -m venv ~/qfm-env

# Activate environment
source ~/qfm-env/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install package
cd ~/quantum-fragment-methods
pip install -e .

# Install Jupyter
pip install jupyterlab
```

#### Verify Installation

```bash
# Check package
python -c "import quantum_fragment_methods; print('Package installed!')"

# Check Jupyter
jupyter --version
```

**Advantages:**
- Works on any system with Python
- No admin privileges required
- Quick setup

**Disadvantages:**
- Must manage dependencies manually
- Less reproducible across systems
- May have version conflicts

---

### Option C: Conda Environment (Alternative)

**Prerequisites:** Conda or Miniconda available

```bash
# Load conda (if needed - cluster-specific)
module load anaconda3  # Or: module load miniconda3

# Create environment
conda create -n qfm-hpc python=3.10 -y

# Activate environment
conda activate qfm-hpc

# Install package
cd ~/quantum-fragment-methods
pip install -e .

# Install Jupyter
conda install -c conda-forge jupyterlab -y
```

#### Verify Installation

```bash
# Check package
python -c "import quantum_fragment_methods; print('Package installed!')"

# Check Jupyter
jupyter --version
```

**Advantages:**
- Good dependency management
- Popular in scientific computing
- Easy environment switching

**Disadvantages:**
- Requires conda installation
- Can be slow
- Large disk space usage

---

### Recommendation

1. **If container runtime available** → Use Option A (containers)
2. **If waiting for container installation** → Use Option B (venv) temporarily
3. **If conda already on cluster** → Use Option C (conda)

**Note:** Environment setup only needs to be done once. The environment persists across sessions.