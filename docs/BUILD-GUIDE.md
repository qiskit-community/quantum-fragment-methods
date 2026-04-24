# Build Guide

This guide provides instructions for setting up Quantum Fragment Methods using Podman containers.

## Table of Contents

- [Quick Start](#quick-start)
- [Container Installation](#container-installation)
  - [Prerequisites](#prerequisites)
  - [Building the Container](#building-the-container)
  - [Running the Container](#running-the-container)
  - [VSCode Integration](#vscode-integration)
  - [Container Management](#container-management)
- [Troubleshooting](#troubleshooting)
- [HPC Usage](#hpc-usage)

---

## Quick Start

The fastest way to get started is using the pre-configured Podman container:

```bash
# Install Podman (macOS)
brew install podman

# Initialize Podman machine (macOS only)
podman machine init
podman machine start

# Build the container
cd quantum-fragment-methods
podman build -t quantum-fragment-methods:latest .

# Start Jupyter server
podman run -d --name qfm-jupyter \
  -p 8888:8888 \
  -v $(pwd):/workspace/quantum-fragment-methods:Z \
  quantum-fragment-methods:latest \
  bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate qfrag-env && jupyter lab --ip=0.0.0.0 --allow-root --no-browser --NotebookApp.token=''"

# Access Jupyter Lab at: http://localhost:8888
```

Then connect VSCode to `http://127.0.0.1:8888` (see [VSCode Integration](#vscode-integration) below).

---

## Container Installation

### Prerequisites

#### Install Podman

**macOS:**
```bash
brew install podman

```

**Linux:**
```bash
# Ubuntu/Debian
sudo apt-get install podman

# Fedora
sudo dnf install podman
```

#### Initialize Podman Machine (macOS only)

```bash
# Create and start the Podman machine with sufficient resources
# Quantum chemistry calculations require substantial memory (12-16 GB recommended)
podman machine init --memory 16384 --cpus 8 --disk-size 100

# Or if already initialized, configure resources:
podman machine stop
podman machine set --memory 16384 --cpus 8
podman machine start

# Verify it's running
podman machine list
```

**⚠️ Important: Memory Requirements**

Quantum chemistry calculations (especially CCSD and quantum circuit optimization) are memory-intensive:
- **Minimum**: 8 GB for small molecules
- **Recommended**: 12-16 GB for typical workflows (alanine, small peptides)
- **Large systems**: 32+ GB for proteins

If your Jupyter kernel dies during computation, increase Podman machine memory:
```bash
podman machine stop
podman machine set --memory 16384  # 16 GB
podman machine start
```

### Building the Container

Navigate to the project directory and build the image:

```bash
cd quantum-fragment-methods
podman build -t quantum-fragment-methods:latest .
```

This will take 10-20 minutes as it installs all dependencies.

### Running the Container

#### Start Jupyter Server for VSCode

```bash
podman run -d --name qfm-jupyter \
  -p 8888:8888 \
  -v $(pwd):/workspace/quantum-fragment-methods:Z \
  quantum-fragment-methods:latest \
  bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate qfrag-env && jupyter lab --ip=0.0.0.0 --allow-root --no-browser --NotebookApp.token=''"
```

**Check that it's running:**
```bash
podman ps
podman logs qfm-jupyter
```

**Access at:** `http://localhost:8888`

#### Alternative: Interactive Shell

If you need to run commands directly in the container:

```bash
podman run -it --rm \
  -v $(pwd):/workspace/quantum-fragment-methods:Z \
  quantum-fragment-methods:latest
```

#### Command Explanation

- `-it` - Interactive mode with terminal
- `--rm` - Automatically remove container when it exits
- `-d` - Run in detached mode (background)
- `-p 8888:8888` - Map port 8888 (Jupyter) from container to host
- `-v $(pwd):/workspace/quantum-fragment-methods:Z` - Mount current directory into container
  - `$(pwd)` - Your current directory on the host
  - `/workspace/quantum-fragment-methods` - Where it appears in the container
  - `:Z` - SELinux label (needed on some Linux systems)
- `quantum-fragment-methods:latest` - The image to use

### VSCode Integration

#### Prerequisites

Install these VSCode extensions:
- **Jupyter** (by Microsoft)
- **Python** (by Microsoft)

#### Connect to Jupyter Server

1. Open any `.ipynb` file in VSCode
2. Click on the kernel selector in the top-right (shows "Select Kernel")
3. Choose "Select Another Kernel..."
4. Select "Existing Jupyter Server..."
5. Enter: `http://127.0.0.1:8888`

#### Test Your Setup

Create a new notebook and verify all packages work:

```python
# Test imports
import pyscf
import qiskit
from qiskit_ibm_runtime import QiskitRuntimeService
import ffsim
import scipy

print("✅ All packages loaded successfully!")
print(f"PySCF version: {pyscf.__version__}")
print(f"Qiskit version: {qiskit.__version__}")
```

**For HPC/SSH workflows and remote execution, see [HPC-JUPYTER.md](HPC-JUPYTER.md)**

### Container Management

#### Open Interactive Shell in Running Container

To explore the container filesystem or run commands interactively:

```bash
# Connect to running container
podman exec -it qfm-jupyter bash

# Once inside, you can:
# - Navigate: cd /workspace/quantum-fragment-methods
# - List files: ls -la
# - View structure: tree /workspace/quantum-fragment-methods -L 2
# - Check Python: python --version
# - Test imports: python -c "from quantum_fragment_methods import embedding"
# - Exit: exit
```

**Quick inspection commands (without entering shell):**
```bash
# View directory structure
podman exec qfm-jupyter tree /workspace/quantum-fragment-methods/quantum_fragment_methods -L 2

# Check Python packages
podman exec qfm-jupyter conda list

# Run a Python command
podman exec qfm-jupyter python -c "import pyscf; print(pyscf.__version__)"
```

#### List Running Containers
```bash
podman ps
```

#### List All Containers (including stopped)
```bash
podman ps -a
```

#### View Container Logs
```bash
# View all logs
podman logs qfm-jupyter

# Follow logs in real-time
podman logs -f qfm-jupyter
```

#### Stop a Container
```bash
podman stop qfm-jupyter
```

#### Start a Stopped Container
```bash
podman start qfm-jupyter
```

#### Remove a Container
```bash
podman rm qfm-jupyter
```

#### List Images
```bash
podman images
```

#### Remove an Image
```bash
podman rmi quantum-fragment-methods:latest
```

#### Clean Up Everything
```bash
# Remove all stopped containers
podman container prune

# Remove unused images
podman image prune -a
```

### What's Included in the Container

The container includes all dependencies:

**System Tools:**
- GCC, G++, Gfortran (C++ compilation)
- CMake
- OpenMPI (parallel computing)
- OpenBLAS (linear algebra)
- Git

**Python Environment:**
- Miniconda with Python 3.10
- Conda environment: `qfrag-env` (auto-activated)

**Python Packages:**
- PySCF (quantum chemistry)
- Qiskit + IBM Runtime (quantum computing)
- Qiskit Addon SQD (sample-based quantum diagonalization)
- ffsim (fermionic quantum simulation)
- scipy (scientific computing)
- qc-pyci (configuration interaction)
- NumPy (numerical computing)
- Jupyter Lab & Notebook
- IPython widgets

**Special Packages:**
- Vayesta (embedding methods - installed)
- PyCI (configuration interaction - cloned to `/workspace/pyci`, **requires manual compilation**)
- SBD solver (selected basis diagonalization - cloned to `/workspace/sbd`, **requires manual compilation**)

**Note on block2:**
- block2 (DMRG solver) is **not available for ARM64 (Apple Silicon)** due to Intel MKL dependency
- On x86_64 HPC systems, block2 will install automatically
- For ARM64 development, use alternative solvers or build on x86_64 HPC

### Compiling PyCI and SBD in the Container

PyCI and SBD are cloned but not compiled during the build. To compile them:

#### Connect to Running Container

```bash
podman exec -it qfm-jupyter bash
```

#### Compile PyCI

```bash
cd /workspace/pyci
make clean
make
pip install .

# Test installation
python -c "import pyci; print('PyCI installed')"
```

#### Compile SBD

```bash
# Set absolute path
cd /workspace
ABS_PATH=$(pwd)
echo "ABS_PATH=$ABS_PATH"

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

# Test installation (use --oversubscribe if you have fewer than 8 cores)
mpirun --allow-run-as-root --oversubscribe -np 8 -x OMP_NUM_THREADS=1 ./diag \
  --fcidump fcidump_Fe4S4.txt \
  --adetfile AlphaDets.txt \
  --method 0 \
  --block 10 \
  --iteration 4 \
  --tolerance 1.0e-4 \
  --adet_comm_size 2 \
  --bdet_comm_size 2 \
  --task_comm_size 2 \
  --init 0 \
  --shuffle 0 \
  --carryover_type 3 \
  --carryover_threshold 1.0e-3 \
  --carryover_adetfile carryover_adet.txt \
  --rdm 1

**Note on MPI Warnings:** You may see many "Read -1, expected..." error messages during the test. These are harmless OpenMPI communication warnings that occur in containerized environments (especially on macOS with Podman). The solver is working correctly if you see Davidson iteration progress with converging energy values. These warnings can be safely ignored.

# Install executable
cd /workspace
mkdir -p executable
cp Compilation_and_Test/diag executable/

# Verify
ls -lh /workspace/executable/diag
```

**Note:** If you need more CPU cores for the SBD test, you can configure the Podman machine:
```bash
# Exit container and stop it
exit
podman stop qfm-jupyter

# Stop Podman machine and increase CPUs
podman machine stop
podman machine set --cpus 8
podman machine start

# Restart container
podman start qfm-jupyter
```

#### Exit Container

```bash
exit
```

Your Jupyter server continues running with PyCI and SBD now available.

---


## Troubleshooting

### Memory and Performance Issues

#### Kernel Dying During Computation

**Symptom**: Jupyter kernel crashes or restarts during long-running calculations (CCSD, circuit optimization)

**Cause**: Insufficient memory allocated to Podman machine

**Solution**:
```bash
# Check current memory allocation
podman machine inspect | grep -A 5 "Resources"

# Stop machine and increase memory
podman machine stop
podman machine set --memory 16384  # 16 GB (adjust based on your system)
podman machine start

# Restart your container
podman start qfm-jupyter
```

**Memory Guidelines**:
- Small molecules (< 10 atoms): 8 GB minimum
- Medium molecules (10-20 atoms, alanine): 12-16 GB recommended
- Large systems (peptides, proteins): 32+ GB, run on HPC

### Container Issues

#### Port Already in Use

If port 8888 is already in use:
```bash
# Use a different port (e.g., 8889)
podman run -it --rm \
  -p 8889:8888 \
  -v $(pwd):/workspace/quantum-fragment-methods:Z \
  quantum-fragment-methods:latest \
  jupyter lab --ip=0.0.0.0 --allow-root --no-browser
```

Then access at `http://127.0.0.1:8889`

## Tips for Development

1. **Use volume mounts** to keep your code on the host
2. **Use `--rm`** for temporary containers
3. **Use `-d` (detached)** for long-running Jupyter sessions
4. **Commit changes** to Git on your host (not in container)
5. **Rebuild image** when updating dependencies

## HPC Usage

For running on HPC systems with SSH port forwarding, Slurm/PBS integration, and remote troubleshooting, see **[HPC.md](HPC.md)**.

## Next Steps

- Try the example notebooks in `examples/`
- Tackle a hard protein from the `suggestions` list
- Read the main [README.md](../README.md) 
- Check [HPC-JUPYTER.md](HPC-JUPYTER.md) for HPC workflows

## Support

For issues with:
- **Container setup**: Check the Container Installation section
- **Podman itself**: See [Podman documentation](https://podman.io/)
- **Package functionality**: See main project documentation
