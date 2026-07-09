# Quantum Fragment Methods Container

FROM --platform=linux/amd64 public.ecr.aws/ubuntu/ubuntu:22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Add labels for documentation
LABEL maintainer="Thaddeus Pellegrini"
LABEL description="Complete environment for quantum fragment methods with PySCF, GPU4PySCF, Qiskit, Qiskit Addon SQD, Vayesta, PyCI, SBD, QRMI, Fulqrum, and Block2. Includes CUDA 12.6 for GPU acceleration on HPC clusters."
LABEL version="2.0"

# Set working directory
WORKDIR /workspace

# ============================================================================
# PHASE 1: Install System Dependencies
RUN apt-get update && apt-get install -y \
    # C++ Build Tools
    gcc \
    g++ \
    gfortran \
    make \
    cmake \
    # Version Control
    git \
    # Download Tools
    wget \
    curl \
    # MPI for Parallel Computing
    openmpi-bin \
    libopenmpi-dev \
    # Linear Algebra Libraries
    libopenblas-dev \
    # OpenMP for Parallelization
    libomp-dev \
    # Python Development
    python3 \
    python3-pip \
    python3-dev \
    # Graphviz for Qiskit visualization
    graphviz \
    libgraphviz-dev \
    # Utilities
    vim \
    nano \
    tree \
    && rm -rf /var/lib/apt/lists/*

# ============================================================================
# PHASE 2: Install CUDA Toolkit 12.6 for GPU Support
# IBM SCC H200 cluster automatically passes through GPUs to containers
# Container needs CUDA Toolkit for GPU4PySCF to work
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    apt-get update && \
    apt-get install -y cuda-toolkit-12-6 && \
    rm cuda-keyring_1.1-1_all.deb && \
    rm -rf /var/lib/apt/lists/*

# Set CUDA environment variables
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
ENV CUDA_HOME="/usr/local/cuda"

# ============================================================================
# PHASE 3: Install Miniconda for Python Environment Management
# Install Miniconda for x86_64 
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh

# Add conda to PATH
ENV PATH="/opt/conda/bin:${PATH}"

# Create .condarc to use only conda-forge and avoid TOS issues
RUN echo "channels:" > /root/.condarc && \
    echo "  - conda-forge" >> /root/.condarc && \
    echo "channel_priority: strict" >> /root/.condarc && \
    echo "auto_activate_base: false" >> /root/.condarc

# Initialize conda for bash
RUN conda init bash

# Create conda environment with Python 3.12 using conda-forge only
# Pin Python to 3.12.* to prevent upgrades to 3.13
RUN conda create -n qfrag-env python=3.12.* --override-channels -c conda-forge -y

# ============================================================================
# PHASE 4: Install Conda Packages
# Pin Python 3.12 to prevent conda from upgrading to 3.13
RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && \
    conda activate qfrag-env && \
    conda install python=3.12.* --override-channels -c conda-forge \
        --no-update-deps -y && \
    conda install --override-channels -c conda-forge \
        cmake=4.2.3 \
        h5py=3.15.1 \
        scipy=1.17.1 \
        xcfun=2.1.1 \
        -y"

# ============================================================================
# PHASE 5: Install Python Packages via pip
RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && \
    conda activate qfrag-env && \
    pip install --no-cache-dir \
        numpy==1.26.4 \
        pyscf==2.9.0 \
        ffsim==0.0.70 \
        qiskit==2.3.0 \
        qiskit-ibm-runtime==0.45.1 \
        qc-pyci==0.6.3 \
        cvxpy>=1.1 \
        matplotlib==3.10.8 \
        pandas==3.0.1 \
        seaborn==0.13.2 \
        jupyter \
        jupyterlab \
        ipykernel \
        ipywidgets \
        pytest \
        pytest-cov"

# Install GPU4PySCF for GPU-accelerated quantum chemistry (CUDA 12.x version)
# Works with IBM SCC H200 GPUs via automatic GPU passthrough
RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && \
    conda activate qfrag-env && \
    pip install --no-cache-dir gpu4pyscf-cuda12x && \
    pip install --no-cache-dir cutensor-cu12"

# Install QRMI (Qiskit Runtime Model Interface)
# Note: QRMI requires Rust compilation which fails on ARM64 emulation
# Use pip install from PyPI instead of building from source
RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && \
    conda activate qfrag-env && \
    pip install --no-cache-dir 'qrmi[ibm]'"

# Install qiskit-addon-sqd Python package from PyPI
RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && \
    conda activate qfrag-env && \
    pip install --no-cache-dir qiskit-addon-sqd==0.12.1"

# Install Fulqrum (Full Quantum Resource Utilization Manager)
# CRITICAL: Must clone with --recurse-submodules to get qiskit-addon-sqd-hpc as a Git submodule
# Fulqrum uses qiskit-addon-sqd-hpc headers which are included as a submodule in the Fulqrum repo
RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && \
    conda activate qfrag-env && \
    git clone --recurse-submodules https://github.com/qiskit-community/fulqrum.git /workspace/fulqrum && \
    cd /workspace/fulqrum && \
    pip install ."

# Install block2 from preview repository (x86_64 wheel)
RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && \
    conda activate qfrag-env && \
    pip install --no-cache-dir block2 --extra-index-url=https://block-hczhai.github.io/block2-preview/pypi/"
# Install high-priority additional libraries (commented out for current build)
# Uncomment to include in future builds:
# RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && \
#     conda activate qfrag-env && \
#     pip install --no-cache-dir \
#         basis-set-exchange \
#         ase \
#         mpi4py \
#         tqdm \
#         py3Dmol"


# ============================================================================
# PHASE 6: Clone and Install Vayesta
# Clone Vayesta and install without building pyscf (use conda version)
RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && \
    conda activate qfrag-env && \
    git clone https://github.com/BoothGroup/Vayesta.git /workspace/Vayesta && \
    cd /workspace/Vayesta && \
    pip install --no-build-isolation --no-deps ."

# ============================================================================
# PHASE 7: Clone PyCI (compilation will be done interactively)
# Clone PyCI repository
RUN git clone https://github.com/theochem/pyci.git /workspace/pyci

# Set environment variables for PyCI compilation
ENV CC=gcc
ENV CXX=g++

# Note: PyCI compilation requires manual steps - see docs/installation.md
# Users should run: cd /workspace/pyci && make && pip install .

# ============================================================================
# PHASE 8: Clone SBD Solver (compilation will be done interactively)
# Clone SBD repository
RUN git clone https://github.com/r-ccs-cms/sbd.git /workspace/sbd

# Note: SBD requires complex setup - see docs/installation.md steps 75-126
# This includes fixing header paths and creating Configuration file

# ============================================================================
# PHASE 9: Install quantum-fragment-methods Package
# Copy the quantum-fragment-methods package into the container
COPY . /workspace/quantum-fragment-methods

# Install quantum-fragment-methods in editable mode
RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && \
    conda activate qfrag-env && \
    cd /workspace/quantum-fragment-methods && \
    pip install -e ."

# ============================================================================
# PHASE 10: Setup Environment Variables
# Set MPI environment variables
ENV OMPI_CXX=g++
ENV OMP_NUM_THREADS=1

# Add workspace to Python path
ENV PYTHONPATH="/workspace:${PYTHONPATH}"

# ============================================================================
# PHASE 11: Create Helper Directories
RUN mkdir -p /workspace/executable && \
    mkdir -p /workspace/data
    #mkdir -p /workspace/results

# ============================================================================
# PHASE 12: Setup Conda Activation in bashrc
RUN echo "source /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate qfrag-env" >> ~/.bashrc && \
    echo "echo ''" >> ~/.bashrc && \
    echo "echo ' Quantum Fragment Methods Environment'" >> ~/.bashrc && \
    echo "echo ' Python packages: PySCF, Qiskit, ffsim, scipy'" >> ~/.bashrc && \
    echo "echo ' Special tools: Vayesta, PyCI, SBD'" >> ~/.bashrc && \
    echo "echo ' Working directory: /workspace'" >> ~/.bashrc && \
    echo "echo ' Architecture: x86_64 via emulation'" >> ~/.bashrc && \
    echo "echo ''" >> ~/.bashrc

# Set default command to bash with conda environment activated
CMD ["/bin/bash"]