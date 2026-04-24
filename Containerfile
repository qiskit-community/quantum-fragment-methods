# Quantum Fragment Methods Container
# Quantum Fragment Methods Container

FROM --platform=linux/amd64 ubuntu:22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Add labels for documentation
LABEL maintainer="Thaddeus Pellegrini"
LABEL description="Complete environment for quantum fragment methods with PySCF, Qiskit, Vayesta, PyCI, and SBD"
LABEL version="1.0"

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
# PHASE 2: Install Miniconda for Python Environment Management
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

# Create conda environment with Python 3.11.15 using conda-forge only
RUN conda create -n qfrag-env python=3.11.15 --override-channels -c conda-forge -y

# ============================================================================
# PHASE 3: Install Conda Packages
RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && \
    conda activate qfrag-env && \
    conda install --override-channels -c conda-forge \
        cmake=4.2.3 \
        h5py=3.15.1 \
        scipy=1.17.1 \
        xcfun=2.1.1 \
        -y"

# ============================================================================
# PHASE 4: Install Python Packages via pip
RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && \
    conda activate qfrag-env && \
    pip install --no-cache-dir \
        numpy==1.26.4 \
        pyscf==2.9.0 \
        ffsim==0.0.70 \
        qiskit==2.3.0 \
        qiskit-ibm-runtime==0.45.1 \
        qiskit-addon-sqd==0.12.1 \
        qc-pyci==0.6.3 \
        matplotlib==3.10.8 \
        pandas==3.0.1 \
        seaborn==0.13.2 \
        jupyter \
        jupyterlab \
        ipykernel \
        ipywidgets \
        pytest \
        pytest-cov"

# Install block2 from preview repository (x86_64 wheel)
RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && \
    conda activate qfrag-env && \
    pip install --no-cache-dir block2 --extra-index-url=https://block-hczhai.github.io/block2-preview/pypi/"

# ============================================================================
# PHASE 5: Clone and Install Vayesta
# Clone Vayesta and install without building pyscf (use conda version)
RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && \
    conda activate qfrag-env && \
    git clone https://github.com/BoothGroup/Vayesta.git /workspace/Vayesta && \
    cd /workspace/Vayesta && \
    pip install --no-build-isolation --no-deps ."

# ============================================================================
# PHASE 6: Clone PyCI (compilation will be done interactively)
# Clone PyCI repository
RUN git clone https://github.com/theochem/pyci.git /workspace/pyci

# Set environment variables for PyCI compilation
ENV CC=gcc
ENV CXX=g++

# Note: PyCI compilation requires manual steps - see docs/installation.md
# Users should run: cd /workspace/pyci && make && pip install .

# ============================================================================
# PHASE 7: Clone SBD Solver (compilation will be done interactively)
# Clone SBD repository
RUN git clone https://github.com/r-ccs-cms/sbd.git /workspace/sbd

# Note: SBD requires complex setup - see docs/installation.md steps 75-126
# This includes fixing header paths and creating Configuration file

# ============================================================================
# PHASE 8: Install quantum-fragment-methods Package
# Copy the quantum-fragment-methods package into the container
COPY . /workspace/quantum-fragment-methods

# Install quantum-fragment-methods in editable mode
RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && \
    conda activate qfrag-env && \
    cd /workspace/quantum-fragment-methods && \
    pip install -e ."

# ============================================================================
# PHASE 9: Setup Environment Variables
# Set MPI environment variables
ENV OMPI_CXX=g++
ENV OMP_NUM_THREADS=1

# Add workspace to Python path
ENV PYTHONPATH="/workspace:${PYTHONPATH}"

# ============================================================================
# PHASE 10: Create Helper Directories
RUN mkdir -p /workspace/executable && \
    mkdir -p /workspace/data
    #mkdir -p /workspace/results

# ============================================================================
# PHASE 11: Setup Conda Activation in bashrc
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
