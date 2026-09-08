# Quantum Fragment Methods Container

FROM --platform=linux/amd64 public.ecr.aws/ubuntu/ubuntu:22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Add labels for documentation
LABEL maintainer="Thaddeus Pellegrini"
LABEL description="Quantum fragment methods: PySCF, GPU4PySCF, Qiskit, qiskit-addon-sqd, Vayesta, sbd-eigensolver, QRMI, Fulqrum. CUDA 12.8 for GPU acceleration on HPC clusters."
LABEL version="4.0"

# Set working directory
WORKDIR /workspace

# ============================================================================
# PHASE 1: System Dependencies
#
# - gcc/g++/gfortran/make: C/C++/Fortran build toolchain for Python extensions
#   (PySCF, Vayesta, sbd-eigensolver pybind11 extension)
# - git/wget/curl: version control and download tools
# - openmpi-bin/libopenmpi-dev: MPI toolchain; mpicc on PATH lets
#   sbd-eigensolver's setup.py auto-discover MPI without MPI_HOME
# - libopenblas-dev: BLAS/LAPACK for PySCF and SBD Davidson kernels
# - libomp-dev: OpenMP runtime for sbd-eigensolver CPU backend
# - python3/python3-pip/python3-dev: system Python (used only to bootstrap
#   Miniconda in Phase 3; conda Python takes over afterwards)
# - vim/nano/tree: interactive debugging utilities
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    gfortran \
    make \
    git \
    wget \
    curl \
    openmpi-bin \
    libopenmpi-dev \
    libopenblas-dev \
    libomp-dev \
    python3 \
    python3-pip \
    python3-dev \
    vim \
    nano \
    tree \
    && rm -rf /var/lib/apt/lists/*

# ============================================================================
# PHASE 2: CUDA Toolkit + NVIDIA userspace libraries
#
# IBM SCC H100 nodes run driver 590.48.01 (requires CUDA >= 12.8 runtime).
# The host OS has no NVIDIA userspace libs — the container must supply them.
#
# - cuda-toolkit-12-8: provides libcudart.so 12.8 (compatible with driver 590.x)
# - libnvidia-compute-590: provides libcuda.so stub at the exact driver version
#   so the kernel module and userspace ABI match.
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    apt-get update && \
    apt-get install -y cuda-toolkit-12-8 libnvidia-compute-590 && \
    rm cuda-keyring_1.1-1_all.deb && \
    rm -rf /var/lib/apt/lists/*

# Set CUDA environment variables
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
ENV CUDA_HOME="/usr/local/cuda"

# ============================================================================
# PHASE 3: Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh

ENV PATH="/opt/conda/bin:${PATH}"

RUN echo "channels:" > /root/.condarc && \
    echo "  - conda-forge" >> /root/.condarc && \
    echo "channel_priority: strict" >> /root/.condarc && \
    echo "auto_activate_base: false" >> /root/.condarc

RUN conda init bash

# Pin Python to 3.12.* — conda would otherwise upgrade to 3.13
RUN conda create -n qfrag-env python=3.12.* --override-channels -c conda-forge -y

# ============================================================================
# PHASE 4: Conda packages
#
# - cmake: required by Vayesta's --no-build-isolation pip install
# - h5py: HDF5 bindings used by EWF dumpfile (ewf_dumpfile.h5)
# - scipy: linear algebra utilities throughout
# (xcfun removed — no imports in this codebase)
RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && \
    conda activate qfrag-env && \
    conda install python=3.12.* --override-channels -c conda-forge \
        --no-update-deps -y && \
    conda install --override-channels -c conda-forge \
        cmake=4.2.3 \
        h5py=3.15.1 \
        scipy=1.17.1 \
        -y"

# ============================================================================
# PHASE 5: Python packages via pip
#
# Install order matters for sbd-eigensolver: pybind11 and mpi4py must be
# present before the sbd-eigensolver sdist is built so setup.py can find
# pybind11 include paths and link mpi4py headers.
#
# Packages removed vs prior image:
#   - qc-pyci: no imports in codebase
#   - cvxpy: no imports in codebase
#   - seaborn: no imports in codebase
#   - pandas: demo-notebook only; add locally if needed
#   - block2: no imports in codebase
RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && \
    conda activate qfrag-env && \
    pip install --no-cache-dir \
        numpy==1.26.4 \
        pyscf==2.9.0 \
        ffsim==0.0.70 \
        qiskit==2.3.0 \
        qiskit-ibm-runtime==0.45.1 \
        matplotlib==3.10.8 \
        jupyter \
        jupyterlab \
        ipykernel \
        ipywidgets \
        pytest \
        pytest-cov"

# GPU4PySCF — cuda12x wheel bundles its own CUDA 12.x runtime (ships 12.9,
# compatible with driver 590.x). libcuda.so is provided by libnvidia-compute-590.
RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && \
    conda activate qfrag-env && \
    pip install --no-cache-dir gpu4pyscf-cuda12x && \
    pip install --no-cache-dir cutensor-cu12"

# QRMI — requires Rust compilation; install from PyPI pre-built wheel
RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && \
    conda activate qfrag-env && \
    pip install --no-cache-dir 'qrmi[ibm]'"

# qiskit-addon-sqd — >=0.13.1 required for SPMD sci_solver support used by
# sbd-eigensolver's solve_sci_batch across MPI ranks
RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && \
    conda activate qfrag-env && \
    pip install --no-cache-dir 'qiskit-addon-sqd>=0.13.1'"

# sbd-eigensolver — Python bindings for SBD (Selected Basis Diagonalization).
# Replaces the former cmake build of r-ccs-cms/sbd that produced /opt/executable/diag.
# setup.py auto-discovers MPI via mpicc (on PATH from Phase 1 openmpi-bin) and
# BLAS via BLAS_LIBS=openblas (default). CPU backend only; GPU (Thrust) requires
# NVHPC nvc++ — add NVHPC_HOME and rebuild for the GPU container image.
RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && \
    conda activate qfrag-env && \
    pip install --no-cache-dir pybind11 mpi4py && \
    pip install --no-cache-dir sbd-eigensolver"

# Smoke test — confirms CPU backend compiled and loads correctly
RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && \
    conda activate qfrag-env && \
    python -c \"import sbd; print('sbd backends:', sbd.available_backends()); assert 'cpu' in sbd.available_backends()\""

# ============================================================================
# PHASE 6: Vayesta (EWF embedding framework)
# Placed before Fulqrum so a Fulqrum build failure cannot block Vayesta.
RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && \
    conda activate qfrag-env && \
    git clone https://github.com/BoothGroup/Vayesta.git /opt/Vayesta && \
    cd /opt/Vayesta && \
    pip install --no-build-isolation --no-deps ."

# Fulqrum — alternative sci_solver backend (fulqrum classical_backend option
# in fermion_local.py). The qiskit-addon-sqd-hpc submodule has no pyproject.toml
# so only the top-level fulqrum package is installed.
RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && \
    conda activate qfrag-env && \
    git clone --recurse-submodules https://github.com/qiskit-community/fulqrum.git /opt/fulqrum && \
    cd /opt/fulqrum && \
    pip install ."

# ============================================================================
# PHASE 7: quantum-fragment-methods package
# Baked-in copy at image build time so the editable install works before
# /workspace is bind-mounted at runtime.
COPY . /opt/quantum-fragment-methods

RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && \
    conda activate qfrag-env && \
    cd /opt/quantum-fragment-methods && \
    pip install -e ."

# ============================================================================
# PHASE 8: Runtime environment
ENV OMPI_CXX=g++
ENV OMP_NUM_THREADS=1

# /workspace takes precedence over the baked-in copy at runtime (GPFS mount)
ENV PYTHONPATH="/workspace:${PYTHONPATH}"

# ============================================================================
# PHASE 9: Helper directories
RUN mkdir -p /opt/data

# ============================================================================
# PHASE 10: Shell setup
RUN echo "source /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate qfrag-env" >> ~/.bashrc && \
    echo "echo ''" >> ~/.bashrc && \
    echo "echo ' Quantum Fragment Methods Environment'" >> ~/.bashrc && \
    echo "echo ' Python packages: PySCF, Qiskit, ffsim, scipy'" >> ~/.bashrc && \
    echo "echo ' Special tools: Vayesta, sbd-eigensolver, Fulqrum, QRMI'" >> ~/.bashrc && \
    echo "echo ' Working directory: /workspace'" >> ~/.bashrc && \
    echo "echo ''" >> ~/.bashrc

CMD ["/bin/bash"]
