# quantum-fragment-methods

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Status](https://img.shields.io/badge/status-research%20release-green.svg)](https://github.com/qiskit-community/quantum-fragment-methods)

## About

A Python package for quantum chemistry tailored toward fragment-based embedding and quantum algorithms applied to macromolecule simulations. This codebase provides open access to the methodology from the paper *"Molecular Quantum Computations on a Protein"*, a collaborative effort between IBM Quantum and the Cleveland Clinic Foundation.

## Overview

The Quantum Fragment Methods framework enables scalable, high-accuracy quantum simulations of large molecular systems — such as peptides and proteins — by embedding them into quantum subspaces. The package features:

- **Embedder objects**: EWF with configurable bath and truncation parameters
- **Rule-based solver assignment**: Priority-based automatic solver selection per fragment
- **Quantum solvers**: SQD (production), ext-SQD (planned)
- **Classical solvers**: FCI, CCSD
- **HPC-ready**: Multi-system Slurm pipelines with checkpoint-aware resumption

## Documentation

- **[Local Build Guide](docs/BUILD-GUIDE.md)** — laptop / workstation setup with Podman or Docker
- **[HPC Build Guide](docs/IBM-SCC-BUILD-GUIDE.md)** — cluster deploy (Slurm, transferring images, native SBD)
- **[Tutorial](examples/notebook_demos/ewf_sqd_demo/ewf_sqd.ipynb)** — interactive Jupyter notebook with usage examples
- **[Contributing Guide](docs/contributing.md)** — guidelines for contributors
- **[Examples](examples/)** — additional example guides

## Quick Start

### Local vs HPC

| Target | Guide | Typical use |
|--------|--------|-------------|
| **Local** | [BUILD-GUIDE.md](docs/BUILD-GUIDE.md) | Interactive demos (e.g. H₂ / N₂ / glycine notebooks), Jupyter on a laptop/workstation |
| **HPC** | [IBM-SCC-BUILD-GUIDE.md](docs/IBM-SCC-BUILD-GUIDE.md) | Larger fragments / MPI SBD, container transferred to the cluster |

Both paths share the same package and YAML configs. Prefer local for development and small demos; use HPC when walltime, cores, or memory exceed a single workstation.

1. **Install**: Follow the [Local Build Guide](docs/BUILD-GUIDE.md) or [HPC Build Guide](docs/IBM-SCC-BUILD-GUIDE.md)
2. **Configure**: Create a `.env` file (copy from `.env.example`) and set QPU credentials
3. **Learn**: Work through the [Tutorial](examples/notebook_demos/ewf_sqd_demo/ewf_sqd.ipynb) notebook
4. **Explore**: Check out [Examples](examples/) for more use cases

### Credentials — `.env` files (notebooks) / `.qrmi_config` (HPC)

Credentials are **never** stored in YAML configs, notebooks, or scripts.

**Notebooks** — copy the `.env.example` in the demo directory:

```bash
# e.g. for sqd_demos:
cp examples/notebook_demos/sqd_demos/.env.example \
   examples/notebook_demos/sqd_demos/.env
# fill in your IBM Quantum IAM API key and CRN, then load in the notebook:
#   from dotenv import load_dotenv; load_dotenv()
```

**HPC pipelines** — follow the `.qrmi_config` setup in
[IBM-SCC-BUILD-GUIDE.md](docs/IBM-SCC-BUILD-GUIDE.md). The `run_workflow.sh`
script sources these vars and injects them into the container via
`--container-env` (export the variable first, then pass the name — no `=value`
in the flag).

### Configuration File

Each demo has its own self-contained YAML config. All tunable parameters live
there — no credentials.

- **Workflow settings**: Basis set and molecular geometry (`xyz_file`, `basis`)
- **Embedder parameters**: Bath type, truncation threshold, fragmentation scheme
- **Solver selection**: `orbital_threshold` controls the FCI / SQD boundary
- **QPU configuration**: Backend name (`provider: qrmi`), shots, DD / twirling options
- **SQD settings**: LUCJ reps (`lucj.n_reps`), iterations, batches, `classical_backend`

Starting points:

| Demo | Config |
|------|--------|
| Glycine notebook (laptop) | `examples/notebook_demos/ewf_sqd_demo/config_glycine_sto-3g.yaml` *(local only — gitignored)* |
| Alanine HPC pipeline | [`examples/hpc_demos/alanine_ewf_sqd_demo/config_alanine_sto-3g.yaml`](examples/hpc_demos/alanine_ewf_sqd_demo/config_alanine_sto-3g.yaml) |
| N₂ SQD standalone | [`examples/notebook_demos/sqd_demos/config_N2_sto-3g_demo.yaml`](examples/notebook_demos/sqd_demos/config_N2_sto-3g_demo.yaml) |
| 20S proteasome ΔΔE_bind | [`examples/hpc_demos/proteasome_ewf_sqd_demo/`](examples/hpc_demos/proteasome_ewf_sqd_demo/) |

See [`quantum_fragment_methods/config/qpu_config.yaml.example`](quantum_fragment_methods/config/qpu_config.yaml.example) for the full annotated template.

### SQD classical backend (`classical_backend`)

SQD uses a LUCJ circuit (routed by `ffsim.qiskit.lucj_pass_manager`) submitted
via `QRMIBackend`, then diagonalizes selected CI subspaces with SBD.

Under `sqd:` in your config, set:

```yaml
sqd:
  # python = stock qiskit-addon-sqd; hpc = C++ qiskit-addon-sqd-hpc bindings
  classical_backend: hpc   # or: python
```

| Value | Preprocessing | When to use |
|-------|----------------|-------------|
| `python` | `qiskit-addon-sqd` (pure Python) | Default if HPC bindings are not installed; good for debugging |
| `hpc` | `qiskit-addon-sqd-hpc` (C++ via nanobind) | Preferred for production / larger batches; requires the HPC extension in the environment |

SBD remains the eigensolver in both cases. For local containers, install and
compile SBD as described in the [Local Build Guide](docs/BUILD-GUIDE.md); on
clusters, follow the [HPC Build Guide](docs/IBM-SCC-BUILD-GUIDE.md).

### System Requirements

- **Memory**: 16 GB RAM minimum (laptop demos); 2 TB per node recommended for protein-scale HPC runs
- **CPU**: 8+ cores recommended for local; 96-core GPU nodes used in production
- **Disk**: 20 GB free space
- **OS**: macOS, Linux, or Windows with WSL2
- **Python**: 3.12

## Project Structure

```
quantum-fragment-methods/
├── quantum_fragment_methods/        # Main package
│   ├── application/
│   │   ├── embedding/               # EWF embedder + base classes
│   │   └── solvers/
│   │       ├── quantum_zoo/
│   │       │   ├── sqd.py           # SOLE maintenance point: SQDSolver +
│   │       │   │                    #   diagonalize_fermionic_hamiltonian,
│   │       │   │                    #   make_sbd_sci_solver, make_fulqrum_sci_solver,
│   │       │   │                    #   counts_to_bit_array, HPC monkey-patch
│   │       │   └── ext_sqd.py       # ext-SQD stub (planned)
│   │       └── classical_zoo/       # FCI, CCSD solvers
│   ├── workflow.py                  # QFWorkflow orchestrator
│   ├── qpu/                         # QPU backend abstraction
│   │   ├── qiskit_ibm_runtime.py    # IBMQuantumBackend (token/CRN, local use)
│   │   └── qrmi.py                  # QRMIBackend (env-var auth, HPC + notebooks)
│   ├── config/                      # Config template
│   └── tests/                       # Unit tests
├── examples/
│   ├── notebook_demos/
│   │   ├── ewf_sqd_demo/            # Glycine EWF+SQD tutorial (QRMI)
│   │   │   ├── ewf_sqd.ipynb
│   │   │   ├── .env.example
│   │   │   └── system_glycine.xyz
│   │   └── sqd_demos/               # H₂, N₂ STO-3G, N₂ 6-31G (QRMI)
│   │       ├── sqd_H2_sto-3g_demo.ipynb
│   │       ├── sqd_N2_sto-3g_demo.ipynb
│   │       ├── sqd_N2_6-31G_demo.ipynb
│   │       └── .env.example
│   └── hpc_demos/
│       ├── alanine_ewf_sqd_demo/    # 4-step Slurm pipeline (alanine)
│       ├── proteasome_ewf_sqd_demo/ # 5-step Slurm pipeline — ixazomib + bortezomib ΔΔE_bind
│       │   ├── 01_meanfield.py / .slurm
│       │   ├── 02_fragments.py / .slurm
│       │   ├── 03_solve.py          # classical (FCI/CCSD) mode
│       │   ├── 03_solve_ccsd.py     # QPU mode — step 3a: CCSD reference
│       │   ├── 03_solve_sqd_submit.py  # QPU mode — step 3b: submit & exit
│       │   ├── 03_solve_sqd_collect.py # QPU mode — step 3c: retrieve + SBD
│       │   ├── 04_reconstruct.py / .slurm
│       │   ├── 05_binding_energy.py / .slurm
│       │   └── run_workflow.sh      # submit all 4 systems; --qpu flag available
│       └── N2_hpc_sqd_demo/         # N₂ direct SQD reference workflow
├── docs/
│   ├── BUILD-GUIDE.md
│   ├── IBM-SCC-BUILD-GUIDE.md
│   └── contributing.md
├── Containerfile
└── pyproject.toml
```

## Citation

If you use this software in your research, please cite:

### Software Citation

```bibtex
@software{quantum_fragment_methods,
  title = {Quantum Fragment Methods},
  author = {Akhil Shajan, Danil Kaliakin, Fangchun Liang, Subhamoy Bhowmik, Susanta Das, Zhen Li, Milana Bazayeva, and Thaddeus Pellegrini},
  year = {2026},
  version = {0.0.1},
  url = {https://github.com/qiskit-community/quantum-fragment-methods}
}
```

### Paper Citation

This software is based on the methodology described by the developers in:

```bibtex
@article{shajan2024molecular,
  title = {Molecular Quantum Computations on a Protein},
  author = {Akhil Shajan, Danil Kaliakin, Fangchun Liang, Thaddeus Pellegrini, Hakan Doga, Subhamoy Bhowmik, Susanta Das, Antonio Mezzacapo, Mario Motta, and Kenneth M. Merz Jr},
  journal = {arXiv preprint arXiv:2512.17130},
  year = {2025},
  url = {https://arxiv.org/abs/2512.17130}
}
```

## Testing

The package includes unit tests for the workflow orchestration system and SBD bindings.

### Running Tests

```bash
# Run all tests
pytest quantum_fragment_methods/tests/

# Run workflow tests only
pytest quantum_fragment_methods/tests/test_workflow.py -v

# Run tests excluding those requiring Vayesta
pytest quantum_fragment_methods/tests/test_workflow.py -v -m "not requires_vayesta"

# Run with coverage report
pytest quantum_fragment_methods/tests/test_workflow.py --cov=quantum_fragment_methods.workflow --cov-report=html
```

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- IBM Quantum team
- Cleveland Clinic Foundation

## Contact

For questions or support, please open an issue on the repository or contact
[thaddeus@blannix.com](mailto:thaddeus@blannix.com).
