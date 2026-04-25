# quantum-fragment-methods

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Status](https://img.shields.io/badge/status-alpha-orange.svg)](IMPLEMENTATION_STATUS.md)

## About

A Python package for quantum chemistry tailored toward fragment-based embedding and quantum algorithms applied to macromolecule simulations. This codebase provides open access to the methodology from the paper *"Molecular Quantum Computations on a Protein"*, a collaborative effort between IBM Quantum and the Cleveland Clinic Foundation. 

## Overview

The Quantum Fragment Methods framework enables scalable, high-accuracy quantum simulations of large molecular systems—such as peptides and proteins—by embedding them into quantum subspaces. The package features:

- **Embedder objects**: EWF with configurable parameters
- **Rule-based solver assignment**: Priority-based automatic solver selection  
- **Quantum solvers**: SQD, ext-SQD (planned)
- **Classical solvers**: FCI, CCSD

## Documentation

- **[Build Guide](docs/BUILD-GUIDE.md)** - Installation instructions using Podman/Docker
- **[Tutorial](examples/local_demo/tutorial.ipynb)** - Interactive Jupyter notebook with usage examples
- **[Contributing Guide](docs/contributing.md)** - Guidelines for contributors
- **[Examples](examples/)** - Additional example guides

## Quick Start

1. **Install**: Follow the [Build Guide](docs/BUILD-GUIDE.md) for complete setup instructions
2. **Configure**: Create a `config.yaml` file to define your workflow parameters (see below)
3. **Learn**: Work through the [Tutorial](examples/local_demo/tutorial.ipynb) notebook
4. **Explore**: Check out [Examples](examples/) for more use cases

### Configuration File

The `config.yaml` file is the main entry point for configuring your quantum fragment calculations. It defines:

- **Workflow settings**: Basis set and molecular geometry
- **Embedder parameters**: Bath type, truncation thresholds, fragmentation scheme (EWF, DMET, MBE)
- **QPU configuration**: Backend selection, credentials, and sampler options
- **Solver settings**: Algorithm-specific parameters for SQD, ext-SQD, and classical solvers

See [examples/local_demo/config.yaml](examples/local_demo/config.yaml) for a complete template with detailed comments.

### System Requirements

- **Memory**: 12-16 GB RAM minimum 
- **CPU**: 8+ cores recommended
- **Disk**: 20 GB free space
- **OS**: macOS, Linux, or Windows with WSL2 (Windows Subsystem for Linux)

## Project Structure

```
quantum-fragment-methods/
├── quantum_fragment_methods/     # Main package
│   ├── application/              # Scientific application layer
│   │   ├── embedding/            # Embedding methods (EWF, etc.)
│   │   └── solvers/              # Quantum and classical solvers
│   │       ├── quantum_zoo/      # Quantum solvers (SQD, ext-SQD, etc.)
│   │       └── classical_zoo/    # Classical solvers (FCI, CCSD, etc.)
│   ├── workflow.py               # Workflow orchestrator
│   ├── qpu/                      # QPU backend abstraction
│   │   ├── base.py               # Base QPU interface 
│   │   └── qiskit_ibm_runtime.py # IBM Quantum backend 
│   ├── config/                   # Configuration management
│   └── tests/                    # Unit tests
├── examples/                     # Example notebooks and workflows
│   └──  local_demo/              # Alanine demo (works on laptop, MVP code)
├── docs/                         # Documentation
└── pyproject.toml                # Package configuration
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

The package includes comprehensive test coverage for the workflow orchestration system.

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

For questions or support, please open an issue on the repository or contact the maintainers.

---

**Note**: This package is under active development. APIs may change between versions.
