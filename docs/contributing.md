# Contributing to Quantum Fragment Methods

Thank you for your interest in contributing to the Quantum Fragment Methods project!  This document provides guidelines for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Project Architecture](#project-architecture)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Code Style](#code-style)
- [Submitting Changes](#submitting-changes)

## Code of Conduct

This project adheres to a code of conduct that all contributors are expected to follow. Please be respectful and constructive in all interactions. The open-source tools are provided with the intention of contributing to the community of scientists and researchers working to advance medicine, healthcare, and life sciences. The tools are made freely available in support of future innovations which may improve the health and wellbeing of all. 

We encourage this same spirit to be applied to derivative works and contribute to solving previously intractible problems in human health as a community. 

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/qiskit-community/quantum-fragment-methods/tree/main/quantum_fragment_methods`
3. Add upstream remote: `git remote add upstream https://github.com/qiskit-community/quantum-fragment-methods/tree/main/quantum_fragment_methods`

## Development Setup

All development is done within the Podman/Docker container. See the [Build Guide](BUILD-GUIDE.md) for complete setup instructions.

### Quick Setup

```bash
# Build the container
podman build -t quantum-fragment-methods:latest .

# Start container with volume mount
podman run -d --name qfm-dev \
  -p 8888:8888 \
  -v $(pwd):/workspace/quantum-fragment-methods:Z \
  quantum-fragment-methods:latest \
  bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate qfrag-env && jupyter lab --ip=0.0.0.0 --allow-root --no-browser --NotebookApp.token=''"

# Access container shell for development
podman exec -it qfm-dev bash
```

### Inside the Container

The container includes all dependencies pre-installed:
- Python 3.12 with conda environment
- PySCF and quantum chemistry libraries
- Development tools (pytest, black, ruff, mypy)
- Jupyter Lab for interactive development

## Project Architecture

The project follows an object-oriented architecture with clear separation of concerns:

### Embedders (`quantum_fragment_methods/embedding/`)

Embedders partition molecular systems into fragments. All embedders inherit from `BaseEmbedder`:

- **Base classes**: `BaseEmbedder`, `EmbeddingResult`, `Fragment`
- **Implementations**: `EWF`
- **Key methods**: `create_fragments()`, `reconstruct_energy()`

### Solvers (`quantum_fragment_methods/solvers/`)

Solvers compute ground state energies for fragments. All solvers inherit from `BaseSolver`:

- **Base classes**: `BaseSolver`, `SolverResult`
- **Classical solvers**: `FCI`, `CCSD` (in `classical_zoo/`)
- **Quantum solvers**: `SQD`, `ext-SQD`, (in `quantum_zoo/`)
- **Key method**: `solve()`

### Workflow (`quantum_fragment_methods/workflow.py`)

The `QFWorkflow` class orchestrates the entire calculation:

- Accepts embedder objects
- Manages solver assignment via priority-based rules
- Coordinates fragment creation and solving

## Making Changes

### Adding a New Embedder

1. Create a new file in `quantum_fragment_methods/embedding/`
2. Inherit from `BaseEmbedder`
3. Implement required methods:
   - `create_fragments(mf, **kwargs) -> EmbeddingResult`
   - `reconstruct_energy(fragment_results, embedding_result) -> float`
4. Add configuration parameters in `__init__`
5. Export in `embedding/__init__.py`

### Adding a New Solver

1. Create a new file in `quantum_fragment_methods/solvers/classical_zoo/` or `quantum_zoo/`
2. Inherit from `BaseSolver`
3. Implement required method:
   - `solve(*args, **kwargs) -> SolverResult`
4. Add configuration parameters in `__init__`
5. Export in the appropriate `__init__.py`

### General Workflow

1. Create a new branch for your feature or bugfix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes following the code style guidelines below

3. Add tests for new functionality

4. Update documentation as needed

## Testing

Run the test suite before submitting changes:

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=quantum_fragment_methods --cov-report=html

# Run specific test file
pytest quantum_fragment_methods/tests/test_embedding.py
```

## Code Style

This project follows PEP 8 style guidelines with some modifications:

- Line length: 100 characters (enforced by Black)
- Use type hints where appropriate
- Write docstrings for all public functions, classes, and modules (Google style)

### Formatting

We use Black for code formatting and Ruff for linting:

```bash
# Format code
black quantum_fragment_methods/

# Check linting
ruff check quantum_fragment_methods/

# Auto-fix linting issues
ruff check --fix quantum_fragment_methods/
```

### Type Checking

We use mypy for static type checking:

```bash
mypy quantum_fragment_methods/
```

### Example: Embedder Class

```python
from __future__ import annotations
from typing import Dict, Any

from .base import BaseEmbedder, EmbeddingResult


class MyEmbedder(BaseEmbedder):
    """
    My custom embedding method.
    
    Parameters
    ----------
    param1 : int
        Description of param1
    param2 : float, optional
        Description of param2 (default: 1.0)
    """
    
    def __init__(self, param1: int, param2: float = 1.0, **kwargs):
        """Initialize embedder."""
        super().__init__(param1=param1, param2=param2, **kwargs)
        self.param1 = param1
        self.param2 = param2
    
    def create_fragments(self, mf, **kwargs) -> EmbeddingResult:
        """
        Create fragments from mean-field calculation.
        
        Parameters
        ----------
        mf : pyscf.scf object
            Mean-field calculation result
        **kwargs : dict
            Additional parameters
            
        Returns
        -------
        EmbeddingResult
            Container with fragments and embedding information
        """
        # Implementation here
        pass
    
    def reconstruct_energy(
        self,
        fragment_results: Dict[int | str, Any],
        embedding_result: EmbeddingResult
    ) -> float:
        """
        Reconstruct total energy from fragment results.
        
        Parameters
        ----------
        fragment_results : dict
            Dictionary mapping fragment_id to solver results
        embedding_result : EmbeddingResult
            Original embedding result
            
        Returns
        -------
        float
            Total reconstructed energy
        """
        # Implementation here
        pass
```

### Example: Solver Class

```python
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..base import BaseSolver, SolverResult


class MySolver(BaseSolver):
    """
    My custom solver.
    
    Parameters
    ----------
    param1 : float, optional
        Description of param1 (default: 1e-6)
    """
    
    def __init__(self, param1: float = 1e-6, **kwargs):
        """Initialize solver."""
        super().__init__(**kwargs)
        self.param1 = param1
    
    def solve(
        self,
        h1e: NDArray,
        h2e: NDArray,
        norb: int,
        nelec: tuple[int, int] | int,
        **kwargs
    ) -> SolverResult:
        """
        Solve for ground state.
        
        Parameters
        ----------
        h1e : np.ndarray
            One-electron Hamiltonian
        h2e : np.ndarray
            Two-electron integrals
        norb : int
            Number of orbitals
        nelec : tuple or int
            Number of electrons
        **kwargs : dict
            Additional parameters
            
        Returns
        -------
        SolverResult
            Ground state energy and properties
        """
        # Implementation here
        pass
```

## Submitting Changes

1. Commit your changes with clear, descriptive commit messages:
   ```bash
   git commit -m "Add feature: description of what you added"
   ```

2. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

3. Create a Pull Request (PR) from your fork to the main repository

4. In your PR description:
   - Describe what changes you made and why
   - Reference any related issues
   - Include any relevant test results or benchmarks

5. Wait for review and address any feedback

## Pull Request Checklist

Before submitting a PR, ensure:

- [ ] Code follows the style guidelines
- [ ] All tests pass
- [ ] New tests added for new functionality
- [ ] Documentation updated (docstrings, README, etc.)
- [ ] CHANGELOG.md updated (if applicable)
- [ ] No merge conflicts with main branch
- [ ] Commit messages are clear and descriptive
- [ ] New embedders/solvers properly inherit from base classes
- [ ] Exports added to appropriate `__init__.py` files

## Questions?

If you have questions about contributing, please open an issue or contact the maintainer.

## License

By contributing to this project, you agree that your contributions will be licensed under the Apache License 2.0.
