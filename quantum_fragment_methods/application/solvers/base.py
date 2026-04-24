# This code is a Qiskit project.
#
# (C) Copyright IBM and Cleveland Clinic Foundation 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Base classes for solvers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class SolverResult:
    """
    Container for solver results.

    Attributes
    ----------
    energy : float
        Ground state energy
    wavefunction : np.ndarray, optional
        Wavefunction coefficients
    rdm1 : np.ndarray, optional
        One-particle reduced density matrix
    rdm2 : np.ndarray, optional
        Two-particle reduced density matrix
    metadata : dict
        Additional solver-specific information
    """

    def __init__(
        self,
        energy: float,
        wavefunction: np.ndarray | None = None,
        rdm1: np.ndarray | None = None,
        rdm2: np.ndarray | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """Initialize solver result."""
        self.energy = energy
        self.wavefunction = wavefunction
        self.rdm1 = rdm1
        self.rdm2 = rdm2
        self.metadata = metadata or {}

    def __repr__(self):
        return f"SolverResult(energy={self.energy:.8f})"


class BaseSolver(ABC):
    """
    Abstract base class for all solvers.

    All solvers (classical and quantum) must implement the solve() method.
    """

    def __init__(self, **kwargs):
        """
        Initialize solver with configuration.

        Parameters
        ----------
        **kwargs : dict
            Solver-specific configuration options
        """
        self.config = kwargs

    @abstractmethod
    def solve(self, *args, **kwargs) -> SolverResult:
        """
        Solve for the ground state.

        The signature and parameters depend on the specific solver implementation.
        Common patterns include:

        - FCI: solve(h1e, h2e, norb, nelec, **kwargs)
        - CCSD: solve(mf, **kwargs)
        - Quantum: solve(hamiltonian, n_qubits, **kwargs)

        Parameters
        ----------
        *args : tuple
            Positional arguments (solver-specific)
        **kwargs : dict
            Keyword arguments (solver-specific)

        Returns
        -------
        SolverResult
            Ground state energy, wavefunction, and density matrices
        """
        pass

    @property
    def name(self) -> str:
        """Return solver name."""
        return self.__class__.__name__

