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

"""Base classes for embedding methods."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class Fragment:
    """
    Container for fragment information.

    Attributes
    ----------
    fragment_id : int or str
        Unique identifier for the fragment
    atom_indices : list of int
        Indices of atoms in this fragment
    orbital_indices : list of int
        Indices of orbitals in this fragment
    n_orbitals : int
        Number of orbitals in the fragment
    n_electrons : int or tuple
        Number of electrons (total or (alpha, beta))
    metadata : dict
        Additional fragment-specific information
    """

    def __init__(
        self,
        fragment_id: int | str,
        atom_indices: list[int],
        orbital_indices: list[int],
        n_electrons: int | tuple[int, int],
        metadata: dict[str, Any] | None = None,
    ):
        """Initialize fragment."""
        self.fragment_id = fragment_id
        self.atom_indices = atom_indices
        self.orbital_indices = orbital_indices
        self.n_orbitals = len(orbital_indices)
        self.n_electrons = n_electrons
        self.metadata = metadata or {}

    def __repr__(self):
        return (
            f"Fragment(id={self.fragment_id}, "
            f"n_orbitals={self.n_orbitals}, "
            f"n_electrons={self.n_electrons})"
        )


class EmbeddingResult:
    """
    Container for embedding results.

    Attributes
    ----------
    fragments : dict
        Dictionary mapping fragment_id to Fragment objects
    embedding_matrices : dict, optional
        Embedding/projection matrices for each fragment
    mean_field_energy : float, optional
        Mean-field reference energy
    metadata : dict
        Additional embedding-specific information
    """

    def __init__(
        self,
        fragments: dict[int | str, Fragment],
        embedding_matrices: dict[int | str, np.ndarray] | None = None,
        mean_field_energy: float | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """Initialize embedding result."""
        self.fragments = fragments
        self.embedding_matrices = embedding_matrices or {}
        self.mean_field_energy = mean_field_energy
        self.metadata = metadata or {}

    def __repr__(self):
        return f"EmbeddingResult(n_fragments={len(self.fragments)})"


class BaseEmbedder(ABC):
    """
    Abstract base class for all embedding methods.

    All embedding methods (EWF, DMET, MBE) must implement the core methods
    for fragment creation and energy reconstruction.
    """

    def __init__(self, **kwargs):
        """
        Initialize embedder with configuration.

        Parameters
        ----------
        **kwargs : dict
            Embedder-specific configuration options
        """
        self.config = kwargs

    @abstractmethod
    def create_fragments(self, mf, **kwargs) -> EmbeddingResult:
        """
        Create fragments from mean-field calculation.

        Parameters
        ----------
        mf : pyscf.scf object
            Mean-field calculation result (RHF, UHF, etc.)
        **kwargs : dict
            Additional parameters for fragment creation

        Returns
        -------
        EmbeddingResult
            Container with fragments and embedding information
        """
        pass

    @abstractmethod
    def reconstruct_energy(
        self, fragment_results: dict[int | str, Any], embedding_result: EmbeddingResult
    ) -> float:
        """
        Reconstruct total energy from fragment results.

        Parameters
        ----------
        fragment_results : dict
            Dictionary mapping fragment_id to solver results
        embedding_result : EmbeddingResult
            Original embedding result with fragment information

        Returns
        -------
        float
            Total reconstructed energy
        """
        pass

    @property
    def name(self) -> str:
        """Return embedder name."""
        return self.__class__.__name__

    def __repr__(self) -> str:
        """String representation of embedder."""
        config_str = ", ".join(f"{k}={v}" for k, v in self.config.items())
        return f"{self.name}({config_str})"
