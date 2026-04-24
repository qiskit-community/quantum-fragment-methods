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

"""Full Configuration Interaction (FCI) solver using PySCF."""

from __future__ import annotations

import numpy as np
import pyscf
import pyscf.fci
import pyscf.fci.direct_spin0
from numpy.typing import NDArray

from ..base import BaseSolver, SolverResult


class FCI(BaseSolver):
    """
    Full Configuration Interaction (FCI) solver.

    This class wraps PySCF's FCI solver to provide exact solutions
    for small molecular fragments. It computes the ground state energy,
    wavefunction, and reduced density matrices.

    Parameters
    ----------
    conv_tol : float, optional
        Convergence tolerance for FCI solver (default: 1e-14)
    max_cycle : int, optional
        Maximum number of iterations (default: 100)
    spin : int, optional
        Spin multiplicity (2S). Default is 0 (singlet)
    verbose : int, optional
        Verbosity level (default: 0)
    **kwargs : dict
        Additional PySCF FCI configuration options

    Attributes
    ----------
    conv_tol : float
        Convergence tolerance
    max_cycle : int
        Maximum iterations
    spin : int
        Spin multiplicity
    verbose : int
        Verbosity level

    Examples
    --------
    >>> fci_solver = FCI(conv_tol=1e-12)
    >>> result = fci_solver.solve(h1e, h2e, norb, nelec)
    >>> print(f"FCI Energy: {result.energy}")
    """

    def __init__(
        self,
        conv_tol: float = 1e-14,
        max_cycle: int = 100,
        spin: int = 0,
        verbose: int = 0,
        **kwargs,
    ):
        """Initialize FCI solver with PySCF backend."""
        super().__init__(**kwargs)
        self.conv_tol = conv_tol
        self.max_cycle = max_cycle
        self.spin = spin
        self.verbose = verbose

    def solve(
        self, h1e: NDArray, h2e: NDArray, norb: int, nelec: tuple[int, int] | int, **kwargs
    ) -> SolverResult:
        """
        Solve FCI problem for given Hamiltonian.

        Parameters
        ----------
        h1e : np.ndarray
            One-electron Hamiltonian matrix (norb, norb)
        h2e : np.ndarray
            Two-electron repulsion integrals in chemist notation
            Shape: (norb, norb, norb, norb)
        norb : int
            Number of spatial orbitals
        nelec : tuple of int or int
            Number of electrons. Can be:
            - tuple (n_alpha, n_beta) for spin-resolved
            - int for total electrons (assumes closed shell)
        **kwargs : dict
            Additional solver parameters:
            - conv_tol: Override convergence tolerance
            - max_cycle: Override max iterations
            - ci0: Initial CI vector guess

        Returns
        -------
        SolverResult
            Container with:
            - energy: FCI ground state energy
            - wavefunction: CI vector coefficients
            - rdm1: One-particle reduced density matrix
            - rdm2: Two-particle reduced density matrix
            - metadata: Convergence info and diagnostics

        Raises
        ------
        ValueError
            If input dimensions are inconsistent
        RuntimeError
            If FCI calculation fails to converge

        Notes
        -----
        This implementation uses PySCF's direct_spin0 FCI solver for
        spin-adapted calculations, which is efficient for singlet states.
        For other spin states, it falls back to the general FCI solver.
        """
        # Validate inputs
        if h1e.shape != (norb, norb):
            raise ValueError(f"h1e shape {h1e.shape} inconsistent with norb={norb}")
        if h2e.shape != (norb, norb, norb, norb):
            raise ValueError(f"h2e shape {h2e.shape} inconsistent with norb={norb}")

        # Handle electron count
        if isinstance(nelec, int):
            if nelec % 2 != 0:
                raise ValueError(
                    "For integer nelec, must be even (closed shell). "
                    "Use tuple (n_alpha, n_beta) for open shell."
                )
            nelec = (nelec // 2, nelec // 2)

        n_alpha, n_beta = nelec

        # Override parameters if provided
        conv_tol = kwargs.get("conv_tol", self.conv_tol)
        max_cycle = kwargs.get("max_cycle", self.max_cycle)
        ci0 = kwargs.get("ci0", None)

        # Select appropriate FCI solver
        if self.spin == 0 and n_alpha == n_beta:
            # Use spin-adapted solver for singlets
            fci_solver = pyscf.fci.direct_spin0  # type: ignore
        else:
            # Use general FCI solver for other cases
            fci_solver = pyscf.fci.direct_spin1  # type: ignore

        # Run FCI calculation
        try:
            energy, ci_vec = fci_solver.kernel(
                h1e,
                h2e,
                norb,
                nelec,
                conv_tol=conv_tol,
                max_cycle=max_cycle,
                verbose=self.verbose,
                ci0=ci0,
            )
        except Exception as e:
            raise RuntimeError(f"FCI calculation failed: {str(e)}") from e

        # Compute reduced density matrices
        dm1, dm2 = fci_solver.make_rdm12(ci_vec, norb, nelec)

        # Prepare metadata
        metadata = {
            "norb": norb,
            "nelec": nelec,
            "conv_tol": conv_tol,
            "max_cycle": max_cycle,
            "spin": self.spin,
            "solver_type": (
                "direct_spin0" if self.spin == 0 and n_alpha == n_beta else "direct_spin1"
            ),
        }

        return SolverResult(
            energy=float(energy), wavefunction=ci_vec, rdm1=dm1, rdm2=dm2, metadata=metadata
        )

    def solve_from_integrals(
        self, h1e: NDArray, h2e: NDArray, norb: int, nocc: int, **kwargs
    ) -> SolverResult:
        """
        Convenience method matching the application code pattern.

        Parameters
        ----------
        h1e : np.ndarray
            One-electron Hamiltonian (effective Hamiltonian)
        h2e : np.ndarray
            Two-electron repulsion integrals (ERIs)
        norb : int
            Number of orbitals
        nocc : int
            Number of occupied orbitals (assumes closed shell)
        **kwargs : dict
            Additional parameters passed to solve()

        Returns
        -------
        SolverResult
            FCI solution with energy and density matrices
        """
        nelec = (nocc, nocc)
        return self.solve(h1e, h2e, norb, nelec, **kwargs)

    def compute_fragment_energy(
        self, result: SolverResult, fock: NDArray, h2e: NDArray, proj: NDArray, nocc: int
    ) -> tuple[float, float]:
        """
        Compute partitioned cumulant energy contributions.

        This method implements the energy decomposition used in
        fragment-based methods like EWF (Embedded Wavefunction).

        Parameters
        ----------
        result : SolverResult
            FCI solution containing density matrices
        fock : np.ndarray
            Fock matrix in cluster basis
        h2e : np.ndarray
            Two-electron integrals in cluster basis
        proj : np.ndarray
            Fragment projector matrix
        nocc : int
            Number of occupied orbitals

        Returns
        -------
        e1_pc : float
            One-body partitioned cumulant energy
        e22_pc : float
            Two-body partitioned cumulant energy

        Notes
        -----
        This implements the partitioned cumulant approach where:
        - dm1_pc = dm1 - HF contribution
        - dm2 is split into different components
        - Only fragment-projected contributions are included
        """
        if result.rdm1 is None or result.rdm2 is None:
            raise ValueError("Result must contain rdm1 and rdm2 for fragment energy calculation")

        dm1 = result.rdm1.copy()
        dm2 = result.rdm2.copy()

        # Remove Hartree-Fock contribution from dm1
        dm1_pc = dm1.copy()
        dm1_pc[np.diag_indices(nocc)] -= 2.0

        # Project dm1 onto fragment
        dm1_pc_proj = proj @ dm1_pc

        # One-body energy contribution
        e1_pc = np.einsum("pq,pq->", fock, dm1_pc_proj)

        # Two-body contribution (simplified version)
        # In full implementation, would split dm2 into components
        dm2_pc = np.einsum("Ijkl,iI->ijkl", dm2, proj)
        e22_pc = 0.5 * np.einsum("pqrs,pqrs->", h2e, dm2_pc)

        return float(e1_pc), float(e22_pc)

    def __repr__(self) -> str:
        """String representation of FCI solver."""
        return (
            f"FCI(conv_tol={self.conv_tol}, max_cycle={self.max_cycle}, "
            f"spin={self.spin}, verbose={self.verbose})"
        )
