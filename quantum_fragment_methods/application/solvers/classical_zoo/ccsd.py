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

"""Coupled Cluster Singles and Doubles (CCSD) solver using PySCF."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..base import BaseSolver, SolverResult


class CCSD(BaseSolver):
    """
    Coupled Cluster Singles and Doubles (CCSD) solver.

    This class wraps PySCF's CCSD solver to provide accurate correlation
    energies for molecular fragments. CCSD is more efficient than FCI
    for larger systems while maintaining good accuracy.

    Parameters
    ----------
    conv_tol : float, optional
        Convergence tolerance for CCSD solver (default: 1e-7)
    max_cycle : int, optional
        Maximum number of iterations (default: 50)
    diis_space : int, optional
        DIIS space size for convergence acceleration (default: 6)
    verbose : int, optional
        Verbosity level (default: 0)
    **kwargs : dict
        Additional PySCF CCSD configuration options

    Attributes
    ----------
    conv_tol : float
        Convergence tolerance
    max_cycle : int
        Maximum iterations
    diis_space : int
        DIIS space size
    verbose : int
        Verbosity level

    Examples
    --------
    >>> ccsd_solver = CCSD(conv_tol=1e-9)
    >>> result = ccsd_solver.solve(mf)
    >>> print(f"CCSD Energy: {result.energy}")

    >>> # With RDM computation
    >>> result = ccsd_solver.solve(mf, compute_rdms=True)
    >>> print(f"RDM1 shape: {result.rdm1.shape}")
    """

    def __init__(
        self,
        conv_tol: float = 1e-7,
        max_cycle: int = 50,
        diis_space: int = 6,
        verbose: int = 0,
        **kwargs,
    ):
        """Initialize CCSD solver with PySCF backend."""
        super().__init__(**kwargs)
        self.conv_tol = conv_tol
        self.max_cycle = max_cycle
        self.diis_space = diis_space
        self.verbose = verbose

    def solve(self, mf, compute_rdms: bool = False, **kwargs) -> SolverResult:
        """
        Solve CCSD problem from mean-field object.

        Parameters
        ----------
        mf : pyscf.scf object
            Mean-field calculation result (RHF, UHF, etc.)
        compute_rdms : bool, optional
            Whether to compute reduced density matrices (default: False)
        **kwargs : dict
            Additional solver parameters:
            - conv_tol: Override convergence tolerance
            - max_cycle: Override max iterations
            - diis_space: Override DIIS space

        Returns
        -------
        SolverResult
            Container with:
            - energy: CCSD total energy
            - wavefunction: None (CCSD doesn't store full wavefunction)
            - rdm1: One-particle RDM (if compute_rdms=True)
            - rdm2: Two-particle RDM (if compute_rdms=True)
            - metadata: Convergence info and diagnostics

        Raises
        ------
        ImportError
            If PySCF is not installed
        TypeError
            If mf is not a valid mean-field object
        RuntimeError
            If CCSD calculation fails

        Notes
        -----
        This implementation uses PySCF's CCSD solver. For restricted
        mean-field objects (RHF), it uses RCCSD. For unrestricted (UHF),
        it uses UCCSD.
        """
        try:
            import pyscf.cc
        except ImportError as e:
            raise ImportError(
                "PySCF is required for CCSD solver. Install with: pip install pyscf"
            ) from e

        # Validate mean-field object
        if not hasattr(mf, "mo_coeff"):
            raise TypeError("Invalid mean-field object. Must have mo_coeff attribute.")

        # Override parameters if provided
        conv_tol = kwargs.get("conv_tol", self.conv_tol)
        max_cycle = kwargs.get("max_cycle", self.max_cycle)
        diis_space = kwargs.get("diis_space", self.diis_space)

        # Create CCSD object
        try:
            ccsd = pyscf.cc.CCSD(mf)
            ccsd.conv_tol = conv_tol
            ccsd.max_cycle = max_cycle
            ccsd.diis_space = diis_space
            ccsd.verbose = self.verbose
        except Exception as e:
            raise RuntimeError(f"Failed to create CCSD object: {str(e)}") from e

        # Run CCSD calculation
        try:
            e_corr, t1, t2 = ccsd.kernel()
            converged = ccsd.converged
        except Exception as e:
            raise RuntimeError(f"CCSD calculation failed: {str(e)}") from e

        # Total energy
        e_tot = ccsd.e_tot

        # Compute RDMs if requested
        rdm1 = None
        rdm2 = None
        if compute_rdms:
            try:
                rdm1 = ccsd.make_rdm1()
                rdm2 = ccsd.make_rdm2()
            except Exception:
                # Some CCSD implementations may not support RDM2
                rdm1 = ccsd.make_rdm1()
                rdm2 = None

        # Prepare metadata
        metadata = {
            "converged": converged,
            "e_corr": float(e_corr),
            "e_hf": float(mf.e_tot),
            "conv_tol": conv_tol,
            "max_cycle": max_cycle,
            "diis_space": diis_space,
            "norb": mf.mol.nao,
            "nelec": mf.mol.nelectron,
        }

        return SolverResult(
            energy=float(e_tot),
            wavefunction=None,  # CCSD doesn't store full wavefunction
            rdm1=rdm1,
            rdm2=rdm2,
            metadata=metadata,
        )

    def solve_from_integrals(
        self, h1e: NDArray, h2e: NDArray, norb: int, nocc: int, **kwargs
    ) -> SolverResult:
        """
        Solve CCSD from Hamiltonian integrals.

        This method creates a temporary mean-field object from the
        provided integrals and then solves CCSD.

        Parameters
        ----------
        h1e : np.ndarray
            One-electron Hamiltonian matrix (norb, norb)
        h2e : np.ndarray
            Two-electron repulsion integrals in chemist notation
            Shape: (norb, norb, norb, norb)
        norb : int
            Number of spatial orbitals
        nocc : int
            Number of occupied orbitals (assumes closed shell)
        **kwargs : dict
            Additional parameters passed to solve()

        Returns
        -------
        SolverResult
            CCSD solution with energy and density matrices

        Notes
        -----
        This method constructs a fake mean-field object with the
        provided integrals. It's useful for fragment calculations
        where you have effective Hamiltonians.
        """
        try:
            from pyscf import gto, scf
        except ImportError as e:
            raise ImportError(
                "PySCF is required for CCSD solver. Install with: pip install pyscf"
            ) from e

        # Create a fake molecule with correct number of electrons
        nelec = 2 * nocc
        mol = gto.Mole()
        mol.nelectron = nelec
        mol.nao = norb
        mol.build(dump_input=False, parse_arg=False)

        # Create fake mean-field object
        mf = scf.RHF(mol)
        mf.get_hcore = lambda *args: h1e
        mf.get_ovlp = lambda *args: np.eye(norb)
        mf._eri = h2e

        # Set up MO coefficients (identity for simplicity)
        mf.mo_coeff = np.eye(norb)
        mf.mo_occ = np.zeros(norb)
        mf.mo_occ[:nocc] = 2.0
        mf.mo_energy = np.diag(h1e)

        # Compute mean-field energy
        dm = mf.make_rdm1()
        vhf = mf.get_veff(dm=dm)
        mf.e_tot = np.einsum("ij,ji->", h1e + 0.5 * vhf, dm)

        # Solve CCSD
        return self.solve(mf, **kwargs)

    @staticmethod
    def split_dm2(nocc: int, dm1: NDArray, dm2: NDArray) -> tuple[NDArray, NDArray, NDArray]:
        """
        Split 2-RDM into HF, 1-body correlation, and 2-body correlation components.

        This decomposition is used in partitioned cumulant energy calculations
        to separate different correlation contributions.

        Parameters
        ----------
        nocc : int
            Number of occupied orbitals
        dm1 : np.ndarray
            One-particle density matrix (with HF part removed)
        dm2 : np.ndarray
            Two-particle density matrix

        Returns
        -------
        dm2_0 : np.ndarray
            HF component of 2-RDM
        dm2_1 : np.ndarray
            1-body correlation component
        dm2_2 : np.ndarray
            2-body correlation component (cumulant)

        Notes
        -----
        The decomposition follows: dm2 = dm2_0 + dm2_1 + dm2_2
        where dm2_2 contains the true 2-body correlation effects.
        """
        dm2_2 = dm2.copy()
        dm2_1 = np.zeros_like(dm2)
        dm2_0 = np.zeros_like(dm2)

        # Build dm2_1 from dm1
        for i in range(nocc):
            dm2_1[i, i, :, :] += dm1 * 2
            dm2_1[:, :, i, i] += dm1 * 2
            dm2_1[:, i, i, :] -= dm1
            dm2_1[i, :, :, i] -= dm1.T

        # Build dm2_0 (HF part)
        for i in range(nocc):
            for j in range(nocc):
                dm2_0[i, i, j, j] += 4
                dm2_0[i, j, j, i] -= 2

        # Extract cumulant
        dm2_2 -= dm2_0 + dm2_1

        return dm2_0, dm2_1, dm2_2

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
            CCSD solution containing density matrices
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

        Raises
        ------
        ValueError
            If result doesn't contain required RDMs

        Notes
        -----
        This implements the partitioned cumulant approach where:
        - dm1_pc = dm1 - HF contribution
        - dm2 is split into different components using split_dm2()
        - Only fragment-projected contributions are included
        """
        if result.rdm1 is None:
            raise ValueError(
                "Result must contain rdm1 for fragment energy calculation. "
                "Call solve() with compute_rdms=True."
            )

        if result.rdm2 is None:
            raise ValueError(
                "Result must contain rdm2 for fragment energy calculation. "
                "Call solve() with compute_rdms=True."
            )

        dm1 = result.rdm1.copy()
        dm2 = result.rdm2.copy()

        # Remove Hartree-Fock contribution from dm1
        dm1_pc = dm1.copy()
        dm1_pc[np.diag_indices(nocc)] -= 2.0

        # Split dm2 into components
        dm2_0, dm2_1, dm2_2 = self.split_dm2(nocc, dm1_pc, dm2)

        # Project dm1 onto fragment
        dm1_pc_proj = proj @ dm1_pc

        # One-body energy contribution
        e1_pc = np.einsum("pq,pq->", fock, dm1_pc_proj)

        # Two-body contribution (only cumulant part)
        dm2_pc = np.einsum("Ijkl,iI->ijkl", dm2_2, proj)
        e22_pc = 0.5 * np.einsum("pqrs,pqrs->", h2e, dm2_pc)

        return float(e1_pc), float(e22_pc)

    def __repr__(self) -> str:
        """String representation of CCSD solver."""
        return (
            f"CCSD(conv_tol={self.conv_tol}, max_cycle={self.max_cycle}, "
            f"diis_space={self.diis_space}, verbose={self.verbose})"
        )
