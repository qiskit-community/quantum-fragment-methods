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

"""
Comprehensive test suite for classical solvers (CCSD and FCI).

This module tests the solver implementations following Test-Driven Development (TDD).
Tests are organized by solver type and functionality.
"""

import numpy as np
import pytest

# ============================================================================
# FCI Solver Tests
# ============================================================================

class TestFCIInitialization:
    """Test FCI solver initialization and configuration."""

    def test_initialization_default(self):
        """Test FCI initialization with default parameters."""
        from quantum_fragment_methods.application.solvers.classical_zoo import FCI

        fci = FCI()
        assert fci.conv_tol == 1e-14
        assert fci.max_cycle == 100
        assert fci.spin == 0
        assert fci.verbose == 0

    def test_initialization_custom_parameters(self):
        """Test FCI initialization with custom parameters."""
        from quantum_fragment_methods.application.solvers.classical_zoo import FCI

        fci = FCI(conv_tol=1e-12, max_cycle=200, spin=2, verbose=4)
        assert fci.conv_tol == 1e-12
        assert fci.max_cycle == 200
        assert fci.spin == 2
        assert fci.verbose == 4

    def test_solver_name(self):
        """Test that solver has correct name."""
        from quantum_fragment_methods.application.solvers.classical_zoo import FCI

        fci = FCI()
        assert fci.name == "FCI"


class TestFCISolve:
    """Test FCI solve method with various inputs."""

    @pytest.fixture
    def h2_integrals(self):
        """Provide H2 molecule integrals for testing."""
        # Simple H2 molecule in minimal basis
        norb = 2
        nelec = (1, 1)  # 2 electrons

        # One-electron Hamiltonian (approximate)
        h1e = np.array([[-1.25, -0.48], [-0.48, -1.25]])

        # Two-electron integrals (chemist notation)
        h2e = np.zeros((norb, norb, norb, norb))
        h2e[0, 0, 0, 0] = 0.67
        h2e[1, 1, 1, 1] = 0.67
        h2e[0, 0, 1, 1] = 0.66
        h2e[1, 1, 0, 0] = 0.66
        h2e[0, 1, 0, 1] = 0.66
        h2e[1, 0, 1, 0] = 0.66
        h2e[0, 1, 1, 0] = 0.18
        h2e[1, 0, 0, 1] = 0.18

        return h1e, h2e, norb, nelec

    @pytest.mark.requires_pyscf
    def test_solve_basic(self, h2_integrals):
        """Test basic FCI solve with H2 integrals."""
        from quantum_fragment_methods.application.solvers.classical_zoo import FCI

        h1e, h2e, norb, nelec = h2_integrals
        fci = FCI()

        result = fci.solve(h1e, h2e, norb, nelec)

        # Check result structure
        assert result.energy is not None
        assert isinstance(result.energy, float)
        assert result.wavefunction is not None
        assert result.rdm1 is not None
        assert result.rdm2 is not None

    @pytest.mark.requires_pyscf
    def test_solve_with_integer_nelec(self, h2_integrals):
        """Test FCI solve with integer electron count (closed shell)."""
        from quantum_fragment_methods.application.solvers.classical_zoo import FCI

        h1e, h2e, norb, _ = h2_integrals
        fci = FCI()

        # Use integer for closed shell
        result = fci.solve(h1e, h2e, norb, nelec=2)

        assert result.energy is not None
        assert result.metadata["nelec"] == (1, 1)

    @pytest.mark.requires_pyscf
    def test_solve_rdm_shapes(self, h2_integrals):
        """Test that RDMs have correct shapes."""
        from quantum_fragment_methods.application.solvers.classical_zoo import FCI

        h1e, h2e, norb, nelec = h2_integrals
        fci = FCI()

        result = fci.solve(h1e, h2e, norb, nelec)

        # Check RDM shapes
        assert result.rdm1.shape == (norb, norb)
        assert result.rdm2.shape == (norb, norb, norb, norb)

    @pytest.mark.requires_pyscf
    def test_solve_metadata(self, h2_integrals):
        """Test that metadata contains expected information."""
        from quantum_fragment_methods.application.solvers.classical_zoo import FCI

        h1e, h2e, norb, nelec = h2_integrals
        fci = FCI(conv_tol=1e-12, max_cycle=150)

        result = fci.solve(h1e, h2e, norb, nelec)

        # Check metadata
        assert "norb" in result.metadata
        assert "nelec" in result.metadata
        assert "conv_tol" in result.metadata
        assert "max_cycle" in result.metadata
        assert result.metadata["norb"] == norb
        assert result.metadata["conv_tol"] == 1e-12


class TestFCIErrorHandling:
    """Test FCI error handling for invalid inputs."""

    def test_invalid_h1e_shape(self):
        """Test error for inconsistent h1e shape."""
        from quantum_fragment_methods.application.solvers.classical_zoo import FCI

        fci = FCI()
        h1e = np.zeros((3, 3))  # Wrong shape
        h2e = np.zeros((2, 2, 2, 2))
        norb = 2
        nelec = (1, 1)

        with pytest.raises(ValueError, match="h1e shape.*inconsistent"):
            fci.solve(h1e, h2e, norb, nelec)

    def test_invalid_h2e_shape(self):
        """Test error for inconsistent h2e shape."""
        from quantum_fragment_methods.application.solvers.classical_zoo import FCI

        fci = FCI()
        h1e = np.zeros((2, 2))
        h2e = np.zeros((3, 3, 3, 3))  # Wrong shape
        norb = 2
        nelec = (1, 1)

        with pytest.raises(ValueError, match="h2e shape.*inconsistent"):
            fci.solve(h1e, h2e, norb, nelec)

    def test_odd_nelec_integer(self):
        """Test error for odd integer electron count."""
        from quantum_fragment_methods.application.solvers.classical_zoo import FCI

        fci = FCI()
        h1e = np.zeros((2, 2))
        h2e = np.zeros((2, 2, 2, 2))
        norb = 2
        nelec = 3  # Odd number

        with pytest.raises(ValueError, match="must be even"):
            fci.solve(h1e, h2e, norb, nelec)


class TestFCIConvenienceMethods:
    """Test FCI convenience methods."""

    @pytest.mark.requires_pyscf
    def test_solve_from_integrals(self):
        """Test solve_from_integrals convenience method."""
        from quantum_fragment_methods.application.solvers.classical_zoo import FCI

        fci = FCI()
        h1e = np.array([[-1.25, -0.48], [-0.48, -1.25]])
        h2e = np.zeros((2, 2, 2, 2))
        h2e[0, 0, 0, 0] = 0.67
        h2e[1, 1, 1, 1] = 0.67

        result = fci.solve_from_integrals(h1e, h2e, norb=2, nocc=1)

        assert result.energy is not None
        assert result.metadata["nelec"] == (1, 1)


# ============================================================================
# CCSD Solver Tests
# ============================================================================


class TestCCSDInitialization:
    """Test CCSD solver initialization and configuration."""

    def test_initialization_default(self):
        """Test CCSD initialization with default parameters."""
        from quantum_fragment_methods.application.solvers.classical_zoo import CCSD

        ccsd = CCSD()
        assert ccsd.conv_tol == 1e-7
        assert ccsd.max_cycle == 50
        assert ccsd.diis_space == 6
        assert ccsd.verbose == 0

    def test_initialization_custom_parameters(self):
        """Test CCSD initialization with custom parameters."""
        from quantum_fragment_methods.application.solvers.classical_zoo import CCSD

        ccsd = CCSD(conv_tol=1e-9, max_cycle=100, diis_space=8, verbose=4)
        assert ccsd.conv_tol == 1e-9
        assert ccsd.max_cycle == 100
        assert ccsd.diis_space == 8
        assert ccsd.verbose == 4

    def test_solver_name(self):
        """Test that solver has correct name."""
        from quantum_fragment_methods.application.solvers.classical_zoo import CCSD

        ccsd = CCSD()
        assert ccsd.name == "CCSD"


class TestCCSDSolve:
    """Test CCSD solve method with various inputs."""

    @pytest.fixture
    def h2_mf(self):
        """Provide H2 mean-field object for testing."""
        pytest.importorskip("pyscf")
        from pyscf import gto, scf

        mol = gto.Mole()
        mol.atom = "H 0 0 0; H 0 0 0.74"
        mol.basis = "sto-3g"
        mol.build()

        mf = scf.RHF(mol)
        mf.kernel()

        return mf

    @pytest.mark.requires_pyscf
    def test_solve_from_mf(self, h2_mf):
        """Test CCSD solve from mean-field object."""
        from quantum_fragment_methods.application.solvers.classical_zoo import CCSD

        ccsd = CCSD()
        result = ccsd.solve(h2_mf)

        # Check result structure
        assert result.energy is not None
        assert isinstance(result.energy, float)
        assert result.metadata is not None

    @pytest.mark.requires_pyscf
    def test_solve_with_rdms(self, h2_mf):
        """Test CCSD solve with RDM computation."""
        from quantum_fragment_methods.application.solvers.classical_zoo import CCSD

        ccsd = CCSD()
        result = ccsd.solve(h2_mf, compute_rdms=True)

        assert result.rdm1 is not None
        assert result.rdm2 is not None
        assert result.rdm1.shape == (h2_mf.mol.nao, h2_mf.mol.nao)

    @pytest.mark.requires_pyscf
    def test_solve_metadata(self, h2_mf):
        """Test that metadata contains expected information."""
        from quantum_fragment_methods.application.solvers.classical_zoo import CCSD

        ccsd = CCSD(conv_tol=1e-9)
        result = ccsd.solve(h2_mf)

        # Check metadata
        assert "converged" in result.metadata
        assert "e_corr" in result.metadata
        assert "conv_tol" in result.metadata
        assert result.metadata["conv_tol"] == 1e-9

    @pytest.mark.requires_pyscf
    def test_solve_convergence(self, h2_mf):
        """Test that CCSD converges for simple system."""
        from quantum_fragment_methods.application.solvers.classical_zoo import CCSD

        ccsd = CCSD()
        result = ccsd.solve(h2_mf)

        assert result.metadata["converged"] is True


class TestCCSDFromIntegrals:
    """Test CCSD solve from integrals."""

    @pytest.mark.requires_pyscf
    def test_solve_from_integrals(self):
        """Test CCSD solve from Hamiltonian integrals."""
        from quantum_fragment_methods.application.solvers.classical_zoo import CCSD

        ccsd = CCSD()

        # Simple 2-orbital system
        h1e = np.array([[-1.25, -0.48], [-0.48, -1.25]])
        h2e = np.zeros((2, 2, 2, 2))
        h2e[0, 0, 0, 0] = 0.67
        h2e[1, 1, 1, 1] = 0.67
        h2e[0, 0, 1, 1] = 0.66
        h2e[1, 1, 0, 0] = 0.66

        result = ccsd.solve_from_integrals(h1e, h2e, norb=2, nocc=1)

        assert result.energy is not None
        assert isinstance(result.energy, float)


class TestCCSDErrorHandling:
    """Test CCSD error handling for invalid inputs."""

    def test_invalid_mf_object(self):
        """Test error for invalid mean-field object."""
        from quantum_fragment_methods.application.solvers.classical_zoo import CCSD

        ccsd = CCSD()

        with pytest.raises((TypeError, AttributeError)):
            ccsd.solve(None)

    @pytest.mark.requires_pyscf
    def test_unconverged_warning(self):
        """Test handling of unconverged CCSD."""
        from quantum_fragment_methods.application.solvers.classical_zoo import CCSD

        pytest.importorskip("pyscf")
        from pyscf import gto, scf

        # Create a system that might not converge easily
        mol = gto.Mole()
        mol.atom = "H 0 0 0; H 0 0 0.74"
        mol.basis = "sto-3g"
        mol.build()

        mf = scf.RHF(mol)
        mf.kernel()

        # Use very tight convergence to potentially fail
        ccsd = CCSD(conv_tol=1e-15, max_cycle=1)
        result = ccsd.solve(mf)

        # Should still return a result even if not converged
        assert result.energy is not None


# ============================================================================
# Solver Comparison Tests
# ============================================================================


class TestSolverComparison:
    """Compare solver behaviors for consistency."""

    @pytest.mark.requires_pyscf
    def test_solvers_return_lower_energy_than_hf(self):
        """Test that correlation methods lower the energy compared to HF."""
        pytest.importorskip("pyscf")
        from pyscf import gto, scf

        from quantum_fragment_methods.application.solvers.classical_zoo import CCSD

        mol = gto.Mole()
        mol.atom = "H 0 0 0; H 0 0 0.74"
        mol.basis = "sto-3g"
        mol.build()

        mf = scf.RHF(mol)
        mf.kernel()
        e_hf = mf.e_tot

        # CCSD should give lower energy than HF
        ccsd = CCSD()
        ccsd_result = ccsd.solve(mf)

        assert ccsd_result.energy < e_hf, "CCSD should lower energy compared to HF"
        assert ccsd_result.metadata["e_corr"] < 0, "Correlation energy should be negative"


# ============================================================================
# Integration Tests
# ============================================================================


class TestSolverIntegration:
    """Test solver integration with workflow."""

    @pytest.mark.requires_pyscf
    def test_solver_result_structure(self):
        """Test that all solvers return consistent SolverResult structure."""
        from quantum_fragment_methods.application.solvers.base import SolverResult
        from quantum_fragment_methods.application.solvers.classical_zoo import CCSD, FCI

        pytest.importorskip("pyscf")
        from pyscf import ao2mo, gto, scf

        mol = gto.Mole()
        mol.atom = "H 0 0 0; H 0 0 0.74"
        mol.basis = "sto-3g"
        mol.build()

        mf = scf.RHF(mol)
        mf.kernel()

        # Test FCI - need to restore 4D ERI
        fci = FCI()
        h1e = mf.get_hcore()
        norb = mf.mol.nao
        h2e = ao2mo.restore(1, mf._eri, norb)
        fci_result = fci.solve(h1e, h2e, norb, (1, 1))
        assert isinstance(fci_result, SolverResult)

        # Test CCSD
        ccsd = CCSD()
        ccsd_result = ccsd.solve(mf)
        assert isinstance(ccsd_result, SolverResult)
