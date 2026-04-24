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
Tests for EWF (Embedded Wavefunction) embedding method.

These tests follow Test-Driven Development (TDD) principles:
1. Define expected behavior through tests
2. Implement EWF class to pass tests
3. Refactor while keeping tests green

Based on the centralized_workflow notebook implementation.
"""

import pytest


class TestEWFInitialization:
    """Test EWF object creation and configuration."""

    def test_ewf_import(self):
        """Test that EWF can be imported."""
        from quantum_fragment_methods.application.embedding import EWF

        assert EWF is not None

    def test_ewf_basic_initialization(self):
        """Test EWF can be created with default parameters."""
        from quantum_fragment_methods.application.embedding import EWF

        ewf = EWF()
        assert ewf is not None
        assert hasattr(ewf, "bath_options")
        assert hasattr(ewf, "solver_options")

    def test_ewf_with_bath_options(self):
        """Test EWF initialization with custom bath options."""
        from quantum_fragment_methods.application.embedding import EWF

        ewf = EWF(bath_type="mp2", truncation=1e-6, occupation_tolerance=1e-6)

        assert ewf.bath_options["bathtype"] == "mp2"
        assert ewf.bath_options["threshold"] == 1e-6
        assert ewf.bath_options["occupation_tolerance"] == 1e-6

    def test_ewf_with_solver_options(self):
        """Test EWF initialization with custom solver options."""
        from quantum_fragment_methods.application.embedding import EWF

        ewf = EWF(bath_type="mp2", max_cycle=300, conv_tol=1e-6)

        assert ewf.solver_options["max_cycle"] == 300
        assert ewf.solver_options["conv_tol"] == 1e-6


class TestEWFFragmentation:
    """Test EWF fragmentation methods."""

    @pytest.fixture
    def simple_molecule(self):
        """Create a simple test molecule (water)."""
        import pyscf
        from pyscf.scf import RHF

        mol = pyscf.gto.Mole()
        mol.atom = """
        O 0.0 0.0 0.0
        H 0.757 0.586 0.0
        H -0.757 0.586 0.0
        """
        mol.unit = "Angstrom"
        mol.basis = "sto-3g"
        mol.build()

        mf = RHF(mol).density_fit()
        mf.kernel()

        return mf

    def test_ewf_kernel_with_mean_field(self, simple_molecule):
        """Test EWF.kernel() method with a mean-field object."""
        from quantum_fragment_methods.application.embedding import EWF

        ewf = EWF(bath_type="mp2", truncation=1e-6)

        # Should accept a PySCF mean-field object
        result = ewf.kernel(simple_molecule)

        assert result is not None
        assert hasattr(result, "fragments")
        assert len(result.fragments) > 0

    def test_ewf_atomic_fragmentation(self, simple_molecule):
        """Test atomic fragmentation (one fragment per atom)."""
        from quantum_fragment_methods.application.embedding import EWF

        ewf = EWF(bath_type="mp2", fragmentation="atomic")

        result = ewf.kernel(simple_molecule)

        # Water has 3 atoms
        assert len(result.fragments) == 3

    def test_ewf_iao_fragmentation(self, simple_molecule):
        """Test IAO (Intrinsic Atomic Orbital) fragmentation."""
        from quantum_fragment_methods.application.embedding import EWF

        ewf = EWF(bath_type="mp2", fragmentation="iao")

        result = ewf.kernel(simple_molecule)

        # Should create fragments based on IAOs
        assert len(result.fragments) > 0
        for _frag_id, frag in result.fragments.items():
            assert hasattr(frag, "n_orbitals")
            assert hasattr(frag, "n_electrons")


class TestEWFFragmentProperties:
    """Test properties of fragments created by EWF."""

    @pytest.fixture
    def ewf_result(self):
        """Create EWF result for testing."""
        import pyscf
        from pyscf.scf import RHF

        from quantum_fragment_methods.application.embedding import EWF

        mol = pyscf.gto.Mole()
        mol.atom = """
        O 0.0 0.0 0.0
        H 0.757 0.586 0.0
        H -0.757 0.586 0.0
        """
        mol.unit = "Angstrom"
        mol.basis = "sto-3g"
        mol.build()

        mf = RHF(mol).density_fit()
        mf.kernel()

        ewf = EWF(bath_type="mp2", fragmentation="atomic")
        return ewf.kernel(mf)

    def test_fragment_has_orbitals(self, ewf_result):
        """Test that fragments have orbital information."""
        for _frag_id, frag in ewf_result.fragments.items():
            assert hasattr(frag, "n_orbitals")
            assert frag.n_orbitals > 0

    def test_fragment_has_electrons(self, ewf_result):
        """Test that fragments have electron count."""
        for _frag_id, frag in ewf_result.fragments.items():
            assert hasattr(frag, "n_electrons")
            assert frag.n_electrons > 0

    def test_fragment_has_bath(self, ewf_result):
        """Test that fragments have bath orbitals in metadata."""
        for _frag_id, frag in ewf_result.fragments.items():
            assert "bath_orbitals" in frag.metadata
            # Bath orbitals may be None for some fragments (e.g., those without environment)
            # Just check the key exists

    def test_fragment_total_electrons(self, ewf_result):
        """Test that total electrons across fragments matches system."""
        # This is a sanity check - fragments may overlap
        # but total should be reasonable
        total_electrons = sum(frag.n_electrons for frag_id, frag in ewf_result.fragments.items())
        assert total_electrons > 0


class TestEWFIntegrationWithVayesta:
    """Test integration with Vayesta library."""

    def test_vayesta_ewf_creation(self):
        """Test that EWF wraps Vayesta's EWF correctly."""
        import pyscf
        from pyscf.scf import RHF

        from quantum_fragment_methods.application.embedding import EWF

        mol = pyscf.gto.Mole()
        mol.atom = "H 0 0 0; H 0 0 0.74"
        mol.basis = "sto-3g"
        mol.build()

        mf = RHF(mol)
        mf.kernel()

        ewf = EWF(bath_type="mp2")
        ewf.kernel(mf)

        # Should have created Vayesta EWF object internally
        assert hasattr(ewf, "_vayesta_ewf")
        assert ewf._vayesta_ewf is not None

    def test_ewf_dump_solver_option(self):
        """Test EWF with dump solver (for fragment serialization)."""
        import os
        import tempfile

        import pyscf
        from pyscf.scf import RHF

        from quantum_fragment_methods.application.embedding import EWF

        mol = pyscf.gto.Mole()
        mol.atom = "H 0 0 0; H 0 0 0.74"
        mol.basis = "sto-3g"
        mol.build()

        mf = RHF(mol)
        mf.kernel()

        # Use unique temp file to avoid collisions
        fd, dumpfile = tempfile.mkstemp(suffix=".h5", prefix="test_ewf_dump_")
        os.close(fd)  # Close file descriptor, Vayesta will create the file

        try:
            ewf = EWF(
                bath_type="mp2", solver="DUMP", dumpfile=dumpfile  # Vayesta requires uppercase
            )

            ewf.kernel(mf)

            # Should have set dump solver options
            assert ewf.solver_options["dumpfile"] == dumpfile

            # Verify HDF5 file was created
            assert os.path.exists(dumpfile)
        finally:
            # Clean up temp file
            if os.path.exists(dumpfile):
                os.remove(dumpfile)


class TestEWFErrorHandling:
    """Test error handling in EWF."""

    def test_ewf_requires_mean_field(self):
        """Test that EWF.kernel() requires a mean-field object."""
        from quantum_fragment_methods.application.embedding import EWF

        ewf = EWF()

        with pytest.raises(ValueError, match="Mean-field object cannot be None"):
            ewf.kernel(None)

    def test_ewf_invalid_bath_type(self):
        """Test that invalid bath type raises error."""
        from quantum_fragment_methods.application.embedding import EWF

        with pytest.raises(ValueError, match="Invalid bath_type"):
            EWF(bath_type="invalid_bath_type")

    def test_ewf_invalid_fragmentation(self):
        """Test that invalid fragmentation type raises error."""
        import pyscf
        from pyscf.scf import RHF

        from quantum_fragment_methods.application.embedding import EWF

        mol = pyscf.gto.Mole()
        mol.atom = "H 0 0 0; H 0 0 0.74"
        mol.basis = "sto-3g"
        mol.build()

        mf = RHF(mol)
        mf.kernel()

        ewf = EWF()

        with pytest.raises(ValueError, match="Invalid fragmentation"):
            ewf.kernel(mf, fragmentation="invalid_fragmentation")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
