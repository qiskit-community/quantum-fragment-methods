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
Comprehensive test suite for QFWorkflow.

This module tests the workflow orchestration, solver assignment rules,
and integration with embedding methods.
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest

# ============================================================================
# QFWorkflow Initialization Tests
# ============================================================================


class TestQFWorkflowInitialization:
    """Test QFWorkflow initialization and configuration."""

    def test_initialization_with_embedder(self, h2_xyz):
        """Test workflow initialization with embedder."""
        from quantum_fragment_methods.application.embedding import EWF
        from quantum_fragment_methods.workflow import QFWorkflow

        embedder = EWF(bath_size=4, truncation=1e-6)
        workflow = QFWorkflow(
            geometry=h2_xyz, basis="sto-3g", embedder=embedder, save_path="test_results/"
        )

        assert workflow.geometry == h2_xyz
        assert workflow.basis == "sto-3g"
        assert workflow.embedder == embedder
        assert workflow.save_path == "test_results/"
        assert workflow.mf is None
        assert workflow.embedding_result is None
        assert len(workflow.solver_rules) == 0
        assert workflow.default_solver is None

    def test_initialization_without_embedder(self, h2_xyz):
        """Test workflow initialization without embedder (for full system)."""
        from quantum_fragment_methods.workflow import QFWorkflow

        workflow = QFWorkflow(geometry=h2_xyz, basis="sto-3g", embedder=None)

        assert workflow.embedder is None


# ============================================================================
# Solver Rule Tests
# ============================================================================


class TestSolverRules:
    """Test solver rule creation and assignment."""

    def test_solver_rule_creation(self):
        """Test SolverRule initialization."""
        from quantum_fragment_methods.application.solvers.classical_zoo import CCSD
        from quantum_fragment_methods.workflow import SolverRule

        rule = SolverRule(
            solver_factory=lambda frag: CCSD(system=frag),
            condition=lambda frag: frag.n_orbitals < 10,
            priority=10,
        )

        assert rule.priority == 10
        assert callable(rule.solver_factory)
        assert callable(rule.condition)

    def test_solver_rule_applies_to(self):
        """Test SolverRule.applies_to() method."""
        from quantum_fragment_methods.application.embedding.base import Fragment
        from quantum_fragment_methods.workflow import SolverRule

        # Create rule for small fragments
        rule = SolverRule(
            solver_factory=lambda frag: Mock(),
            condition=lambda frag: frag.n_orbitals < 6,
            priority=10,
        )

        # Small fragment should match
        small_frag = Fragment(
            fragment_id=0, atom_indices=[0], orbital_indices=[0, 1, 2, 3], n_electrons=4
        )
        assert rule.applies_to(small_frag) is True

        # Large fragment should not match
        large_frag = Fragment(
            fragment_id=1, atom_indices=[1], orbital_indices=list(range(10)), n_electrons=10
        )
        assert rule.applies_to(large_frag) is False

    def test_solver_rule_default_condition(self):
        """Test SolverRule with None condition (applies to all)."""
        from quantum_fragment_methods.application.embedding.base import Fragment
        from quantum_fragment_methods.workflow import SolverRule

        rule = SolverRule(
            solver_factory=lambda frag: Mock(), condition=None, priority=0  # Should apply to all
        )

        frag = Fragment(fragment_id=0, atom_indices=[0], orbital_indices=[0, 1], n_electrons=2)
        assert rule.applies_to(frag) is True

    def test_add_solver_rule(self, h2_xyz):
        """Test adding solver rules to workflow."""
        from quantum_fragment_methods.application.embedding import EWF
        from quantum_fragment_methods.application.solvers.classical_zoo import CCSD, FCI
        from quantum_fragment_methods.workflow import QFWorkflow

        embedder = EWF(bath_size=4)
        workflow = QFWorkflow(geometry=h2_xyz, basis="sto-3g", embedder=embedder)

        # Add first rule
        workflow.add_solver_rule(
            solver_factory=lambda frag: FCI(system=frag),
            condition=lambda frag: frag.n_orbitals < 6,
            priority=20,
        )
        assert len(workflow.solver_rules) == 1
        assert workflow.solver_rules[0].priority == 20

        # Add second rule
        workflow.add_solver_rule(
            solver_factory=lambda frag: CCSD(system=frag), condition=None, priority=10
        )
        assert len(workflow.solver_rules) == 2

        # Rules should be sorted by priority (highest first)
        assert workflow.solver_rules[0].priority == 20
        assert workflow.solver_rules[1].priority == 10

    def test_set_default_solver(self, h2_xyz):
        """Test setting default solver."""
        from quantum_fragment_methods.application.embedding import EWF
        from quantum_fragment_methods.application.solvers.classical_zoo import CCSD
        from quantum_fragment_methods.workflow import QFWorkflow

        embedder = EWF(bath_size=4)
        workflow = QFWorkflow(geometry=h2_xyz, basis="sto-3g", embedder=embedder)

        workflow.set_default_solver(lambda frag: CCSD(system=frag))
        assert workflow.default_solver is not None
        assert callable(workflow.default_solver)


# ============================================================================
# Mean-Field Tests
# ============================================================================


class TestMeanField:
    """Test mean-field calculation in workflow."""

    @pytest.mark.requires_pyscf
    def test_run_mean_field_from_xyz_string(self, h2_xyz):
        """Test mean-field calculation from XYZ string."""
        from quantum_fragment_methods.application.embedding import EWF
        from quantum_fragment_methods.workflow import QFWorkflow

        embedder = EWF(bath_size=4)
        workflow = QFWorkflow(geometry=h2_xyz, basis="sto-3g", embedder=embedder)

        workflow.run_mean_field()

        assert workflow.mf is not None
        assert workflow.mol is not None
        assert hasattr(workflow.mf, "e_tot")
        assert hasattr(workflow.mf, "mo_coeff")

        # Check that energy is reasonable for H2
        assert -2.0 < workflow.mf.e_tot < -1.0

    @pytest.mark.requires_pyscf
    def test_run_mean_field_from_file(self, tmp_path):
        """Test mean-field calculation from XYZ file."""
        from quantum_fragment_methods.application.embedding import EWF
        from quantum_fragment_methods.workflow import QFWorkflow

        # Write proper XYZ file with header
        xyz_content = """2
H2 molecule
H 0 0 0
H 0 0 0.74
"""
        xyz_file = tmp_path / "h2.xyz"
        xyz_file.write_text(xyz_content)

        embedder = EWF(bath_size=4)
        workflow = QFWorkflow(geometry=str(xyz_file), basis="sto-3g", embedder=embedder)

        workflow.run_mean_field()

        assert workflow.mf is not None
        assert workflow.mol is not None


# ============================================================================
# Fragment Creation Tests
# ============================================================================


class TestFragmentCreation:
    """Test fragment creation with different embedders."""

    @pytest.mark.requires_vayesta
    @pytest.mark.requires_pyscf
    def test_create_fragments_with_ewf(self, h2_xyz):
        """Test fragment creation with EWF embedder."""
        from quantum_fragment_methods.application.embedding import EWF
        from quantum_fragment_methods.workflow import QFWorkflow

        embedder = EWF(bath_size=4, truncation=1e-6)
        workflow = QFWorkflow(geometry=h2_xyz, basis="sto-3g", embedder=embedder)

        workflow.run_mean_field()
        result = workflow.create_fragments()

        assert result is not None
        assert workflow.embedding_result is not None
        assert len(workflow.embedding_result.fragments) > 0

        # Check fragment properties
        for _frag_id, fragment in workflow.embedding_result.fragments.items():
            assert fragment.n_orbitals > 0
            assert fragment.n_electrons > 0
            assert isinstance(fragment.atom_indices, list)

    def test_create_fragments_without_mean_field(self, h2_xyz):
        """Test that create_fragments fails without mean-field."""
        from quantum_fragment_methods.application.embedding import EWF
        from quantum_fragment_methods.workflow import QFWorkflow

        embedder = EWF(bath_size=4)
        workflow = QFWorkflow(geometry=h2_xyz, basis="sto-3g", embedder=embedder)

        with pytest.raises(RuntimeError, match="Must run mean-field"):
            workflow.create_fragments()


# ============================================================================
# Solver Assignment Tests
# ============================================================================


class TestSolverAssignment:
    """Test solver assignment to fragments."""

    def test_assign_solvers_priority_order(self):
        """Test that solvers are assigned based on priority."""
        from quantum_fragment_methods.application.embedding.base import EmbeddingResult, Fragment
        from quantum_fragment_methods.application.solvers.classical_zoo import CCSD, FCI
        from quantum_fragment_methods.workflow import QFWorkflow

        # Create mock workflow with fragments
        workflow = QFWorkflow(geometry="", basis="sto-3g", embedder=Mock())

        # Create test fragments
        small_frag = Fragment(
            fragment_id=0, atom_indices=[0], orbital_indices=[0, 1, 2, 3], n_electrons=4
        )
        large_frag = Fragment(
            fragment_id=1, atom_indices=[1], orbital_indices=list(range(10)), n_electrons=10
        )

        workflow.embedding_result = EmbeddingResult(fragments={0: small_frag, 1: large_frag})

        # Add rules (higher priority first)
        workflow.add_solver_rule(
            solver_factory=lambda frag: FCI(),
            condition=lambda frag: frag.n_orbitals < 6,
            priority=20,
        )
        workflow.add_solver_rule(solver_factory=lambda frag: CCSD(), condition=None, priority=10)

        # Assign solvers
        solvers = workflow._assign_solvers()

        # Small fragment should get FCI (higher priority)
        assert isinstance(solvers[0], FCI)
        # Large fragment should get CCSD (default rule)
        assert isinstance(solvers[1], CCSD)

    def test_assign_solvers_no_match_with_default(self):
        """Test solver assignment when no rule matches but default exists."""
        from quantum_fragment_methods.application.embedding.base import EmbeddingResult, Fragment
        from quantum_fragment_methods.application.solvers.classical_zoo import CCSD
        from quantum_fragment_methods.workflow import QFWorkflow

        workflow = QFWorkflow(geometry="", basis="sto-3g", embedder=Mock())

        frag = Fragment(fragment_id=0, atom_indices=[0], orbital_indices=[0, 1], n_electrons=2)
        workflow.embedding_result = EmbeddingResult(fragments={0: frag})

        # Add rule that won't match
        workflow.add_solver_rule(
            solver_factory=lambda frag: Mock(),
            condition=lambda frag: frag.n_orbitals > 100,
            priority=10,
        )

        # Set default solver
        workflow.set_default_solver(lambda frag: CCSD())

        solvers = workflow._assign_solvers()
        assert isinstance(solvers[0], CCSD)

    def test_assign_solvers_no_match_no_default(self):
        """Test that assignment fails when no rule matches and no default."""
        from quantum_fragment_methods.application.embedding.base import EmbeddingResult, Fragment
        from quantum_fragment_methods.workflow import QFWorkflow

        workflow = QFWorkflow(geometry="", basis="sto-3g", embedder=Mock())

        frag = Fragment(fragment_id=0, atom_indices=[0], orbital_indices=[0, 1], n_electrons=2)
        workflow.embedding_result = EmbeddingResult(fragments={0: frag})

        # Add rule that won't match
        workflow.add_solver_rule(
            solver_factory=lambda frag: Mock(),
            condition=lambda frag: frag.n_orbitals > 100,
            priority=10,
        )

        # No default solver set
        with pytest.raises(RuntimeError, match="No solver rule matched"):
            workflow._assign_solvers()


# ============================================================================
# Fragment Solving Tests
# ============================================================================


class TestFragmentSolving:
    """Test solving fragments with different data formats."""

    def test_solve_fragments_with_hamiltonian(self):
        """Test solving fragments with pre-computed Hamiltonian."""
        from quantum_fragment_methods.application.embedding.base import EmbeddingResult, Fragment
        from quantum_fragment_methods.application.solvers.base import SolverResult
        from quantum_fragment_methods.application.solvers.classical_zoo import CCSD
        from quantum_fragment_methods.workflow import QFWorkflow

        workflow = QFWorkflow(geometry="", basis="sto-3g", embedder=Mock())

        # Create fragment with Hamiltonian data
        h1e = np.random.rand(4, 4)
        h1e = h1e + h1e.T  # Make symmetric
        h2e = np.random.rand(4, 4, 4, 4)

        frag = Fragment(
            fragment_id=0,
            atom_indices=[0],
            orbital_indices=[0, 1, 2, 3],
            n_electrons=4,
            metadata={"hamiltonian": {"h1e": h1e, "h2e": h2e}},
        )

        workflow.embedding_result = EmbeddingResult(fragments={0: frag})
        workflow.add_solver_rule(solver_factory=lambda frag: CCSD(), condition=None, priority=0)

        # Mock the solver's solve_from_integrals method
        with patch.object(CCSD, "solve_from_integrals") as mock_solve:
            mock_solve.return_value = SolverResult(energy=-1.5)

            results = workflow.solve_fragments()

            assert 0 in results
            assert results[0].energy == -1.5
            mock_solve.assert_called_once()

    def test_solve_fragments_with_vayesta_fragment(self):
        """Test solving fragments with Vayesta fragment data."""
        from quantum_fragment_methods.application.embedding.base import EmbeddingResult, Fragment
        from quantum_fragment_methods.application.solvers.base import SolverResult
        from quantum_fragment_methods.application.solvers.classical_zoo import CCSD
        from quantum_fragment_methods.workflow import QFWorkflow

        workflow = QFWorkflow(geometry="", basis="sto-3g", embedder=Mock())

        # Create mock Vayesta fragment with cluster
        mock_cluster = Mock()
        mock_cluster.norb = 4
        mock_cluster.nelec = 4
        mock_cluster.get_heff = Mock(return_value=np.random.rand(4, 4))
        mock_cluster.get_eris_bare = Mock(return_value=np.random.rand(4, 4, 4, 4))

        mock_vfrag = Mock()
        mock_vfrag.cluster = mock_cluster

        frag = Fragment(
            fragment_id=0,
            atom_indices=[0],
            orbital_indices=[0, 1, 2, 3],
            n_electrons=4,
            metadata={"vayesta_fragment": mock_vfrag},
        )

        workflow.embedding_result = EmbeddingResult(fragments={0: frag})
        workflow.add_solver_rule(solver_factory=lambda frag: CCSD(), condition=None, priority=0)

        # Mock the solver's solve_from_integrals method
        with patch.object(CCSD, "solve_from_integrals") as mock_solve:
            mock_solve.return_value = SolverResult(energy=-2.0)

            results = workflow.solve_fragments()

            assert 0 in results
            assert results[0].energy == -2.0
            mock_cluster.get_heff.assert_called_once()
            mock_cluster.get_eris_bare.assert_called_once()

    def test_solve_fragments_missing_data(self):
        """Test that solving fails when fragment has no Hamiltonian data."""
        from quantum_fragment_methods.application.embedding.base import EmbeddingResult, Fragment
        from quantum_fragment_methods.application.solvers.classical_zoo import CCSD
        from quantum_fragment_methods.workflow import QFWorkflow

        workflow = QFWorkflow(geometry="", basis="sto-3g", embedder=Mock())

        # Fragment with no Hamiltonian data
        frag = Fragment(
            fragment_id=0,
            atom_indices=[0],
            orbital_indices=[0, 1],
            n_electrons=2,
            metadata={},  # No hamiltonian, mean_field, or vayesta_fragment
        )

        workflow.embedding_result = EmbeddingResult(fragments={0: frag})
        workflow.add_solver_rule(solver_factory=lambda frag: CCSD(), condition=None, priority=0)

        with pytest.raises(RuntimeError, match="missing Hamiltonian data"):
            workflow.solve_fragments()

    def test_solve_fragments_without_embedding_result(self):
        """Test that solving fails without embedding result."""
        from quantum_fragment_methods.workflow import QFWorkflow

        workflow = QFWorkflow(geometry="", basis="sto-3g", embedder=Mock())

        with pytest.raises(RuntimeError, match="Must create fragments"):
            workflow.solve_fragments()


# ============================================================================
# Integration Tests
# ============================================================================


class TestWorkflowIntegration:
    """Integration tests for complete workflow."""

    @pytest.mark.integration
    @pytest.mark.requires_pyscf
    @pytest.mark.requires_vayesta
    @pytest.mark.slow
    def test_full_workflow_h2(self, h2_xyz):
        """Test complete workflow on H2 molecule."""
        from quantum_fragment_methods.application.embedding import EWF
        from quantum_fragment_methods.application.solvers.classical_zoo import CCSD
        from quantum_fragment_methods.workflow import QFWorkflow

        embedder = EWF(bath_size=4, truncation=1e-6)
        workflow = QFWorkflow(geometry=h2_xyz, basis="sto-3g", embedder=embedder)

        # Add solver rule
        workflow.add_solver_rule(solver_factory=lambda frag: CCSD(), condition=None, priority=0)

        # Run workflow
        result = workflow.run()

        assert result is not None
        assert hasattr(result, "total_energy")
        assert hasattr(result, "fragment_results")
        assert hasattr(result, "mf_energy")

        # Energy should be reasonable for H2 with EWF fragmentation
        # Each fragment includes nuclear repulsion and bath orbitals
        # Total energy is sum of fragment energies (not mean-field + correlation)
        assert -5.0 < result.total_energy < -2.0

        # Should have 2 fragments (one per H atom)
        assert len(result.fragment_results) == 2


# ============================================================================
# WorkflowResult Tests
# ============================================================================


class TestWorkflowResult:
    """Test WorkflowResult container."""

    def test_workflow_result_creation(self):
        """Test WorkflowResult initialization."""
        from quantum_fragment_methods.workflow import WorkflowResult

        result = WorkflowResult(
            total_energy=-1.5, fragment_results={0: Mock(), 1: Mock()}, mf_energy=-1.4
        )

        assert result.total_energy == -1.5
        assert len(result.fragment_results) == 2
        assert result.mf_energy == -1.4

    def test_workflow_result_repr(self):
        """Test WorkflowResult string representation."""
        from quantum_fragment_methods.workflow import WorkflowResult

        result = WorkflowResult(total_energy=-1.5, fragment_results={}, mf_energy=-1.4)

        repr_str = repr(result)
        assert "total_energy" in repr_str or "-1.5" in str(result.total_energy)
