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

# quantum_fragment_methods/workflow.py

from .application.embedding.base import EmbeddingResult


class SolverRule:
    """Rule for assigning solvers to fragments based on fragment properties."""

    def __init__(self, solver_factory, condition=None, priority=0):
        """
        Args:
            solver_factory: Callable that creates a solver instance for a fragment
            condition: Callable that takes a fragment and returns True if rule applies
                      If None, applies to all fragments (default rule)
            priority: Higher priority rules are evaluated first
        """
        self.solver_factory = solver_factory
        self.condition = condition if condition is not None else lambda frag: True
        self.priority = priority

    def applies_to(self, fragment):
        """Check if this rule applies to the given fragment."""
        return self.condition(fragment)

    def create_solver(self, fragment):
        """Create solver instance for the fragment."""
        return self.solver_factory(fragment)


class QFWorkflow:
    """Orchestrator for quantum fragment calculations (EWF, DMET, MBE)."""

    def __init__(
        self, geometry, basis, embedder=None, save_path="results/", fragmentation="atomic", **kwargs
    ):
        """
        Initialize quantum fragment workflow.

        Parameters
        ----------
        geometry : str or array-like
            Molecular geometry (XYZ string or coordinates)
        basis : str
            Basis set name (e.g., 'sto-3g', '6-31g')
        embedder : BaseEmbedder, optional
            Embedding method instance (EWF, DMET, or MBE). If None, workflow
            can only run mean-field calculations (no fragmentation).
        save_path : str, optional
            Directory for saving results (default: 'results/')
        fragmentation : str, optional
            Fragmentation scheme for EWF: 'atomic' or 'iao' (default: 'atomic')
        **kwargs : dict
            Additional workflow configuration options

        Examples
        --------
        >>> from quantum_fragment_methods.application.embedding import EWF
        >>> embedder = EWF(bath_type='mp2', truncation=1e-6)
        >>> workflow = QFWorkflow(
        ...     geometry=xyz_string,
        ...     basis='sto-3g',
        ...     embedder=embedder,
        ...     fragmentation='iao'
        ... )
        """
        self.geometry = geometry
        self.basis = basis
        self.embedder = embedder
        self.save_path = save_path
        self.fragmentation = fragmentation
        self.embedding_options = kwargs  # Store additional options for embedder
        self.solver_rules = []
        self.default_solver = None
        self.mf = None
        self.embedding_result = None

    def add_solver_rule(self, solver_factory, condition=None, priority=0):
        """
        Add a rule for assigning solvers to fragments.

        Args:
            solver_factory: Callable that creates a solver for a fragment
            condition: Callable(fragment) -> bool. If None, applies to all fragments.
            priority: Higher priority rules are checked first

        Examples:
            # Assign CCSD to fragments with < 10 orbitals
            workflow.add_solver_rule(
                lambda frag: CCSDSolver(frag),
                condition=lambda frag: frag.n_orbitals < 10,
                priority=10
            )

            # Assign FCI to small fragments
            workflow.add_solver_rule(
                lambda frag: FCISolver(frag),
                condition=lambda frag: frag.n_orbitals < 6,
                priority=20  # Higher priority, checked first
            )

            # Default solver for all other fragments
            workflow.add_solver_rule(
                lambda frag: DMRGSolver(frag),
                condition=None,  # Applies to all
                priority=0
            )
        """
        rule = SolverRule(solver_factory, condition, priority)
        self.solver_rules.append(rule)
        # Keep rules sorted by priority (highest first)
        self.solver_rules.sort(key=lambda r: r.priority, reverse=True)

    def set_default_solver(self, solver_factory):
        """
        Set default solver for fragments that don't match any rule.

        Args:
            solver_factory: Callable that creates a solver for a fragment
        """
        self.default_solver = solver_factory

    def run_mean_field(self):
        """
        Run Hartree-Fock mean-field calculation using PySCF.

        This method:
        1. Parses the geometry (XYZ file or string)
        2. Creates a PySCF Mole object
        3. Runs RHF calculation with density fitting
        4. Stores the mean-field object for fragment creation
        """
        import os

        import pyscf
        from pyscf.scf import RHF

        # Parse geometry
        if isinstance(self.geometry, str):
            if os.path.isfile(self.geometry):
                # Read from XYZ file
                with open(self.geometry) as f:
                    lines = f.readlines()
                    atom_data = "".join(lines[2:])  # Skip first two lines (atom count and comment)
            else:
                # Assume it's an XYZ string
                atom_data = self.geometry
        else:
            raise ValueError("geometry must be a file path or XYZ string")

        # Create PySCF molecule
        mol = pyscf.gto.Mole()
        mol.atom = atom_data
        mol.unit = "Angstrom"
        mol.basis = self.basis
        mol.verbose = 4
        mol.build()

        # Run Hartree-Fock with density fitting
        mf = RHF(mol).density_fit()
        mf.kernel()

        # Store mean-field object
        self.mf = mf
        self.mol = mol

        print(f"Mean-Field (Hartree-Fock) energy = {mf.e_tot} Ha")

        return mf

    def create_fragments(self) -> EmbeddingResult:
        """
        Create fragments using the embedder.

        Returns
        -------
        EmbeddingResult
            Container with fragments and embedding information
        """
        if self.mf is None:
            raise RuntimeError("Must run mean-field calculation first")

        # Pass fragmentation scheme and any additional options to embedder
        self.embedding_result = self.embedder.create_fragments(
            self.mf, fragmentation=self.fragmentation, **self.embedding_options
        )
        return self.embedding_result

    def _assign_solvers(self):
        """
        Assign solvers to fragments based on rules.
        Called after fragments are created.
        """
        if self.embedding_result is None:
            raise RuntimeError("Fragments must be created before assigning solvers")

        fragment_solvers = {}

        for fragment_id, fragment in self.embedding_result.fragments.items():
            # Find first matching rule
            solver = None
            for rule in self.solver_rules:
                if rule.applies_to(fragment):
                    solver = rule.create_solver(fragment)
                    break

            # Use default solver if no rule matched
            if solver is None:
                if self.default_solver is None:
                    raise RuntimeError(
                        f"No solver rule matched fragment '{fragment_id}' and no default solver set"
                    )
                solver = self.default_solver(fragment)

            fragment_solvers[fragment_id] = solver

        return fragment_solvers

    def solve_fragments(self):
        """
        Solve each fragment with assigned solvers.

        Returns
        -------
        dict
            Dictionary mapping fragment_id to SolverResult
        """
        if self.embedding_result is None:
            raise RuntimeError("Must create fragments before solving")

        # Assign solvers to fragments
        solvers = self._assign_solvers()

        # Solve each fragment
        fragment_results = {}

        for fragment_id, fragment in self.embedding_result.fragments.items():
            solver = solvers[fragment_id]

            print(f"Solving fragment {fragment_id} with {solver.name}...")

            # Get fragment data from metadata
            if "hamiltonian" in fragment.metadata:
                # Fragment has pre-computed Hamiltonian integrals
                h1e = fragment.metadata["hamiltonian"]["h1e"]
                h2e = fragment.metadata["hamiltonian"]["h2e"]
                norb = fragment.n_orbitals
                nelec = (
                    fragment.n_electrons
                    if isinstance(fragment.n_electrons, int)
                    else sum(fragment.n_electrons)
                )
                nocc = nelec // 2

                # Solve using integrals
                if hasattr(solver, "solve_from_integrals"):
                    result = solver.solve_from_integrals(h1e, h2e, norb, nocc)
                else:
                    raise NotImplementedError(
                        f"Solver {solver.name} does not support solve_from_integrals(). "
                        "Fragment Hamiltonians are available but solver requires mean-field object."
                    )
            elif "mean_field" in fragment.metadata:
                # Fragment has mean-field object
                frag_mf = fragment.metadata["mean_field"]
                result = solver.solve(frag_mf)
            elif "vayesta_fragment" in fragment.metadata:
                # Extract Hamiltonian from Vayesta fragment
                vfrag = fragment.metadata["vayesta_fragment"]

                # Check if Vayesta used DUMP solver (writes to HDF5)
                # The dumpfile path is stored in the embedding result metadata
                dumpfile = self.embedding_result.metadata.get("dumpfile", None)

                # Fallback: try to get from vayesta_ewf object
                if dumpfile is None and "vayesta_ewf" in self.embedding_result.metadata:
                    vayesta_ewf = self.embedding_result.metadata["vayesta_ewf"]
                    if hasattr(vayesta_ewf, "opts") and hasattr(vayesta_ewf.opts, "solver_options"):
                        solver_opts = vayesta_ewf.opts.solver_options
                        if isinstance(solver_opts, dict) and "dumpfile" in solver_opts:
                            dumpfile = solver_opts["dumpfile"]
                        elif hasattr(solver_opts, "dumpfile"):
                            dumpfile = solver_opts.dumpfile

                if dumpfile:
                    # Hamiltonians are in HDF5 file, need to read them
                    import os

                    import h5py

                    try:
                        if not os.path.exists(dumpfile):
                            raise FileNotFoundError(f"Dumpfile {dumpfile} does not exist")

                        with h5py.File(dumpfile, "r") as f:
                            # Find the fragment group in HDF5
                            frag_key = f"fragment_{fragment_id}"
                            if frag_key in f:
                                frag_group = f[frag_key]
                                h1e = frag_group["heff"][:]
                                h2e = frag_group["eris"][:]
                                norb = int(frag_group.attrs["norb"])
                                nocc = int(frag_group.attrs["nocc"])
                            else:
                                raise KeyError(
                                    f"Fragment {fragment_id} not found in HDF5 file {dumpfile}. "
                                    f"Available keys: {list(f.keys())}"
                                )
                    except Exception as e:
                        raise RuntimeError(
                            f"Failed to read Vayesta cluster data from HDF5 file {dumpfile}: {e}"
                        ) from e

                    # Solve using integrals - always compute RDMs for energy reconstruction
                    if hasattr(solver, "solve_from_integrals"):
                        result = solver.solve_from_integrals(
                            h1e, h2e, norb, nocc, compute_rdms=True
                        )

                        # Store additional data needed for partitioned cumulant energy
                        # Load c_frag and c_cluster from HDF5 for fragment projector
                        try:
                            with h5py.File(dumpfile, "r") as f:
                                frag_group = f[frag_key]
                                c_frag = frag_group["c_frag"][:] if "c_frag" in frag_group else None
                                c_cluster = (
                                    frag_group["c_cluster"][:]
                                    if "c_cluster" in frag_group
                                    else None
                                )

                                # Store in result metadata for energy reconstruction
                                if c_frag is not None:
                                    result.metadata["c_frag"] = c_frag
                                if c_cluster is not None:
                                    result.metadata["c_cluster"] = c_cluster
                                result.metadata["norb"] = norb
                                result.metadata["nocc"] = nocc
                        except Exception as e:
                            print(
                                f"Warning: Could not load c_frag/c_cluster for fragment {fragment_id}: {e}"
                            )
                    else:
                        raise NotImplementedError(
                            f"Solver {solver.name} does not support solve_from_integrals(). "
                            "Vayesta DUMP solver requires solvers that can work with Hamiltonian integrals."
                        )

                # Get cluster Hamiltonian from Vayesta fragment (in-memory)
                elif hasattr(vfrag, "cluster") and vfrag.cluster is not None:
                    cluster = vfrag.cluster

                    # Extract Hamiltonian integrals from Vayesta cluster
                    # Vayesta clusters have different methods depending on version
                    if hasattr(cluster, "get_heff"):
                        h1e = cluster.get_heff()
                    elif hasattr(cluster, "heff"):
                        h1e = cluster.heff
                    else:
                        raise AttributeError(
                            f"Vayesta cluster for fragment {fragment_id} has no 'get_heff()' or 'heff' attribute. "
                            "Cannot extract one-electron Hamiltonian."
                        )

                    if hasattr(cluster, "get_eris_bare"):
                        h2e = cluster.get_eris_bare()
                    elif hasattr(cluster, "eris"):
                        h2e = cluster.eris
                    else:
                        raise AttributeError(
                            f"Vayesta cluster for fragment {fragment_id} has no 'get_eris_bare()' or 'eris' attribute. "
                            "Cannot extract two-electron integrals."
                        )

                    norb = cluster.norb
                    nelec = cluster.nelec if isinstance(cluster.nelec, int) else sum(cluster.nelec)
                    nocc = nelec // 2

                    # Solve using integrals
                    if hasattr(solver, "solve_from_integrals"):
                        result = solver.solve_from_integrals(h1e, h2e, norb, nocc)
                    else:
                        raise NotImplementedError(
                            f"Solver {solver.name} does not support solve_from_integrals(). "
                            "Vayesta fragments require solvers that can work with Hamiltonian integrals."
                        )
                else:
                    raise RuntimeError(
                        f"Fragment {fragment_id} has Vayesta fragment but no cluster data. "
                        "Ensure EWF kernel() has been run."
                    )
            else:
                raise RuntimeError(
                    f"Fragment {fragment_id} missing Hamiltonian data. "
                    "Fragments must contain 'hamiltonian', 'mean_field', or 'vayesta_fragment' in metadata."
                )

            fragment_results[fragment_id] = result
            # For EWF, print correlation energy (not total cluster energy which overlaps)
            if hasattr(result, "metadata") and "e_corr" in result.metadata:
                e_corr = result.metadata["e_corr"]
                print(f"  Fragment {fragment_id} correlation energy: {e_corr:.8f} Ha")
            else:
                print(f"  Fragment {fragment_id} energy: {result.energy:.8f} Ha")

        return fragment_results

    def reconstruct_energy(self, fragment_results):
        """
        Reconstruct total energy using the embedder.

        Parameters
        ----------
        fragment_results : dict
            Dictionary mapping fragment_id to solver results

        Returns
        -------
        float
            Total reconstructed energy
        """
        if self.embedding_result is None:
            raise RuntimeError("Must create fragments before reconstructing energy")

        return self.embedder.reconstruct_energy(fragment_results, self.embedding_result)

    def run(self):
        """Execute full workflow."""
        self.run_mean_field()
        self.create_fragments()
        fragment_results = self.solve_fragments()
        total_energy = self.reconstruct_energy(fragment_results)
        return WorkflowResult(
            total_energy,
            fragment_results,
            self.mf.e_tot if self.mf else None,
            self.embedding_result,
        )


class WorkflowResult:
    """Container for results."""

    def __init__(self, total_energy, fragment_results, mf_energy=None, embedding_result=None):
        self.total_energy = total_energy
        self.fragment_results = fragment_results
        self.mf_energy = mf_energy
        self.embedding_result = embedding_result

    @property
    def fragment_energies(self):
        """Get dictionary of fragment energies."""
        return {frag_id: result.energy for frag_id, result in self.fragment_results.items()}

    @property
    def fragments(self):
        """Get list of fragment objects for iteration."""
        if self.embedding_result is None:
            return []
        return list(self.embedding_result.fragments.values())
