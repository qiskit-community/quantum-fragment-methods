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

"""Embedded Wave Function (EWF) method."""

from __future__ import annotations

from typing import Any

from .base import BaseEmbedder, EmbeddingResult, Fragment


class EWF(BaseEmbedder):
    """
    Embedded Wave Function (EWF) embedding method using Vayesta.

    EWF partitions the system into fragments and solves each fragment
    with an embedded wavefunction approach, accounting for the environment
    through an effective Hamiltonian.

    Parameters
    ----------
    bath_type : str, optional
        Type of bath orbitals: 'mp2', 'dmet' (default: 'mp2')
    truncation : float, optional
        Truncation threshold for bath orbital selection (default: 1e-6)
    bath_options : dict, optional
        Additional bath configuration options
    solver_options : dict, optional
        Solver-specific configuration options
    **kwargs : dict
        Additional configuration options

    Examples
    --------
    >>> ewf = EWF(bath_type='mp2', truncation=1e-6)
    >>> result = ewf.kernel(mf)
    >>> print(f"Number of fragments: {len(result.fragments)}")
    """

    def __init__(
        self,
        bath_type: str = "mp2",
        truncation: float = 1e-6,
        bath_options: dict[str, Any] | None = None,
        solver_options: dict[str, Any] | None = None,
        **kwargs,
    ):
        """Initialize EWF embedding method."""
        super().__init__(
            bath_type=bath_type,
            truncation=truncation,
            bath_options=bath_options,
            solver_options=solver_options,
            **kwargs,
        )
        self.bath_type = bath_type
        self.truncation = truncation
        self.bath_options = bath_options or {}
        self.solver_options = solver_options or {}
        self._vayesta_ewf = None

        # Validate bath_type
        valid_bath_types = ["mp2", "dmet", "full"]
        if bath_type not in valid_bath_types:
            raise ValueError(
                f"Invalid bath_type: {bath_type}. " f"Must be one of {valid_bath_types}"
            )

        # Set default bath options
        if "bathtype" not in self.bath_options:
            self.bath_options["bathtype"] = bath_type
        if "threshold" not in self.bath_options:
            self.bath_options["threshold"] = truncation

        # Extract bath/solver options from kwargs
        # Bath options
        for key in ["occupation_tolerance", "dmet_threshold", "canonicalize"]:
            if key in kwargs:
                self.bath_options[key] = kwargs.pop(key)

        # Solver options (note: 'solver' is passed separately to Vayesta, not in solver_options)
        for key in ["max_cycle", "conv_tol", "dumpfile"]:
            if key in kwargs:
                self.solver_options[key] = kwargs.pop(key)

        # Store solver separately (not in solver_options dict)
        self._solver = kwargs.pop("solver", None)

    def kernel(self, mf, fragmentation: str = "atomic", **kwargs):
        """
        Run EWF calculation using Vayesta.

        Parameters
        ----------
        mf : pyscf.scf object
            Mean-field calculation result
        fragmentation : str, optional
            Fragmentation scheme: 'atomic' or 'iao' (default: 'atomic')
        **kwargs : dict
            Additional parameters:
            - atom_indices: List of atom indices for atomic fragmentation
            - solver: Solver name (default: 'DUMP' for fragment dumping)
            - dumpfile: HDF5 file path for dump solver

        Returns
        -------
        EmbeddingResult
            Container with fragments and embedding information
        """
        # Validate mean-field object
        if mf is None:
            raise ValueError("Mean-field object cannot be None")

        # Validate fragmentation type
        valid_frag = ["atomic", "iao"]
        if fragmentation.lower() not in valid_frag:
            raise ValueError(
                f"Invalid fragmentation: {fragmentation}. " f"Must be one of {valid_frag}"
            )

        try:
            import os
            import tempfile

            import vayesta.ewf
        except ImportError as e:
            raise ImportError(
                "Vayesta library is required for EWF. Install with: pip install vayesta"
            ) from e

        # Get solver and atom indices from kwargs
        solver = kwargs.get("solver", self._solver if self._solver else "DUMP")
        atom_indices = kwargs.get("atom_indices", None)

        # Handle dumpfile - create unique temp file if using DUMP solver
        dumpfile = kwargs.get("dumpfile", self.solver_options.get("dumpfile"))
        if solver.upper() == "DUMP" and dumpfile is None:
            # Create unique temp file
            fd, dumpfile = tempfile.mkstemp(suffix=".h5", prefix="ewf_cluster_")
            os.close(fd)  # Close file descriptor, Vayesta will open it
            self.solver_options["dumpfile"] = dumpfile
        elif dumpfile is not None:
            self.solver_options["dumpfile"] = dumpfile

        # Create Vayesta EWF object
        self._vayesta_ewf = vayesta.ewf.EWF(
            mf, solver=solver, bath_options=self.bath_options, solver_options=self.solver_options
        )

        # Add fragments based on fragmentation scheme
        if fragmentation.lower() == "atomic":
            # Atomic fragmentation - one fragment per atom
            if atom_indices is None:
                atom_indices = range(mf.mol.natm)

            for atom_idx in atom_indices:
                with self._vayesta_ewf.iao_fragmentation() as f:
                    f.add_atomic_fragment(atom_idx)

        elif fragmentation.lower() == "iao":
            # IAO fragmentation
            with self._vayesta_ewf.iao_fragmentation() as f:
                if atom_indices is not None:
                    for atom_idx in atom_indices:
                        f.add_atomic_fragment(atom_idx)
                else:
                    # Add all atoms
                    for atom_idx in range(mf.mol.natm):
                        f.add_atomic_fragment(atom_idx)
        else:
            raise ValueError(
                f"Unknown fragmentation scheme: {fragmentation}. " "Use 'atomic' or 'iao'."
            )

        # Run EWF kernel
        self._vayesta_ewf.kernel()

        # Convert Vayesta fragments to our Fragment objects
        fragments = {}
        for i, vfrag in enumerate(self._vayesta_ewf.fragments):
            # Extract fragment properties
            frag_id = i
            atom_indices = getattr(vfrag, "atoms", [])

            # Get orbital and electron counts from cluster
            n_orbitals = 0
            n_electrons = getattr(vfrag, "nelectron", 0)
            bath_orbitals = None

            if hasattr(vfrag, "cluster") and vfrag.cluster is not None:
                cluster = vfrag.cluster
                # Total orbitals in cluster (fragment + bath)
                if hasattr(cluster, "norb"):
                    n_orbitals = cluster.norb
                # Bath orbitals
                if hasattr(cluster, "norb_bath"):
                    bath_orbitals = cluster.norb_bath

            # Fallback to nao if cluster not available
            if n_orbitals == 0:
                n_orbitals = getattr(vfrag, "nao", 0)

            # Create metadata
            metadata = {"bath_orbitals": bath_orbitals, "vayesta_fragment": vfrag}

            # Create Fragment object
            fragment = Fragment(
                fragment_id=frag_id,
                atom_indices=atom_indices if isinstance(atom_indices, list) else [atom_indices],
                orbital_indices=list(range(n_orbitals)) if n_orbitals > 0 else [],
                n_electrons=n_electrons,
                metadata=metadata,
            )
            fragments[frag_id] = fragment

        # Create EmbeddingResult with dumpfile path if available
        metadata = {
            "vayesta_ewf": self._vayesta_ewf,
            "fragmentation": fragmentation,
            "bath_type": self.bath_type,
        }

        # Add dumpfile path to metadata if it exists
        if "dumpfile" in self.solver_options:
            metadata["dumpfile"] = self.solver_options["dumpfile"]

        result = EmbeddingResult(fragments=fragments, mean_field_energy=mf.e_tot, metadata=metadata)

        return result

    def create_fragments(self, mf, **kwargs) -> EmbeddingResult:
        """
        Create fragments from mean-field calculation.

        This is an alias for kernel() to maintain compatibility with
        the BaseEmbedder interface.

        Parameters
        ----------
        mf : pyscf.scf object
            Mean-field calculation result
        **kwargs : dict
            Additional parameters passed to kernel()

        Returns
        -------
        EmbeddingResult
            Container with fragments and embedding matrices
        """
        return self.kernel(mf, **kwargs)

    def reconstruct_energy(
        self, fragment_results: dict[int | str, Any], embedding_result: EmbeddingResult
    ) -> float:
        """
        Reconstruct total energy from fragment results using partitioned cumulant approach.

        This method implements the same energy reconstruction as the working notebook,
        computing partitioned cumulant energies (e1_pc and e22_pc) for each fragment
        and summing them with the mean-field energy.

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

        Raises
        ------
        ValueError
            If required data is not available

        Notes
        -----
        This bypasses Vayesta's energy reconstruction and implements manual
        partitioned cumulant aggregation following the working notebook approach.
        """
        import numpy as np

        # Get mean-field energy
        e_mf = embedding_result.mean_field_energy
        if e_mf is None:
            raise ValueError("Mean-field energy not available in embedding result")

        # Get Vayesta EWF object for accessing mean-field data
        vayesta_ewf = embedding_result.metadata.get("vayesta_ewf")
        if vayesta_ewf is None:
            raise ValueError("Vayesta EWF object not found in embedding result metadata")

        # Get mean-field object to access global matrices
        mf = vayesta_ewf.mf

        # Load global overlap and Fock matrices
        ovlp = mf.get_ovlp()
        fock = mf.get_fock()

        # Check if we have fragment results from external solvers
        if not fragment_results:
            print("Warning: No fragment results provided. Returning mean-field energy.")
            return e_mf

        # Get dumpfile path to load c_frag and c_cluster
        dumpfile = embedding_result.metadata.get("dumpfile")
        if dumpfile is None:
            print("Warning: No dumpfile found. Attempting Vayesta energy reconstruction.")
            # Fall back to trying Vayesta's reconstruction
            try:
                e_corr_total = vayesta_ewf.get_e_corr()
                return e_mf + e_corr_total if e_corr_total is not None else e_mf
            except Exception as e:
                print(f"Vayesta reconstruction failed: {e}. Returning mean-field energy.")
                return e_mf

        # Aggregate partitioned cumulant energies from all fragments
        e1_pc_total = 0.0
        e22_pc_total = 0.0

        try:
            import h5py
        except ImportError as e:
            raise ImportError("h5py is required for reading cluster data") from e

        for frag_id, result in fragment_results.items():
            # Convert frag_id to int if needed
            frag_idx = int(frag_id) if isinstance(frag_id, str) else frag_id

            # Check if result has RDMs (required for partitioned cumulant)
            if result.rdm1 is None or result.rdm2 is None:
                print(f"Warning: Fragment {frag_id} missing RDMs. Skipping energy contribution.")
                continue

            # Load fragment data from HDF5
            try:
                with h5py.File(dumpfile, "r") as f:
                    frag_key = f"fragment_{frag_idx}"
                    if frag_key not in f:
                        print(f"Warning: Fragment {frag_key} not found in HDF5. Skipping.")
                        continue

                    frag_group = f[frag_key]
                    h2e = frag_group["eris"][:]
                    c_frag = frag_group["c_frag"][:]
                    c_cluster = frag_group["c_cluster"][:]
                    nocc = int(frag_group.attrs["nocc"])

            except Exception as e:
                print(f"Warning: Could not load data for fragment {frag_id}: {e}")
                continue

            # Compute fragment projector
            proj = c_frag.T @ ovlp @ c_cluster
            proj = proj.T @ proj

            # Get Fock matrix in cluster basis
            fock_cls = c_cluster.T @ fock @ c_cluster

            # Compute partitioned cumulant energies
            dm1 = result.rdm1.copy()
            dm2 = result.rdm2.copy()

            # Remove HF contribution from dm1
            dm1_pc = dm1.copy()
            dm1_pc[np.diag_indices(nocc)] -= 2.0

            # Split dm2 using the CCSD solver's method
            from ..solvers.classical_zoo.ccsd import CCSD

            dm2_0, dm2_1, dm2_2 = CCSD.split_dm2(nocc, dm1_pc, dm2)

            # Project dm1 onto fragment
            dm1_pc_proj = proj @ dm1_pc

            # One-body energy contribution
            e1_pc = np.einsum("pq,pq->", fock_cls, dm1_pc_proj)

            # Two-body contribution (only cumulant part)
            dm2_pc = np.einsum("Ijkl,iI->ijkl", dm2_2, proj)
            e22_pc = 0.5 * np.einsum("pqrs,pqrs->", h2e, dm2_pc)

            e1_pc_total += e1_pc
            e22_pc_total += e22_pc

        # Total energy = mean-field + partitioned cumulant contributions
        total_energy = e_mf + e1_pc_total + e22_pc_total

        print("\nPartitioned Cumulant Energy Reconstruction:")
        print(f"  Mean-field energy:     {e_mf:.8f} Ha")
        print(f"  1-body PC correction:  {e1_pc_total:.8f} Ha")
        print(f"  2-body PC correction:  {e22_pc_total:.8f} Ha")
        print(f"  Total energy:          {total_energy:.8f} Ha")

        return float(total_energy)
