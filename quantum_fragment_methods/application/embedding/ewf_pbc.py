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

### -----  Notice of file modification -------
# This file has been developed based on the ewf.py file for applying the 
# fragmentation embedding to periodic systems
### -------------------------------------------

from __future__ import annotations

from typing import Any

from .base import BaseEmbedder, EmbeddingResult, Fragment
import logging

logger = logging.getLogger(__name__)

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
    approx_cumulant : bool, optional
        Controls which cumulant is used for the 2-body fragment energy term
        in ``reconstruct_energy``.

        ``True`` (default, matches Vayesta ``approx_cumulant=True``):
            Use the **approximate cumulant**
            λ̃ = dm2_full − (γ_HF⊗γ_HF)_antisym.
            The cross-terms dm2_1 = (γ_HF⊗δγ + δγ⊗γ_HF) are kept inside
            λ̃ and handled at the mean-field level by the global
            ½ Tr[V_HF·δγ_demo] correction.  This is the formulation used
            by Vayesta's ``get_dmet_energy(approx_cumulant=True)`` and gives
            energies that match Vayesta.

        ``False`` (matches Vayesta ``approx_cumulant=False``):
            Use the **true cumulant**
            λ = dm2_full − (γ_CC⊗γ_CC)_antisym,
            extracted via ``CCSD.split_dm2``.  The cross-terms dm2_1 are
            fully removed from the 2-body term.  This corresponds to
            Vayesta's ``get_dmet_energy(approx_cumulant=False)`` and gives
            a slightly different energy.
    **kwargs : dict
        Additional configuration options

    Examples
    --------
    >>> ewf = EWF(bath_type='mp2', truncation=1e-6)
    >>> result = ewf.kernel(mf)
    >>> print(f"Number of fragments: {len(result.fragments)}")

    >>> # Use true cumulant (matches Vayesta approx_cumulant=False)
    >>> ewf = EWF(bath_type='mp2', truncation=1e-6, approx_cumulant=False)
    """

    def __init__(
        self,
        bath_type: str = "mp2",
        truncation: float = 1e-6,
        bath_options: dict[str, Any] | None = None,
        solver_options: dict[str, Any] | None = None,
        approx_cumulant: bool = True,
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
        self.approx_cumulant = approx_cumulant
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
            mf, solver=solver, bath_options=self.bath_options, energy_functional='wf',
            solver_options=self.solver_options
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
        logger.debug(f"Number of fragments: {len(self._vayesta_ewf.fragments)}")
        # Convert Vayesta fragments to our Fragment objects
        fragments = {}
        for i, vfrag in enumerate(self._vayesta_ewf.fragments):
            logger.debug("index: %s, vfrag: %s, sym_parent: %s", i, vfrag, vfrag.sym_parent)
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

        # -----------------------------------------------------------------
        # CHANGE 1: store e_mf per unit cell via Vayesta's own property.
        # Vayesta computes  e_mf = mf.energy_tot(...) / ncells  so for
        # molecules (ncells=1) this equals mf.e_tot; for PBC supercells it
        # divides by the number of k-points, giving the per-cell energy that
        # get_dmet_energy() returns.
        # -----------------------------------------------------------------
        e_mf_per_cell = self._vayesta_ewf.e_mf
        result = EmbeddingResult(fragments=fragments, mean_field_energy=e_mf_per_cell, metadata=metadata)

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
        self,
        fragment_results: dict[int | str, Any],
        embedding_result: EmbeddingResult,
        approx_cumulant: bool | None = None,
    ) -> float:
        """
        Reconstruct total energy from fragment results using the partitioned
        cumulant approach, matching Vayesta's ``get_dmet_energy`` logic.

        This replicates ``Embedding.get_dmet_energy(part_cumulant=True,
        approx_cumulant=<approx_cumulant>)`` from Vayesta using the cluster
        data stored in the HDF5 dumpfile and the RDMs returned by the
        external solver.

        The total energy is assembled as::

            E_total = E_nuc + E_exxdiv           # non-electronic (from Vayesta)
                    + Σ_x w_x (e1_x + e2_x)      # fragment cumulant terms
                    + ½ Tr[V_HF(γ_MF) · δγ_demo] # global non-cumulant correction

        where the sum runs over symmetry-unique fragments and
        ``δγ_demo = 2 γ_demo − γ_MF``.

        Parameters
        ----------
        fragment_results : dict
            Dictionary mapping fragment_id to solver results.  Each result
            must expose ``rdm1`` (shape norb×norb) and ``rdm2``
            (shape norb×norb×norb×norb) in the cluster MO basis.
        embedding_result : EmbeddingResult
            Original embedding result returned by ``kernel()``.
        approx_cumulant : bool or None, optional
            Override the instance-level ``self.approx_cumulant`` flag for
            this call only.  If ``None`` (default), the value set in
            ``__init__`` is used.

            ``True``: approximate cumulant λ̃ = dm2_full − dm2_0.
                Matches Vayesta ``get_dmet_energy(approx_cumulant=True)``.
            ``False``: true cumulant λ via ``CCSD.split_dm2``.
                Matches Vayesta ``get_dmet_energy(approx_cumulant=False)``.

        Returns
        -------
        float
            Total reconstructed energy per unit cell (Ha).

        Raises
        ------
        ValueError
            If required data (e_mf, vayesta_ewf, dumpfile) is not available.

        Notes
        -----
        Key differences vs. the original ``ewf.py`` implementation:

        1. ``e_mf`` is read from ``vayesta_ewf.e_mf`` (per-cell) instead of
           the raw supercell ``mf.e_tot``.
        2. The 1-body cluster operator is ``c_cluster.T @ mf.get_hcore() @
           c_cluster`` (bare core Hamiltonian projected into the cluster),
           NOT the dumpfile ``heff = fock − v_act`` and NOT the Fock matrix.
        3. The 1-body contraction uses the full correlated ``dm1`` (not
           ``δγ = dm1 − dm1_HF``), matching Vayesta's
           ``einsum('xj,xi,ij->', h1e_eff, p_frag, dm1)``.
        4. A global ``½ Tr[V_HF(γ_MF) · δγ_demo]`` correction is added after
           the fragment loop using a democratically assembled 1-RDM.
        5. The exxdiv correction ``−madelung × nocc / ncells`` is added for
           PBC calculations where ``mf.exxdiv is not None``.
        6. The 2-body cumulant is controlled by ``approx_cumulant``:

           - ``True``:  λ̃ = dm2_full − dm2_0 (HF⊗HF block only).
             Cross-terms dm2_1 stay inside λ̃ and are handled by the global
             ½ Tr[V_HF·δγ] correction — matches Vayesta exactly.
           - ``False``: λ  = dm2_full − dm2_0 − dm2_1 (full antisym γ_CC⊗γ_CC).
             Extracted via ``CCSD.split_dm2``; cross-terms fully removed from
             the 2-body term — matches ``Vayesta approx_cumulant=False``.
        """
        import numpy as np

        # ------------------------------------------------------------------
        # Resolve approx_cumulant: per-call argument overrides instance default
        # ------------------------------------------------------------------
        if approx_cumulant is None:
            approx_cumulant = self.approx_cumulant
        print(f"reconstruct_energy: approx_cumulant={approx_cumulant}")

        # ------------------------------------------------------------------
        # Setup
        # ------------------------------------------------------------------
        e_mf = embedding_result.mean_field_energy  # per unit cell (change 1)
        if e_mf is None:
            raise ValueError("Mean-field energy not available in embedding result")

        vayesta_ewf = embedding_result.metadata.get("vayesta_ewf")
        if vayesta_ewf is None:
            raise ValueError("Vayesta EWF object not found in embedding result metadata")

        # mf is the FoldedSCF object (BVK supercell) stored by Vayesta after
        # init_mf().  For a KRHF(2×2×2) input it is a FoldedSCF wrapping the
        # original KRHF; for a Gamma-point calculation it is the original mf.
        mf = vayesta_ewf.mf
        ovlp = mf.get_ovlp()
        ncells = vayesta_ewf.ncells  # 1 for molecules / Gamma; len(kpts) for PBC

        # ------------------------------------------------------------------
        # For k-point PBC (FoldedSCF): the primitive cell and kpts live on
        # vayesta_ewf.kcell / vayesta_ewf.kpts.
        # Madelung constant must be evaluated on the PRIMITIVE cell with all
        # k-points (matching Vayesta qemb.py:359):
        #   madelung = pyscf.pbc.tools.madelung(self.mol, self.mf.kpt)
        # where self.mol == kcell (primitive cell) and self.mf.kpt == [0,0,0].
        # Using mf.mol (the BVK supercell) gives a wrong Madelung constant.
        #
        # e_nuc must be the primitive-cell nuclear repulsion, i.e.
        # kcell.energy_nuc() / 1  (already per primitive cell).
        # mf.mol.energy_nuc() is the supercell e_nuc; dividing by ncells gives
        # the same value, so both paths are correct for e_nuc.
        # ------------------------------------------------------------------
        has_exxdiv = hasattr(mf, "exxdiv") and mf.exxdiv is not None
        madelung = None
        if has_exxdiv:
            import pyscf.pbc.tools
            # Vayesta uses the SUPERCELL (mf.mol) and Gamma kpt (mf.kpt) for Madelung.
            # (qemb.py:359: pyscf.pbc.tools.madelung(self.mol, self.mf.kpt))
            # self.mol = FoldedSCF supercell; self.mf.kpt = [0,0,0]
            kpt_gamma = getattr(mf, "kpt", np.zeros(3))
            madelung = pyscf.pbc.tools.madelung(mf.mol, kpt_gamma)

        # ------------------------------------------------------------------
        # Early exit: no fragment results
        # ------------------------------------------------------------------
        if not fragment_results:
            print("Warning: No fragment results provided. Returning mean-field energy.")
            return e_mf

        # ------------------------------------------------------------------
        # Early exit: no dumpfile
        # ------------------------------------------------------------------
        dumpfile = embedding_result.metadata.get("dumpfile")
        if dumpfile is None:
            print("Warning: No dumpfile found. Attempting Vayesta energy reconstruction.")
            try:
                e_corr_total = vayesta_ewf.get_e_corr()
                return e_mf + e_corr_total if e_corr_total is not None else e_mf
            except Exception as e:
                print(f"Vayesta reconstruction failed: {e}. Returning mean-field energy.")
                return e_mf

        try:
            import h5py
        except ImportError as e:
            raise ImportError("h5py is required for reading cluster data") from e

        # ------------------------------------------------------------------
        # Fragment loop — accumulate e1 (heff·γ_CC) and e2 (½ g·λ)
        # and simultaneously build the democratic 1-RDM in the MO basis
        # for the global correction (change 4).
        # ------------------------------------------------------------------
        e1_total = 0.0   # Σ w_x · e1_x   (heff contracted with full dm1)
        e2_total = 0.0   # Σ w_x · e2_x   (cumulant 2-body term)

        nmo = vayesta_ewf.nmo
        mo_coeff = vayesta_ewf.mo_coeff   # (nao, nmo) global MO coefficients

        # democratic 1-RDM accumulated in MO basis (correlation part only,
        # i.e. without the HF diagonal 2 on occupied indices)
        dm1_demo_mo = np.zeros((nmo, nmo))

        for frag_id, result in fragment_results.items():

            frag_idx = int(frag_id) if isinstance(frag_id, str) else frag_id
            frag = vayesta_ewf.fragments[frag_idx]

            # Skip symmetric children — their contribution is counted through
            # symmetry_factor on the parent.
            if frag.sym_parent is not None:
                continue

            inner_weight = frag.opts.sym_factor    # sub-symmetry factor (usually 1.0)
            outer_weight = frag.symmetry_factor    # translation / point-group factor
            total_weight = inner_weight * outer_weight
            #print('sym_factor, symmetry_factor:', frag.opts.sym_factor, frag.symmetry_factor)

            if result.rdm1 is None or result.rdm2 is None:
                print(f"Warning: Fragment {frag_id} missing RDMs. Skipping energy contribution.")
                continue

            # Load cluster data from HDF5 dumpfile
            try:
                with h5py.File(dumpfile, "r") as h5f:
                    frag_key = f"fragment_{frag_idx}"
                    if frag_key not in h5f:
                        print(f"Warning: Fragment {frag_key} not found in HDF5. Skipping.")
                        continue

                    grp = h5f[frag_key]
                    h2e      = grp["eris"][:]       # bare ERIs in cluster MO basis
                    c_frag   = grp["c_frag"][:]     # (nao, nfrag) fragment orbitals
                    c_cluster = grp["c_cluster"][:]  # (nao, norb) cluster active MOs
                    nocc    = int(grp.attrs["nocc"])

            except Exception as exc:
                print(f"Warning: Could not load data for fragment {frag_id}: {exc}")
                continue

            # Fragment projector: P = Rᵀ R,  R = c_cluster.T @ S @ c_frag
            # Shape: (norb, norb)  — same algebra as Vayesta's get_fragment_projector
            r_proj = c_frag.T @ ovlp @ c_cluster   # (nfrag, norb)
            proj   = r_proj.T @ r_proj              # (norb, norb)

            # ----------------------------------------------------------
            # CHANGE 2: h1e_eff = c_cluster.T @ hcore @ c_cluster
            #
            # Vayesta's get_fragment_dmet_energy with part_cumulant=True:
            #   h1e_eff = c_act.T @ get_hcore_for_energy() @ c_act
            #
            # This is the PURE CORE HAMILTONIAN projected into the cluster.
            # It is NOT the dumpfile's 'heff' (which is fock − v_act) and
            # NOT the dumpfile's 'fock'.  We reconstruct it from c_cluster
            # and the global mf hcore.
            # ----------------------------------------------------------
            hcore_ao = mf.get_hcore()
            h1e_eff  = c_cluster.T @ hcore_ao @ c_cluster   # (norb, norb)

            dm1 = result.rdm1.copy()   # full correlated 1-RDM in cluster MO basis
            dm2 = result.rdm2.copy()   # full correlated 2-RDM in cluster MO basis

            # ----------------------------------------------------------
            # CHANGE 3: one-body energy uses full γ_CC (not δγ)
            # Vayesta:  e1 = einsum('xj,xi,ij->', h1e_eff, p_frag, dm1_CC)
            # i.e. project one index of h1e_eff onto the fragment and
            # contract with the FULL correlated dm1 (not δγ).
            # ----------------------------------------------------------
            h1e_proj = h1e_eff @ proj   # (norb, norb): one index projected onto fragment
            e1 = np.einsum("ij,ij->", h1e_proj, dm1)
            # Equivalently: einsum('xj,xi,ij->', h1e_eff, proj, dm1)

            # ----------------------------------------------------------
            # Two-body cumulant energy
            #
            # result.rdm2 is the FULL 2-RDM in CHEMIST notation returned by
            # pyscf's make_rdm2(with_dm1=True).  Vayesta stores this as
            # RRDM_WaveFunction.dm2 and then extracts the (approx) cumulant
            # by calling wf.make_rdm2(with_dm1=False, approx_cumulant=...).
            #
            # We replicate that method exactly (rdm.py:46-64):
            #
            # approx_cumulant=True  (default):
            #   dm1_v = dm1_full.copy()
            #   dm1_v[diag(nocc)] -= 1          # dm1_v = dm1_full - I_occ
            #   for i in range(nocc):
            #       dm2[i, i, :, :] -= 2 * dm1_v
            #       dm2[:, :, i, i] -= 2 * dm1_v
            #       dm2[:, i, i, :] += dm1_v
            #       dm2[i, :, :, i] += dm1_v
            #   (no j-loop; HF pure-diagonal handled globally in Phase 2)
            #
            # approx_cumulant=False  (true cumulant):
            #   dm2 -= make_rdm2_non_cumulant()
            #   where dm2_nc = dm1⊗dm1 - ½ dm1⊗̃dm1  (full antisym outer product)
            #   using the FULL correlated dm1 (with_mf=True, i.e. +2 on occ diag)
            # ----------------------------------------------------------

            dm2_cumulant = dm2.copy()

            if approx_cumulant:
                # approx_cumulant=True → approximate cumulant λ̃
                # Mirror RRDM_WaveFunction.make_rdm2(with_dm1=False, approx_cumulant=True)
                # rdm.py:51-57: subtract partial non-cumulant, keeping Δdm1² cross-terms.
                # dm1_v = dm1_full − I_occ  (removes only the HF-diagonal from dm1)
                dm1_v = dm1.copy()
                dm1_v[np.diag_indices(nocc)] -= 1.0   # dm1_v = dm1_full - I_occ
                for i in range(nocc):
                    dm2_cumulant[i, i, :, :] -= 2.0 * dm1_v
                    dm2_cumulant[:, :, i, i] -= 2.0 * dm1_v
                    dm2_cumulant[:, i, i, :] += dm1_v
                    dm2_cumulant[i, :, :, i] += dm1_v
            else:
                # approx_cumulant=False → true cumulant λ
                # Mirror RRDM_WaveFunction.make_rdm2(with_dm1=False, approx_cumulant=False)
                # rdm.py:49-50: subtract full antisymmetrised outer product.
                # dm2_nc = dm1⊗dm1 − ½ dm1⊗̃dm1  (full correlated dm1, chemist conv)
                dm2_nc = (
                    np.einsum("ij,kl->ijkl", dm1, dm1)
                    - 0.5 * np.einsum("ij,kl->iklj", dm1, dm1)
                )
                dm2_cumulant -= dm2_nc

            # Project first index of cumulant onto fragment
            # Vayesta: einsum("xjkl,xi,ijkl->", eris, p_frag, dm2) / 2
            #       =  einsum("pqrs,pqrs->", eris, einsum("Ijkl,iI->ijkl", dm2, proj)) / 2
            dm2_proj = np.einsum("Ijkl,iI->ijkl", dm2_cumulant, proj)
            e2 = 0.5 * np.einsum("pqrs,pqrs->", h2e, dm2_proj)

            #print(f'  Fragment {frag_idx}: e1={e1:.8f}  e2={e2:.8f}')
            e1_total  += total_weight * e1
            e2_total  += total_weight * e2

            # ----------------------------------------------------------
            # Accumulate democratic 1-RDM (MO basis, correlation part)
            # for the global correction (change 4).
            #
            # Vayesta make_rdm1_demo_rhf (rdm.py:41):
            #   dm1 += einsum('xi,ij,px,qj->pq', px, dm1x, rx, rx)
            # where px = cluster|frag|cluster = proj  (norb, norb)
            #       rx = mo|cluster            (nmo, norb)
            #
            # Expanding the einsum:
            #   dm1[p,q] = Σ_{x,i,j} px[x,i] · dm1x[i,j] · rx[p,x] · rx[q,j]
            #            = Σ_{p}(rx @ proj)[p,i] · dm1x[i,j] · rx[q,j]
            #            = (rx @ proj @ dm1x @ rx.T)[p,q]
            #
            # proj is on the LEFT index only — NOT on both sides.
            # ----------------------------------------------------------
            rx = mo_coeff.T @ ovlp @ c_cluster        # (nmo, norb)
            dm1_frag_corr = dm1.copy()
            dm1_frag_corr[np.diag_indices(nocc)] -= 2.0   # strip HF part (with_mf=False)
            # Vayesta: einsum("xi,ij,px,qj->pq", px, dm1x, rx, rx)
            #        = rx @ proj @ dm1x @ rx.T  (proj on LEFT index of dm1x)
            dm1_demo_mo += total_weight * (rx @ proj @ dm1_frag_corr @ rx.T)

        # ------------------------------------------------------------------
        # CHANGE 4: global non-cumulant correction
        #
        # Vayesta get_dmet_elec_energy (qemb.py:1295-1304):
        #
        #   approx_cumulant=True:
        #     ddm1   = 2*γ_demo − γ_MF   (Δdm1)
        #     vhf_mf = get_veff_for_energy(with_exxdiv=False)  ← from HF density
        #     e_global = ½ Tr[vhf_mf · ddm1]
        #
        #   approx_cumulant=False:
        #     dm1_demo = γ_demo  (full democratic DM)
        #     vhf_demo = get_veff_for_energy(dm1=dm1_demo, with_exxdiv=False)
        #     e_global = ½ Tr[vhf_demo · dm1_demo]
        #
        # IMPORTANT: for PBC, mf.get_veff() has exxdiv baked in.
        # Vayesta strips it via get_veff_for_energy(with_exxdiv=False).
        # We must subtract v_exxdiv_ao before the contraction.
        # ------------------------------------------------------------------
        nocc_global = vayesta_ewf.nocc

        # Full democratic 1-RDM in MO basis (with MF contribution)
        dm1_demo_mo_full = dm1_demo_mo.copy()
        dm1_demo_mo_full[np.diag_indices(nocc_global)] += 2.0   # with_mf=True
        dm1_demo_mo_full = (dm1_demo_mo_full + dm1_demo_mo_full.T) / 2.0   # symmetrize

        # Transform to AO basis
        dm1_demo_ao = mo_coeff @ dm1_demo_mo_full @ mo_coeff.T

        # exxdiv correction matrix (only for PBC)
        v_exxdiv_ao = np.zeros_like(dm1_demo_ao)
        if has_exxdiv:
            sc = ovlp @ mo_coeff[:, :nocc_global]
            v_exxdiv_ao = -madelung * sc @ sc.T

        if approx_cumulant:
            # approx_cumulant=True path (Vayesta qemb.py:1299-1301):
            #   ddm1 = 2*γ_demo − γ_MF
            #   vhf  = V_HF(γ_MF) − v_exxdiv   (built from HF density)
            #   e_global = ½ Tr[vhf · ddm1]
            dm1_mf_ao = mf.make_rdm1()
            ddm1 = 2.0 * dm1_demo_ao - dm1_mf_ao
            vhf_mf = np.asarray(mf.get_veff()) - v_exxdiv_ao
            e_global_nc = 0.5 * np.sum(vhf_mf * ddm1)
        else:
            # approx_cumulant=False path (Vayesta qemb.py:1297-1298):
            #   vhf_demo = V_HF(γ_demo) − v_exxdiv   (built from democratic dm1)
            #   e_global = ½ Tr[vhf_demo · γ_demo]
            vhf_demo = np.asarray(mf.get_veff(dm=dm1_demo_ao)) - v_exxdiv_ao
            e_global_nc = 0.5 * np.sum(vhf_demo * dm1_demo_ao)

        # ------------------------------------------------------------------
        # CHANGE 5: exxdiv correction for PBC
        #
        # Vayesta get_dmet_energy adds get_exxdiv()[0] = −madelung*nocc/ncells
        # on top of get_dmet_elec_energy().
        #
        # KEY: Vayesta's get_dmet_energy is a STANDALONE total energy builder:
        #
        #   E_total = get_dmet_elec_energy()   ← uses V_HF WITHOUT exxdiv
        #           + e_nuc
        #           + e_exxdiv
        #
        # It does NOT add e_mf. We replicate this structure: assemble the
        # total energy from scratch rather than as e_mf + correction, so
        # that we do not double-count exxdiv or e_nuc which are already
        # folded into e_mf (via mf.energy_tot = energy_elec_with_exxdiv + e_nuc).
        # ------------------------------------------------------------------
        e_exxdiv = 0.0
        if has_exxdiv:
            e_exxdiv = -madelung * nocc_global / ncells

        # ------------------------------------------------------------------
        # Assemble total energy — Vayesta-style, from scratch
        #
        # get_dmet_elec_energy()  =  Σ_x w_x (e1_x + e2_x)  +  e_global_nc
        # get_dmet_energy()       =  get_dmet_elec_energy()  +  e_nuc  +  e_exxdiv
        #
        # e_nuc is mf.mol.energy_nuc() / ncells (same for PBC Cell and Mole)
        # ------------------------------------------------------------------
        e_elec = (e1_total + e2_total) / ncells + e_global_nc / ncells
        e_nuc  = mf.mol.energy_nuc() / ncells
        total_energy = e_elec + e_nuc + e_exxdiv

        print("\nPartitioned Cumulant Energy Reconstruction:")
        print(f"  Electronic DMET energy (per cell): {e_elec:.8f} Ha")
        print(f"    1-body fragment contributions:   {e1_total / ncells:.8f} Ha")
        print(f"    2-body cumulant contributions:   {e2_total / ncells:.8f} Ha")
        print(f"    Global non-cumulant correction:  {e_global_nc / ncells:.8f} Ha")
        print(f"  Nuclear repulsion (per cell):      {e_nuc:.8f} Ha")
        print(f"  Exxdiv correction:                 {e_exxdiv:.8f} Ha")
        print(f"  Total energy (per cell):           {total_energy:.8f} Ha")
        print(f"  Reference E(MF) (per cell):        {e_mf:.8f} Ha")
        print(f"  Implied E(corr):                   {total_energy - e_mf:.8f} Ha")

        return float(total_energy)
