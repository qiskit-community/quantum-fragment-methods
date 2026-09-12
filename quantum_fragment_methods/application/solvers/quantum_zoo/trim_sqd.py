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

"""TrimSQD solver — SQD with screening-based subspace selection.

TrimSQD (PR #369 in qiskit-addon-sqd) improves on standard SQD by screening a
larger candidate pool before diagonalization.  Each iteration:

1. Draws one disjoint pool of ``samples_per_batch × num_batches`` bitstrings.
2. Diagonalizes each batch independently (screening step).
3. Retains the highest-weight CI strings from every batch (trimming).
4. Merges the survivors and performs a second merged diagonalization (reporting step).
5. Carries the trimmed merged result forward to the next iteration.

This doubles the effective subspace coverage at the cost of one extra FCI call
per iteration, and typically converges faster and to lower energies than standard
SQD with the same total sample budget.

Two policy classes are available:

* :class:`TrimPolicy` — partitions sampled bitstrings across batches.
* :class:`SectorTrimPolicy` — partitions the alpha/beta CI string arrays
  directly, keeping each spin sector disjoint across batches.

``TrimSQDSolver`` subclasses ``SQDSolver`` and reuses all QPU logic (LUCJ
circuit construction, job submission, checkpointing).  Only the classical
post-processing step is overridden.

Requires ``qiskit-addon-sqd >= 0.14`` (the trim-sqd branch / PR #369).
"""

from __future__ import annotations

import logging
from functools import partial
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from qiskit_addon_sqd.fermion import (
    SCIResult,
    diagonalize_fermionic_hamiltonian as _addon_diagonalize_fermionic_hamiltonian,
)

from quantum_fragment_methods.application.solvers.base import SolverResult
from quantum_fragment_methods.application.solvers.quantum_zoo.sqd import (
    SQDSolver,
    counts_to_bit_array,
    make_sbd_sci_solver,
)
from quantum_fragment_methods.qpu.base import QPUBackend

logger = logging.getLogger(__name__)

__all__ = ["TrimSQDSolver"]


def _import_trim_policy(sector: bool = False):
    """Import TrimPolicy or SectorTrimPolicy with a helpful error on missing branch."""
    try:
        if sector:
            from qiskit_addon_sqd.trim import SectorTrimPolicy
            return SectorTrimPolicy
        else:
            from qiskit_addon_sqd.trim import TrimPolicy
            return TrimPolicy
    except ImportError as exc:
        raise ImportError(
            "TrimSQD requires qiskit-addon-sqd >= 0.14 (the trim-sqd branch). "
            "Install from the PR branch:\n"
            "  pip install git+https://github.com/Qiskit/qiskit-addon-sqd.git@trim-sqd\n"
            "or from the local checkout:\n"
            "  pip install -e /path/to/qiskit-addon-sqd"
        ) from exc


class TrimSQDSolver(SQDSolver):
    """SQD solver using TrimSQD screening-based subspace selection.

    Inherits all QPU logic from :class:`~.SQDSolver` — LUCJ circuit
    construction, job submission, and checkpoint management are identical.
    Only the classical post-processing step is overridden to use
    ``TrimPolicy`` or ``SectorTrimPolicy`` instead of the default
    ``StandardPolicy``.

    Configuration (``config["sqd"]``) supports all standard SQD keys plus:

    .. code-block:: yaml

        sqd:
          # --- Standard SQD keys (all supported) ---
          classical_backend: python
          symmetrize_spin: true
          n_batches: 5
          iterations: 5
          samples_per_batch: 300
          max_dim: 2000

          # --- TrimSQD-specific keys ---
          trim:
            trim_ratio: 0.5          # fraction of each batch retained (0 < x ≤ 1)
            max_strings_per_trim:    # optional int cap on strings kept per batch/spin
            max_carryover:           # optional int cap on merged carryover strings
            sector_trim: false       # true → SectorTrimPolicy; false → TrimPolicy
            partition_carryover: true  # SectorTrimPolicy only

    Notes
    -----
    ``carryover_threshold`` is **not** passed to ``diagonalize_fermionic_hamiltonian``
    when a policy is active — the two are mutually exclusive in the addon API.
    Carryover is controlled entirely by ``trim_ratio`` / ``max_carryover`` instead.
    """

    def __init__(self, qpu_backend: QPUBackend, config: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__(qpu_backend, config=config, **kwargs)
        self.trim_config: Dict[str, Any] = (self.config.get("trim") or {})

    # ------------------------------------------------------------------
    # Override only the post-processing step
    # ------------------------------------------------------------------

    def _sbd_postprocessing(
        self,
        h1e: np.ndarray,
        h2e: np.ndarray,
        counts: Dict[str, int],
        norb: int,
        nelec: Tuple[int, int],
        workflow_path: Path,
    ) -> SolverResult:
        """Classical post-processing using TrimSQD policy (qiskit-addon-sqd 0.14+)."""
        from qiskit_addon_sqd.fermion import solve_sci_batch

        # ── Standard config keys ──────────────────────────────────────────────
        iterations        = self.sqd_config.get("iterations", 5)
        n_batches         = self.sqd_config.get("n_batches", 5)
        samples_per_batch = self.sqd_config.get("samples_per_batch", 300)
        energy_tol        = self.sqd_config.get("energy_tol", 1.0e-8)
        occupancies_tol   = self.sqd_config.get("occupancies_tol", 1.0e-5)
        symmetrize_spin   = self.sqd_config.get("symmetrize_spin", True)
        classical_backend = self.sqd_config.get("classical_backend", "python")
        max_dim           = self.sqd_config.get("max_dim", None)
        seed              = self.sqd_config.get("seed", None)

        # ── TrimSQD-specific config keys ──────────────────────────────────────
        trim_ratio            = self.trim_config.get("trim_ratio", 0.5)
        max_strings_per_trim  = self.trim_config.get("max_strings_per_trim", None)
        max_carryover         = self.trim_config.get("max_carryover", None)
        use_sector_trim       = self.trim_config.get("sector_trim", False)
        partition_carryover   = self.trim_config.get("partition_carryover", True)

        print(
            f"  TrimSQD: {iterations} iter × {n_batches} batches × {samples_per_batch} samples"
            f"  trim_ratio={trim_ratio}"
            f"  {'SectorTrim' if use_sector_trim else 'Trim'}Policy"
            f"  backend={classical_backend}",
            flush=True,
        )

        # ── Build policy ──────────────────────────────────────────────────────
        if use_sector_trim:
            PolicyCls = _import_trim_policy(sector=True)
            policy = PolicyCls(
                trim_ratio=trim_ratio,
                max_strings_per_trim=max_strings_per_trim,
                max_carryover=max_carryover,
                partition_carryover=partition_carryover,
            )
        else:
            PolicyCls = _import_trim_policy(sector=False)
            policy = PolicyCls(
                trim_ratio=trim_ratio,
                max_strings_per_trim=max_strings_per_trim,
                max_carryover=max_carryover,
            )

        # ── BitArray ─────────────────────────────────────────────────────────
        bit_array = counts_to_bit_array(counts, num_bits=2 * norb)

        # ── HF bootstrap occupancies ──────────────────────────────────────────
        num_elec_a, num_elec_b = nelec
        hf_occ_a = np.array([1.0 if i < num_elec_a else 0.0 for i in range(norb)])
        hf_occ_b = np.array([1.0 if i < num_elec_b else 0.0 for i in range(norb)])
        initial_occupancies = (hf_occ_a, hf_occ_b)

        # ── sci_solver ───────────────────────────────────────────────────────
        if classical_backend == "sbd":
            sci_solver = make_sbd_sci_solver(self.sbd_config)
        else:
            sci_solver = partial(solve_sci_batch, spin_sq=0.0)

        # ── Diagonalize ───────────────────────────────────────────────────────
        # NOTE: carryover_threshold is NOT passed — it is mutually exclusive
        # with policy= in the addon 0.14 API.
        result = _addon_diagonalize_fermionic_hamiltonian(
            h1e,
            h2e,
            bit_array,
            samples_per_batch=samples_per_batch,
            norb=norb,
            nelec=nelec,
            num_batches=n_batches,
            energy_tol=energy_tol,
            occupancies_tol=occupancies_tol,
            max_iterations=iterations,
            policy=policy,
            symmetrize_spin=symmetrize_spin,
            max_dim=max_dim,
            sci_solver=sci_solver,
            initial_occupancies=initial_occupancies,
            seed=seed,
        )

        # ── Unpack result ─────────────────────────────────────────────────────
        rdm1 = result.rdm1
        rdm2 = result.rdm2
        if rdm1 is None and result.sci_state is not None:
            dm_a, dm_b = result.sci_state.rdm(rank=1, spin_summed=False)
            rdm1 = dm_a + dm_b
        if rdm2 is None and result.sci_state is not None:
            dm2_aa, dm2_ab, dm2_bb = result.sci_state.rdm(rank=2, spin_summed=False)
            rdm2 = dm2_aa + dm2_ab + dm2_ab.transpose(2, 3, 0, 1) + dm2_bb

        amplitudes = result.sci_state.amplitudes if result.sci_state is not None else None
        ci_strs_a  = result.sci_state.ci_strs_a  if result.sci_state is not None else None
        ci_strs_b  = result.sci_state.ci_strs_b  if result.sci_state is not None else None

        return SolverResult(
            energy=result.energy,
            wavefunction=amplitudes,
            rdm1=rdm1,
            rdm2=rdm2,
            metadata={
                "ci_strs_a":    ci_strs_a,
                "ci_strs_b":    ci_strs_b,
                "occupancies":  result.orbital_occupancies,
                "norb":         norb,
                "nelec":        nelec,
                "policy":       type(policy).__name__,
                "trim_ratio":   trim_ratio,
            },
        )
