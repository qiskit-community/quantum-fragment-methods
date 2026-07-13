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

"""SQD diagonalization with SBD as the selected-CI solver.

This module wraps :func:`qiskit_addon_sqd.fermion.diagonalize_fermionic_hamiltonian`
and injects an SBD-backed ``sci_solver``. The configuration-recovery loop itself
comes from ``qiskit-addon-sqd``; only the eigensolver integration is QFM-specific.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
from qiskit.primitives import BitArray
from qiskit_addon_sqd.fermion import (
    SCIResult,
    SCIState,
    diagonalize_fermionic_hamiltonian as _addon_diagonalize_fermionic_hamiltonian,
)

from quantum_fragment_methods.application.solvers.quantum_zoo.utils.sbd_interface import (
    SBDInterface,
)

logger = logging.getLogger(__name__)

__all__ = [
    "SCIResult",
    "SCIState",
    "counts_to_bit_array",
    "diagonalize_fermionic_hamiltonian",
    "make_sbd_sci_solver",
]


def counts_to_bit_array(
    counts: Mapping[str, int],
    *,
    num_bits: int | None = None,
) -> BitArray:
    """Convert a QPU counts dictionary into a :class:`~qiskit.primitives.BitArray`.

    Args:
        counts: Measurement counts mapping bitstrings to shot counts.
        num_bits: Number of bits per shot. Defaults to the longest key length.

    Returns:
        BitArray suitable for ``qiskit-addon-sqd``.
    """
    if not counts:
        raise ValueError("counts dictionary must contain at least one bitstring.")
    if num_bits is None:
        num_bits = max(len(bitstring) for bitstring in counts)
    return BitArray.from_counts(dict(counts), num_bits=num_bits)


def _write_fcidump(
    h1e: np.ndarray,
    h2e: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    fcidump_path: Path,
) -> None:
    """Write an FCIDUMP file for the SBD solver."""
    from pyscf import tools

    tools.fcidump.from_integrals(
        str(fcidump_path),
        h1e,
        h2e,
        norb,
        nelec,
        nuc=0.0,
    )


def make_sbd_sci_solver(
    sbd_config: dict[str, Any],
    workflow_path: str | Path,
) -> Callable[
    [list[tuple[np.ndarray, np.ndarray]], np.ndarray, np.ndarray, int, tuple[int, int]],
    list[SCIResult],
]:
    """Build an ``sci_solver`` callback that diagonalizes batches with SBD.

    Args:
        sbd_config: SBD configuration. Must include ``exe_path``.
        workflow_path: Directory for FCIDUMP files and per-batch SBD workdirs.

    Returns:
        Callable matching the ``sci_solver`` signature expected by
        ``qiskit-addon-sqd``.
    """
    sbd_exe_path = sbd_config.get("exe_path")
    if not sbd_exe_path:
        raise ValueError("exe_path must be specified in sbd_config")

    work_root = Path(workflow_path)
    work_root.mkdir(parents=True, exist_ok=True)
    sbd_interface = SBDInterface(sbd_exe_path=sbd_exe_path, config=sbd_config)
    call_count = {"n": 0}

    def sci_solver(
        ci_strings: list[tuple[np.ndarray, np.ndarray]],
        one_body_tensor: np.ndarray,
        two_body_tensor: np.ndarray,
        norb: int,
        nelec: tuple[int, int],
    ) -> list[SCIResult]:
        call_count["n"] += 1
        iteration = call_count["n"]
        iteration_dir = work_root / f"iteration_{iteration}"
        iteration_dir.mkdir(parents=True, exist_ok=True)

        fcidump_path = iteration_dir / "fcidump.txt"
        _write_fcidump(one_body_tensor, two_body_tensor, norb, nelec, fcidump_path)

        logger.info(
            "SBD sci_solver iteration %s: diagonalizing %s batch(es)",
            iteration,
            len(ci_strings),
        )

        results: list[SCIResult] = []
        for batch_idx, (ci_strs_a, _ci_strs_b) in enumerate(ci_strings):
            # SBD AlphaDets currently expect a single spin sector; when spins are
            # symmetrized, alpha and beta strings match. Prefer alpha otherwise.
            batch_work_dir = iteration_dir / f"batch_{batch_idx}"
            batch_work_dir.mkdir(parents=True, exist_ok=True)

            sbd_result = sbd_interface.run_sbd_solver(
                fcidump_path=str(fcidump_path),
                ci_strs_alpha=np.asarray(ci_strs_a).tolist(),
                norb=norb,
                nelec=nelec,
                work_dir=str(batch_work_dir),
                cpus_per_batch=sbd_config.get("cpus_per_batch", 4),
            )

            energy = sbd_result["energy"]
            if energy is None:
                raise RuntimeError(
                    f"SBD did not report an energy for iteration {iteration}, "
                    f"batch {batch_idx} (see {batch_work_dir})"
                )

            ci_strs_a_out = np.asarray(sbd_result["ci_strs_a"])
            ci_strs_b_out = np.asarray(sbd_result["ci_strs_b"])
            sci_state = SCIState(
                amplitudes=sbd_result["amplitudes"],
                ci_strs_a=ci_strs_a_out,
                ci_strs_b=ci_strs_b_out,
                norb=norb,
                nelec=nelec,
            )
            results.append(
                SCIResult(
                    energy=float(energy),
                    sci_state=sci_state,
                    orbital_occupancies=tuple(sbd_result["occupancies"]),
                    rdm1=sbd_result.get("rdm1"),
                    rdm2=sbd_result.get("rdm2"),
                )
            )

        return results

    return sci_solver


def diagonalize_fermionic_hamiltonian(
    one_body_tensor: np.ndarray,
    two_body_tensor: np.ndarray,
    counts: Mapping[str, int],
    samples_per_batch: int,
    norb: int,
    nelec: tuple[int, int],
    *,
    num_batches: int = 1,
    energy_tol: float = 1e-8,
    occupancies_tol: float = 1e-5,
    max_iterations: int = 100,
    symmetrize_spin: bool = True,
    max_dim: int | tuple[int, int] | None = None,
    include_configurations: list[int]
    | tuple[list[int], list[int]]
    | np.ndarray
    | None = None,
    initial_occupancies: tuple[np.ndarray, np.ndarray] | None = None,
    carryover_threshold: float = 1e-4,
    callback: Callable[[list[SCIResult]], None] | None = None,
    workflow_path: str = "",
    sbd_config: dict[str, Any] | None = None,
    seed: int | np.random.Generator | None = None,
) -> SCIResult:
    """Run SQD using ``qiskit-addon-sqd`` with SBD as the SCI solver.

    Args:
        one_body_tensor: One-body Hamiltonian tensor.
        two_body_tensor: Two-body Hamiltonian tensor.
        counts: Sampled bitstring counts from the QPU.
        samples_per_batch: Subsamples per configuration-recovery batch.
        norb: Number of spatial orbitals.
        nelec: ``(n_alpha, n_beta)`` electron counts.
        num_batches: Batches per configuration-recovery iteration.
        energy_tol: Energy convergence tolerance.
        occupancies_tol: Occupancy convergence tolerance.
        max_iterations: Maximum configuration-recovery iterations.
        symmetrize_spin: Merge alpha/beta CI strings (required for current SBD).
        max_dim: Optional CI subspace dimension cap.
        include_configurations: Configurations always included in the subspace.
        initial_occupancies: Optional initial orbital occupancies.
        carryover_threshold: Coefficient threshold for carryover determinants.
        callback: Optional callback after each iteration.
        workflow_path: Directory for SBD scratch files.
        sbd_config: SBD configuration (must include ``exe_path``).
        seed: RNG seed for recovery/subsampling.

    Returns:
        Best :class:`~qiskit_addon_sqd.fermion.SCIResult` found by SQD.
    """
    if sbd_config is None:
        raise ValueError("sbd_config must be provided for SBD solver integration")

    bit_array = counts_to_bit_array(counts, num_bits=2 * norb)
    sci_solver = make_sbd_sci_solver(sbd_config, workflow_path or ".")

    return _addon_diagonalize_fermionic_hamiltonian(
        one_body_tensor,
        two_body_tensor,
        bit_array,
        samples_per_batch,
        norb,
        nelec,
        num_batches=num_batches,
        energy_tol=energy_tol,
        occupancies_tol=occupancies_tol,
        max_iterations=max_iterations,
        sci_solver=sci_solver,
        symmetrize_spin=symmetrize_spin,
        max_dim=max_dim,
        include_configurations=include_configurations,
        initial_occupancies=initial_occupancies,
        carryover_threshold=carryover_threshold,
        callback=callback,
        seed=seed,
    )
