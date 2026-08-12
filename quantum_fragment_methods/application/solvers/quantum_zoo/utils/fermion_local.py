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
and injects an SBD-backed ``sci_solver``. Classical configuration recovery /
subsampling can use stock ``qiskit-addon-sqd`` (``classical_backend=\"python\"``)
or the C++ bindings from ``qiskit-addon-sqd-hpc`` (``classical_backend=\"hpc\"``).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from qiskit.primitives import BitArray
from qiskit_addon_sqd import fermion as sqd_fermion
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

_VALID_CLASSICAL_BACKENDS = frozenset({"python", "hpc"})


def _seed_to_int(rand_seed: Any) -> int | None:
    """Map a numpy Generator / int / None to an integer seed for HPC RNG."""
    if rand_seed is None:
        return None
    if isinstance(rand_seed, np.random.Generator):
        return int(rand_seed.integers(0, 2**63 - 1))
    return int(rand_seed)


def _as_contig_bool(matrix: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(matrix, dtype=bool)


def _as_contig_f64(vector: np.ndarray | Sequence[float]) -> np.ndarray:
    return np.ascontiguousarray(vector, dtype=np.float64)


def _hpc_postselect(
    bitstring_matrix: np.ndarray,
    probabilities: np.ndarray,
    *,
    hamming_right: int,
    hamming_left: int,
) -> tuple[np.ndarray, np.ndarray]:
    import qiskit_addon_sqd_hpc as hpc

    out_bs, out_p = hpc.postselect_by_hamming_right_and_left(
        _as_contig_bool(bitstring_matrix),
        _as_contig_f64(probabilities),
        int(hamming_right),
        int(hamming_left),
    )
    return np.asarray(out_bs, dtype=bool), np.asarray(out_p, dtype=np.float64)


def _hpc_recover(
    bitstring_matrix: np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    avg_occupancies: tuple[np.ndarray, np.ndarray],
    num_elec_a: int,
    num_elec_b: int,
    rand_seed: np.random.Generator | int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    import qiskit_addon_sqd_hpc as hpc

    occ_a, occ_b = avg_occupancies
    out_bs, out_p = hpc.recover_configurations(
        _as_contig_bool(bitstring_matrix),
        _as_contig_f64(probabilities),
        _as_contig_f64(occ_a),
        _as_contig_f64(occ_b),
        int(num_elec_a),
        int(num_elec_b),
        _seed_to_int(rand_seed),
    )
    return np.asarray(out_bs, dtype=bool), np.asarray(out_p, dtype=np.float64)


def _hpc_subsample(
    bitstring_matrix: np.ndarray,
    probabilities: np.ndarray,
    samples_per_batch: int,
    num_batches: int,
    rand_seed: np.random.Generator | int | None = None,
) -> list[np.ndarray]:
    import qiskit_addon_sqd_hpc as hpc

    batches = hpc.subsample(
        _as_contig_bool(bitstring_matrix),
        _as_contig_f64(probabilities),
        int(samples_per_batch),
        int(num_batches),
        _seed_to_int(rand_seed),
    )
    return [np.asarray(batch, dtype=bool) for batch in batches]


@contextmanager
def _use_hpc_classical_preprocess() -> Iterator[None]:
    """Temporarily route addon preprocess helpers to C++ HPC bindings."""
    try:
        import qiskit_addon_sqd_hpc  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "classical_backend='hpc' requires the qiskit-addon-sqd-hpc package. "
            "Install it in the container (pip install -e /workspace/qiskit-addon-sqd-hpc) "
            "or set classical_backend='python'."
        ) from exc

    originals = {
        "postselect_by_hamming_right_and_left": (
            sqd_fermion.postselect_by_hamming_right_and_left
        ),
        "recover_configurations": sqd_fermion.recover_configurations,
        "subsample": sqd_fermion.subsample,
    }
    sqd_fermion.postselect_by_hamming_right_and_left = _hpc_postselect
    sqd_fermion.recover_configurations = _hpc_recover
    sqd_fermion.subsample = _hpc_subsample
    logger.info("Using qiskit-addon-sqd-hpc for classical SQD preprocessing")
    try:
        yield
    finally:
        sqd_fermion.postselect_by_hamming_right_and_left = originals[
            "postselect_by_hamming_right_and_left"
        ]
        sqd_fermion.recover_configurations = originals["recover_configurations"]
        sqd_fermion.subsample = originals["subsample"]



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
    max_iterations: int = 0,
) -> Callable[
    [list[tuple[np.ndarray, np.ndarray]], np.ndarray, np.ndarray, int, tuple[int, int]],
    list[SCIResult],
]:
    """Build an ``sci_solver`` callback that diagonalizes batches with SBD.

    Args:
        sbd_config: SBD configuration. Must include ``exe_path``.
        workflow_path: Directory for FCIDUMP files and per-batch SBD workdirs.
        max_iterations: Total SBD iterations expected (used for progress display).

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
    best_energy = {"e": None}

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

        logger.debug(
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

        # Print one concise line per SBD iteration: energy + delta from previous
        iter_energy = min(r.energy for r in results)
        prev = best_energy["e"]
        delta_str = f"  Δ={iter_energy - prev:+.6f} Ha" if prev is not None else ""
        total_str = f"/{max_iterations}" if max_iterations > 0 else ""
        print(
            f"    SBD iter {iteration:2d}{total_str}"
            f"  E={iter_energy:.8f} Ha{delta_str}",
            flush=True,
        )
        best_energy["e"] = iter_energy

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
    classical_backend: str = "python",
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
        classical_backend: ``\"python\"`` for stock ``qiskit-addon-sqd``
            preprocessing, or ``\"hpc\"`` for ``qiskit-addon-sqd-hpc``.

    Returns:
        Best :class:`~qiskit_addon_sqd.fermion.SCIResult` found by SQD.
    """
    if sbd_config is None:
        raise ValueError("sbd_config must be provided for SBD solver integration")

    backend = classical_backend.lower().strip()
    if backend not in _VALID_CLASSICAL_BACKENDS:
        raise ValueError(
            f"classical_backend must be one of {sorted(_VALID_CLASSICAL_BACKENDS)}, "
            f"got {classical_backend!r}"
        )

    bit_array = counts_to_bit_array(counts, num_bits=2 * norb)
    sci_solver = make_sbd_sci_solver(sbd_config, workflow_path or ".", max_iterations=max_iterations)

    kwargs = dict(
        one_body_tensor=one_body_tensor,
        two_body_tensor=two_body_tensor,
        bit_array=bit_array,
        samples_per_batch=samples_per_batch,
        norb=norb,
        nelec=nelec,
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

    if backend == "hpc":
        with _use_hpc_classical_preprocess():
            return _addon_diagonalize_fermionic_hamiltonian(**kwargs)

    logger.info("Using qiskit-addon-sqd (Python) for classical SQD preprocessing")
    return _addon_diagonalize_fermionic_hamiltonian(**kwargs)
