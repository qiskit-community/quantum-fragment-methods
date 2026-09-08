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
from functools import partial
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

try:
    from sbd.sbd_solver import solve_sci_batch as _sbd_solve_sci_batch
    from sbd.device_config import DeviceConfig as _SBDDeviceConfig
except ImportError as _sbd_import_err:  # pragma: no cover
    _sbd_solve_sci_batch = None  # type: ignore[assignment]
    _SBDDeviceConfig = None  # type: ignore[assignment]
    _sbd_import_err_msg = (
        "sbd-eigensolver is not installed.  "
        "Run: pip install sbd-eigensolver"
    )

logger = logging.getLogger(__name__)

__all__ = [
    "SCIResult",
    "SCIState",
    "counts_to_bit_array",
    "diagonalize_fermionic_hamiltonian",
    "make_sbd_sci_solver",
]

_VALID_CLASSICAL_BACKENDS = frozenset({"python", "hpc", "fulqrum"})


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


def make_sbd_sci_solver(
    sbd_config: dict[str, Any],
    device: str = "cpu",
    mpi_comm: Any | None = None,
) -> Callable[
    [list[tuple[np.ndarray, np.ndarray]], np.ndarray, np.ndarray, int, tuple[int, int]],
    list[SCIResult],
]:
    """Build an ``sci_solver`` backed by ``sbd-eigensolver`` Python bindings.

    Returns a ``partial(solve_sci_batch, ...)`` that matches the ``sci_solver``
    signature expected by :func:`diagonalize_fermionic_hamiltonian`.  No
    subprocess is spawned and no FCIDUMP files are written — the bindings handle
    all internal I/O via a managed temp directory.

    Args:
        sbd_config: SBD configuration dict.  Keys map directly to ``TPB_SBD``
            struct fields (``method``, ``eps``, ``max_it``, ``max_nb``,
            ``max_time``, ``do_rdm``, ``do_shuffle``, ``carryover_type``,
            ``ratio``, ``threshold``, ``bit_length``, etc.).
            ``exe_path``, ``cpus_per_batch``, ``device``, ``mpi_ranks``,
            and ``mri_ranks`` (legacy misspelling of ``mpi_ranks``) are
            consumed here and not forwarded to ``solve_sci_batch``.
        device: ``'cpu'`` (default) or ``'gpu'`` / ``'gpu-omp'``.
            Overridden by ``sbd_config.get('device')`` when present.
        mpi_comm: MPI communicator.  Defaults to ``MPI.COMM_WORLD`` inside
            ``solve_sci_batch`` when ``None``.

    Returns:
        Callable matching the ``sci_solver`` signature.

    Raises:
        ImportError: if ``sbd-eigensolver`` is not installed.
    """
    if _sbd_solve_sci_batch is None:  # pragma: no cover
        raise ImportError(_sbd_import_err_msg)

    # Resolve device — config key takes precedence over kwarg
    resolved_device = sbd_config.get("device", device)
    device_config = _SBDDeviceConfig(device=resolved_device)

    # Strip non-TPB_SBD keys so _create_sbd_config doesn't choke on unknowns
    _strip = {"exe_path", "cpus_per_batch", "device", "mpi_ranks", "mri_ranks"}
    tpb_config = {k: v for k, v in sbd_config.items() if k not in _strip}

    logger.debug(
        "make_sbd_sci_solver: device=%s  tpb_keys=%s",
        resolved_device,
        sorted(tpb_config),
    )

    return partial(
        _sbd_solve_sci_batch,
        sbd_config=tpb_config,
        device_config=device_config,
        mpi_comm=mpi_comm,
    )


def make_fulqrum_sci_solver(
    workflow_path: str | Path,
    max_iterations: int = 0,
) -> Callable[
    [list[tuple[np.ndarray, np.ndarray]], np.ndarray, np.ndarray, int, tuple[int, int]],
    list[SCIResult],
]:
    """Build an ``sci_solver`` callback backed by Fulqrum.

    STUB: currently delegates to ``qiskit_addon_sqd.fermion.solve_fermion``
    (PySCF FCI on the selected subspace).  Replace the body of the inner
    ``sci_solver()`` function with real Fulqrum calls once the colleague
    use-case example is available.

    Expected Fulqrum interface (to be confirmed)::

        subspace = fulqrum.Subspace(ci_strings, norb, nelec)
        result   = fulqrum.solve(subspace, one_body_tensor, two_body_tensor)
        return [SCIResult(energy=result.energy, sci_state=..., ...)]

    Args:
        workflow_path: Directory for any scratch files (unused by stub).
        max_iterations: Total iterations expected (used for progress display).

    Returns:
        Callable matching the ``sci_solver`` signature expected by
        ``qiskit-addon-sqd``.
    """
    call_count = {"n": 0}
    best_energy = {"e": None}

    def sci_solver(
        ci_strings: list[tuple[np.ndarray, np.ndarray]],
        one_body_tensor: np.ndarray,
        two_body_tensor: np.ndarray,
        norb: int,
        nelec: tuple[int, int],
    ) -> list[SCIResult]:
        # ── STUB ──────────────────────────────────────────────────────────────
        # TODO: replace this block with real Fulqrum calls once the use-case
        # example is available from the colleague.  The function signature and
        # return type must remain identical.
        # ──────────────────────────────────────────────────────────────────────
        from qiskit_addon_sqd.fermion import solve_fermion

        call_count["n"] += 1
        results: list[SCIResult] = []

        for ci_strs_a, ci_strs_b in ci_strings:
            sci_result = solve_fermion(
                (ci_strs_a, ci_strs_b),
                one_body_tensor,
                two_body_tensor,
                norb=norb,
                nelec=nelec,
            )
            results.append(sci_result)

        iter_energy = min(r.energy for r in results)
        prev = best_energy["e"]
        delta_str = f"  Δ={iter_energy - prev:+.6f} Ha" if prev is not None else ""
        total_str = f"/{max_iterations}" if max_iterations > 0 else ""
        print(
            f"    Fulqrum(stub) iter {call_count['n']:2d}{total_str}"
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
        workflow_path: Directory for SBD scratch files (Fulqrum only; ignored
            by the ``sbd-eigensolver`` backend which manages its own temp dir).
        sbd_config: SBD configuration dict passed to
            :func:`make_sbd_sci_solver`.  Must contain at least ``device``
            (``'cpu'`` or ``'gpu'``); all other keys are forwarded to
            ``TPB_SBD``.  Not required for ``fulqrum`` backend.
        seed: RNG seed for recovery/subsampling.
        classical_backend: ``\"python\"`` for stock ``qiskit-addon-sqd``
            preprocessing, or ``\"hpc\"`` for ``qiskit-addon-sqd-hpc``.

    Returns:
        Best :class:`~qiskit_addon_sqd.fermion.SCIResult` found by SQD.
    """
    backend = classical_backend.lower().strip()
    if backend not in _VALID_CLASSICAL_BACKENDS:
        raise ValueError(
            f"classical_backend must be one of {sorted(_VALID_CLASSICAL_BACKENDS)}, "
            f"got {classical_backend!r}"
        )

    if backend not in ("fulqrum",) and sbd_config is None:
        raise ValueError("sbd_config must be provided for SBD solver integration")

    bit_array = counts_to_bit_array(counts, num_bits=2 * norb)

    # ── Fulqrum backend ───────────────────────────────────────────────────────
    if backend == "fulqrum":
        sci_solver = make_fulqrum_sci_solver(
            workflow_path or ".", max_iterations=max_iterations
        )
        logger.info("Using Fulqrum sci_solver for classical SQD diagonalization")
        return _addon_diagonalize_fermionic_hamiltonian(
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

    # ── SBD backend (python or hpc classical preprocessing) ──────────────────
    sci_solver = make_sbd_sci_solver(sbd_config)

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
