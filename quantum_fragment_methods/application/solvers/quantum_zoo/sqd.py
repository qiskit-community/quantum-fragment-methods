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

"""Sample-based Quantum Diagonalization (SQD) solver.

This module implements the SQD algorithm which combines quantum sampling
with classical post-processing to solve fermionic Hamiltonians.

It is the **single maintenance point** for all SQD-related logic:

* LUCJ circuit construction (via ffsim's built-in LUCJ pass manager)
* Fermionic Hamiltonian diagonalization helpers (formerly ``fermion_local``)
* SBD and Fulqrum sci_solver factories
* The SQDSolver class itself

The ``utils/`` sub-package re-exports from here for backward compatibility.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple

import numpy as np
from qiskit.primitives import BitArray
from qiskit_addon_sqd import fermion as sqd_fermion
from qiskit_addon_sqd.fermion import (
    SCIResult,
    SCIState,
    diagonalize_fermionic_hamiltonian as _addon_diagonalize_fermionic_hamiltonian,
)

from quantum_fragment_methods.application.solvers.base import BaseSolver, SolverResult
from quantum_fragment_methods.qpu.base import QPUBackend

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional SBD bindings (graceful degradation when not installed)
# ---------------------------------------------------------------------------

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

_VALID_CLASSICAL_BACKENDS = frozenset({"python", "hpc", "fulqrum"})

__all__ = [
    # Fermionic diagonalization helpers (public API)
    "SCIResult",
    "SCIState",
    "counts_to_bit_array",
    "diagonalize_fermionic_hamiltonian",
    "make_sbd_sci_solver",
    "make_fulqrum_sci_solver",
    # Solver class
    "SQDSolver",
]


# ===========================================================================
# Fermionic diagonalization helpers
# (formerly quantum_zoo/utils/fermion_local.py)
# ===========================================================================

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
        classical_backend: ``"python"`` for stock ``qiskit-addon-sqd``
            preprocessing, or ``"hpc"`` for ``qiskit-addon-sqd-hpc``.

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


# ===========================================================================
# SQDSolver
# ===========================================================================

class SQDSolver(BaseSolver):
    """Sample-based Quantum Diagonalization solver.

    This solver implements the SQD algorithm which uses quantum hardware
    to sample from a LUCJ ansatz, then performs classical post-processing
    with the SBD solver to obtain accurate ground state energies.

    Circuit construction delegates entirely to ``ffsim``'s built-in LUCJ
    pass manager (``ffsim.qiskit.lucj_pass_manager``), which handles qubit
    layout, routing, and transpilation for IBM heavy-hex backends natively.

    Attributes:
        qpu_backend: Quantum hardware backend for sampling
        config: Configuration dictionary with SQD parameters
    """

    def __init__(self, qpu_backend: QPUBackend, config: Optional[Dict[str, Any]] = None, **kwargs):
        """Initialize SQD solver.

        Args:
            qpu_backend: QPUBackend instance for quantum hardware access
            config: Configuration dictionary containing:
                - lucj: LUCJ ansatz parameters (n_reps, optimization_level, …)
                - sqd: SQD algorithm parameters
                - sbd: SBD solver configuration
                - transpilation: Circuit transpilation options (unused; layout
                  is handled by ffsim's pass manager)
            **kwargs: Additional solver options
        """
        super().__init__(**kwargs)
        self.qpu_backend = qpu_backend
        self.config = config or {}

        # Extract sub-configurations
        self.lucj_config = self.config.get("lucj", {})
        self.sqd_config = self.config
        self.sbd_config = self.config.get("sbd", {})
        self.transpilation_config = self.config.get("transpilation", {})
        self.circuit_save_dir = self.config.get("circuit_save_dir", None)

        logger.info(f"Initialized SQDSolver with backend: {qpu_backend}")

    def solve(
        self,
        h1e: np.ndarray,
        h2e: np.ndarray,
        norb: int,
        nelec: Tuple[int, int],
        mf: Optional[Any] = None,
        t1: Optional[np.ndarray] = None,
        t2: Optional[np.ndarray] = None,
        workflow_path: Optional[str] = None,
        wait_for_completion: bool = False,
        max_wait_time: int = 300,
        poll_interval: int = 30,
        force_resubmit: bool = False,
        **kwargs,
    ) -> SolverResult:
        """Solve fermionic Hamiltonian using SQD algorithm.

        This method implements a checkpoint-based workflow that can handle long QPU queue times:
        1. Submit QPU job and save job_id
        2. Optionally wait for completion (or return early if job is queued)
        3. Retrieve results when job completes
        4. Run SBD post-processing

        Args:
            h1e: One-body Hamiltonian tensor in MO basis (norb x norb)
            h2e: Two-body Hamiltonian tensor in MO basis (norb x norb x norb x norb)
            norb: Number of spatial orbitals
            nelec: Tuple of (n_alpha, n_beta) electrons
            mf: Mean-field object (for getting CCSD amplitudes if not provided)
            t1: CCSD single excitation amplitudes (optional)
            t2: CCSD double excitation amplitudes (optional)
            workflow_path: Path for storing intermediate results
            wait_for_completion: If True, poll until job completes. If False, raise error
                if job not ready (allows workflow to be resumed later). Default: False
            max_wait_time: Maximum time (seconds) to wait if wait_for_completion=True.
                Default: 300 (5 minutes). Set to large value for long waits.
            poll_interval: Time (seconds) between status checks. Default: 30
            force_resubmit: If True, ignore existing checkpoints and resubmit job.
                Useful when previous job failed. Default: False
            **kwargs: Additional options

        Returns:
            SolverResult containing energy, wavefunction, and RDMs

        Raises:
            ValueError: If required parameters are missing
            RuntimeError: If QPU job fails or is not ready (when wait_for_completion=False)
            TimeoutError: If job doesn't complete within max_wait_time
        """
        logger.debug(f"Starting SQD solve: norb={norb}, nelec={nelec}")

        # Setup workflow directory
        if workflow_path is None:
            workflow_path = os.getcwd()
        workflow_path = Path(workflow_path)
        workflow_path.mkdir(parents=True, exist_ok=True)

        # Check for existing job_id
        job_id_file = workflow_path / "job_id.txt"
        counts_file = workflow_path / "counts.npy"

        # Clear checkpoints if force_resubmit
        if force_resubmit:
            print("  [force-resubmit] Clearing existing checkpoints", flush=True)
            if job_id_file.exists():
                job_id_file.unlink()
            if counts_file.exists():
                counts_file.unlink()

        # Determine workflow stage
        if counts_file.exists():
            # Stage 3: Post-processing (counts already retrieved)
            counts = np.load(counts_file, allow_pickle=True).item()
            print(f"  Resuming from checkpoint: {sum(counts.values())} shots loaded", flush=True)

        elif job_id_file.exists():
            # Stage 2: Job submitted, need to retrieve results
            with open(job_id_file, "r") as f:
                job_id = f.read().strip()
            print(f"  Resuming QPU job {job_id} ...", flush=True)

            counts = self._retrieve_counts_with_wait(
                job_id, workflow_path, wait_for_completion, max_wait_time, poll_interval
            )

        else:
            # Stage 1: Fresh start - need to submit job
            if t1 is None or t2 is None:
                if mf is None:
                    raise ValueError("Either (t1, t2) or mf must be provided")
                logger.debug("Computing CCSD amplitudes from mean-field object...")
                t1, t2 = self._compute_ccsd_amplitudes(mf)

            job_id = self._qpu_sampling(t1, t2, norb, nelec, workflow_path)
            print(f"  QPU job submitted: {job_id}", flush=True)
            print(
                f"  Backend: {self.qpu_backend.config.get('backend_name', 'unknown')}"
                f"  checkpoint: {workflow_path}/job_id.txt",
                flush=True,
            )

            counts = self._retrieve_counts_with_wait(
                job_id, workflow_path, wait_for_completion, max_wait_time, poll_interval
            )

        # Classical post-processing with SBD
        result = self._sbd_postprocessing(h1e, h2e, counts, norb, nelec, workflow_path)
        print(f"  SBD complete — E = {result.energy:.8f} Ha", flush=True)
        return result

    def _compute_ccsd_amplitudes(self, mf: Any) -> Tuple[np.ndarray, np.ndarray]:
        """Compute CCSD amplitudes from mean-field object."""
        try:
            from pyscf import cc
        except ImportError as e:
            raise ImportError("PySCF is required for CCSD calculations") from e

        logger.info("Running CCSD calculation...")
        ccsd = cc.CCSD(mf).run()
        logger.info(f"CCSD energy: {ccsd.e_tot:.8f}")
        return ccsd.t1, ccsd.t2

    def _qpu_sampling(
        self, t1: np.ndarray, t2: np.ndarray, norb: int, nelec: Tuple[int, int], workflow_path: Path
    ) -> str:
        """Build LUCJ circuit and sample on QPU.  Returns job ID."""
        logger.info("Building LUCJ ansatz circuit via ffsim pass manager...")
        circuit = self._build_lucj_circuit(t1, t2, norb, nelec)

        # Save circuits to disk if circuit_save_dir is configured
        self._save_circuits(circuit, workflow_path)

        # Submit to QPU
        logger.info("Submitting job to QPU...")
        job_id = self._submit_to_qpu(circuit)

        # Save job ID
        job_id_file = workflow_path / "job_id.txt"
        with open(job_id_file, "w") as f:
            f.write(job_id)
        logger.info(f"Job ID saved to {job_id_file}")

        return job_id

    def _build_lucj_circuit(
        self, t1: np.ndarray, t2: np.ndarray, norb: int, nelec: Tuple[int, int]
    ) -> Any:
        """Build a hardware-native LUCJ ansatz circuit using ffsim's pass manager.

        Delegates entirely to ``ffsim.qiskit.lucj_pass_manager``, which:
          - Constructs the UCJ operator from t2 (and optionally t1) amplitudes
          - Selects the optimal zigzag qubit layout on the IBM heavy-hex backend
          - Transpiles and folds Rzz angles in one shot

        This replaces the previous manual rustworkx zigzag + Qiskit preset
        pass manager approach (lucj.py) with ffsim's authoritative router.

        Args:
            t1: CCSD single excitation amplitudes (norb × norb)
            t2: CCSD double excitation amplitudes (norb × norb × norb × norb)
            norb: Number of spatial orbitals
            nelec: (n_alpha, n_beta) electron counts

        Returns:
            ISA QuantumCircuit ready for QPU submission (measurements included)
        """
        try:
            import ffsim
            from ffsim.qiskit import lucj_pass_manager
            from qiskit import QuantumCircuit, QuantumRegister
        except ImportError as e:
            raise ImportError(
                "ffsim and qiskit are required for LUCJ circuit construction. "
                "Install with: pip install ffsim qiskit"
            ) from e

        cfg = self.lucj_config
        n_reps = cfg.get("n_reps", 1)
        optimization_level = cfg.get("optimization_level",
                                     self.transpilation_config.get("optimization_level", 0))
        seed = cfg.get("seed_transpiler",
                       self.transpilation_config.get("seed_transpiler", 0))

        logger.info(
            f"Building LUCJ circuit: norb={norb}, nelec={nelec}, "
            f"n_reps={n_reps}, optimization_level={optimization_level}"
        )

        # --- UCJ operator from CCSD amplitudes ---
        ucj_op = ffsim.UCJOpSpinBalanced.from_t_amplitudes(
            t2=t2,
            n_reps=n_reps,
        )

        # --- Abstract circuit: HF reference + UCJ ---
        qubits = QuantumRegister(2 * norb, name="q")
        circuit = QuantumCircuit(qubits)
        circuit.append(ffsim.qiskit.PrepareHartreeFockJW(norb, nelec), qubits)
        circuit.append(ffsim.qiskit.UCJOpSpinBalancedJW(ucj_op), qubits)
        circuit.measure_all()

        logger.info(f"Abstract circuit: {circuit.num_qubits} qubits, depth={circuit.depth()}")

        # --- ffsim pass manager: layout + routing + transpilation in one step ---
        target = self.qpu_backend.backend
        if target is None:
            raise RuntimeError("Backend not initialized. Call qpu_backend.get_backend() first.")

        pm = lucj_pass_manager(
            target=target,
            optimization_level=optimization_level,
            seed_transpiler=seed,
        )
        isa_circuit = pm.run(circuit)

        logger.info(
            f"ISA circuit: {isa_circuit.num_qubits} qubits, "
            f"depth={isa_circuit.depth()}, ops={isa_circuit.count_ops()}"
        )
        return isa_circuit

    def _save_circuits(self, isa_circuit: Any, workflow_path: Path) -> None:
        """Save the ISA circuit to disk in QPY format (when circuit_save_dir is set).

        Args:
            isa_circuit: Hardware-native ISA circuit produced by ffsim pass manager
            workflow_path: Per-fragment working directory
        """
        save_dir_name = self.circuit_save_dir
        if not save_dir_name:
            return

        try:
            from qiskit.qpy import dump as qpy_dump
        except ImportError:
            logger.warning("qiskit.qpy not available — skipping circuit save")
            return

        save_dir = workflow_path / save_dir_name
        save_dir.mkdir(parents=True, exist_ok=True)

        isa_path = save_dir / "circuit_isa.qpy"
        with open(isa_path, "wb") as f:
            qpy_dump(isa_circuit, f)
        logger.info(f"ISA circuit saved: {isa_path}")
        print(f"  Circuit saved to: {isa_path}", flush=True)

    def _submit_to_qpu(self, circuit: Any) -> str:
        """Submit circuit to QPU and return job ID."""
        sampler = self.qpu_backend.create_sampler()
        job_id = self.qpu_backend.submit_job([circuit], sampler)
        logger.info(f"Job submitted to QPU. Job ID: {job_id}")
        return job_id

    def _retrieve_counts(self, job_id: str, workflow_path: Path) -> Dict[str, int]:
        """Retrieve measurement counts from completed QPU job."""
        logger.info(f"Retrieving results for job {job_id}...")

        counts_file = workflow_path / "counts.npy"
        if counts_file.exists():
            logger.info(f"Loading counts from {counts_file}")
            return np.load(counts_file, allow_pickle=True).item()

        try:
            status = self.qpu_backend.get_job_status(job_id)
            logger.info(f"Job status: {status}")
        except Exception as e:
            logger.error(f"Failed to get job status: {e}")
            raise RuntimeError(
                f"Failed to retrieve job status for {job_id}. "
                "Please check your QPU connection and try again."
            ) from e

        if status in ["COMPLETED", "DONE"]:
            try:
                result = self.qpu_backend.get_job_result(job_id)
                counts = result[0].data.meas.get_counts()
                np.save(counts_file, counts)
                logger.info(f"Counts saved to {counts_file}  ({sum(counts.values())} shots)")
                return counts
            except Exception as e:
                raise RuntimeError(
                    f"Failed to retrieve results for completed job {job_id}. "
                    "The job may have expired or been cancelled."
                ) from e

        elif status in ["QUEUED", "VALIDATING", "RUNNING"]:
            raise RuntimeError(
                f"Job {job_id} is still {status}. "
                "Please wait for job completion and run the retrieval step again."
            )
        elif status in ["CANCELLED", "ERROR", "FAILED"]:
            raise RuntimeError(
                f"Job {job_id} {status}. "
                "Please check the IBM Quantum dashboard for details and resubmit if needed."
            )
        else:
            raise RuntimeError(f"Job {job_id} has unknown status: {status}.")

    def _retrieve_counts_with_wait(
        self,
        job_id: str,
        workflow_path: Path,
        wait_for_completion: bool,
        max_wait_time: int,
        poll_interval: int,
    ) -> Dict[str, int]:
        """Retrieve counts with optional polling for job completion."""
        counts_file = workflow_path / "counts.npy"
        if counts_file.exists():
            logger.info(f"Loading existing counts from {counts_file}")
            return np.load(counts_file, allow_pickle=True).item()

        try:
            status = self.qpu_backend.get_job_status(job_id)
            logger.info(f"Job status: {status}")
        except Exception as e:
            raise RuntimeError(
                f"Failed to retrieve job status for {job_id}. "
                "Please check your QPU connection and try again."
            ) from e

        if status in ["COMPLETED", "DONE"]:
            return self._retrieve_completed_job(job_id, workflow_path)

        if status in ["CANCELLED", "ERROR", "FAILED"]:
            raise RuntimeError(
                f"Job {job_id} {status}. "
                "Please check the IBM Quantum dashboard for details and resubmit if needed."
            )

        # Still in progress
        if not wait_for_completion:
            raise RuntimeError(
                f"Job {job_id} is still {status}.\n\n"
                f"QPU jobs can take up to 24 hours to complete.\n"
                f"To wait automatically, re-run solve() with wait_for_completion=True.\n"
                f"Once complete, simply re-run solve() — it will resume from the checkpoint."
            )

        print(f"  Polling job {job_id} every {poll_interval}s (max {max_wait_time}s) ...", flush=True)
        start_time = time.time()

        while True:
            elapsed = time.time() - start_time
            if elapsed > max_wait_time:
                raise TimeoutError(
                    f"Job {job_id} did not complete within {max_wait_time} seconds.\n"
                    f"Current status: {status}\n"
                    f"Re-run solve() with a larger max_wait_time, or re-run later."
                )

            time.sleep(poll_interval)

            try:
                status = self.qpu_backend.get_job_status(job_id)
                elapsed_str = f"{int(elapsed // 60)}m{int(elapsed % 60):02d}s"
                print(f"  [{elapsed_str}] {job_id} — {status}", flush=True)
            except Exception as e:
                logger.warning(f"Failed to check status: {e}")
                continue

            if status in ["COMPLETED", "DONE"]:
                elapsed_m, elapsed_s = int(elapsed // 60), int(elapsed % 60)
                print(f"  ✓ Job completed ({elapsed_m}m{elapsed_s:02d}s)", flush=True)
                return self._retrieve_completed_job(job_id, workflow_path)

            if status in ["CANCELLED", "ERROR", "FAILED"]:
                raise RuntimeError(
                    f"Job {job_id} {status}. "
                    "Please check the IBM Quantum dashboard for details and resubmit if needed."
                )

    def _retrieve_completed_job(self, job_id: str, workflow_path: Path) -> Dict[str, int]:
        """Retrieve and save results from a completed job."""
        try:
            result = self.qpu_backend.get_job_result(job_id)
            counts = result[0].data.meas.get_counts()
            counts_file = workflow_path / "counts.npy"
            np.save(counts_file, counts)
            logger.info(f"Counts saved to {counts_file}  ({sum(counts.values())} shots)")
            return counts
        except Exception as e:
            raise RuntimeError(
                f"Failed to retrieve results for completed job {job_id}. "
                "The job may have expired or been cancelled."
            ) from e

    def _sbd_postprocessing(
        self,
        h1e: np.ndarray,
        h2e: np.ndarray,
        counts: Dict[str, int],
        norb: int,
        nelec: Tuple[int, int],
        workflow_path: Path,
    ) -> SolverResult:
        """Classical post-processing with SBD solver."""
        logger.debug("Starting SBD post-processing via qiskit-addon-sqd")

        iterations = self.sqd_config.get("iterations", 5)
        n_batches = self.sqd_config.get("n_batches", 10)
        samples_per_batch = self.sqd_config.get("samples_per_batch", 3000)
        energy_tol = self.sqd_config.get("energy_tol", 1.0e-8)
        occupancies_tol = self.sqd_config.get("occupancies_tol", 1.0e-5)
        carryover_threshold = self.sqd_config.get("carryover_threshold", 1.0e-4)
        symmetrize_spin = self.sqd_config.get("symmetrize_spin", True)
        classical_backend = self.sqd_config.get("classical_backend", "python")

        print(
            f"  SBD: {iterations} iter × {n_batches} batches × {samples_per_batch} samples"
            f"  backend={classical_backend}",
            flush=True,
        )

        sqd_workflow_path = workflow_path / "sqd_diagonalizer"
        sqd_workflow_path.mkdir(parents=True, exist_ok=True)

        result = diagonalize_fermionic_hamiltonian(
            h1e,
            h2e,
            counts,
            symmetrize_spin=symmetrize_spin,
            samples_per_batch=samples_per_batch,
            norb=norb,
            nelec=nelec,
            num_batches=n_batches,
            energy_tol=energy_tol,
            occupancies_tol=occupancies_tol,
            max_iterations=iterations,
            carryover_threshold=carryover_threshold,
            workflow_path=str(sqd_workflow_path),
            sbd_config=self.sbd_config,
            classical_backend=classical_backend,
        )

        rdm1 = result.rdm1 if result.rdm1 is not None else result.sci_state.rdm(rank=1)
        rdm2 = result.rdm2 if result.rdm2 is not None else result.sci_state.rdm(rank=2)

        solver_result = SolverResult(
            energy=result.energy,
            wavefunction=result.sci_state.amplitudes,
            rdm1=rdm1,
            rdm2=rdm2,
            metadata={
                "ci_strs_a": result.sci_state.ci_strs_a,
                "ci_strs_b": result.sci_state.ci_strs_b,
                "occupancies": result.orbital_occupancies,
                "norb": norb,
                "nelec": nelec,
            },
        )

        logger.debug(f"SBD post-processing complete. Energy: {solver_result.energy:.8f}")
        return solver_result

    def solve_from_integrals(
        self, h1e: np.ndarray, h2e: np.ndarray, norb: int, nocc: int, **kwargs
    ) -> SolverResult:
        """Convenience method for workflow integration (closed-shell fragments).

        Parameters
        ----------
        h1e : np.ndarray
            One-electron Hamiltonian (effective Hamiltonian)
        h2e : np.ndarray
            Two-electron repulsion integrals (ERIs)
        norb : int
            Number of orbitals
        nocc : int
            Number of occupied orbitals (assumes closed shell)
        **kwargs : dict
            Additional parameters passed to solve()
        """
        from pyscf import gto, scf

        mol = gto.M()
        mol.nelectron = 2 * nocc
        mol.spin = 0
        mol.nao = norb
        mol.verbose = 0

        mf = scf.RHF(mol)
        mf.get_hcore = lambda *args: h1e
        mf.get_ovlp = lambda *args: np.eye(norb)
        mf._eri = h2e
        mf.kernel()

        if "wait_for_completion" not in kwargs:
            kwargs["wait_for_completion"] = True
        if "max_wait_time" not in kwargs:
            kwargs["max_wait_time"] = 86400  # 24 hours

        nelec = (nocc, nocc)
        return self.solve(h1e, h2e, norb, nelec, mf=mf, **kwargs)

    def _write_fcidump(
        self,
        h1e: np.ndarray,
        h2e: np.ndarray,
        norb: int,
        nelec: Tuple[int, int],
        fcidump_path: Path,
    ) -> None:
        """Write Hamiltonian to FCIDUMP format."""
        try:
            from pyscf import tools
        except ImportError as e:
            raise ImportError("PySCF is required for FCIDUMP generation") from e

        n_alpha, n_beta = nelec
        total_elec = n_alpha + n_beta
        with open(fcidump_path, "w") as f:
            tools.fcidump.write_head(f, norb, total_elec, ms=n_alpha - n_beta)
            tools.fcidump.write_eri(f, h2e, norb, tol=1e-15)
            tools.fcidump.write_hcore(f, h1e, norb, tol=1e-15)
        logger.info(f"FCIDUMP written to {fcidump_path}")

    @property
    def name(self) -> str:
        """Return solver name."""
        return "SQD"
