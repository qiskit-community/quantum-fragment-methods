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
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from quantum_fragment_methods.application.solvers.base import BaseSolver, SolverResult
from quantum_fragment_methods.qpu.base import QPUBackend

logger = logging.getLogger(__name__)


class SQDSolver(BaseSolver):
    """Sample-based Quantum Diagonalization solver.

    This solver implements the SQD algorithm which uses quantum hardware
    to sample from a LUCJ ansatz, then performs classical post-processing
    with the SBD solver to obtain accurate ground state energies.

    Attributes:
        qpu_backend: Quantum hardware backend for sampling
        config: Configuration dictionary with SQD parameters
    """

    def __init__(self, qpu_backend: QPUBackend, config: Optional[Dict[str, Any]] = None, **kwargs):
        """Initialize SQD solver.

        Args:
            qpu_backend: QPUBackend instance for quantum hardware access
            config: Configuration dictionary containing:
                - lucj: LUCJ ansatz parameters
                - sqd: SQD algorithm parameters
                - sbd: SBD solver configuration
                - transpilation: Circuit transpilation options
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
            h1e: One-body Hamiltonian tensor (norb x norb)
            h2e: Two-body Hamiltonian tensor (norb x norb x norb x norb)
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

        Notes:
            For long queue times (up to 24 hours), use wait_for_completion=False and
            re-run solve() later. The method will detect existing job_id and resume
            from the appropriate checkpoint.

            If a job fails, use force_resubmit=True to clear checkpoints and resubmit.
        """
        logger.info(f"Starting SQD solve for system with {norb} orbitals, {nelec} electrons")

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
            logger.info("=" * 60)
            logger.info("FORCE RESUBMIT: Clearing existing checkpoints")
            logger.info("=" * 60)
            if job_id_file.exists():
                job_id_file.unlink()
                logger.info(f"Removed {job_id_file}")
            if counts_file.exists():
                counts_file.unlink()
                logger.info(f"Removed {counts_file}")

        # Determine workflow stage
        if counts_file.exists():
            # Stage 3: Post-processing (counts already retrieved)
            logger.info("=" * 60)
            logger.info("RESUMING: Counts already retrieved, running post-processing")
            logger.info("=" * 60)
            counts = np.load(counts_file, allow_pickle=True).item()
            logger.info(f"Loaded {sum(counts.values())} shots from {counts_file}")

        elif job_id_file.exists():
            # Stage 2: Job submitted, need to retrieve results
            with open(job_id_file, "r") as f:
                job_id = f.read().strip()
            logger.info("=" * 60)
            logger.info(f"RESUMING: Found existing job {job_id}")
            logger.info("=" * 60)

            # Try to retrieve counts (with optional waiting)
            counts = self._retrieve_counts_with_wait(
                job_id, workflow_path, wait_for_completion, max_wait_time, poll_interval
            )

        else:
            # Stage 1: Fresh start - need to submit job
            # Validate inputs and get CCSD amplitudes
            if t1 is None or t2 is None:
                if mf is None:
                    raise ValueError("Either (t1, t2) or mf must be provided")
                logger.info("Computing CCSD amplitudes from mean-field object...")
                t1, t2 = self._compute_ccsd_amplitudes(mf)

            logger.info("=" * 60)
            logger.info("STEP 1: Submitting QPU Job")
            logger.info("=" * 60)
            job_id = self._qpu_sampling(t1, t2, norb, nelec, workflow_path)
            logger.info(f"✓ Job submitted: {job_id}")

            # Try to retrieve counts (with optional waiting)
            counts = self._retrieve_counts_with_wait(
                job_id, workflow_path, wait_for_completion, max_wait_time, poll_interval
            )

        # Step 3: Classical Post-Processing
        logger.info("=" * 60)
        logger.info("STEP 3: Classical Post-Processing with SBD")
        logger.info("=" * 60)
        result = self._sbd_postprocessing(h1e, h2e, counts, norb, nelec, workflow_path)
        logger.info(f"✓ SBD post-processing complete")

        logger.info("=" * 60)
        logger.info(f"SQD SOLVE COMPLETE - Final energy: {result.energy:.8f}")
        logger.info("=" * 60)
        return result

    def _compute_ccsd_amplitudes(self, mf: Any) -> Tuple[np.ndarray, np.ndarray]:
        """Compute CCSD amplitudes from mean-field object.

        Args:
            mf: PySCF mean-field object

        Returns:
            Tuple of (t1, t2) CCSD amplitudes
        """
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
        """Build LUCJ circuit and sample on QPU.

        Args:
            t1: CCSD single excitation amplitudes
            t2: CCSD double excitation amplitudes
            norb: Number of spatial orbitals
            nelec: Tuple of (n_alpha, n_beta) electrons
            workflow_path: Path for storing results

        Returns:
            Job ID string for the submitted QPU job
        """
        # Build LUCJ circuit
        logger.info("Building LUCJ ansatz circuit...")
        circuit = self._build_lucj_circuit(t1, t2, norb, nelec)

        # Transpile for hardware
        logger.info("Transpiling circuit for hardware...")
        isa_circuit = self._transpile_circuit(circuit)

        # Submit to QPU
        logger.info("Submitting job to QPU...")
        job_id = self._submit_to_qpu(isa_circuit)

        # Save job ID
        job_id_file = workflow_path / "job_id.txt"
        with open(job_id_file, "w") as f:
            f.write(job_id)
        logger.info(f"Job ID saved to {job_id_file}")

        return job_id

    def _build_lucj_circuit(
        self, t1: np.ndarray, t2: np.ndarray, norb: int, nelec: Tuple[int, int]
    ) -> Any:
        """Build LUCJ ansatz circuit from CCSD amplitudes.

        Args:
            t1: CCSD single excitation amplitudes
            t2: CCSD double excitation amplitudes
            norb: Number of spatial orbitals
            nelec: Tuple of (n_alpha, n_beta) electrons

        Returns:
            Quantum circuit implementing LUCJ ansatz

        Raises:
            ImportError: If required packages are not installed
        """
        from quantum_fragment_methods.application.solvers.quantum_zoo.utils.lucj import (
            build_lucj_circuit,
        )

        # Build circuit using utility function
        circuit = build_lucj_circuit(t1, t2, norb, nelec, self.lucj_config)

        # Add measurements
        circuit.measure_all()

        logger.info(f"Circuit built: {circuit.num_qubits} qubits, depth {circuit.depth()}")

        return circuit

    def _transpile_circuit(self, circuit: Any) -> Any:
        """Transpile circuit for target hardware backend.

        Args:
            circuit: Quantum circuit to transpile

        Returns:
            Transpiled ISA circuit
        """
        try:
            import ffsim.qiskit
            from qiskit.transpiler import PassManager
            from qiskit.transpiler.passes import RemoveIdentityEquivalent
            from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
            from qiskit_ibm_runtime.transpiler.passes import FoldRzzAngle
        except ImportError as e:
            raise ImportError("qiskit and ffsim are required for transpilation") from e

        # Get transpilation options
        optimization_level = self.transpilation_config.get("optimization_level", 0)
        seed = self.transpilation_config.get("seed_transpiler", 0)

        # Get backend from QPU
        backend = self.qpu_backend.backend
        if backend is None:
            raise RuntimeError("Backend not initialized. Call qpu_backend.get_backend() first.")

        # Generate pass manager
        pm = generate_preset_pass_manager(
            optimization_level=optimization_level, backend=backend, seed_transpiler=seed
        )

        # Add custom passes
        pm.pre_init = ffsim.qiskit.PRE_INIT
        pm.post_init = PassManager([RemoveIdentityEquivalent()])
        pm.post_optimization = PassManager(
            [
                FoldRzzAngle(),
                RemoveIdentityEquivalent(target=backend.target),
            ]
        )

        # Run transpilation
        isa_circuit = pm.run(circuit)

        logger.info(f"Transpiled circuit: {isa_circuit.count_ops()}")
        logger.info(f"Circuit depth: {isa_circuit.depth()}")

        return isa_circuit

    def _submit_to_qpu(self, circuit: Any) -> str:
        """Submit circuit to QPU and return job ID.

        Args:
            circuit: Transpiled quantum circuit

        Returns:
            Job ID string
        """
        # Create sampler
        sampler = self.qpu_backend.create_sampler()

        # Submit job
        job_id = self.qpu_backend.submit_job([circuit], sampler)

        logger.info(f"Job submitted to QPU. Job ID: {job_id}")

        return job_id

    def _retrieve_counts(self, job_id: str, workflow_path: Path) -> Dict[str, int]:
        """Retrieve measurement counts from completed QPU job.

        Args:
            job_id: Job identifier
            workflow_path: Path for storing counts

        Returns:
            Dictionary of measurement counts

        Raises:
            RuntimeError: If job is not completed and needs to wait
        """
        logger.info(f"Retrieving results for job {job_id}...")

        # Check if counts already saved
        counts_file = workflow_path / "counts.npy"
        if counts_file.exists():
            logger.info(f"Loading counts from {counts_file}")
            counts = np.load(counts_file, allow_pickle=True).item()
            return counts

        # Check job status
        try:
            status = self.qpu_backend.get_job_status(job_id)
            logger.info(f"Job status: {status}")
        except Exception as e:
            logger.error(f"Failed to get job status: {e}")
            raise RuntimeError(
                f"Failed to retrieve job status for {job_id}. "
                "Please check your QPU connection and try again."
            ) from e

        # Handle different job states
        if status in ["COMPLETED", "DONE"]:
            # Job completed successfully - retrieve results
            try:
                result = self.qpu_backend.get_job_result(job_id)
                pub_result = result[0]
                counts = pub_result.data.meas.get_counts()

                # Save counts
                np.save(counts_file, counts)
                logger.info(f"Counts saved to {counts_file}")
                logger.info(f"Total shots: {sum(counts.values())}")

                return counts
            except Exception as e:
                logger.error(f"Failed to retrieve job results: {e}")
                raise RuntimeError(
                    f"Failed to retrieve results for completed job {job_id}. "
                    "The job may have expired or been cancelled."
                ) from e

        elif status in ["QUEUED", "VALIDATING", "RUNNING"]:
            # Job is still in progress
            logger.warning(f"Job not completed yet (status: {status}).")
            raise RuntimeError(
                f"Job {job_id} is still {status}. "
                f"Please wait for job completion and run the retrieval step again. "
                f"You can monitor the job status in section 7 of the notebook."
            )

        elif status in ["CANCELLED", "ERROR", "FAILED"]:
            # Job failed or was cancelled
            logger.error(f"Job {status}: {job_id}")
            raise RuntimeError(
                f"Job {job_id} {status}. "
                "Please check the IBM Quantum dashboard for details and resubmit if needed."
            )

        else:
            # Unknown status
            logger.warning(f"Unknown job status: {status}")
            raise RuntimeError(
                f"Job {job_id} has unknown status: {status}. "
                "Please check the IBM Quantum dashboard."
            )

    def _retrieve_counts_with_wait(
        self,
        job_id: str,
        workflow_path: Path,
        wait_for_completion: bool,
        max_wait_time: int,
        poll_interval: int,
    ) -> Dict[str, int]:
        """Retrieve counts with optional polling for job completion.

        Args:
            job_id: Job identifier
            workflow_path: Path for storing counts
            wait_for_completion: If True, poll until complete. If False, fail if not ready
            max_wait_time: Maximum time to wait (seconds)
            poll_interval: Time between status checks (seconds)

        Returns:
            Dictionary of measurement counts

        Raises:
            RuntimeError: If job not ready and wait_for_completion=False
            TimeoutError: If job doesn't complete within max_wait_time
        """
        counts_file = workflow_path / "counts.npy"

        # Check if counts already saved
        if counts_file.exists():
            logger.info(f"Loading existing counts from {counts_file}")
            counts = np.load(counts_file, allow_pickle=True).item()
            return counts

        # Check job status
        try:
            status = self.qpu_backend.get_job_status(job_id)
            logger.info(f"Job status: {status}")
        except Exception as e:
            logger.error(f"Failed to get job status: {e}")
            raise RuntimeError(
                f"Failed to retrieve job status for {job_id}. "
                "Please check your QPU connection and try again."
            ) from e

        # If job is complete, retrieve immediately
        if status in ["COMPLETED", "DONE"]:
            return self._retrieve_completed_job(job_id, workflow_path)

        # If job failed, raise error
        if status in ["CANCELLED", "ERROR", "FAILED"]:
            logger.error(f"Job {status}: {job_id}")
            raise RuntimeError(
                f"Job {job_id} {status}. "
                "Please check the IBM Quantum dashboard for details and resubmit if needed."
            )

        # Job is still in progress (QUEUED, VALIDATING, RUNNING)
        if not wait_for_completion:
            # Don't wait - inform user to try again later
            logger.warning(f"Job {job_id} is {status}. Not waiting for completion.")
            raise RuntimeError(
                f"Job {job_id} is still {status}.\n\n"
                f"QPU jobs can take up to 24 hours to complete.\n"
                f"To wait automatically, re-run solve() with wait_for_completion=True.\n"
                f"Once complete, simply re-run solve() - it will resume from the checkpoint."
            )

        # Wait for completion with polling
        logger.info(
            f"Waiting for job completion (max {max_wait_time}s, checking every {poll_interval}s)..."
        )
        start_time = time.time()

        while True:
            elapsed = time.time() - start_time

            if elapsed > max_wait_time:
                raise TimeoutError(
                    f"Job {job_id} did not complete within {max_wait_time} seconds.\n"
                    f"Current status: {status}\n"
                    f"The job is still running. You can:\n"
                    f"1. Re-run solve() with a larger max_wait_time\n"
                    f"2. Re-run solve() later (it will resume from checkpoint)"
                )

            # Wait before next check
            time.sleep(poll_interval)

            # Check status again
            try:
                status = self.qpu_backend.get_job_status(job_id)
                logger.info(f"[{int(elapsed)}s] Job status: {status}")
            except Exception as e:
                logger.warning(f"Failed to check status: {e}")
                continue

            # Check if complete
            if status in ["COMPLETED", "DONE"]:
                logger.info(f"✓ Job completed after {int(elapsed)} seconds")
                return self._retrieve_completed_job(job_id, workflow_path)

            # Check if failed
            if status in ["CANCELLED", "ERROR", "FAILED"]:
                raise RuntimeError(
                    f"Job {job_id} {status}. Please check the IBM Quantum dashboard for details and resubmit if needed."
                )

    def _retrieve_completed_job(self, job_id: str, workflow_path: Path) -> Dict[str, int]:
        """Retrieve and save results from a completed job.

        Args:
            job_id: Job identifier
            workflow_path: Path for storing counts

        Returns:
            Dictionary of measurement counts
        """
        try:
            result = self.qpu_backend.get_job_result(job_id)
            pub_result = result[0]
            counts = pub_result.data.meas.get_counts()

            # Save counts
            counts_file = workflow_path / "counts.npy"
            np.save(counts_file, counts)
            logger.info(f"Counts saved to {counts_file}")
            logger.info(f"Total shots: {sum(counts.values())}")

            return counts
        except Exception as e:
            logger.error(f"Failed to retrieve job results: {e}")
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
        """Classical post-processing with SBD solver.

        Args:
            h1e: One-body Hamiltonian
            h2e: Two-body Hamiltonian
            counts: Measurement counts from QPU
            norb: Number of orbitals
            nelec: Number of electrons
            workflow_path: Working directory

        Returns:
            SolverResult with final energy and state
        """
        from quantum_fragment_methods.application.solvers.quantum_zoo.utils.fermion_local import (
            diagonalize_fermionic_hamiltonian,
        )

        logger.info("Running SQD post-processing with custom HPC-integrated algorithm...")

        # Get SQD algorithm parameters
        iterations = self.sqd_config.get("iterations", 5)
        n_batches = self.sqd_config.get("n_batches", 10)
        samples_per_batch = self.sqd_config.get("samples_per_batch", 3000)
        energy_tol = self.sqd_config.get("energy_tol", 1.0e-8)
        occupancies_tol = self.sqd_config.get("occupancies_tol", 1.0e-5)
        carryover_threshold = self.sqd_config.get("carryover_threshold", 1.0e-4)
        symmetrize_spin = self.sqd_config.get("symmetrize_spin", True)

        logger.info(
            f"SQD parameters: iterations={iterations}, n_batches={n_batches}, "
            f"samples_per_batch={samples_per_batch}"
        )

        # Create workflow directory for SQD
        sqd_workflow_path = workflow_path / "sqd_diagonalizer"
        sqd_workflow_path.mkdir(parents=True, exist_ok=True)

        # Run the SQD algorithm with HPC integration
        result = diagonalize_fermionic_hamiltonian(
            h1e,
            h2e,
            counts,
            symmetrize_spin = symmetrize_spin,
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
        )

        # Convert to SolverResult
        solver_result = SolverResult(
            energy=result.energy,
            wavefunction=result.sci_state.amplitudes,
            rdm1=result.sci_state.rdm(rank=1),
            rdm2=result.sci_state.rdm(rank=2),
            metadata={
                "ci_strs_a": result.sci_state.ci_strs_a,
                "ci_strs_b": result.sci_state.ci_strs_b,
                "occupancies": result.occupancies,
                "norb": norb,
                "nelec": nelec,
            },
        )

        logger.info(f"SQD post-processing complete. Energy: {solver_result.energy:.8f}")
        return solver_result

    def solve_from_integrals(
        self, h1e: np.ndarray, h2e: np.ndarray, norb: int, nocc: int, **kwargs
    ) -> SolverResult:
        """
        Convenience method for workflow integration.

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

        Returns
        -------
        SolverResult
            SQD solution with energy and density matrices
        """
        from pyscf import gto, scf

        # Create a fake molecule for the fragment with proper dimensions
        mol = gto.M()
        mol.nelectron = 2 * nocc
        mol.spin = 0
        mol.nao = norb
        mol.verbose = 0

        # Create mean-field object with fragment Hamiltonian
        mf = scf.RHF(mol)
        mf.get_hcore = lambda *args: h1e
        mf.get_ovlp = lambda *args: np.eye(norb)  # Identity overlap for orthonormal basis
        mf._eri = h2e
        mf.kernel()

        # Now call solve with the mean-field object
        # Default to wait_for_completion=True for workflow integration
        # This enables "set it and forget it" behavior
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
        """Write Hamiltonian to FCIDUMP format.

        Args:
            h1e: One-body Hamiltonian
            h2e: Two-body Hamiltonian
            norb: Number of orbitals
            nelec: Tuple of (n_alpha, n_beta)
            fcidump_path: Output file path
        """
        try:
            from pyscf import tools
        except ImportError as e:
            raise ImportError("PySCF is required for FCIDUMP generation") from e

        n_alpha, n_beta = nelec
        total_elec = n_alpha + n_beta

        # Write FCIDUMP
        with open(fcidump_path, "w") as f:
            tools.fcidump.write_head(f, norb, total_elec, ms=n_alpha - n_beta)
            tools.fcidump.write_eri(f, h2e, norb, tol=1e-15)
            tools.fcidump.write_hcore(f, h1e, norb, tol=1e-15)

        logger.info(f"FCIDUMP written to {fcidump_path}")

    @property
    def name(self) -> str:
        """Return solver name."""
        return "SQD"
