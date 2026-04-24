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

# This code is a Qiskit project.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Modified fermion functions for SQD algorithm with HPC integration.

This module contains a customized version of diagonalize_fermionic_hamiltonian
that integrates with HPC workflows using subprocess-based SBD solver calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from pyscf import fci
from pyscf.fci.selected_ci import (
    _as_SCIvector,
    make_rdm1,
    make_rdm1s,
    make_rdm2,
    make_rdm2s,
    spin_square,
)

from qiskit_addon_sqd.configuration_recovery import recover_configurations
from qiskit_addon_sqd.counts import (
    counts_to_arrays,
    bitstring_matrix_to_integers,
)
from qiskit_addon_sqd.subsampling import postselect_by_hamming_right_and_left, subsample

import os
from pathlib import Path


@dataclass(frozen=True)
class SCIState:
    """The amplitudes and determinants describing a quantum state."""

    amplitudes: np.ndarray
    """An :math:`M \\times N` array where :math:`M =` len(``ci_strs_a``)
    and :math:`N` = len(``ci_strs_b``). ``amplitudes[i][j]`` is the
    amplitude of the determinant pair (``ci_strs_a[i]``, ``ci_strs_b[j]``).
    """

    ci_strs_a: np.ndarray
    """The alpha determinants."""

    ci_strs_b: np.ndarray
    """The beta determinants."""

    norb: int
    """The number of spatial orbitals."""

    nelec: tuple[int, int]
    """The numbers of alpha and beta electrons."""

    def __post_init__(self):
        """Validate dimensions of inputs."""
        object.__setattr__(
            self, "amplitudes", np.asarray(self.amplitudes)
        )  # Convert to ndarray if not already
        if self.amplitudes.shape != (len(self.ci_strs_a), len(self.ci_strs_b)):
            raise ValueError(
                f"'amplitudes' shape must be ({len(self.ci_strs_a)}, {len(self.ci_strs_b)}) "
                f"but got {self.amplitudes.shape}"
            )

    def rdm(self, rank: int = 1, spin_summed: bool = False) -> np.ndarray:
        """Compute reduced density matrix."""
        myci = fci.selected_ci.SelectedCI()
        ci_strs = (self.ci_strs_a, self.ci_strs_b)
        sci_vec = _as_SCIvector(self.amplitudes, ci_strs)

        if rank == 1:
            if spin_summed:
                return make_rdm1(sci_vec, self.norb, self.nelec)
            else:
                return make_rdm1s(sci_vec, self.norb, self.nelec)
        elif rank == 2:
            if spin_summed:
                return make_rdm2(sci_vec, self.norb, self.nelec)
            else:
                return make_rdm2s(sci_vec, self.norb, self.nelec)
        else:
            raise ValueError(f"Unsupported rank: {rank}")

    def spin_square(self) -> tuple[float, float]:
        """Compute spin square."""
        myci = fci.selected_ci.SelectedCI()
        ci_strs = (self.ci_strs_a, self.ci_strs_b)
        sci_vec = _as_SCIvector(self.amplitudes, ci_strs)
        return spin_square(sci_vec, self.norb, self.nelec)

    def orbital_occupancies(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute orbital occupancies."""
        dm1s = self.rdm(rank=1, spin_summed=False)
        return (np.diagonal(dm1s[0]), np.diagonal(dm1s[1]))


@dataclass(frozen=True)
class SCIResult:
    """Result from SCI calculation."""

    energy: float
    """The energy."""

    sci_state: SCIState
    """The SCI state."""

    occupancies: tuple[np.ndarray, np.ndarray]
    """The orbital occupancies."""


def _unique_with_order_preserved(arr: np.ndarray) -> np.ndarray:
    """Return unique elements preserving order."""
    _, indices = np.unique(arr, return_index=True)
    return arr[np.sort(indices)]


def _write_fcidump(
    h1e: np.ndarray,
    h2e: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    fcidump_path: Path,
) -> None:
    """Write FCIDUMP file for SBD solver."""
    from pyscf import tools

    # Create a fake molecule for FCIDUMP
    from pyscf import gto, scf

    mol = gto.M()
    mol.nelectron = sum(nelec)
    mol.spin = nelec[0] - nelec[1]

    # Create fake mean-field object
    mf = scf.RHF(mol)
    mf.mo_coeff = np.eye(norb)
    mf.mo_energy = np.zeros(norb)

    # Write FCIDUMP
    tools.fcidump.from_integrals(
        str(fcidump_path),
        h1e,
        h2e,
        norb,
        nelec,
        nuc=0.0,
    )


def diagonalize_fermionic_hamiltonian(
    one_body_tensor: np.ndarray,
    two_body_tensor: np.ndarray,
    counts: dict,
    samples_per_batch: int,
    norb: int,
    nelec: tuple[int, int],
    *,
    num_batches: int = 1,
    energy_tol: float = 1e-8,
    occupancies_tol: float = 1e-5,
    max_iterations: int = 100,
    symmetrize_spin: bool = False,
    max_dim: int | tuple[int, int] | None = None,
    include_configurations: list[int] | tuple[list[int], list[int]] | np.ndarray | None = None,
    initial_occupancies: tuple[np.ndarray, np.ndarray] | None = None,
    carryover_threshold: float = 1e-4,
    callback: Callable[[list[SCIResult]], None] | None = None,
    workflow_path: str = "",
    sbd_config: dict | None = None,
    seed: int | np.random.Generator | None = None,
) -> SCIResult:
    """
    Run the sample-based quantum diagonalization (SQD) algorithm.

    This is a modified version that uses SBDInterface for HPC integration.

    Args:
        one_body_tensor: The one-body tensor of the Hamiltonian.
        two_body_tensor: The two-body tensor of the Hamiltonian.
        counts: Dictionary of sampled bitstrings and their counts.
        samples_per_batch: The number of bitstrings to include in each subsampled batch.
        norb: The number of spatial orbitals.
        nelec: The numbers of alpha and beta electrons.
        num_batches: The number of batches to subsample in each iteration.
        energy_tol: Numerical tolerance for convergence of the energy.
        occupancies_tol: Numerical tolerance for convergence of occupancies.
        max_iterations: Limit on the number of configuration recovery iterations.
        symmetrize_spin: Whether to merge spin-alpha and spin-beta CI strings.
        max_dim: Limit on the dimension of the spin sectors.
        include_configurations: Configurations to always include.
        initial_occupancies: Initial guess for the average occupancies.
        carryover_threshold: Threshold for carrying over bitstrings.
        callback: A callback function called after each iteration.
        workflow_path: Path for storing intermediate files.
        seed: A seed for the pseudorandom number generator.

    Returns:
        The estimate of the energy and the SCI state.
    """
    if max_iterations < 1:
        raise ValueError("Maximum number of iterations must be at least 1.")

    n_alpha, n_beta = nelec
    if symmetrize_spin and n_alpha != n_beta:
        raise ValueError(
            "Spin symmetrization is only possible if the numbers of alpha and beta "
            f"electrons are equal. Instead, got {n_alpha} and {n_beta}."
        )

    if max_dim is None:
        max_dim_a = max_dim_b = None
    elif isinstance(max_dim, tuple):
        max_dim_a, max_dim_b = max_dim
    else:
        max_dim_a = max_dim_b = max_dim
    if symmetrize_spin and max_dim_a != max_dim_b:
        raise ValueError(
            "When requesting spin symmetrization, the maximum dimension must be "
            "the same for both spin alpha and spin beta. "
            f"Instead, got {max_dim_a} and {max_dim_b}"
        )

    if include_configurations is None:
        include_a: list[int] | np.ndarray = np.array([], dtype=int)
        include_b: list[int] | np.ndarray = np.array([], dtype=int)
    elif isinstance(include_configurations, tuple):
        include_a, include_b = include_configurations
    else:
        include_a = include_configurations
        include_b = include_configurations

    rng = np.random.default_rng(seed)
    current_occupancies = initial_occupancies
    best_result = None

    include_a = np.unique(include_a)
    include_b = np.unique(include_b)
    carryover_strings_a = np.array([], dtype=np.int64)
    carryover_strings_b = np.array([], dtype=np.int64)

    # Convert counts into bitstring matrix and probability array
    bs_mat, raw_probs = counts_to_arrays(counts)
    raw_bitstrings = bs_mat

    # Spin square hardcoded to zero for now
    spin_sq = 0

    # Run configuration recovery loop
    for iteration in range(max_iterations):
        if current_occupancies is None:
            # Postselect bitstrings with correct electron numbers
            bitstrings, probs = postselect_by_hamming_right_and_left(
                raw_bitstrings, raw_probs, hamming_right=n_alpha, hamming_left=n_beta
            )
            if not bitstrings.size:
                raise ValueError(
                    "The input counts did not contain any valid bitstrings. "
                    "Either pass counts that contain at least one valid bitstring "
                    "(with the correct right and left Hamming weights), or specify initial_occupancies."
                )
        else:
            # Use occupancy information to refine configurations
            bitstrings, probs = recover_configurations(
                raw_bitstrings, raw_probs, current_occupancies, n_alpha, n_beta, rand_seed=rng
            )

        # Subsample batches of bitstrings
        subsamples = subsample(
            bitstrings,
            probs,
            samples_per_batch=samples_per_batch,
            num_batches=num_batches,
            rand_seed=rng,
        )

        # Convert bitstrings to CI strings
        ci_strings = []
        for samples in subsamples:
            # Get the single-spin bitstrings and counts
            samples_a, counts_a = np.unique(
                bitstring_matrix_to_integers(samples[:, norb:]), return_counts=True
            )
            samples_b, counts_b = np.unique(
                bitstring_matrix_to_integers(samples[:, :norb]), return_counts=True
            )
            if symmetrize_spin:
                # Merge the bitstrings for spin alpha and spin beta
                samples = np.concatenate((samples_a, samples_b))
                counts = np.concatenate((counts_a, counts_b))
                samples = samples[np.argsort(counts)[::-1]]
                strs = np.concatenate((include_a, include_b, carryover_strings_a, samples))
                strs_a = strs_b = _unique_with_order_preserved(strs)[:max_dim_a]
            else:
                # Sort by marginal probability
                samples_a = samples_a[np.argsort(counts_a)[::-1]]
                samples_b = samples_b[np.argsort(counts_b)[::-1]]
                strs_a = np.concatenate((include_a, carryover_strings_a, samples_a))
                strs_b = np.concatenate((include_b, carryover_strings_b, samples_b))
                strs_a = _unique_with_order_preserved(strs_a)[:max_dim_a]
                strs_b = _unique_with_order_preserved(strs_b)[:max_dim_b]
            strs_a.sort()
            strs_b.sort()
            ci_strings.append((strs_a, strs_b))

        # Initialize SBD interface if config provided
        if sbd_config is None:
            raise ValueError("sbd_config must be provided for SBD solver integration")

        sbd_exe_path = sbd_config.get("exe_path")
        if not sbd_exe_path:
            raise ValueError("exe_path must be specified in sbd_config")

        from quantum_fragment_methods.application.solvers.quantum_zoo.utils.sbd_interface import (
            SBDInterface,
        )

        sbd_interface = SBDInterface(sbd_exe_path=sbd_exe_path, config=sbd_config)

        print(f"Iteration {iteration + 1}: Handling {num_batches} batches")
        print("####################")

        # Process batches using SBD interface
        results = []

        # Create FCIDUMP file from Hamiltonian
        fcidump_path = Path(workflow_path) / "fcidump.txt"
        _write_fcidump(one_body_tensor, two_body_tensor, norb, nelec, fcidump_path)

        for j, (ci_strs_a, ci_strs_b) in enumerate(ci_strings):
            print(f"Processing batch {j + 1}/{num_batches}...")

            # Create batch work directory
            batch_work_dir = Path(workflow_path) / f"batch_{j}"
            batch_work_dir.mkdir(parents=True, exist_ok=True)

            # Run SBD solver for this batch
            sbd_result = sbd_interface.run_sbd_solver(
                fcidump_path=str(fcidump_path),
                ci_strs_alpha=ci_strs_a.tolist(),
                norb=norb,
                nelec=nelec,
                work_dir=str(batch_work_dir),
                cpus_per_batch=sbd_config.get("cpus_per_batch", 4),
            )

            # Construct SCIState from SBD result
            sci_state = SCIState(
                amplitudes=sbd_result["amplitudes"],
                ci_strs_a=sbd_result["ci_strs_a"],
                ci_strs_b=sbd_result["ci_strs_b"],
                norb=norb,
                nelec=nelec,
            )

            # Create SCIResult
            results.append(
                SCIResult(
                    energy=sbd_result["energy"],
                    sci_state=sci_state,
                    occupancies=sbd_result["occupancies"],
                )
            )

        print("####################")
        print("Completed batches")
        print("####################")

        # Call callback if provided
        if callback is not None:
            callback(results)

        # Find best result
        best_result = min(results, key=lambda x: x.energy)

        # Check convergence
        if iteration > 0 and best_result is not None:
            energy_change = abs(best_result.energy - prev_energy)
            occ_change = max(
                np.max(np.abs(best_result.occupancies[0] - prev_occupancies[0])),
                np.max(np.abs(best_result.occupancies[1] - prev_occupancies[1])),
            )

            print(f"Energy change: {energy_change:.2e}, Occupancy change: {occ_change:.2e}")

            if energy_change < energy_tol and occ_change < occupancies_tol:
                print(f"Converged after {iteration + 1} iterations")
                break

        if best_result is None:
            raise RuntimeError("No valid results obtained from SBD solver")

        prev_energy = best_result.energy
        prev_occupancies = best_result.occupancies
        current_occupancies = best_result.occupancies

        # Update carryover strings
        carryover_strings_a = []
        carryover_strings_b = []
        for result in results:
            amplitudes = result.sci_state.amplitudes
            ci_strs_a = result.sci_state.ci_strs_a
            ci_strs_b = result.sci_state.ci_strs_b

            # Find configurations with large coefficients
            for i, str_a in enumerate(ci_strs_a):
                for j, str_b in enumerate(ci_strs_b):
                    if abs(amplitudes[i, j]) > carryover_threshold:
                        carryover_strings_a.append(str_a)
                        carryover_strings_b.append(str_b)

        carryover_strings_a = np.unique(carryover_strings_a)
        carryover_strings_b = np.unique(carryover_strings_b)

    if best_result is None:
        raise RuntimeError("SQD algorithm did not produce any valid results")

    return best_result
