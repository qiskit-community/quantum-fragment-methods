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

"""Interface for SBD (Selected Basis Diagonalization) solver.

This module provides utilities for interfacing with the SBD solver,
including conversion of CI strings to bitstrings, RDM extraction,
and energy/coefficient parsing from SBD output files.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class SBDInterface:
    """Wrapper for SBD solver interactions.

    This class provides methods to:
    - Convert CI strings to SBD-compatible bitstring format
    - Execute SBD solver with appropriate parameters
    - Extract RDMs, energies, and coefficients from SBD output
    """

    def __init__(self, sbd_exe_path: str, config: Optional[Dict[str, Any]] = None):
        """Initialize SBD interface.

        Args:
            sbd_exe_path: Path to SBD executable
            config: Configuration dictionary with SBD parameters
        """
        self.sbd_exe_path = sbd_exe_path
        self.config = config or {}
        logger.info(f"Initialized SBDInterface with executable: {sbd_exe_path}")

    @staticmethod
    def convert_ci_strs_to_bitstrings(ci_strs_alpha: List[int]) -> List[str]:
        """Convert a list of CI strings into a list of bitstrings.

        Args:
            ci_strs_alpha: List of CI string integers

        Returns:
            List of bitstring representations
        """
        bitstring_list = []
        for ci_str in ci_strs_alpha:
            bitstring_list.append(bin(ci_str)[2:])
        return bitstring_list

    @staticmethod
    def format_bitstrings(bitstring_list: List[str], norb: int) -> List[str]:
        """Add missing zeros corresponding to empty orbitals on left side of bitstrings.

        Args:
            bitstring_list: List of bitstrings
            norb: Number of orbitals

        Returns:
            List of formatted bitstrings with proper zero-padding
        """
        final_bitstring_list = []
        for bitstring in bitstring_list:
            final_bitstring_list.append(bitstring.zfill(norb))
        return final_bitstring_list

    @staticmethod
    def write_into_alphadets(batch_folder_path: str, final_bitstring_list: List[str]) -> None:
        """Write the bitstrings in the file format supported by SBD solver.

        Args:
            batch_folder_path: Path to batch folder
            final_bitstring_list: List of formatted bitstrings
        """
        alpha_det_file_path = os.path.join(batch_folder_path, "AlphaDets.txt")
        with open(alpha_det_file_path, "w") as alpha_det_file:
            for bitstring in final_bitstring_list:
                alpha_det_file.write(bitstring + "\n")
        logger.info(
            f"Wrote {len(final_bitstring_list)} alpha determinants to {alpha_det_file_path}"
        )

    @staticmethod
    def gen_dets(ci_strs_alpha: List[int], norb: int) -> List[str]:
        """Convert address to bitstrings and format bitstrings.

        Args:
            ci_strs_alpha: List of CI string integers
            norb: Number of orbitals

        Returns:
            List of formatted bitstrings
        """
        bitstring_list = SBDInterface.convert_ci_strs_to_bitstrings(ci_strs_alpha)
        final_bitstring_list = SBDInterface.format_bitstrings(bitstring_list, norb)
        return final_bitstring_list

    @staticmethod
    def get_rdm1_and_rdm2(batch_folder_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Open 1pRDM.txt and 2pRDM.txt of SBD and convert them into PySCF format.

        Args:
            batch_folder_path: Path to batch folder containing RDM files

        Returns:
            Tuple of (rdm1, rdm2) as numpy arrays
        """
        # First get the 1RDM from 1pRDM.txt
        rdm1_path = os.path.join(batch_folder_path, "1pRDM.txt")
        with open(rdm1_path) as f:
            f1 = f.readlines()
        f1 = [x.split() for x in f1]
        f1 = [[int(x[0]), int(x[1]), float(x[2])] for x in f1]
        norb = max([x[0] for x in f1]) + 1

        r1 = np.zeros((norb, norb))
        for i, j, D in f1:
            r1[i, j] = D

        # Second get the 2RDM from 2pRDM.txt
        rdm2_path = os.path.join(batch_folder_path, "2pRDM.txt")
        with open(rdm2_path) as f:
            f2 = f.readlines()
        f2 = [x.split() for x in f2]
        f2 = [[int(x[0]), int(x[1]), int(x[2]), int(x[3]), float(x[4])] for x in f2]

        r2 = np.zeros((norb, norb, norb, norb))
        for i, k, j, l, D in f2:
            r2[i, j, k, l] = D

        logger.info(
            f"Extracted RDMs from {batch_folder_path}: rdm1 shape={r1.shape}, rdm2 shape={r2.shape}"
        )
        return r1, r2

    @staticmethod
    def extract_sci_coeff(file_path: str) -> List[float]:
        """Extract SCI coefficients from SBD matrixformwf.txt file.

        Extracts the first column from a space-separated matrixformwf.txt.
        First column in matrixformwf.txt corresponds to SCI coefficients.

        Args:
            file_path: The path to the matrixformwf.txt

        Returns:
            List containing the values of the SCI coefficients produced in SBD
        """
        sci_coeff = []
        try:
            with open(file_path, "r") as f:
                for line in f:
                    # Remove leading/trailing whitespace and split the line by spaces
                    columns = line.strip().split()
                    if columns:  # Ensure the line is not empty
                        sci_coeff.append(float(columns[0]))
            logger.info(f"Extracted {len(sci_coeff)} SCI coefficients from {file_path}")
        except FileNotFoundError:
            logger.error(f"Error: The file '{file_path}' was not found.")
            raise
        except Exception as e:
            logger.error(f"An error occurred while extracting SCI coefficients: {e}")
            raise
        return sci_coeff

    @staticmethod
    def extract_energy(filepath: str) -> Optional[float]:
        """Extract SCI energy (without nuclear part) from SBD log file.

        Finds the line "One-Body + Two-Body energy" in a SBD log file
        and extracts the float number appearing on this line.

        Args:
            filepath: The path to the SBD log file

        Returns:
            The extracted float number if found, otherwise None
        """
        try:
            with open(filepath, "r") as f:
                for line in f:
                    if "One-Body + Two-Body energy" in line:
                        # Use a regular expression to find a float number in the line
                        match = re.search(r"[-+]?\d+\.\d+(?:[eE][-+]?\d+)?", line)
                        if match:
                            energy = float(match.group(0))
                            logger.info(f"Extracted SCI energy: {energy:.8f} from {filepath}")
                            return energy
            logger.warning(f"Energy line not found in {filepath}")
            return None
        except FileNotFoundError:
            logger.error(f"Error: File not found at {filepath}")
            return None
        except Exception as e:
            logger.error(f"An error occurred while extracting energy: {e}")
            return None

    def run_sbd_solver(
        self,
        fcidump_path: str,
        ci_strs_alpha: List[int],
        norb: int,
        nelec: Tuple[int, int],
        work_dir: str,
        cpus_per_batch: int = 4,
        **kwargs,
    ) -> Dict[str, Any]:
        """Run SBD solver on a batch of CI strings.

        Args:
            fcidump_path: Path to FCIDUMP file
            ci_strs_alpha: List of alpha CI strings
            norb: Number of orbitals
            nelec: Tuple of (n_alpha, n_beta) electrons
            work_dir: Working directory for SBD execution
            cpus_per_batch: Number of CPUs to use
            **kwargs: Additional SBD options

        Returns:
            Dictionary containing:
                - energy: SCI energy
                - rdm1: One-particle reduced density matrix
                - rdm2: Two-particle reduced density matrix
                - amplitudes: SCI coefficients
                - ci_strs_a: Alpha CI strings
                - ci_strs_b: Beta CI strings
        """
        # Create working directory
        os.makedirs(work_dir, exist_ok=True)
        logger.info(f"Running SBD solver in {work_dir}")

        # Generate and write alpha determinants
        alpha_det = self.gen_dets(ci_strs_alpha, norb)
        self.write_into_alphadets(work_dir, alpha_det)

        # Setup SBD command
        omp_threads = int(cpus_per_batch / 2)
        fci_dump_path = os.path.abspath(fcidump_path)
        adet_file_path = os.path.join(os.path.abspath(work_dir), "AlphaDets.txt")

        # User-defined options
        sbd_user_options = (
            f"mpirun --allow-run-as-root -np {cpus_per_batch} "
            f"-x OMP_NUM_THREADS={omp_threads} {self.sbd_exe_path} "
            f"--fcidump {fci_dump_path} --adetfile {adet_file_path}"
        )

        # Additional options (can be customized via config)
        sbd_other_options = (
            " --method 0 --block 10 --iteration 4 --tolerance 1.0e-4 "
            "--adet_comm_size 2 --bdet_comm_size 2 --task_comm_size 2 "
            "--init 0 --shuffle 0 --carryover_ratio 0.5 --rdm 1 "
            "--dump_matrix_form_wf matrixformwf.txt"
        )

        # Merge options from config if provided
        if "sbd_options" in self.config:
            sbd_other_options = " " + self.config["sbd_options"]

        sbd_call = sbd_user_options + sbd_other_options
        sbd_log_path = os.path.join(work_dir, "sbd_solver_logfile.log")

        # Run SBD solver
        logger.info(f"Executing SBD command: {sbd_call}")
        with open(sbd_log_path, "w") as logfile:
            process = subprocess.run(
                sbd_call.split(),
                env=os.environ,
                cwd=work_dir,
                stdout=logfile,
                stderr=logfile,
            )

        if process.returncode != 0:
            logger.warning(f"SBD solver returned non-zero exit code: {process.returncode}")

        # Extract results
        rdm1, rdm2 = self.get_rdm1_and_rdm2(work_dir)
        energy_sci = self.extract_energy(sbd_log_path)

        # Extract SCI coefficients
        matrixformwf_path = os.path.join(work_dir, "matrixformwf.txt")
        sci_coeff_raw = self.extract_sci_coeff(matrixformwf_path)
        sci_coeff_np = np.array(sci_coeff_raw)
        sci_coeff_formatted = sci_coeff_np.reshape(len(ci_strs_alpha), len(ci_strs_alpha))

        # Calculate occupancies
        avg_occupancy_per_MO = np.diagonal(rdm1)
        avg_occupancy_per_spin_orb = avg_occupancy_per_MO / 2
        avg_occs = (avg_occupancy_per_spin_orb, avg_occupancy_per_spin_orb)

        result = {
            "energy": energy_sci,
            "rdm1": rdm1,
            "rdm2": rdm2,
            "amplitudes": sci_coeff_formatted,
            "ci_strs_a": ci_strs_alpha,
            "ci_strs_b": ci_strs_alpha,  # SBD currently uses same for both spins
            "occupancies": avg_occs,
            "norb": norb,
            "nelec": nelec,
        }

        logger.info(f"SBD solver completed. Energy: {energy_sci:.8f}")
        return result
