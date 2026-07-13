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

"""Utility modules for quantum solvers."""

from __future__ import annotations

from quantum_fragment_methods.application.solvers.quantum_zoo.utils.lucj import (
    build_lucj_circuit,
    get_zigzag_physical_layout,
)
from quantum_fragment_methods.application.solvers.quantum_zoo.utils.sbd_interface import (
    SBDInterface,
)
from quantum_fragment_methods.application.solvers.quantum_zoo.utils.fermion_local import (
    SCIResult,
    SCIState,
    counts_to_bit_array,
    diagonalize_fermionic_hamiltonian,
    make_sbd_sci_solver,
)

__all__ = [
    "SBDInterface",
    "SCIResult",
    "SCIState",
    "build_lucj_circuit",
    "counts_to_bit_array",
    "diagonalize_fermionic_hamiltonian",
    "get_zigzag_physical_layout",
    "make_sbd_sci_solver",
]
