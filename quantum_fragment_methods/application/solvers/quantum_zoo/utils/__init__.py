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

"""
Backwards-compatibility shim for ``quantum_zoo.utils``.

All public symbols previously defined in ``fermion_local.py`` and ``lucj.py``
now live in :mod:`quantum_fragment_methods.application.solvers.quantum_zoo.sqd`.
This module re-exports them so that any code (tests, notebooks, older scripts)
that still imports from ``quantum_zoo.utils`` continues to work without
modification.

Do not add new logic here — ``sqd.py`` is the single maintenance point.
"""

from quantum_fragment_methods.application.solvers.quantum_zoo.sqd import (  # noqa: F401
    SQDSolver,
    counts_to_bit_array,
    diagonalize_fermionic_hamiltonian,
    make_fulqrum_sci_solver,
    make_sbd_sci_solver,
)

__all__ = [
    "SQDSolver",
    "counts_to_bit_array",
    "diagonalize_fermionic_hamiltonian",
    "make_fulqrum_sci_solver",
    "make_sbd_sci_solver",
]
