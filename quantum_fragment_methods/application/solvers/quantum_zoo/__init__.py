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

"""Quantum solvers for fragment problems."""

from __future__ import annotations

from quantum_fragment_methods.application.solvers.quantum_zoo.ext_sqd import extSQD
from quantum_fragment_methods.application.solvers.quantum_zoo.sqd import SQDSolver
from quantum_fragment_methods.application.solvers.quantum_zoo.sqdrift import SqDRIFT

# Backward compatibility alias
SQD = SQDSolver

__all__ = ["extSQD", "SQD", "SQDSolver", "SqDRIFT"]
