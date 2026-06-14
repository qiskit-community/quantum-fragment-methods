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

#!/usr/bin/env python3
"""
Quantum and Classical Solvers

This module provides various solvers for quantum chemistry calculations,
including both classical and quantum methods.
"""

from __future__ import annotations

from .base import BaseSolver, SolverResult
from .classical_zoo.ccsd import CCSD
from .classical_zoo.fci import FCI
from .quantum_zoo.ext_sqd import extSQD
from .quantum_zoo.sqd import SQDSolver

# Backward compatibility alias
SQD = SQDSolver

__all__ = [
    "BaseSolver",
    "SolverResult",
    "RHF",
    "CCSD",
    "FCI",
    "extSQD",
    "SQD",
    "SQDSolver",
]
