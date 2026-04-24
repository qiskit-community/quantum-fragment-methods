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
Application Layer

This module contains the scientific application logic including
embedding methods and solver implementations.
"""

from __future__ import annotations

# Re-export embedding module
from . import embedding, solvers

# Re-export commonly used classes for convenience
from .embedding import (
    DMET,
    EWF,
    MBE,
    BaseEmbedder,
    EmbeddingResult,
    Fragment,
)
from .solvers import (
    CCSD,
    DMRG,
    FCI,
    HCI,
    RHF,
    SQD,
    BaseSolver,
    SolverResult,
    SqDRIFT,
    extSQD,
)

__all__ = [
    # Modules
    "embedding",
    "solvers",
    # Embedding classes
    "BaseEmbedder",
    "EmbeddingResult",
    "Fragment",
    "DMET",
    "EWF",
    "MBE",
    # Solver classes
    "BaseSolver",
    "SolverResult",
    "RHF",
    "CCSD",
    "DMRG",
    "FCI",
    "HCI",
    "extSQD",
    "SQD",
    "SqDRIFT",
]
