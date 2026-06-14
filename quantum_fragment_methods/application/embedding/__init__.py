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
Embedding Methods

This module provides various quantum embedding methods for fragment-based
quantum chemistry calculations.
"""

from __future__ import annotations

from .base import BaseEmbedder, EmbeddingResult, Fragment
from .ewf import EWF

__all__ = [
    "BaseEmbedder",
    "EmbeddingResult",
    "Fragment",
    "EWF"
]
