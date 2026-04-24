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

"""Quantum Processing Unit (QPU) backend abstractions.

This module provides a unified interface for interacting with different
quantum hardware providers through abstract base classes and concrete
implementations.

Available backends:
- IBMQuantumBackend: IBM Quantum hardware via Qiskit Runtime
"""

from quantum_fragment_methods.qpu.base import QPUBackend
from quantum_fragment_methods.qpu.qiskit_ibm_runtime import IBMQuantumBackend

__all__ = ["QPUBackend", "IBMQuantumBackend"]
