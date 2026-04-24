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

"""Quantum Fragment Methods - Framework for embedded quantum simulations."""

from __future__ import annotations

from quantum_fragment_methods.workflow import QFWorkflow, WorkflowResult

__version__ = "0.0.1"

# Import application layer modules
from quantum_fragment_methods.application import embedding, solvers

__all__ = ["embedding", "solvers", "QFWorkflow", "WorkflowResult", "__version__"]
