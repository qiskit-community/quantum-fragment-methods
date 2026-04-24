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
Pytest configuration and shared fixtures for quantum-fragment-methods tests.
"""

import sys
from pathlib import Path

import pytest

# Add package root to path for imports
package_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(package_root))


@pytest.fixture(scope="session")
def test_data_dir():
    """Return path to test data directory."""
    return Path(__file__).parent / "test_data"


@pytest.fixture(scope="session")
def water_xyz():
    """Return water molecule XYZ string."""
    return """3
Water molecule
O 0.0 0.0 0.0
H 0.757 0.586 0.0
H -0.757 0.586 0.0
"""


@pytest.fixture(scope="session")
def h2_xyz():
    """Return H2 molecule XYZ string (without comment line for direct use)."""
    return """H 0 0 0
H 0 0 0.74
"""


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "requires_vayesta: marks tests that require Vayesta library")
    config.addinivalue_line("markers", "requires_pyscf: marks tests that require PySCF library")
