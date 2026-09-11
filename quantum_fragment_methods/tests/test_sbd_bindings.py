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
Unit tests for the sbd-eigensolver-backed make_sbd_sci_solver.

These tests use unittest.mock to patch sbd internals so they run locally
without sbd-eigensolver installed.  An optional integration marker runs the
real binding when the package is available.
"""

from __future__ import annotations

import importlib
from functools import partial
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dummy_sbd_config(**extra):
    """Minimal sbd_config that matches new YAML schema."""
    cfg = {
        "device": "cpu",
        "mri_ranks": 1,
        "method": "davidson",
        "eps": 1e-8,
        "max_it": 50,
    }
    cfg.update(extra)
    return cfg


# ---------------------------------------------------------------------------
# Import guard: ImportError when sbd is absent
# ---------------------------------------------------------------------------

class TestImportGuard:
    """make_sbd_sci_solver raises a clear ImportError when sbd is not installed."""

    def test_raises_import_error_when_sbd_missing(self):
        import quantum_fragment_methods.application.solvers.quantum_zoo.sqd as fl

        # Temporarily pretend sbd didn't import
        original = fl._sbd_solve_sci_batch
        try:
            fl._sbd_solve_sci_batch = None
            with pytest.raises(ImportError, match="sbd-eigensolver"):
                fl.make_sbd_sci_solver({"device": "cpu"})
        finally:
            fl._sbd_solve_sci_batch = original


# ---------------------------------------------------------------------------
# make_sbd_sci_solver with mocked sbd
# ---------------------------------------------------------------------------

class TestMakeSbdSciSolverMocked:
    """make_sbd_sci_solver returns the right partial with mocked sbd."""

    @pytest.fixture(autouse=True)
    def _patch_sbd(self):
        """Inject mock solve_sci_batch and DeviceConfig into sqd module."""
        mock_solve = MagicMock(name="solve_sci_batch")
        mock_device_cls = MagicMock(name="DeviceConfig")
        mock_device_cls.return_value = MagicMock(name="DeviceConfig_instance")

        import quantum_fragment_methods.application.solvers.quantum_zoo.sqd as fl
        with (
            patch.object(fl, "_sbd_solve_sci_batch", mock_solve),
            patch.object(fl, "_SBDDeviceConfig", mock_device_cls),
        ):
            self._mock_solve = mock_solve
            self._mock_device_cls = mock_device_cls
            yield

    def test_returns_partial(self):
        from quantum_fragment_methods.application.solvers.quantum_zoo.sqd import (
            make_sbd_sci_solver,
        )
        solver = make_sbd_sci_solver(_make_dummy_sbd_config())
        assert isinstance(solver, partial)

    def test_partial_wraps_solve_sci_batch(self):
        from quantum_fragment_methods.application.solvers.quantum_zoo.sqd import (
            make_sbd_sci_solver,
        )
        solver = make_sbd_sci_solver(_make_dummy_sbd_config())
        assert solver.func is self._mock_solve

    def test_device_config_cpu(self):
        from quantum_fragment_methods.application.solvers.quantum_zoo.sqd import (
            make_sbd_sci_solver,
        )
        make_sbd_sci_solver(_make_dummy_sbd_config(device="cpu"))
        self._mock_device_cls.assert_called_once_with(device="cpu")

    def test_device_config_kwarg_default(self):
        """device kwarg default is 'cpu' when config has no 'device' key."""
        from quantum_fragment_methods.application.solvers.quantum_zoo.sqd import (
            make_sbd_sci_solver,
        )
        cfg = {k: v for k, v in _make_dummy_sbd_config().items() if k != "device"}
        make_sbd_sci_solver(cfg)
        self._mock_device_cls.assert_called_once_with(device="cpu")

    def test_device_config_kwarg_overridden_by_config_key(self):
        """sbd_config['device'] takes precedence over the device= kwarg."""
        from quantum_fragment_methods.application.solvers.quantum_zoo.sqd import (
            make_sbd_sci_solver,
        )
        make_sbd_sci_solver(_make_dummy_sbd_config(device="gpu"), device="cpu")
        self._mock_device_cls.assert_called_once_with(device="gpu")

    def test_non_tpb_keys_stripped(self):
        """exe_path, cpus_per_batch, device, mri_ranks must not reach solve_sci_batch."""
        from quantum_fragment_methods.application.solvers.quantum_zoo.sqd import (
            make_sbd_sci_solver,
        )
        cfg = _make_dummy_sbd_config()
        cfg["exe_path"] = "/opt/executable/diag"
        cfg["cpus_per_batch"] = 8
        solver = make_sbd_sci_solver(cfg)
        forwarded = solver.keywords["sbd_config"]
        assert "exe_path"      not in forwarded
        assert "cpus_per_batch" not in forwarded
        assert "device"        not in forwarded
        assert "mpi_ranks"     not in forwarded

    def test_tpb_keys_forwarded(self):
        """TPB_SBD knobs are passed through untouched."""
        from quantum_fragment_methods.application.solvers.quantum_zoo.sqd import (
            make_sbd_sci_solver,
        )
        cfg = _make_dummy_sbd_config(method="davidson", eps=1e-10, max_it=100)
        solver = make_sbd_sci_solver(cfg)
        forwarded = solver.keywords["sbd_config"]
        assert forwarded["method"] == "davidson"
        assert forwarded["eps"]    == 1e-10
        assert forwarded["max_it"] == 100

    def test_mpi_comm_none_by_default(self):
        from quantum_fragment_methods.application.solvers.quantum_zoo.sqd import (
            make_sbd_sci_solver,
        )
        solver = make_sbd_sci_solver(_make_dummy_sbd_config())
        assert solver.keywords["mpi_comm"] is None

    def test_mpi_comm_forwarded(self):
        from quantum_fragment_methods.application.solvers.quantum_zoo.sqd import (
            make_sbd_sci_solver,
        )
        fake_comm = MagicMock(name="MPI.COMM_WORLD")
        solver = make_sbd_sci_solver(_make_dummy_sbd_config(), mpi_comm=fake_comm)
        assert solver.keywords["mpi_comm"] is fake_comm

    def test_old_exe_path_config_does_not_raise(self):
        """Old-style config with exe_path is silently stripped — no ValueError."""
        from quantum_fragment_methods.application.solvers.quantum_zoo.sqd import (
            make_sbd_sci_solver,
        )
        old_cfg = {
            "exe_path": "/opt/executable/diag",
            "cpus_per_batch": 8,
            "method": "davidson",
        }
        # Should not raise (backward compat: keys stripped, not rejected)
        solver = make_sbd_sci_solver(old_cfg)
        assert isinstance(solver, partial)


# ---------------------------------------------------------------------------
# utils/__init__ no longer exports SBDInterface
# ---------------------------------------------------------------------------

class TestUtilsInit:
    def test_sbd_interface_not_in_all(self):
        from quantum_fragment_methods.application.solvers.quantum_zoo import utils
        assert "SBDInterface" not in utils.__all__

    def test_sbd_interface_not_importable(self):
        with pytest.raises((ImportError, AttributeError)):
            from quantum_fragment_methods.application.solvers.quantum_zoo.utils import SBDInterface  # noqa: F401

    def test_make_sbd_sci_solver_importable(self):
        from quantum_fragment_methods.application.solvers.quantum_zoo.utils import make_sbd_sci_solver
        assert callable(make_sbd_sci_solver)


# ---------------------------------------------------------------------------
# Optional: real sbd integration (skip if not installed)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    importlib.util.find_spec("sbd") is None,
    reason="sbd-eigensolver not installed",
)
class TestSbdIntegration:
    """Integration tests that run only when sbd-eigensolver is installed."""

    def test_available_backends_includes_cpu(self):
        import sbd
        backends = sbd.available_backends()
        assert "cpu" in backends, f"Expected 'cpu' in {backends}"

    def test_make_sbd_sci_solver_real(self):
        from quantum_fragment_methods.application.solvers.quantum_zoo.sqd import (
            make_sbd_sci_solver,
        )
        solver = make_sbd_sci_solver({"device": "cpu", "method": "davidson"})
        assert isinstance(solver, partial)
        assert callable(solver)
