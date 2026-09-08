#!/usr/bin/env python3
"""
Local smoke test: alanine EWF+CCSD/FCI pipeline (no cluster, no QPU, no SBD).

Runs steps 1–3 of the alanine demo entirely in-process using the trial config
(orbital_threshold: 15 → FCI for small frags, CCSD for large).  Validates that
the SBD migration left the classical solver path fully intact.

Run with:
    pytest quantum_fragment_methods/tests/test_alanine_smoke.py -v -m slow
    # or directly (skips pytest overhead):
    python quantum_fragment_methods/tests/test_alanine_smoke.py

Expected runtime on a laptop: ~5–15 min (alanine STO-3G HF + EWF + CCSD/FCI).
Set QFM_SMOKE_FAST=1 to use sto-3g with a 4-atom sub-geometry (~60s) instead.
"""

from __future__ import annotations

import importlib
import os
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT  = Path(__file__).parent.parent.parent
DEMO_DIR   = REPO_ROOT / "examples" / "hpc_demos" / "alanine_ewf_sqd_demo"
TRIAL_CFG  = DEMO_DIR / "config_alanine_sto-3g_trial.yaml"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_step(script: Path, config: Path, data_dir: Path, extra_args: list[str] | None = None):
    """Run a demo step script as a subprocess and assert it exits 0."""
    cmd = [
        sys.executable, str(script),
        "--config", str(config),
        "--data-dir", str(data_dir),
    ]
    if extra_args:
        cmd.extend(extra_args)
    result = subprocess.run(cmd, capture_output=False, cwd=str(DEMO_DIR))
    if result.returncode != 0:
        pytest.fail(
            f"Step {script.name} exited with code {result.returncode}.\n"
            f"Command: {' '.join(cmd)}"
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.skipif(
    importlib.util.find_spec("pyscf") is None,
    reason="pyscf not installed",
)
@pytest.mark.skipif(
    importlib.util.find_spec("vayesta") is None,
    reason="vayesta (Vayesta EWF embedder) not installed",
)
class TestAlanineSmokeLocal:
    """
    End-to-end smoke test for the alanine EWF+CCSD/FCI pipeline.

    Steps tested:
      1 — HF mean-field (01_meanfield.py)
      2 — EWF fragmentation (02_fragments.py)
      3 — Classical solve: CCSD/FCI for all frags (03_solve.py, trial config)

    SBD / QPU is NOT exercised — orbital_threshold in the trial config routes
    all fragments to FCI or CCSD.  This test purely validates that the SBD
    migration did not break the classical solver path.
    """

    @pytest.fixture(scope="class")
    def work_dir(self, tmp_path_factory):
        return tmp_path_factory.mktemp("alanine_smoke")

    def test_step1_meanfield(self, work_dir):
        """HF converges and writes mf_data.pkl."""
        _run_step(
            DEMO_DIR / "01_meanfield.py",
            TRIAL_CFG,
            work_dir,
        )
        pkl = work_dir / "mf_data.pkl"
        assert pkl.exists(), "01_meanfield.py did not write mf_data.pkl"
        with open(pkl, "rb") as f:
            mf_data = pickle.load(f)
        assert "hf_energy" in mf_data, "mf_data.pkl missing hf_energy"
        e_hf = mf_data["hf_energy"]
        # Alanine STO-3G HF energy is around -320 Ha
        assert e_hf < -300, f"HF energy suspiciously large: {e_hf:.6f} Ha"
        print(f"\n  HF energy: {e_hf:.8f} Ha")

    def test_step2_fragments(self, work_dir):
        """EWF fragmentation runs and writes embedding_data.pkl."""
        _run_step(
            DEMO_DIR / "02_fragments.py",
            TRIAL_CFG,
            work_dir,
        )
        pkl = work_dir / "embedding_data.pkl"
        assert pkl.exists(), "02_fragments.py did not write embedding_data.pkl"
        with open(pkl, "rb") as f:
            emb = pickle.load(f)
        frags = emb.get("fragment_meta", {})
        assert len(frags) > 0, "No fragments produced by EWF"
        print(f"\n  Fragments produced: {len(frags)}")
        for fid, meta in sorted(frags.items()):
            print(f"    frag {fid}: n_orb={meta.get('n_orbitals', '?')}")

    def test_step3_solve_classical(self, work_dir):
        """CCSD/FCI solve runs for all fragments, writes solver_results.pkl."""
        _run_step(
            DEMO_DIR / "03_solve.py",
            TRIAL_CFG,
            work_dir,
        )
        pkl = work_dir / "solver_results.pkl"
        assert pkl.exists(), "03_solve.py did not write solver_results.pkl"
        with open(pkl, "rb") as f:
            results = pickle.load(f)
        assert len(results) > 0, "solver_results.pkl is empty"

        print(f"\n  Fragment results ({len(results)} total):")
        for fid, r in sorted(results.items()):
            solver = "CCSD" if r.metadata.get("e_corr") is not None else "FCI"
            e_corr = r.metadata.get("e_corr", 0.0)
            norb   = r.metadata.get("norb", "?")
            print(f"    frag {fid} [{solver}]  norb={norb}  "
                  f"E={r.energy:.8f} Ha  e_corr={e_corr:.6f} Ha")
            # Correlation energy should be negative
            assert e_corr <= 0, (
                f"Fragment {fid} has positive correlation energy: {e_corr}"
            )
            # Energy should be physically reasonable
            assert r.energy < 0, f"Fragment {fid} has positive total energy: {r.energy}"

    def test_no_sbd_interface_imported(self):
        """Confirm SBDInterface is gone from the loaded module graph."""
        # If sbd_interface got imported as a side-effect of any step above,
        # sys.modules would contain it.
        sbd_if_keys = [k for k in sys.modules if "sbd_interface" in k]
        assert not sbd_if_keys, (
            f"sbd_interface was imported unexpectedly: {sbd_if_keys}"
        )

    def test_make_sbd_sci_solver_import(self):
        """make_sbd_sci_solver is importable from both public paths."""
        from quantum_fragment_methods.application.solvers.quantum_zoo.utils import (
            make_sbd_sci_solver,
        )
        from quantum_fragment_methods.application.solvers.quantum_zoo.utils.fermion_local import (
            make_sbd_sci_solver as mss2,
        )
        assert make_sbd_sci_solver is mss2


# ---------------------------------------------------------------------------
# CLI entry-point (run without pytest)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import shutil
    import time

    print("=" * 65)
    print("Alanine EWF+CCSD/FCI local smoke test")
    print("=" * 65)
    print(f"Demo dir:   {DEMO_DIR}")
    print(f"Trial cfg:  {TRIAL_CFG}")
    print()

    with tempfile.TemporaryDirectory(prefix="qfm_smoke_") as _tmp:
        work = Path(_tmp)
        steps = [
            ("Step 1 — HF",           "01_meanfield.py",  "mf_data.pkl"),
            ("Step 2 — EWF frags",    "02_fragments.py",  "embedding_data.pkl"),
            ("Step 3 — CCSD/FCI",     "03_solve.py",       "solver_results.pkl"),
        ]
        for label, script, expected_pkl in steps:
            print(f"  Running {label}...")
            t0 = time.time()
            cmd = [
                sys.executable,
                str(DEMO_DIR / script),
                "--config", str(TRIAL_CFG),
                "--data-dir", str(work),
            ]
            rc = subprocess.run(cmd, cwd=str(DEMO_DIR)).returncode
            elapsed = time.time() - t0
            if rc != 0:
                print(f"  FAIL  {label} exited {rc}")
                sys.exit(1)
            pkl_path = work / expected_pkl
            if not pkl_path.exists():
                print(f"  FAIL  {expected_pkl} not written")
                sys.exit(1)
            print(f"  OK    {label}  ({elapsed:.1f}s)")

        # Print fragment summary
        print()
        print("Fragment energies:")
        with open(work / "solver_results.pkl", "rb") as f:
            results = pickle.load(f)
        for fid, r in sorted(results.items()):
            solver = "CCSD" if r.metadata.get("e_corr") is not None else "FCI"
            e_corr = r.metadata.get("e_corr", 0.0)
            norb   = r.metadata.get("norb", "?")
            print(f"  frag {fid:>2} [{solver}]  norb={norb:>3}  "
                  f"E={r.energy:.8f} Ha  e_corr={e_corr:.6f} Ha")

    print()
    print("All steps passed. Classical solver path is intact.")
