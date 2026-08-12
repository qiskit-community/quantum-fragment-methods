"""
Sanity-check script: print trial run (CCSD/FCI) per-fragment energies and
compare NOONs for fragments that have both trial and production RDMs saved.

Run inside the container after step 3 has written solver_results.pkl:
  python check_trial_energies.py
"""
import os
import pickle
import numpy as np

DEMO = os.path.dirname(os.path.abspath(__file__))
TRIAL_PKL  = os.path.join(DEMO, "data/alanine_sto-3g_trial/solver_results.pkl")
PROD_RDM   = os.path.join(DEMO, "data/alanine_sto-3g/rdms")
TRIAL_RDM  = os.path.join(DEMO, "data/alanine_sto-3g_trial/rdms")

# ---------------------------------------------------------------------------
# Trial per-fragment energies
# ---------------------------------------------------------------------------
print("=" * 65)
print("Trial run per-fragment energies  (CCSD frags 0-5, FCI frags 6-12)")
print("=" * 65)
print(f"  {'Frag':>4}  {'solver':>6}  {'norb':>5}  {'e_total':>18}  {'e_corr':>14}")
print(f"  {'-'*4}  {'-'*6}  {'-'*5}  {'-'*18}  {'-'*14}")

if os.path.exists(TRIAL_PKL):
    with open(TRIAL_PKL, "rb") as f:
        trial_results = pickle.load(f)
    for fid, r in sorted(trial_results.items()):
        solver = "CCSD" if "e_corr" in r.metadata else "FCI"
        ec     = r.metadata.get("e_corr", float("nan"))
        norb   = r.metadata.get("norb", "?")
        print(f"  {fid:>4}  {solver:>6}  {norb:>5}  {r.energy:>18.8f}  {ec:>14.8f}")
else:
    print("  trial solver_results.pkl not found")

# ---------------------------------------------------------------------------
# NOON comparison: SQD-1M vs CCSD (trial) for fragments whose RDMs are saved
# ---------------------------------------------------------------------------
print()
print("=" * 65)
print("NOON comparison: SQD (1M shots) vs CCSD trial reference")
print("=" * 65)

any_found = False
for fid in range(13):
    sqd_path  = os.path.join(PROD_RDM,  f"fragment_{fid}_rdm1.npy")
    ref_path  = os.path.join(TRIAL_RDM, f"fragment_{fid}_rdm1.npy")
    if not os.path.exists(sqd_path):
        continue
    any_found = True
    sqd_rdm1 = np.load(sqd_path)
    sqd_noons = np.sort(np.linalg.eigvalsh(sqd_rdm1))[::-1]
    diag = np.diag(sqd_rdm1)
    n_int = int(np.sum((diag < 0.01) | (diag > 1.99)))
    print(f"\nFragment {fid} [SQD-1M]  norb={sqd_rdm1.shape[0]}  "
          f"trace={np.trace(sqd_rdm1):.6f}  integer_diag={n_int}/{len(diag)}")
    print(f"  NOONs: {np.array2string(sqd_noons, precision=4, suppress_small=True)}")

    if os.path.exists(ref_path):
        ref_rdm1  = np.load(ref_path)
        ref_noons = np.sort(np.linalg.eigvalsh(ref_rdm1))[::-1]
        print(f"Fragment {fid} [CCSD-ref] norb={ref_rdm1.shape[0]}  "
              f"trace={np.trace(ref_rdm1):.6f}")
        print(f"  NOONs: {np.array2string(ref_noons, precision=4, suppress_small=True)}")
        delta = sqd_noons - ref_noons
        print(f"  Δ NOON (SQD−CCSD): max_abs={np.abs(delta).max():.4f}  "
              f"rms={np.sqrt(np.mean(delta**2)):.4f}")

if not any_found:
    print("  No production RDMs found yet — step 3 still running.")
    print("  Re-run this script after step 3 completes.")
