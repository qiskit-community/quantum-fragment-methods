"""Diagnostic: compare RDM conventions between SQD and CCSD solver results."""
import pickle
import numpy as np

sqd  = pickle.load(open("examples/hpc_demos/alanine_ewf_sqd_demo/data/alanine_sto-3g/solver_results.pkl", "rb"))
ccsd = pickle.load(open("examples/hpc_demos/alanine_ewf_sqd_demo/data/alanine_sto-3g_trial/solver_results.pkl", "rb"))

for fid in [0, 3]:
    print(f"\n=== Fragment {fid} ===")
    for label, res in [("SQD", sqd[fid]), ("CCSD", ccsd[fid])]:
        r1 = res.rdm1
        print(f"  {label} rdm1: shape={r1.shape}  trace={np.trace(r1):.6f}  "
              f"min={r1.min():.4f}  max={r1.max():.4f}")
        print(f"        diag[:6]={np.diag(r1)[:6].round(6)}")
        if res.rdm2 is not None:
            r2 = res.rdm2
            print(f"  {label} rdm2: shape={r2.shape}")
        else:
            print(f"  {label} rdm2: None")
