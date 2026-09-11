g for# Fragment Solving Parallelization Plan

**Status:** Planned — not yet implemented  
**Context:** `QFWorkflow.solve_fragments()` is currently serial. Fragment solving is embarrassingly parallel — each fragment's Hamiltonian is fully independent once the EWF bath is built.

---

## Current Architecture

`QFWorkflow.solve_fragments()` (`workflow.py:243`) loops serially over fragments. Each fragment calls `solver.solve_from_integrals(h1e, h2e, norb, nocc, workflow_path=...)` with zero shared state between fragments after the EWF kernel completes.

The per-fragment checkpoint directories (`save_path/fragment_{id}/`) introduced in the glycine demo make this safe for parallel execution — no shared file state.

---

## Option 1 — `concurrent.futures` (laptop, low effort)

Replace the serial loop with a `ProcessPoolExecutor`:

**Pattern:**
1. Extract `(fragment_id, h1e, h2e, norb, nocc, solver_type, workflow_path)` tuples before the pool
2. Separate QPU fragments (SQD) from classical fragments (FCI/CCSD) — QPU fragments run serially (they are already async via checkpoint)
3. `pool.map(solve_one, classical_tuples)` for FCI/CCSD fragments
4. Collect results, merge with QPU results

**Why processes not threads:** FCI is CPU-bound; Python GIL prevents true thread parallelism for numpy-heavy solvers.

**Caveat:** Vayesta fragment objects may not be picklable — must extract `(h1e, h2e, norb, nocc)` numpy arrays before forking (already done in `solve_fragments`).

**Implementation point:** `QFWorkflow.solve_fragments(parallel="process", max_workers=None)`

---

## Option 2 — MPI via `mpi4py` (HPC production, medium effort)

Natural fit for SCC. Each MPI rank handles a subset of fragments:

```
Rank 0: fragment 0 (FCI), fragment 5 (FCI)
Rank 1: fragment 1 (FCI), fragment 6 (FCI)
Rank 2: fragment 2 (SQD) ← submits to QPU, polls Kingston
Rank 3: fragment 3 (FCI), fragment 7 (FCI)
Rank 4: fragment 4 (FCI), fragment 8 (FCI), fragment 9 (FCI)
```

**Key points:**
- `QFWorkflow.solve_fragments_mpi(comm)` — rank picks `fragment_id % comm.size == comm.rank`
- Results gathered with `comm.gather()` on rank 0
- QPU rank runs the full QRMI submit/poll/checkpoint loop
- `make_sbd_sci_solver` already supports `mpi_comm` passthrough (`sqd.py:239`)
- `QRMI_JOB_QPU_RESOURCES` env var is available on all ranks (set by Slurm SPANK plugin)

**Slurm job script:**
```bash
#SBATCH --ntasks=5          # one per heavy-atom fragment
#SBATCH --gres=qpu:ibm_kingston:1
mpirun -n 5 python run_workflow.py --config config_glycine_sto-3g.yaml
```

**Implementation point:** Add `solve_fragments_mpi(comm)` method to `QFWorkflow`

---

## Option 3 — Dask / Ray (adaptive, high effort)

Scatter fragment tasks as futures, collect results. Overkill for 10 fragments (glycine) but natural for large molecules:
- Alanine: ~15 fragments
- Proteasome complex: 50+ fragments

Dask has a `SLURMCluster` scheduler that maps cleanly to SCC job arrays.

---

## Recommended Roadmap

| Scale | Approach | Effort | Priority |
|-------|----------|--------|----------|
| Laptop demo (glycine) | `ProcessPoolExecutor` in `solve_fragments` | Low | Next |
| HPC production (alanine, proteasome) | `solve_fragments_mpi(comm)` + Slurm | Medium | After laptop |
| Large-scale (100+ fragments) | Dask `SLURMCluster` | High | Future |

---

## Implementation Entry Point

**File:** `quantum_fragment_methods/workflow.py`  
**Method:** `QFWorkflow.solve_fragments()` (line ~243)

Add a `parallel` kwarg that selects execution strategy, keeping the serial path as default for backward compatibility:

```python
def solve_fragments(self, parallel=None, max_workers=None):
    """
    parallel: None (serial), "process" (ProcessPoolExecutor), "mpi" (mpi4py)
    """
```

**QPU fragments always run serially** — they involve long polling loops and are already async via the job_id.txt / counts.npy checkpoint system. Only classical (FCI/CCSD) fragments are parallelized.

---

## Files to Modify

- `quantum_fragment_methods/workflow.py` — add parallel execution strategies
- `quantum_fragment_methods/application/solvers/quantum_zoo/sqd.py` — no changes needed (checkpoint system already safe)
- `examples/hpc_demos/*/` — update Slurm scripts to pass `--ntasks=N`
