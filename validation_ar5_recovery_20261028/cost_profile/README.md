# Frozen solver cost attribution

This diagnostic adds process-CPU timers to **scratch copies** of two solver
headers. It does not change production source, plugin binaries, equations,
Newton directions, tolerances, or acceptance gates. The unified diff and exact
replacement specification are included for review. Every transformed header
is checked against both its original and instrumented SHA256.

The two saved AR5 inner equations and frozen caps are reused from `../frozen`;
snapshots are not duplicated here. Three alternating repetitions per method
compare the original Newton direction with an extended reference budget and
the final v4 direction. Snapshot parsing and result serialization are outside
the timed solve. Both methods converge to the previously validated same roots;
all Newton/Krylov counts and accepted residuals match the archived uninstrumented
replays. Root comparisons themselves are in the adjacent frozen regressions.

`COST_BREAKDOWN.json` reports exclusive CPU categories and explicitly separates
nested timers. In particular, FGMRES total contains matrix products,
preconditioner triangular solves, and Krylov bookkeeping; it must not be added
to those child categories. Hydrodynamic setup similarly contains its own
factorization. Residual/trial evaluation includes the physical weighted
residual, contact response, sparse products, and metric evaluation.

This is a **fresh-workspace frozen inner solve**, not a complete physical step.
Production can reuse hydrodynamic factors across endpoint corrections, so its
cold setup cost cannot be assigned to every subsequent inner solve. These
profiling builds use `-O3`; absolute times should not be substituted for other
archived benchmarks with different build conditions. Concurrent workloads and
timer overhead can affect timings. No full-trajectory speedup is claimed.

The largest single measured category in v4 is repeated preconditioner solves:
roughly 39% (step 595) and 42% (step 683) of process CPU. FGMRES overall accounts
for approximately 53% and 62%; hydrodynamic plus contact factorization accounts
for approximately 21% and 22%. Therefore **factorization alone is not the dominant
cost** in these captured solves. Step 683 reduces residual/trial evaluations
from 344 to 81, but still uses 1112 Krylov iterations versus the reference's 1140.

Reproduce on Linux with the same source release, a C++17 compiler and the vendored
Eigen headers; no LAMMPS, MPI, FFTW or MKL is required:

```sh
python3 run_profile.py --output /tmp/slurry-cost-profile
```

The destination must be new. `--build-only` validates the instrumentation and
compilation without running the frozen problems. `step595.jsonl` and
`step683.jsonl` are raw measurements from the delivered diagnostic. The JSON
summary reports both ratios of mean category CPU to mean total CPU and medians
of the per-run percentages; these are distinct statistics.
