# AR5, volume fraction 0.20: solver recovery investigation

## Reproducing the reported failure

The supplied input uses 1,000 bidisperse AR5 spheroids at volume fraction 0.20,
normal stiffness 300 N/m, sliding and rolling stiffness 240 N/m, both friction
coefficients 10, solvent viscosity 0.01 Pa s, shear rate 100/s and timestep
2.5e-6 s. The initial coordinates are taken directly from the supplied
`particles.data`; they are not repacked for this investigation.

All 435 source-file hashes in the supplied build identity match the source
tree used for reproduction. The reported run used release
`20261028-bidisperse-imex`.

The reported failure occurs at step 723, strain 0.18075. Its inner frozen-cap
convex solve reaches the 40-update Newton limit with residual 3.22117 versus
tolerance 0.00152347. This residual is a resistance-weighted solver norm; it
is not a stress in Pa or a force in N. The user-facing `iterations` option
sets the linear Krylov limit and does not change this Newton limit.

At the last supplied history row, step 400, the relative viscosity is 4.512,
the relative residual is 1.319e-8, and maximum endpoint overlap divided by
the smaller particle's minimum semiaxis is 3.381e-5, or **0.003381%**.
These are measurements at step 400, not at the failed step. The available
record does not show large overlap or a viscosity divergence before failure.

The reported release had a checked direct fallback for a failed linear Newton
direction, but no alternate nonlinear strategy when its 40 updates expired.
The endpoint wrapper enriches `NonlinearFailure` diagnostics and rethrows;
that catch is not a nonlinear recovery mechanism. Its IMEX limit threw a plain
runtime error, which also omitted the endpoint counters.

A controlled diagnostic holds the failed hydrodynamic state, normal springs,
timestep, frozen normal loads, initial velocity and absolute residual
threshold fixed, then removes one friction channel at a time from the
diagnostic equation. The old solver requires 41/46 updates on the two full
AR5 states (steps 595/683), 25/34 with sliding removed, 34/25 with rolling
removed, and only 2/2 with both removed. Both friction channels contribute to
repeated damping; neither is the sole cause in every state. The large normal
stiffness by itself is not causing the observed 40-update failure on these
captured states. These altered equations are causal diagnostics only; the
production model retains both friction channels and their original values.

## Cost in the supplied run

Step 400 spends 1.365 s of its 1.416 s measured total in the solver (96.4%).
The nearfield sparse factorization and associated preconditioner solves use
Eigen's serial `SimplicialLDLT`; requesting 17 CPU cores does not parallelize
this path. Factorization alone is not the dominant measured cost.

Scratch instrumentation of the same frozen AR5 equations gives the following
v4 process-CPU breakdown. Each percentage is the median of three runs.

| Frozen state | FGMRES total | Preconditioner triangular solves, included in FGMRES | Hydro + contact factorization | Residual/trial evaluations |
| --- | ---: | ---: | ---: | ---: |
| Step 595 | 53.14% | 38.62% | 21.27% | 13.10% |
| Step 683 | 61.98% | 41.83% | 22.11% | 7.03% |

On step 683, Newton updates fall from 46 to 24 and trial evaluations from
344 to 81, while Krylov iterations remain 1112 versus 1140. The nonlinear
repair therefore removes much repeated line-search work but leaves expensive
linear work. These are fresh-workspace frozen inner solves, including cold
hydrodynamic setup; that setup is not charged anew to every production
endpoint correction. Instrumentation leaves all archived iteration counts
and accepted residuals unchanged. Portable profiling sources and the full
breakdown are in `validation_ar5_recovery_20261028/cost_profile`.

Wall times from concurrent local runs must not be used as an uncontended
speed benchmark, and this frozen-state breakdown is not a full-trajectory
timing claim.

## Solver change

The candidate retains the same convex objective and its exact force residual.
For a yielded sliding or rolling spring, the radial Huber potential has no
radial curvature in its exact Hessian. A Newton direction can therefore travel
too far across a stick/yield transition and need strong line-search damping.

After an exact Newton direction is accepted with step length at most 0.5, the
next direction uses the Huber tangent quadratic, with curvature
`min(k, cap / norm(trial_history))`. This adds positive radial curvature to
yielded sliding, rolling and twisting terms. It is a temporary change to the
direction matrix, not to the physical stiffness or constitutive law. The
following direction uses the exact Hessian again. Normal contact keeps its
exact current derivative; the original full-objective Armijo test checks any
normal-contact activation along the proposed direction.

The frozen loads, integration timestep, force/mobility residual definition and
final physical acceptance thresholds remain unchanged.
The trace phase `nearfield_imex_huber_majorized` identifies the alternate
direction. `Workspace::huber_majorized_directions` counts its use. The
developer-only `use_huber_globalization=false` switch permits old-solver
direction comparisons on an identical frozen equation.

The primary budget remains 40 updates. If an inner convex solve exhausts it,
the solver continues from its current iterate with a forced majorized
direction. The delivered LAMMPS wrapper allows **1000 total nonlinear updates
per inner solve** by default: 40 primary updates plus at most 960 recovery
updates. Set the environment variable `SLURRY_IMEX_MAX_ITERATIONS` to change
that total ceiling. Integers of at least 40 are accepted; a value of 40
disables the extra recovery allowance. Convergence is checked throughout;
increasing the ceiling does not make an already converged solve perform
additional iterations.

This total is separate from the existing batch setting
`KRYLOV_ITERATIONS=1000`, which controls the linear solve for each Newton
direction through the `iterations` input option. Changing that batch setting
alone never changed the old 40-update nonlinear cap. The existing outer
limit of 24 endpoint corrections and the 30-trial line-search safeguard
remain unchanged. The nearfield linear fast path uses at most
`min(KRYLOV_ITERATIONS, 160)` Krylov iterations, then has a checked direct
fallback. That internal 160-iteration fast-path limit is not a fatal
nonlinear cap.
The nonlinear ceiling applies to each inner solve, not to a shared budget
over all endpoint corrections.

The underlying C++ `Workspace::maximum_recovery_iterations` default remains
80 for controlled library tests; the LAMMPS wrapper overrides it to the
configured total minus 40. The archived v4 integration and frozen tests below
used that 40+80 budget. They are not reruns of the final wrapper's 40+960
configuration. Both configurations retain the same direction algorithm,
physical equation and convergence gates; the larger ceiling changes what
happens only if the shorter budget would be exhausted. Setting the
developer-only recovery budget to zero restores the original hard primary
limit for controlled comparisons.

No unconverged intermediate iterate is committed. Failure at the configured
total limit still stops the run with its residual and budgets.

Recovery trace phases are `nearfield_imex_recovery` and
`nearfield_imex_recovery_majorized`. A recovered physical step also prints
`slurry/bridge IMEX recovery accepted` without requiring trace mode, including
the step, number of recovered inner solves, additional updates and accepted
physical relative residual. Ordinary steps do not print this notice.

Unit validation includes a two-particle problem with known exact sliding,
rolling and twisting roots, a deliberately saturated initial estimate, and
assertions that the alternate path was actually exercised. Separate tests
check positive added curvature and the Huber quadratic upper bound on both
sides of each yield boundary. Production-trajectory results are required in
addition to these small tests. Forced one-update primary-budget tests verify
the recovery root and acceptance, disabled recovery, exhaustion of the total
budget, and invalid negative recovery budgets.

## Validation status

The user requested that long tests stop and the current overlay be packaged.
The release status is `PASS_BOUNDED_VALIDATION_USER_STOPPED`: the captured
nonlinear problems and the two completed N=256 integration tests pass, and
the supplied N=1000 case has an accepted trajectory through step 1974,
strain 0.4935. The N=1000 run did not complete its planned 4000 steps or
strain 1.
The previous release's N=1000, AR5, volume-fraction-0.24 integration test
covered only 200 steps, strain 0.05; it did not reach the strain of this
reported failure.

The initial N=1000 v4 verification was interrupted when the execution
environment stopped. Its last recorded accepted step is 1243, strain
0.31075, with no solver error reported before interruption. This is a partial
trajectory, not a completed 4000-step run. The dump does not preserve the
private sliding and rolling contact histories, so it cannot provide an exact
restart. The fresh run from the same supplied initial particles was then
stopped with Ctrl-C at the user's request, with exit status 130. Its last
accepted step is 1974. The parallel instrumented run was also stopped with
exit status 130, after accepted step 1917, strain 0.47925. Neither log reports
a solver error or accepted nonlinear recovery before the requested stop.

For the production run's recorded steps 0–1974, the maximum of the two
endpoint overlap fractions is **0.0451948%** of the smaller minimum semiaxis,
and the maximum physical relative residual is **9.9966e-8**. These statistics
cover the accepted partial trajectory; they do not turn the interrupted
process into a successful 4000-step completion. No further long integration
is required for this user-requested package.

Recovery must retain the original forces, frozen Coulomb caps, physical
timestep and final force/mobility acceptance tolerances. A failed or
unconverged step must not be committed.

### Completed integration checks

Both cases below use the immutable v4 test binary with a 40+80 nonlinear
budget, normal stiffness
300 N/m, sliding and rolling stiffness 240 N/m, and timestep 2.5e-6 s.
Their initial states and complete histories are archived in
`validation_ar5_recovery_20261028/regression_ar1_n256` and
`regression_ar5_n256`; they are separate regression fixtures, not replacements
for the supplied N=1000 fixture.

| Particles | AR | Volume fraction | Completed steps | Strain | Maximum overlap / smaller minimum semiaxis | Maximum physical relative residual |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 1 | 0.34 | 4000 | 1 | 0.190700% | 9.9942e-8 |
| 256 | 5 | 0.24 | 1000 | 0.25 | 0.0391944% | 9.9964e-8 |

The overlap statistic is the maximum of the recorded solver-endpoint and
actual-endpoint fractions over each complete history. The AR1 run uses
nonlinear recovery once, at step 1871, with nine additional inner updates;
the accepted physical relative residual is 7.1281e-8. The AR5 N=256 run
does not use recovery. See `V4_NONREGRESSION.json` for the binary and fixture
hashes and the full results.

All 54 v4 CTest cases and all 28 Python bridge tests pass, with no Python tests
skipped. Their test-source and executable hashes are recorded in
`test_provenance.json`; this records the final test revision separately from
the earlier test hash in the v4 plugin build provenance. The final release
report records the separate build and short checks for the wrapper's
configurable nonlinear ceiling. The final wrapper rebuild and all 54 CTest
cases pass. Two-step checks on the supplied N=1000 particles at total ceilings
1000, 120 and 2400 reproduce the tested v4 physical prefix exactly; malformed
values and values below 40 are rejected. No new long trajectory was run after
the user requested the stop.

These are finite-transient solver checks. They do not establish steady-state
viscosity, stability to strain 50, or stability after introducing attraction.
The configured nonlinear ceiling is a bounded recovery allowance, not a
mathematical convergence guarantee for every future contact network. The
release's `default_timestep_stability_passed` flag refers only to the finite
intervals documented here, including the N=1000 partial run through step
1974; it does not certify the default strain-50 target.

### Captured failure comparisons

The unmodified local production binary reproduces the same failure class at
step 595 (strain 0.14875), with residual 1.251866906 against tolerance
0.001334425277. The last accepted step has relative viscosity 5.4382 and
maximum endpoint overlap 0.004648% of the smaller minimum semiaxis. Its entire
40-update failing inner solve uses damped directions while all linear
residual checks pass. The local and HPC failure steps differ; their step-zero
stress agrees to roundoff, but their later contact trajectories are not
bitwise identical across build environments.

A separate instrumented `-O1` baseline trajectory fails at step 683. Each
captured inner problem was saved with its original frozen caps and initial
velocity. Comparisons within each row below solve the same equation at the
same acceptance tolerance:

| Captured state | Previous Newton, limit 40 | Previous Newton, extended budget | Huber direction, limit 40 | Tested v4, budget 40+80 | Relative velocity difference from tighter original-Newton reference |
| --- | --- | ---: | --- | --- | ---: |
| AR5, N=1000, step 595 | Fails at 40 | Passes in 41 | Passes in 19 | Passes in 19; no recovery | 6.94e-10 |
| AR5, N=1000, step 683 | Fails at 40 | Passes in 46 | Passes in 24 | Passes in 24; no recovery | 7.64e-10 |
| AR1, N=256, step 1871 | Passes in 12 | Passes in 12 | Fails at 40 | Passes in 49; 9 recovery updates | 1.20e-10 |

For step 683, the old 40-update residual is 3.080958210 against tolerance
0.001474177172. The final solver's relative residual is 4.36e-11. The
original-Newton reference at a 1000-times tighter tolerance converges in 47
updates. The corresponding tighter references require 42 updates for step
595 and 12 for the AR1 state.

The AR1 state exposes a performance tradeoff: the Huber direction is slower
than the original Newton direction there. Its naturally triggered recovery
passes the unchanged tolerance and permits the full AR1 integration to
finish. The method is therefore not a uniform per-state speedup. Portable
replay sources, compressed snapshots, logs and root comparisons are in
`validation_ar5_recovery_20261028/frozen`.

Four alternating repetitions per method on frozen step 683 give median
solver process CPU times of 2.2326655 s for the candidate and 3.44721 s for
the extended original Newton reference: 35.23% less CPU time on this state.
These frozen-problem measurements do not predict the wall-time reduction of
every trajectory.

## Applying the release

Extract the cumulative overlay into the existing project source root. The
overlay preserves `run_slurry_cpu.sbatch` and `sweep_slurry.sh`, including their run
parameters and sweep pairs. Rebuild before submitting another simulation:

```sh
unzip -o /path/to/slurry_ar5_solver_overlay_20261028.zip
python3 apply_solver_limits.py
sbatch build_slurry_cpu.sbatch
```

The settings helper adds the visible `IMEX_MAX_ITERATIONS=1000` setting and
its `SLURRY_IMEX_MAX_ITERATIONS` export to the existing run script, preserving
its AR, volume fraction and other run settings. The wrapper also defaults to
1000 if the environment variable is absent, so the helper makes the setting
editable without being required for the new default to take effect.

Wait for the build job to finish successfully, then submit the existing run
or sweep as usual. Updating source files alone does not update an already
built plugin or an already running job. No stiffness or timestep change is
required by this solver update.

After applying the helper, edit `IMEX_MAX_ITERATIONS` in
`run_slurry_cpu.sbatch` to change its nonlinear ceiling. If using a run script
that does not set or override that variable, a different total ceiling can
also be supplied for one submission, for example:

```sh
sbatch --export=ALL,SLURRY_IMEX_MAX_ITERATIONS=2000 run_slurry_cpu.sbatch
```

This changes the maximum allowed nonlinear work, not the timestep, contact
stiffness, force law or convergence tolerance. The solver still returns as
soon as its acceptance conditions pass.
