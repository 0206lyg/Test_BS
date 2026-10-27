# Strong-friction contact solver update

Release: `20261027-contact-globalization`  
Source baseline: `4855a3a3681306a47a94c098cb0f10cfda2839e2`

## Reproduced failure

The uploaded `slurry_44687480.out` reports AR=2, phi=0.24, N=500 and
dt=1e-6 s. The replay used the repository defaults mu_s=mu_r=10,
kn=100 N/m, kt=kr=80 N/m, D=8 micrometres and seed 1729 growth packing.
Recreating that packing produced
the same final PW overlap 9.72452037e-7, minimum surface gap 7.971045323e-13 m,
and 876 near pairs.

With the original source, the local FFTW/Eigen run failed at **step 61**,
when the active contact count increased from 2 to 3. The proximal Newton
solver used 40 nonlinear iterations and 462 linear iterations, but its
scaled original force/mobility residual remained 5.881659 against a target
5.246692600e-5. These are solver-scaled norms, not forces in newtons.
Accepted line-search steps repeatedly fell to about 1e-8. This is a
globalization stall, not a solution that merely missed the iteration limit
by a small amount. A separate two-particle AR5 fixture also reproduced
the pre-existing failure at step 1 and dt=1e-6 s.

## Numerical change

The production endpoint solver now calls `SlurryBridgeContactSolver::solve`.
It first tries the existing retained-wrench Newton/FGMRES solve. Only a typed
nonlinear convergence failure enters adaptive stiffness continuation using
the original contact return-map formulation. Fatal hydrodynamic, linear
algebra, and nonfinite-value errors are propagated.

The continuation first obtains normal-only equilibrium, then adaptively
restores tangential and rolling spring stiffness from zero to the requested
values. The existing generic implementation also handles twisting when
enabled; the present production model has no twisting. Normal stiffness,
friction coefficients, dt, geometry, hydrodynamics, and committed spring
histories remain fixed. Intermediate stages provide initial guesses only.
Only the final **scale=1** solve, checked against the original equations and
original tolerance, can advance physical time or commit contact history.

No friction coefficient, lubrication regularization, spring stiffness,
timestep, stopping tolerance, particle position, or run-script setting was
changed. The 40-iteration proximal budget was not increased. A converged warm
start returns without another Newton solve. Failed-attempt work and all
endpoint-correction work are included in numerical counters. A nonlinear
failure now prints its actual step, residual, tolerance and iteration counts
even when detailed tracing is disabled.

## Validation

The same AR2/N500 replay completed **100 physical steps** with the modified
solver; the original solver failed at step 61. Continuation rescued steps 61
and 68, including three rejected intermediate stages that were retried with
smaller stiffness increments. Maximum relative residual was 7.4881305576e-8
against the requested 1e-6; maximum endpoint gap inconsistency was
1.02863e-19 m. All essential output values were finite. The run covers strain
0.01. Before the original failure, viscosity histories over steps 0–60
agreed to about 2.0e-10 relative. No physical time step was skipped.

The final source, including diagnostics and complete endpoint counter
accounting, was additionally rebuilt and passed all three contact fixtures
for five steps and the N500 fixture for five steps through the shipped replay
driver. The 100-step candidate and final-source numerical equations are
identical; only diagnostic formatting and counter aggregation differ.

The original AR1/2/5 contact fixtures each passed 100 physical steps at
dt=1e-6 s and unchanged strong-friction parameters with both direct mobility
and FFTW mesh mobility. Maximum relative residuals were:

| Backend | AR1 | AR2 | AR5 |
|---|---:|---:|---:|
| Direct |5.28e-10 |1.93e-8 |8.76e-8 |
| FFTW mesh |6.36e-10 |1.39e-8 |6.54e-8 |

AR5 used continuation at steps 1, 2, 3; the following 97 steps used the original
fast path. Maximum AR5 endpoint gap inconsistency was about 1e-18 m.

Five compiled C++ regression tests passed: `bridge_globalization`,
`contact_condensation`, `contact_rotation`, `bridge_solver`, `bridge_step`.
The new test uses two explicit hard-contact fixtures and independently
reevaluates the original force/torque residual and contact law. It also checks
that frozen data and committed history remain unchanged, a converged warm
root requires zero further iterations, and a fatal operator error propagates
exactly once. Their residuals were 4.31e-12 and 7.81e-12, below respective
tolerances 4.46e-7 and 3.06e-7.

Detailed integration results and the exact N500 replay are retained in
`validation_solver_20261027/`. The reported mesh integration results use the
pinned LAMMPS release and FFTW plus Eigen sparse factorization. The production
FFTW/static oneMKL-PARDISO requirements are unchanged; the Negishi MKL build
and a 100000-step campaign were not run here. Short regression success does
not establish steady-state viscosity or timestep convergence.

## Apply and run

Place the overlay ZIP in `/home/lee6456/slurry`, then run:

```bash
cd /home/lee6456/slurry
unzip -o slurry_solver_overlay_20261027.zip
sbatch build_slurry_cpu.sbatch
```

After the build reports `Build complete`, run:

```bash
sbatch run_slurry_cpu.sbatch
```

The build log must show release `20261027-contact-globalization`.
The overlay does not contain `run_slurry_cpu.sbatch`, so locally edited
parameters and resource requests in that run script are preserved. Rebuild
before starting a new run; existing private runtimes retain their old plugin.
No remote repository branch or existing run directory is changed by applying
the overlay.

## Replay the retained fixtures

The following driver uses a caller-selected executable/plugin and a new
output directory. Its default is 100 steps and 1 thread. Detailed solver traces
are enabled for this diagnostic replay.

```bash
python3 validation_solver_20261027/reproduce.py \
  --lmp /path/to/lmp --plugin /path/to/libslurry.so \
  --output /path/to/new-contact-check --case contacts --backend mesh

python3 validation_solver_20261027/reproduce.py \
  --lmp /path/to/lmp --plugin /path/to/libslurry.so \
  --output /path/to/new-n500-check --case n500 --backend mesh
```

The N500 fixture changes only output frequency and the requested run length
from the original production case; its physical input and initial packing
are retained. The contact fixtures deliberately begin with a small overlap
to exercise sliding/rolling; they are not production packing examples.
