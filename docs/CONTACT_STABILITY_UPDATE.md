# Contact stability and method selection

Release `20261027-calibrated-linear-contact`.

The production integrator is now an explicit Heun predictor–corrector. Each
step solves two **linear** force/mobility systems, using current and predicted
positions, orientations and contact histories. It has no Newton contact solve,
stiffness continuation or endpoint-force iteration. The existing periodic FFT
mobility, compact lubrication, friction coefficients and contact laws remain.

The selected numerical settings are:

| Setting | Old | Selected |
|---|---:|---:|
| Normal stiffness, N/m | 100 | 0.3 |
| Sliding stiffness, N/m | 80 | 0.24 |
| Rolling quasi-force stiffness, N/m | 80 | 0.24 |
| Sliding and rolling friction coefficients | 10, 10 | 10, 10 |
| Time step, s | 1e-6 | 2.5e-6 |
| Steps for the original 0.1 s run | 100000 | 40000 |
| Output interval for the original 1e-4 s cadence | 100 | 40 |

These are numerical spring constants, not measured graphite elastic constants.
The overlap acceptance criterion is less than 5% of the particle radius for
spheres; for an oblate particle, use its smallest semiaxis. No position or
overlap is clipped to enforce this criterion. A new concentration, aspect
ratio or stress regime must be checked against this criterion and against
stiffness and time-step refinement.

## Why changing stiffness is justified

Mari et al. choose spring constants to keep overlap and tangential stretch
below 5% of radius, and require contact relaxation to be resolved while
remaining fast compared with shear. Singh et al. use finite sliding and
rolling springs with a maximum-overlap criterion of 3%. Infinite friction
removes the Coulomb cap; it does not make spring stiffness infinite.

The authors' public LF_DEM source was built and run with the identical AR1,
N500 initial coordinates. Their finite-stiffness explicit implementation also
showed numerical two-cycle oscillations with our old 100/80/80 N/m settings:
at dt=1e-7 s, contact-stress increments had lag-one correlation -0.982 and
97.84% sign alternation. Thus the old instability cannot be attributed simply
to large friction or to an inability of explicit methods to handle friction.

In the author code, approximately 0.30159/0.24127/0.24127 N/m reached strain
0.1 with maximum overlap 0.0964% of radius. Refining time step changed total
relative viscosity by at most 1e-5. Doubling stiffness changed the total
viscosity by up to 0.341%, but the contact contribution by 9.25%; this is not
proof of convergence of every microscopic quantity. The author model uses
Stokes drag plus pair lubrication, so its runtime and viscosity must not be
identified with those of the retained FFT model.

## Current-model verification

All N500 tests below use AR1, phi=0.24, D=8 micrometres, eta=0.01 Pa s,
shear rate 100/s, mu_s=mu_r=10, and the exact original failing initial packing.

* Heun, dt=2.5 microseconds, selected stiffness: **48 steps to 120
  microseconds passed**, covering the previous failure time of 62 microseconds.
  Maximum relative force/mobility residual was 9.95e-8, below 1e-6.
  Maximum prepared endpoint overlap was 0.0202% of radius.
* The one-stage explicit scheme at dt=1 microsecond passed 120 steps for
  kn=0.1, 0.3 and 1 N/m, with kt=kr=0.8kn. At kn=0.3, its maximum total
  viscosity difference from Heun was 0.00922% over shared physical times.
* Raising kn from 0.3 to 1 N/m changed total viscosity by at most 0.806%
  in this short start-up window, but contact stress by 46.9%. The small
  total-viscosity change does not certify contact-stress or steady-state
  stiffness convergence. In particular, this window is shorter than the
  softened normal-contact relaxation time.
* AR1/2/5 pair tests with finite AR2/5 rolling activity were refined
  from 2.5 to 1.25 microseconds. Maximum AR5 contact-stress difference was
  5.07% of the reference peak, and maximum finite rolling-energy difference
  across AR2/5 was 3.83%. At 5 microseconds the AR5 contact transient error
  was 33.5%, so that larger step was rejected despite its smooth total
  viscosity and bounded overlap.
* Independent finite-prestretch tests check damping, force balance and
  convergence to an analytical relaxation ODE. Increasing already inactive
  friction caps leaves those relaxation rates unchanged.

The long N500 runs used the numerical-equation-identical PC candidate before
diagnostic cleanup. The final source separately passes N500 and pair smoke
checks; current velocities are now stored separately from prepared advance
velocities, and rolling projection/timing diagnostics were corrected.
The retained validation tables omit the candidate's defective diagnostic
columns. Sources, exact inputs, relevant comparisons and final checks are
under `validation_contact_stability_20261027/`.

## Performance selection

The same N500 packing, same hydrodynamics and coefficients, one thread and
20 microseconds of physical time were run sequentially:

| Method | Steps | CPU time, s |
|---|---:|---:|
| One-stage explicit, dt=1 microsecond | 20 | 162.219 |
| Heun, dt=2.5 microseconds | 8 | 126.449 |

Heun used **22.05% less CPU** in this bounded benchmark. Final relative
viscosity differed by 0.00157%, and contact stress by 0.841%. This is a
method/settings comparison on the local FFTW/Eigen build, not a forecast of
the complete Negishi campaign. Long-strain stability, steady rheology and
the production MKL/PARDISO build are not certified by these short tests.

## Histories and output semantics

Predictor histories are temporary. The corrector starts from the original
history, transports predicted sliding/rolling rates objectively, integrates
their trapezoidal average, and projects against the corrected endpoint load.
Only the accepted advance commits history. A failed linear solve commits
nothing. Instantaneous atom velocities, forces, stress and energies belong
to `time_s`; separately stored average velocities advance the particles.

Endpoint gaps, rolling projection counts and projection work describe the
prepared step ending at `next_step_end_time_s`. Legacy CSV column names are
retained. `next_step_rolling_projection_work_J` explicitly reports work;
`rolling_plastic_dissipation_W` divides that work by the upcoming dt.
`contact_gap_consistency_m` is NaN because an implicit endpoint-force
consistency condition is not imposed. The actual linear-gap defect remains
reported. Residual diagnostics report the larger of the two linear stages.

## Apply

```bash
cd /home/lee6456/slurry
unzip -o slurry_contact_stability_overlay_20261027.zip
python3 apply_contact_settings.py
sbatch build_slurry_cpu.sbatch
```

After `Build complete`, submit `sbatch run_slurry_cpu.sbatch`.
The settings helper changes only the three stiffnesses, dt, step count and
output interval. It preserves run-specific AR, phi, particle count, friction,
paths and Slurm resources, rounding the duration upward and the output cadence to the nearest new step. It is idempotent and saves the original run script once.
The ZIP deliberately does not overwrite that run script.

## References

* [Mari et al. 2014, Appendix B](https://arxiv.org/html/1403.6793v4).
* [Singh et al. 2020 and supplement, pp. 7–8](https://arxiv.org/pdf/2002.10996).
* [Authors' dataset identifying LF_DEM](https://acdc.alcf.anl.gov/mdf/detail/singh_rolling_friction_prl_2020_v1.4/).
* [Inspected 2020 LF_DEM source](https://github.com/ryseto/LF_DEM/tree/00fa7cc094d6fe4c459faa478116defd3468a33d).
