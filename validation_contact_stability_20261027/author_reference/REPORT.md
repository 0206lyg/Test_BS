# Actual LF_DEM reference benchmark

The unmodified authors' LF_DEM numerical code was built and run on the project's exact AR1, N=500, phi=0.24 initial coordinates. The high numerical stiffness reproduces a time-integration instability even in the authors' code. A softer finite-stiffness reference passes timestep refinement over strain 0.1. This is an independent sphere-model benchmark; its Stokes drag + pair lubrication differs from Test_BS periodic FFT + compact bridge. Its timings are not a solver-only speedup measurement.

## Source provenance and compilation

- Authors' repository: https://github.com/ryseto/LF_DEM
- Exact commit: `00fa7cc094d6fe4c459faa478116defd3468a33d`, dated 2020-08-31.
- Clone and executable: `/tmp/lf_dem_singh_method/LF_DEM/LF_DEM`.
- SuiteSparse 5.4.0 commit `62d1da9b443631240f7d4a8d629014f6983aa2a5`, built locally in `/tmp/suitesparse_lfdem_ref`; CHOLMOD without METIS partition, standard system BLAS/LAPACK.
- LF_DEM source code was not edited. Only its local build configuration was created; a copy is `Makefile_config.mk`.
- Reference runs use `OMP_NUM_THREADS=1`, `OMP_THREAD_LIMIT=1`, and `OPENBLAS_NUM_THREADS=1`. The thread limit matters because CHOLMOD contains explicit `num_threads(...)` pragmas. Other project jobs ran concurrently, so elapsed timings are indicative.
- `System.cpp:888–979` implements explicit Heun predictor–corrector with two linear resistance solves. `SimulationInit.cpp:954` selects it by default. These public-code details do not prove the exact options of every Singh paper run.
- Singh's primary supplement is https://arxiv.org/pdf/2002.10996, pp.7–8. It specifies finite normal/sliding/rolling springs, Coulomb caps, maximum overlap below 3% radius, and lubrication damping without an additional explicit dashpot. It gives no exact timestep or stiffness ratios.

## Physical mapping and controls

`mapping.json` records the exact conversion. Radius a=4e-6 m, medium viscosity eta=0.01 Pa s, imposed shear rate=100 /s. Lengths are divided by a with no axis permutation: both codes impose xz shear. Particle positions come from `linear_validation/ar1_n500_fixture/particles.data` and are in `n500_ar1.dat`.

The hydro force unit is 6*pi*eta*a^2*gamma_dot, hence the stiffness unit is 6*pi*eta*a*gamma_dot = 7.539822368615503e-5 N/m. The dimensionless time is strain, gamma_dot*t.

All runs have mu_s=mu_r=10, kt/kn=kr/kn=0.8 (the project ratio, not a claimed Singh paper ratio), no repulsion, lubrication cutoff 1e-3, and authors' tangential lubrication model. Contact normal/tangential relaxation settings are negative, which reuses the lubrication resistance at contact rather than adding freely chosen numerical damping.

The `fixed_dt=true` branch of this old code loops at the initial output-at-zero sentinel. To avoid changing author code, effectively fixed runs use `fixed_dt=false`, a huge displacement cap, and `dt_max` equal to the requested timestep. Output endpoints may shorten the last step. Adaptive runs use a displacement cap of 1e-4 a.

## Completed stiffness and timestep tests

| Case | kn (N/m) | kt=kr (N/m) | Physical dt | End strain | eta_r at end | Max overlap / a | Elapsed s |
|---|---:|---:|---|---:|---:|---:|---:|
| soft2000 | 0.1507964474 | 0.1206371579 | adaptive, cap 1e-6 s | 0.1 | 1.95016 | 0.00130139 | 23.80 |
| soft4000 | 0.3015928947 | 0.2412743158 | adaptive, cap 1e-6 s | 0.1 | 1.94912 | 0.000963961 | 24.04 |
| soft8000 | 0.6031857895 | 0.4825486316 | adaptive, cap 5e-7 s | 0.1 | 1.95480 | 0.000586495 | 26.46 |
| soft4000_halfdt | 0.3015928947 | 0.2412743158 | cap 2.5e-7 s | 0.1 | 1.94912 | 0.000963972 | 54.28 |

The soft4000 default run actually uses about 5.56–5.88e-7 s after startup. Reducing its cap to 2.5e-7 s changes total viscosity by at most 1e-5 over all common output times (5.13e-6 relative to maximum reference viscosity), contact viscosity by 3.8e-6, and maximum overlap by 1.6e-8 a. Thus this short trajectory is timestep converged to the displayed precision.

Doubling numerical stiffness from soft4000 to soft8000 changes total viscosity by at most 0.00667, about 0.341% of the maximum reference viscosity. Contact viscosity alone changes by as much as 9.25%, especially around contact formation. Do not turn small total-viscosity variation into a claim that every microscopic observable has converged. These are transient strain-0.1 tests, not steady-state rheology or long-run certification. All recorded overlaps are far below either 3% or the user's newly allowed 5% radius criterion.

## Original stiffness instability in the authors' code

Original kn=100 N/m and kt=kr=80 N/m map to kn*=1.326291e6 and kt*=kr*=1.061033e6.

- `stiff_dt1e6s`: completed strain 0.02 in 6.80 s, but eta_r ranges 1.19989–2.13826 and max angular speed/gamma_dot reaches 166.639. Contact-viscosity adjacent-step jumps reach 0.852313. Completion does not pass stability.
- `stiff_dt1e7s`: completed strain 0.02 in 54.11 s; max angular speed/gamma_dot=24.5673. Contact-viscosity increments alternate sign in 97.84% of adjacent pairs, with lag-one correlation -0.982. This is a numerical two-cycle, not a stable trajectory.
- `stiff_dt1e8s`: all-step output reached strain 0.007323 before intentional termination. The large alternating mode disappeared, with angular speed/gamma_dot remaining O(1). A second sparse-output fine run reached strain 0.008181. These trials were deliberately stopped when the parent prioritized the approved stiffness reduction; no completed strain-0.02 or long-run success is claimed.
- `original_adaptive`: author displacement adaptivity alone also produced large angular speeds (~24), so it is not a substitute for matching the spring relaxation times.

## What the source's automatic policies actually do

`System.cpp:1013–1046` chooses dt = disp_max / max(max_velocity, max_sliding_velocity), further caps dt*gamma_dot by disp_max and applies optional dt_min/dt_max. It does not estimate the largest spring relaxation eigenvalue; rolling relative speed is not included in max_sliding_velocity (`System.cpp:1385–1402`).

Optional `adjustContactModelParameters()` (`System.cpp:2584–2655`) exponentially averages overlap, maximum tangential stretch, kn and kt in strain. It sets target kn = average(kn)*average(overlap)/overlap_target and analogous kt, relaxes each toward its target with delta_strain/memory_strain_k, applies user bounds, then updates dt. It defaults OFF, starts after strain 0.2 by default, and its own comments warn about stress fluctuations near DST. It adjusts kn and kt only; kr is not updated to preserve a ratio. Our benchmark stiffnesses are explicitly fixed numerical choices with an independent refinement check, not claimed outputs of this optional tuner.

## Reproduction and raw evidence

Run from the workspace: `python3 author_reference/run_case.py soft4000`. The runner uses the already-built author executable, one CPU thread, and the case's complete `params.txt`. `python3 author_reference/analyze.py` regenerates `summary.json` from raw `data_*.dat`.

Each case stores the original LF_DEM `input_*.dat` with the full invocation and input files, `data_*.dat` with column headers, `console.log`, and `timing.json`. The common initial raw row has eta_r=1.89464, 909 near interactions, and zero contacts. This can be compared with Test_BS only as a different hydrodynamic model baseline.
