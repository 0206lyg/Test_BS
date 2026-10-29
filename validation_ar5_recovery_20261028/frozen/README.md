# AR5 nonlinear-recovery frozen regressions

The AR5 fixtures come from the uploaded N=1000, AR=5, volume fraction 0.20 initial particles and settings: normal stiffness 300 N/m, sliding and rolling stiffness 240 N/m, shear rate 100/s, timestep 2.5 microseconds. The separate AR1 fixture is described below. Each fixture contains the actual **inner** convex problem, initial warm velocity and independently saved frozen normal loads. Do not recompute the loads from corrected gaps or tighten the stored tolerances by another factor of ten.

The O3 diagnostic trajectory reproduced the original local production binary bit-for-bit through its step 595 failure (strain 0.14875). The O1 diagnostic trajectory failed at step 683 (strain 0.17075); the uploaded HPC log failed at step 723. Floating-point trajectories differ between builds, so these step numbers are not interchangeable. All AR5 runs use the same uploaded initial particles and physical settings.

`replay_ar5_recovery.cpp` verifies six branches on the identical captured equation:

1. Delivered solver behavior, recovery disabled: reproduces the 40-update failure.
2. Adaptive Huber majorizer with the same 40-update limit: passes.
3. Production 40+80 bounded budget: passes without requiring recovery for this fixture.
4. Original solver with a 200-update reference budget: passes.
5. Original solver with a 1000-times tighter tolerance: supplies an independent root comparison.
6. Deliberately short primary budget of four updates: exercises bounded recovery while preserving the original tolerance.

The primary fixture at step 595 takes 19 updates with the hybrid direction, versus 41 with extended original Newton. Its velocity differs from the tighter reference by 6.94e-10 in relative norm. The step 683 fixture takes 24 versus 46 updates and agrees to 7.64e-10. Four alternating repetitions per method on step 683 gave median process CPU times of 2.233 versus 3.447 seconds. All physical equations, frozen loads, stiffnesses and timestep remain unchanged. The shared host causes timing variation; these frozen solves do not establish a full-trajectory speedup or stability to strain 50.

From this directory:

```sh
python3 -c "import gzip,pathlib; p=pathlib.Path('ar5_phi020_step595_inner.snapshot.gz'); pathlib.Path(p.stem).write_bytes(gzip.decompress(p.read_bytes()))"
c++ -O2 -std=c++17 -DEIGEN_DONT_PARALLELIZE -DSLURRY_HAVE_EIGEN=1 -I../../lammps_slurry/src -isystem ../../lammps_slurry/vendor/eigen3 replay_ar5_recovery.cpp -o replay_ar5_recovery
./replay_ar5_recovery ar5_phi020_step595_inner.snapshot ar5_phi020_step595_caps.txt
```

Replace `595` with `683` in the decompression and execution commands to replay the second AR5 failure.

## Natural recovery and its performance tradeoff

The independent N=256, AR=1, volume fraction 0.34 regression trajectory exposed a different difficult state at step 1871. The original Newton method solves this frozen equation in 12 updates. The hybrid direction without recovery exhausts its primary 40-update budget; the production solver uses nine additional updates and passes at 49. Its velocity agrees with a 1000-times tighter original-Newton reference to 1.20e-10. This state is slower with the hybrid direction, so the change is not a uniform per-state speedup. It directly tests a naturally triggered recovery, with unchanged timestep, stiffness and residual tolerance.

```sh
python3 -c "import gzip,pathlib; p=pathlib.Path('ar1_phi034_n256_step1871_inner.snapshot.gz'); pathlib.Path(p.stem).write_bytes(gzip.decompress(p.read_bytes()))"
c++ -O2 -std=c++17 -DEIGEN_DONT_PARALLELIZE -DSLURRY_HAVE_EIGEN=1 -I../../lammps_slurry/src -isystem ../../lammps_slurry/vendor/eigen3 replay_natural_recovery.cpp -o replay_natural_recovery
./replay_natural_recovery ar1_phi034_n256_step1871_inner.snapshot ar1_phi034_n256_step1871_caps.txt
```

The default recovery budget of 80 is a finite work limit after the primary 40 updates, not a mathematical guarantee of convergence within 120 updates for every future contact network. Exhausting the total budget still reports failure; no unconverged physical step is accepted.

No LAMMPS, MPI, FFTW or MKL runtime is needed. The `step*_metadata.json` files record fixture and binary hashes, capture provenance and measured comparisons. Archived snapshots remain compressed; decompressed files and compiled executables are disposable local outputs. The focused C++ unit test additionally checks disabled/exhausted/negative recovery budgets and closed-form sliding, rolling and twisting solutions.
