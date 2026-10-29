# High-stiffness frozen-step regression

This captures the original bidisperse AR5, N=1000, volume fraction 0.24 trajectory immediately before step 9. The physical springs are normal 300, sliding 240 and rolling 240 N/m; timestep 2.5 microseconds. The capture plugin deliberately stops after writing the snapshot. That deliberate stop is not a solver failure.

`replay_stiff_imex.cpp` runs the same frozen equation three times: original dyadic Armijo with 40 updates (must reproduce failure), refined line search with the same 40 updates (must pass), and original search with a 100-update reference budget (passes at 41). The accepted solutions agree to relative velocity 2.3e-10. Caps, physical timestep, equations and residual thresholds are unchanged. `frozen_replay.log` records the actual run.

From this directory, decompress without changing the archived fixture:

```sh
python3 -c "import gzip,pathlib; pathlib.Path('ar5_step9.snapshot').write_bytes(gzip.open('ar5_step9.snapshot.gz','rb').read())"
c++ -O2 -std=c++17 -DEIGEN_DONT_PARALLELIZE -DSLURRY_HAVE_EIGEN=1 -I../../lammps_slurry/src -isystem ../../lammps_slurry/vendor/eigen3 -I. replay_stiff_imex.cpp -o replay_stiff_imex
./replay_stiff_imex ar5_step9.snapshot
```

This standalone replay does not require LAMMPS, MPI, FFTW or MKL. Its fixed inputs test the numerical recovery independently of how a new simulation evolves. It does not certify arbitrary density, strain or attraction stability.
