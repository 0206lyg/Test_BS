# N500 short-run solver validation

This bundle preserves AR 2, 5, 10, 20 at volume fraction 0.24 and seed 1729.
The original uploaded logs failed during initial setup; they provide no exact
binary hash or comparable wall time. Final runs use the same stored initial
particle files and record step 0 plus two advancing steps. This is a short-run
solver check, not steady-state viscosity or Stokes-accuracy validation.

Read `summary.json` for the comparison and `final_summary.json` for full checks.
`provenance/` records the matching verified build manifest, exact source hashes,
and final binary hashes. No executable is included. Intermediate solver trials
are excluded. Absolute historical paths in raw launch records are preserved.

To rerun fixed initial data after building the plugin, from this directory:

```sh
python solver500_smoke.py \
  --source fixtures/ar2 --source fixtures/ar5 \
  --source fixtures/ar10 --source fixtures/ar20 \
  --output-root NEW_RESULTS --lammps /absolute/path/to/lmp \
  --plugin /absolute/path/to/libslurry.so \
  --library /absolute/path/to/liblammps.so.0 --threads 8 --steps 2 --timeout 0
python solver500_summarize.py NEW_RESULTS --steps 2 \
  --require-ar 2 --require-ar 5 --require-ar 10 --require-ar 20 --output new_summary.json
```

The rerun driver is diagnostic. Production use goes through
`lammps_slurry/tools/run_bridge.py --build-env /absolute/path/to/build/env.sh`,
which checks the build and creates a private runtime. Use `--mpi` in the diagnostic
driver when the LAMMPS executable was built with MPI; it launches one rank.

## Same-state response comparison

`comparisons/ar2_step0.json` and `comparisons/ar10_step0.json` compare the final
step-0 response with previously converged, frozen solver runs at exactly the same
initial particle file, box, geometry, and physical inputs. The records contain
relative L2 and maximum absolute differences for velocity, hydrodynamic force,
torque, STF stresslet, contact force/torque and angular momentum, plus scalar
stress, viscosity, and power differences. Configured tolerances and measured
residuals accompany the comparisons; no numerical acceptance threshold is imposed.
These compare algorithms for the same model, not against an exact Stokes solution.

The full historical baseline trials are not bundled; their paths and recorded
binary hashes are preserved in the comparison JSON. To repeat a comparison when
those baseline output files are available, use:

```sh
python solver500_compare_step0.py --baseline BASELINE_CASE \
  --candidate FINAL_CASE --output comparison.json
```

## Approved runner

`approved_runner_ar10/` records the actual `run_bridge.py` private-runtime path through step 10. `verification.json` reports its physical residual, motion and runtime hash checks. Initial zero-duration averages are undefined; instantaneous responses remain finite. No steady-state claim is made.
