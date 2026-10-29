# Nearfield validation evidence

`FINAL_VALIDATION.json` states the tested conditions and limits. `FINAL_TIMING.json`
compares the first 20 completed steps on the same host with one thread; it does not
predict a strain-10 runtime. The previous configuration used mesh hydrodynamics
and kn=0.3; the final configuration uses nearfield hydrodynamics and kn=30 N/m.

The selected production binary was built against pinned LAMMPS
`8de817dd79bfe4525d5d39246a212d833e6dee07`, using the Eigen reference sparse backend.
Its SHA is recorded in the report. Binaries and build dependencies are not included.
Build the supplied source for the target system with `build_slurry_cpu.sbatch`.

Reproduce a case from the repository root, replacing the two executable paths:

```bash
python3 validation_nearfield_20261028/run_case.py \
  --fixture validation_nearfield_20261028/fixtures/fixture_ar5_k30 \
  --plugin /absolute/path/to/libslurry.so \
  --lmp /absolute/path/to/lmp \
  --output /absolute/path/to/new_validation_directory \
  --steps 1000 --dt 2.5e-6 --threads 1
```

The output directory must not already exist. The runner stores the exact input,
particle/plugin hashes, full history, console log and numerical summary. A timestep
comparison uses the same fixture with 400 steps at 1.25e-6 or 800 steps at 6.25e-7,
both ending at strain 0.05. Production retains 2.5e-6.

`runs/` includes both final results and explicitly labeled development comparisons.
The k3 IMEX run was deliberately interrupted after exceeding the overlap criterion;
the old k30 preconditioner run was interrupted after its timing comparison. These
have `deliberate_stop.json` and return code 130, not a nonlinear-solver failure.
The separate coupled-Coulomb candidate actually failed at step 451 and is preserved
as failure evidence. Final IMEX runs must be identified by their `v2` name and SHA.

The geometric criterion is overlap below 5% of the minimum semiaxis. Passing it and
the force/torque residual checks is not proof of steady rheology. Strong startup
stress includes stored elastic energy. Strain 10, stiffness independence, attraction,
other concentrations and statistical jamming extrapolations are not certified here.
