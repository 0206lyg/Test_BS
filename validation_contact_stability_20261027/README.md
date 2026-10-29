# Contact stability evidence

See ../docs/CONTACT_STABILITY_UPDATE.md for the selection and limits.

* n500/heun_120us: PC v1, 48 steps, dt=2.5 us. The physical equations are
  identical to v2. Only unchanged physical columns are retained; invalid
  v1 projection, gap-consistency and component-timing diagnostics are omitted.
  Endpoint overlap belongs to the prepared next step, at time_s + dt.
* n500/final_source_20us: rebuilt exact delivered source, PC v2, 8 steps.
  Full corrected diagnostics, input and original N500 coordinates are included.
* n500/euler_kn*: original one-stage explicit reference, 120 us.
* fair_benchmark: sequential one-thread, 20 us, same geometry and model.
  The selected comparison excludes dt=5 us because pair transients were inaccurate.
* pc_soft_selected and pair_stiffness03: Heun and Euler pair time refinement.
* author_reference: actual LF_DEM runs with a different hydrodynamic operator.
  Source commit and mapping are recorded in REPORT.md. Build LF_DEM separately
  and set LF_DEM_EXE to its absolute executable path before using run_case.py.
  Archived timing/metadata retain the original absolute experiment paths.
* checks: seven C++ regression checks, 33 Python checks, migration check,
  and final plugin build. Initial failed compilations and subsequent retries
  are retained; final C++ summary reports seven passes.

No binary is supplied: rebuild the plugin with the normal build script.
The portable runner requires explicit executable and plugin paths, for example:

```bash
python3 run_case.py --fixture n500/final_source_20us \
  --lmp /absolute/path/to/lmp --plugin /absolute/path/to/libslurry.so \
  --output replay_n500 --dt 2.5e-6 --steps 48 --threads 1
```

The final-source N500 smoke agrees with the PC v2 timing run within 3.3e-9 in relative viscosity.
These short checks do not certify long-strain stability or steady contact-stress
convergence. The kn=.3 to 1 comparison changes transient contact stress by 46.9%.
