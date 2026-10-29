# Bidisperse transient validation

`FINAL_VALIDATION.json` defines the tested scope. High stiffness is kn=300 and kt=kr=240 N/m at dt=2.5e-6 s. These are finite transient tests, not steady viscosity, strain50, attraction, or a universal stability guarantee.

Each `fixture_*` directory contains a fixed initial state, preparation parameters, exact surface-clearance certificate, and independent count/volume/mixing audit. N1000 uses733 small/267 large particles, with volume-equivalent size ratio1.4; rounding gives50.0120%/49.9880% solid-volume shares. Spatial cell counts are diagnostics, not equilibrium certification.

Each run directory contains all-step history, input, binary/fixture hashes, console and summary. `v1` is the new bidisperse implementation before safeguarded directional line-search refinement; `v2` contains the refinement. Two actual v1 Newton failures are retained. The v2 N1000 AR5 1000-step exploratory run was deliberately interrupted atstep90 and is not a failure or completed1000-step validation.

Reproduce a fixed state with locally built LAMMPS and plugin paths:

```bash
python3 run_validation.py --fixture fixture_ar5_phi024_n1000 --plugin /absolute/path/libslurry.so --lmp /absolute/path/lmp --output /new/output/directory --steps 200 --kn 300 --threads 1
```

`audit_fixture.py FIXTURE` independently checks counts, physical volume, axes, and spatial mixing. `compare_solver_versions.py` compares the preserved paired histories. Binaries and large trajectory dumps are deliberately excluded; source hashes identify the tested runtime.
