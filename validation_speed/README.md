# Same-input speed comparison

`compare_speed.py` runs the baseline and candidate plugins sequentially against the same LAMMPS executable and certified initial particles. It does not run the packer. AR 2, 5, 10 and 20 fixtures contain 500 particles at volume fraction 0.24; thickness 300 nm, viscosity 0.01 Pa s, shear rate 100 s^-1, timestep 1e-6 s, gap floor 1 nm, contact onset 3 nm, friction 0.1, normal stiffness 100 N/m, tangential stiffness 80 N/m, mesh spectral 2, and solver tolerance 1e-6. The clearance reports come from the production geometric packer. `particles.data` and the input template were generated with the current `prepare_bridge.py`; hashes bind all fixture files.

Run from the repository root, with paths to compatible builds:

```sh
python3 validation_speed/compare_speed.py --lmp /absolute/path/lmp --baseline-plugin /absolute/path/baseline/libslurry.so --candidate-plugin /absolute/path/candidate/libslurry.so --threads 1 8 --steps 5 --ars 2 20 --output /absolute/path/new-comparison
```

For an OpenMPI build inside a scheduler allocation, add `--mpi-launcher /absolute/path/mpirun`. This uses exactly one MPI rank bound to the requested number of cores. Run the Python driver once, outside MPI ranks. Ensure the allocation and process CPU affinity contain at least the requested physical cores; the runner records affinity and OpenMP configuration. Keep competing benchmarks/builds off those cores. There is no launch retry. Existing output directories are refused. Baseline and candidate binaries must remain unchanged during the comparison.

Each run preserves `console.log`, `log.lammps`, `history.csv`, trajectory, final data, launch configuration, binary hashes and `result.json`. `summary.json` and `comparison.csv` report final stress, relative viscosity, actual solver residual and numerical differences. The summary also compares every common history step and records iteration/mobility counts. A zero baseline value yields an undefined relative difference (`null`), with the absolute difference retained.

Add `--compare-trajectories` to also compare the final frame by particle ID. The grouped diagnostics include position, velocity, angular momentum, orientation and all 17 `f_hydro` components, separated into hydrodynamic force, torque, stresslet and contact force/torque so different units are not mixed. Each group reports relative L2, absolute L2, component RMS and maximum component error. Quaternions are normalized and sign-aligned, with rotation-angle errors reported separately. Position comparison includes both raw coordinates and nearest periodic-image differences in the restricted triclinic box when the two cells match. Its relative L2 denominator is the baseline coordinate norm and therefore depends on the coordinate origin; the absolute errors are also retained. Zero baseline norm gives relative error zero only for identical vectors, otherwise `null`.

Existing completed runs can be reanalyzed without launching LAMMPS:

```sh
python3 validation_speed/compare_speed.py --reanalyze /absolute/path/existing-comparison
```

This writes `reanalysis.json`, including trajectory comparisons, without modifying the original `summary.json`. The Python functions `read_last_frame`, `compare_trajectories` and `reanalyze` can also be imported for separate analysis. The final timestep and particle ID set must match; missing or incomplete trajectories are explicitly reported rather than treated as agreement.

Wall time includes process startup, plugin load, FFT planning, initial setup, physical steps and output. Summed per-step profile times exclude step zero; output interval is one, so advancing steps are individually observed. These two measurements distinguish first-use costs from repeated-step work. Thread counts and AR values are run in the requested order, baseline then candidate, strictly sequentially. Repeat comparisons with enough steps to assess timing variability; a short run does not establish steady-state viscosity or validate the physical model. The runner reports numerical differences rather than silently accepting a chosen physical-error threshold. Exit zero means all runs completed the requested final step with finite required diagnostics; it is not an assertion that every observed difference is acceptable.

Use `--ars 2 5 10 20` for all supplied fixtures. The `--timeout` option limits each external launch; its default is unlimited. Fixture and certificate SHA256 checks always run before launching any simulation.

`results/numerical_equivalence.json` records the initial AR 2/20, eight-thread, five-step numerical comparison, with four small raw histories in `results/numerical_raw_histories`. That run overlapped compilation: its timings are explicitly excluded from final performance claims. The artifact contains global grouped L2 differences, individual-particle worst cases with their actual denominators and absolute differences, physical power-balance normalization, raw and normalized residuals, binary hashes, and the contact-condensation dimension reduction. Individual nearly zero net forces or torques can have large relative differences even when absolute differences are negligible; both measures are retained.
