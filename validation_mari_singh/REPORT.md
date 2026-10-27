# Local build and integration validation

The updated plugin compiled and loaded into the repository's pinned LAMMPS release. Short contact-containing runs passed at AR 1, 2, and 5 with a timestep of `1e-7 s`, using both direct and mesh periodic mobility. **AR 5 failed at `1e-6 s` in the retained fixture.** These are integration checks, not evidence for a jamming concentration or a converged relative viscosity.

## Environment and scope

- Source base: `4a0a1ea66e02e7d46d64d1bec9602e8cb141897e`, plus this overlay.
- Pinned LAMMPS: `stable_30Sep2026`, commit `8de817dd79bfe4525d5d39246a212d833e6dee07`.
- GNU C++ 13.3.0, CMake 4.4.4, C++17, Release build.
- Serial LAMMPS with `ASPHERE` and `PLUGIN`; plugin OpenMP available, validation run with one thread.
- **Local reference numerical backends: `builtin-reference` FFT and `Eigen-reference` sparse solver. Production FFTW and static oneMKL/PARDISO were unavailable and were not tested.** The production build wrapper and its backend requirements were not relaxed.
- No HPC submission, long rheology run, volume-fraction sweep, or timestep/stiffness/particle-count convergence study was performed.

## Build commands

The following are the actual configuration options, with machine-specific paths replaced by `LAMMPS`, `LAMMPS_BUILD`, `INSTALL`, `REPO`, and `PLUGIN_BUILD`.

```sh
cmake -S LAMMPS/cmake -B LAMMPS_BUILD \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=17 \
  -DCMAKE_INSTALL_PREFIX=INSTALL -DCMAKE_INSTALL_LIBDIR=lib \
  -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_RPATH=INSTALL/lib \
  -DBUILD_MPI=OFF -DBUILD_OMP=OFF -DPKG_ASPHERE=ON -DPKG_PLUGIN=ON \
  -DDOWNLOAD_POTENTIALS=OFF -DENABLE_TESTING=OFF
cmake --build LAMMPS_BUILD -j 4
cmake --install LAMMPS_BUILD

cmake -S REPO/lammps_slurry -B PLUGIN_BUILD \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON \
  -DLAMMPS_SOURCE_DIR=LAMMPS -DLAMMPS_DIR=INSTALL/lib/cmake/LAMMPS \
  -DCMAKE_INSTALL_PREFIX=INSTALL -DSLURRY_ENABLE_LU=ON \
  -DSLURRY_USE_FFTW=OFF -DSLURRY_USE_MKL_PARDISO=OFF
cmake --build PLUGIN_BUILD -j 2
OMP_NUM_THREADS=1 ctest --test-dir PLUGIN_BUILD --output-on-failure
```

The actual local compilation used separate plugin and test target invocations to remain within the 8 GB memory limit. No production source files were changed for the reference backend build.

## Regression tests

Full CTest result: **44/44 passed, zero failures**, in 7.67 seconds. See `ctest.log` for the complete output. All 32 C++ test/packer executables needed by that suite compiled successfully.

Completed checks include the real geometry packer, its self-test, `test_bridge_tools.py` (23 tests), and `test_packing_clearance.py` (3 tests). The compiled contact-rotation, contact-condensation, compact-bridge, and periodic-mobility tests passed. The bridge-tools tests include fixed-volume AR 1/2/5 dimensions and a genuine random 16-particle packing/certification run.

## Contact integration fixtures

`smoke_contact.py` creates two differently oriented spheroids under shear, initially with `0.1 nm` elastic overlap. This deliberate initial overlap guarantees contact coverage; it is **not** the production non-overlapping packing procedure. Each case runs five physical steps.

Every case uses volume-equivalent diameter `D = 8 µm`, reference radius `D/2 = 4 µm`, and additive regularization `epsilon × D/2 = 4 nm`, with `epsilon = 0.001`. Other parameters are `mu_s = mu_r = 10`, `k_n = 100 N/m`, sliding and rolling quasi-force stiffnesses `80 N/m`, viscosity `0.01 Pa s`, and shear rate `100 s^-1`.

| Mobility | Timestep (s) | AR 1 | AR 2 | AR 5 |
|---|---:|---|---|---|
| Direct | `1e-6` | Pass | Pass | **Fail: nonlinear iteration limit at first evolved step** |
| Direct | `1e-7` | Pass | Pass | Pass |
| Mesh, reference FFT | `1e-7` | Pass | Pass | Pass |

All passing cases have one active contact and finite numerical outputs after step zero. Initial interval and selected-window means are intentionally `NaN` until a physical interval has elapsed. AR 2 and AR 5 have nonzero rolling elastic energy, so the rolling path is exercised.

At `1e-7 s`, maximum nonlinear relative residuals are:

| AR | Direct | Mesh |
|---:|---:|---:|
| 1 | `2.99e-9` | `3.56e-9` |
| 2 | `8.00e-8` | `6.83e-9` |
| 5 | `1.43e-8` | `1.39e-8` |

The requested tolerance is `1e-6`. Maximum force-law endpoint gap-consistency errors in these cases are approximately `5.1e-18 m` or smaller. Passing at two different timesteps for five steps does not establish timestep convergence: their simulated durations differ.

## Reproduction and retained evidence

```sh
python3 validation_mari_singh/smoke_contact.py \
  --repo REPO --lmp INSTALL/bin/lmp --plugin PLUGIN_BUILD/libslurry.so \
  --output NEW_OUTPUT --timestep 1e-6 --backend direct
```

The above deliberately reproduces the recorded AR 5 failure and therefore returns a nonzero exit code. Repeat with `--timestep 1e-7` for the passing direct cases, or additionally `--backend mesh` for the passing mesh cases. Use a separate output directory for each run.

- `contact_smoke/`: original `1e-6 s` direct cases, including the AR 5 failure log.
- `contact_smoke_dt1e7/`: passing `1e-7 s` direct cases.
- `contact_smoke_mesh_dt1e7/`: passing `1e-7 s` mesh cases.
- Each directory contains per-AR input, initial/final data where available, history CSV, stdout log, and a machine-readable `summary.json`.
- `validation_metadata.json`, `slurry-numerics.json`, and `REPRODUCE.txt` record model settings and build limitations.
- `ctest.log` records the complete 44-test regression run; the separate Python test logs retain the real-packer checks.

These results do not establish that `mu_s = mu_r = 10` is saturated for spheroids, that relative viscosity 100 is reproduced, or that the effective divergence concentration is 0.30.
