# Release metadata fix — 2026-10-27

The original Mari/Singh overlay changed `prepare_bridge.py` but omitted the corresponding `release.json` hash and release identities in `build_slurry_cpu.sbatch` and `bundle_guard.py`. The user's build correctly stopped before compilation because the updated generator did not match the old release. This was a packaging omission, not evidence of a failed physics solve or failed C++ compilation. The original 44-test suite did not exercise this build preflight.

The corrected release is `20261027-mari-singh-spheroid-contact`. All three release identities agree, and release.json contains the exact current generator hashes and current model metadata. Integrity checks remain enabled. No force law, physical parameter, run-script setting, numerical backend requirement or runtime snapshot policy is changed by this hotfix.

## Apply to the already-installed Mari/Singh overlay

Put `slurry_mari_release_fix_20261027.zip` in `/home/lee6456/slurry` and run:

```bash
cd /home/lee6456/slurry
unzip -o slurry_mari_release_fix_20261027.zip
sbatch build_slurry_cpu.sbatch
```

The new build log must show `Launcher release: 20261027-mari-singh-spheroid-contact`. After the build reports `Build complete`, submit `sbatch run_slurry_cpu.sbatch`.

The hotfix contains the three release/build files, CMake test registration, the new guard test, and this verification record/log. It does not contain `run_slurry_cpu.sbatch`, so existing run settings are preserved. It assumes the original Mari/Singh overlay was already applied. The full Mari/Singh overlay has also been regenerated with this fix.

## Verification

- Applied the original overlay to a pristine archive of base `4a0a1ea66e02e7d46d64d1bec9602e8cb141897e` and reproduced the exact `prepare_bridge.py does not match release` failure.
- Applied this hotfix on top and passed the real `bundle_guard.py check-source` and the new seven-test regression suite.
- The real Bash build launcher passed its real source guard and reached the expected `runtime_snapshot.py build` handoff. Modules and the eventual compilation handoff were stubbed for this test; this does not claim a new production cluster build.
- Altered generators and stale releases are still rejected. Temporary, inert build-identity fixtures also test rejection of changed sources, changed binaries/offline tools and relocated roots.
- Reconfigured the existing local reference build to register the new suite. Full CTest: **45/45 passed**, recorded in `release_fix_ctest.log`. Production C++ code was unchanged by this hotfix.
- Shell syntax and `git diff --check` passed.

The earlier numerical validation scope remains unchanged: reference FFT/Eigen backends locally, no FFTW/MKL cluster build or long rheology validation claimed.
