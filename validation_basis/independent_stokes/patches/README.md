# Deterministic near-quadrature policy for the AR20 reference backend

This local patch specializes the pinned FMM3DBIE backend to panel order **p >= 4**. It preserves the public Fortran ABI and selects the existing high-order branch of `get_quadparams_adap`. It is not a general upstream repair for arbitrary panel order, and it does not establish physical accuracy.

The primary evidence is the upstream source at commit `be6b2b3700b77e9cbf20c512ff4f0922aaee45d9`:

- [`near_field_routs.f`, lines 883–1007](https://github.com/fastalgorithms/fmm3dbie/blob/be6b2b3700b77e9cbf20c512ff4f0922aaee45d9/src/quadratures/near_field_routs.f#L883-L1007) declares local `norder` at line 911 and reads it at line 927 without assignment. Its six-argument signature has no panel-order argument.
- [`ggq-quads.f`, lines 243–291](https://github.com/fastalgorithms/fmm3dbie/blob/be6b2b3700b77e9cbf20c512ff4f0922aaee45d9/src/quadratures/ggq-quads.f#L243-L291) documents dependence on discretization order, then calls the routine for triangle and quadrilateral patch types. All eight calls in this file and all four calls in `adap-quads.f` use that same six-argument ABI. The caller knows `norders`, but the callee cannot infer it from `iptype`.

The patch initializes the local selector to 4 immediately before the existing conditional. This makes the intended p >= 4 branch deterministic without changing its coefficients or any caller, self GGQ quadrature, kernel, or geometry. Its adaptive/far quadrature orders are greater than those of the lower-order branch. Its stopping tolerance is **not uniformly tighter**: for the two highest precision bins, the existing high-order values are `7e-8` and `3e-10` versus `3e-8` and `2e-10` in the lower-order branch. No claim of improved accuracy follows from the branch choice alone.

Requested tolerance is binned. In particular, `1e-7` and `1e-9` both select bin 3; `1e-11` selects bin 4. Repeating two tolerances in one bin is not a quadrature convergence test. The patch also does not alter the on-surface self GGQ rule, whose parameterization must be checked independently.

Build a separate library from the existing combined archive, changing only this object:

```bash
python3 patches/build_quadparams_library.py \
  --upstream /path/to/pinned/fmm3dbie \
  --output-prefix /path/to/new/quadparams_highorder_v1 \
  --fc /path/to/gfortran
python3 patches/probe_quadparams.py \
  --old-library /path/to/pinned/fmm3dbie/lib/libfmm3dbie.so \
  --new-library /path/to/new/quadparams_highorder_v1/lib/libfmm3dbie.so \
  --output /path/to/quadparams_policy_probe.json
```

The builder requires the pinned source SHA, a clean tracked source tree, and a new output directory. It copies the combined archive, compiles the patched single file with the flags used for the current reference build, replaces exactly `near_field_routs.o`, and links a new shared library. It checks every archive member, records hashes and commands, and verifies that the original source, archive, and shared library remain unchanged. The new manifest explicitly says `physical_accuracy_qualified: false`.

The probe uses separate processes for the two libraries to avoid ELF symbol interposition. It checks both patch types, every precision bin and both sides of each threshold, and repeated output. Passing verifies parameter selection and ABI only. Existing density/operator results retain their old library SHA; a mixed old-density/new-operator calculation is a diagnostic, not a qualified reference. New qualification requires the patched operator and valid independent error/convergence evidence at the same AR20 pose.

The local build changed only `near_field_routs.o`; all 131 other archive members are byte-identical. The old shared-library SHA is `72416a77f0f03392cbb19a04368502bd4dd0514762fe2838d989988b46e03e44`; the separate patched library SHA is `c173caecdc37b51ce7887a92d14a746d9463d59d497ae99ecbef15204e66b052`. Both libraries have the same dynamic dependency list. GNU Fortran 13.3 with `-O2 -Wuninitialized -Wmaybe-uninitialized` independently reports that `norder` is used uninitialized at the conditional.

`../results/quadparams_highorder_policy_probe.json` records 28 parameter cases, each repeated three times. All patched cases matched the intended high-order policy. The old compiled library selected the lower-order branch in every observed case; this observation is compiler/build specific and does not make the undefined source behavior valid. The build provenance is in `../results/quadparams_highorder_patch_manifest.json`.
