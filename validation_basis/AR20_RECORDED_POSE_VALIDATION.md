# Recorded AR20 pose: direct reference diagnostics

> **Historical diagnostic record.** The fixed 75 nm calculations and proposed next steps below describe the earlier finite-basis investigation. They do not impose a mandatory 75 nm anchor on the current independent-reference strategy. See [AR20_INDEPENDENT_STATUS.md](AR20_INDEPENDENT_STATUS.md) for the current status. Original results remain preserved.

This study evaluates only the recorded failed orientation at `h/c=0.5` (75 nm for `c=150 nm`). It does not generate another aspect ratio, an orientation table, or a direct 1 nm reference. All results are candidates, not a qualified calibration anchor.

```
theta_i = 0.0049245303592113559
theta_j = 0.041010932014513654
phi     = 1.7016960206944713
AR      = 20
```

`ar20_pair_probe.cpp` preserves raw reciprocity, solves all 17 generalized velocity/strain right-hand sides, records the symmetric matrix separately, and exports actual source coefficients. The direct reference uses `eta=1, c=1`. Matrix scaling follows `SlurryScattering`: translation, `a*Omega`, `a*E`, with conjugate moments force, torque/a, and stresslet/a. The isolated full17 matrix is analytic. The 11-dimensional result is the existing midpoint-map constrained reduction, not an assertion that its lifted response reproduces all 17 modes.

`ar20_boundary_check.py` evaluates the actual velocity of those coefficients on independent global and gap-patch surface points. Its angular integration is analytic; it uses the same MFS representation and is **not an independent Stokes formulation**. Per-RHS errors are normalized by the maximum prescribed boundary velocity of that RHS. In addition to maximum and RMS values it reports translations, rotations, and strain separately. A small linear-system residual is not a no-slip test.

The optional `self_phi` argument replaces the identity diagonal by the actual projected self response `S=P Gself D`, computed with a high-accuracy modal kernel. This is a diagnostic, not a production format change. `ar20_collocation_probe.cpp` instead solves an oversampled, surface-area-weighted least-squares system using actual self and cross fields; it is another diagnostic repair attempt, not an accepted production solver.

## Reproduce

Run from the repository root; choose a working directory with at least several GB free. Basis files and exported coefficients are working data, not checked-in calibration tables.

```bash
mkdir -p /tmp/ar20_pose
c++ -O3 -DNDEBUG -march=native -std=c++17 -I lammps_slurry/src -I lammps_slurry/vendor/eigen3 lammps_slurry/tools/slurry_generate_basis.cpp -o /tmp/ar20_pose/generate
c++ -O3 -DNDEBUG -march=native -std=c++17 -I lammps_slurry/src -I lammps_slurry/vendor/eigen3 validation_basis/ar20_pair_probe.cpp -o /tmp/ar20_pose/probe
c++ -O3 -DNDEBUG -march=native -std=c++17 -I lammps_slurry/src -I lammps_slurry/vendor/eigen3 validation_basis/ar20_collocation_probe.cpp -o /tmp/ar20_pose/collocation
/tmp/ar20_pose/generate --ar 20 --degree 16 --source-rings 192 --source-depth .5 --source-phi 256 --fit-phi 16384 --check-phi 32768 --target-rings 57 --target-phi 111 --output /tmp/ar20_pose/L16.ssb --report /tmp/ar20_pose/L16_generation.json
/tmp/ar20_pose/generate --ar 20 --degree 22 --source-rings 192 --source-depth .5 --source-phi 256 --fit-phi 16384 --check-phi 32768 --target-rings 75 --target-phi 147 --output /tmp/ar20_pose/L22.ssb --report /tmp/ar20_pose/L22_generation.json
/tmp/ar20_pose/probe /tmp/ar20_pose/L16.ssb .5 3 256 /tmp/ar20_pose/L16.json 16
/tmp/ar20_pose/probe /tmp/ar20_pose/L16.ssb .5 3 512 /tmp/ar20_pose/L16_phi512.json 16
/tmp/ar20_pose/probe /tmp/ar20_pose/L16.ssb .5 3 256 /tmp/ar20_pose/L16_self.json 16 16384
/tmp/ar20_pose/probe /tmp/ar20_pose/L22.ssb .5 3 256 /tmp/ar20_pose/L22_restricted16.json 16
/tmp/ar20_pose/probe /tmp/ar20_pose/L22.ssb .5 3 256 /tmp/ar20_pose/L18_self.json 18 16384
/tmp/ar20_pose/probe /tmp/ar20_pose/L22.ssb .5 3 256 /tmp/ar20_pose/L20_self.json 20 16384
/tmp/ar20_pose/probe /tmp/ar20_pose/L22.ssb .5 3 256 /tmp/ar20_pose/L22_self.json 22 16384
python validation_basis/ar20_compare_direct.py /tmp/ar20_pose/L16.json /tmp/ar20_pose/L22_restricted16.json --output /tmp/ar20_pose/restriction.json
python validation_basis/ar20_boundary_check.py /tmp/ar20_pose/L16.json --output /tmp/ar20_pose/L16_boundary.json
python validation_basis/ar20_boundary_check.py /tmp/ar20_pose/L22_self.json --output /tmp/ar20_pose/L22_boundary.json
python validation_basis/ar20_boundary_kernel_check.py /tmp/ar20_pose/L16.json /tmp/ar20_pose/cross_quadrature.json
/tmp/ar20_pose/collocation /tmp/ar20_pose/L16.ssb .5 16 2 512 16384 /tmp/ar20_pose/L16_collocation.json
python validation_basis/ar20_boundary_check.py /tmp/ar20_pose/L16_collocation.json --output /tmp/ar20_pose/L16_collocation_boundary.json
/tmp/ar20_pose/collocation /tmp/ar20_pose/L22.ssb .5 22 2 512 16384 /tmp/ar20_pose/L22_collocation.json
python validation_basis/ar20_boundary_check.py /tmp/ar20_pose/L22_collocation.json --output /tmp/ar20_pose/L22_collocation_boundary.json
```

Repeat the L16 generator/probe with `--source-depth .25` to reproduce the source-placement test. The probe infers the actual source depth from the saved confocal rings.

## Findings before final repair assessment

- Restricting the L22 fitted isolated operator to the L16 harmonic subspace reproduces an independently generated L16 pair matrix to `2.14e-12` in full17 generalized energy. This validates reuse of a fitted operator for degree refinement at fixed source geometry.
- Runtime source azimuth refinement 256→512 changes full17 generalized energy by `3.07e-6`. Direct integration at six worst/closest points confirms the cross field differs from analytic angular integration by `2.39e-4` at 256 samples and `2.06e-10` at 512 samples for the dominant normal-translation RHS.
- The L16 identity-diagonal solution has maximum actual boundary-velocity error 140.27% on the global check and 102.66% on the gap patch. Its self-fit defect contributes at most 4.64%/3.59%; the cross remainder reaches 139.29%/103.15%. At this quadrature the remainder is predominantly a boundary-space truncation defect.
- Replacing the self diagonal by its actual response changes neither conclusion: L16 errors remain 140.16%/102.65%, despite a smaller reciprocity defect. A better linear residual or reciprocal matrix does not certify the field.
- Native L22 with the actual self response improves boundary errors to 66.85%/58.16%. L16→L22 full17 energy changes by 1.58%; clamped-strain energy changes by 0.367%, and force/torque-free strain energy by 0.0187%. The worst boundary error is a normal translation, so this is not a small strain normalization artifact.
- The smaller source depth improves the isolated full-basis weighted residual by about 10.5 times and physical11 residual by about 8.3 times. It must still pass the same pair test.

The dominant residual occurs near the projection of the other spheroid's rim onto the first spheroid, rather than only at the exact closest point. Uniform harmonic degree refinement is expensive and slow. The collocation experiment tests whether the existing span can improve the physical field before deciding whether local enrichment or a different reference solver is needed.


## Why the moment-only criterion is insufficient

The two latest Galerkin refinements pass the nominal 1% moment criterion:

| Refinement | Full17 worst energy change | Force response | Torque/a response | Stresslet/a response |
|---|---:|---:|---:|---:|
| L18 → L20 | 0.77528% | 0.51748% | 0.53474% | 0.51814% |
| L20 → L22 | 0.21459% | 0.05436% | 0.08730% | 0.05457% |

The force, torque, and stresslet columns are relative Frobenius changes of the corresponding raw output rows over all 17 right-hand sides; they do not replace the worst-direction energy criterion. At L20, the actual global and gap-patch maximum no-slip errors are still 78.58% and 71.30%, improving only to 66.85% and 58.16% at L22. Thus the matrix trend cannot be used to certify a Stokes anchor.

The source-placement trial confirms that conclusion. Reducing confocal depth from 0.5 to 0.25 changes full17 energy by only 0.03721% and reduces global self-fit error from 4.64% to 0.596%, while total global error remains 139.31% and gap error 102.96%. Improving the isolated source representation does not remove the dominant cross-field truncation.

The L16 oversampled collocation repair reduces the independent maximum no-slip error to 42.12% globally and 43.13% in the gap patch, but changes full17 energy by 38.34% relative to L16 Galerkin. Its raw reciprocity defect is 0.00585. This is a useful diagnostic of insufficient boundary span, not a replacement calibration matrix.


## Final collocation result and required next repair

The L22 collocation trial also fails: the independent no-slip maximum is **34.79% globally and 31.51% in the gap patch**, with RMS errors 2.669% and 3.976%. The maximum global errors by RHS family are 34.79% for translation, 26.24% for rotation, and 21.93% for strain. The L16→L22 collocation full17 energy change is 20.61%, so this trial does not establish moment convergence either.

The L22 collocation matrix differs from the L22 Galerkin matrix by 11.66% in worst generalized energy; raw force, torque/a, and stresslet/a row-block changes are 5.13%, 6.35%, and 5.14%. These compare the documented trial settings (Galerkin target factor 3/source phi 256; collocation target factor 2/source phi 512), rather than claiming an isolated solver-only discretization effect. Both fail the actual boundary check by a wide margin.

**No direct matrix from this study is qualified for finite-gap calibration or production table generation.** The nominal moment trend and algebraic matching tests cannot override the field failure. No other aspect ratio, gap, or full table was run during this diagnostic study.

The next necessary numerical repair is to enlarge the boundary representation where the other disk's rim projects onto the surface: local or anisotropic enrichment with the zero-flux constraint preserved, or an independently convergent adaptive boundary-integral reference. Merely increasing source azimuth, improving isolated-body fitting, replacing the identity self block, or changing the square solve to least squares has now been tested and is insufficient. Keep the same 75 nm anchor and exact pose, then recheck actual no-slip plus all force, torque, and stresslet responses under independent refinements. The anchor or tolerance must not be changed to bypass this failure.

The recorded environment had an 8 GiB memory limit. Streaming serialization reduced the L22 generator peak to 2.10 GiB (396.7s); the L22 collocation trial used 2.05 GiB (305.5s). Its dense QR system had 29,400 rows and 3,172 columns. The source/basis files are not included because they are large reproducible working data; small raw matrices, comparisons, build hashes, and boundary checks are preserved in `ar20_v4_results/`, with `numerical_summary.json` as the compact entry point.
