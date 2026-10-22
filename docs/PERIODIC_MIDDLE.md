# Periodic middle-range mobility

`slurry_periodic_mobility.h` supplies the periodic middle response used by
`fix slurry/bridge`. It retains the previously selected ellipsoid force,
torque, and five stresslet modes. The compact near correction remains v4.
The periodic implementation introduces one explicit finite-distance
approximation: the unresolved high-frequency response is retained on each
isolated body and its cross-body and image interactions are omitted.

## Coordinates and normalization

The eleven moments act **on the fluid**:

- `g = (F_x,F_y,F_z,T_x,T_y,T_z,S_1,...,S_5)`;
- conjugate rates are `q = (U,Omega,-E_STF)`;
- the STF basis is `(xx-yy)/sqrt(2)`,
  `(xx+yy-2zz)/sqrt(6)`, and normalized `xy,xz,yz` tensors.

The ellipsoid shape matrix is `A_i = Q_i diag(a_i²,b_i²,c_i²) Q_i^T`.
The cell matrix `H` has lattice vectors as columns, `V=det(H)>0`, and
`k=2*pi*H^(-T)*n` for integer `n`. For `k != 0`, put

```
nk = k / |k|,  Pk = I - nk nk^T,
a_i(nk) = sqrt(nk^T A_i nk),  z_i = |k| a_i(nk),
B(nk) = [ -cross(nk)/2,  STF_1 nk, ..., STF_5 nk ],
Psi_i(k) = exp(-i k.x_i) [ sinc(z_i) I,
                           -i 3 j1(z_i)/a_i(nk) B(nk) ].
```

Here `sinc(z)=sin(z)/z` and `j1` is the spherical Bessel function. The
untruncated periodic Galerkin mobility would be

```
M_ij = sum_{k != 0} Psi_i(k)^* Pk Psi_j(k) / (eta V |k|²).
```

The physical `1/(eta V |k|²)` measure is included. A half lattice is stored
with twice that weight. No four auxiliary directions or fitted self
rescaling are added. A spherical **physical wavenumber** cutoff, rather
than a fixed integer cube, preserves the set of modes under integer
changes of the lattice basis.

The zero mode is removed. This fixes the mean disturbance velocity and
uses the usual pressure-gradient/mean-force compensation if the applied
force has a nonzero mean. The present shear run has no imposed net body
force; internal contact forces sum to zero. The implementation should not
be interpreted as predicting mean sedimentation/backflow without a
separately specified pressure or flux condition.

## Finite-cost approximation

The default is the positive split

```
M_low,ij = sum_{0 < |k| <= kmax}
             exp(-|k|²/(4 xi²)) Psi_i(k)^* Pk Psi_j(k)
             / (eta V |k|²),
M_approx = M_low + blockdiag(M_high,self,i),
kmax = 2 xi sqrt(-log(spectral_tolerance)).
```

`--spectral 2` means `xi*a_max=2`; it is not a lattice index or an error
percentage. The default spectral tolerance is `1e-8`. It controls only
truncation of the Gaussian low-frequency sum. It does **not** bound the
error caused by omitting the high-frequency cross terms. At a reciprocal
mode's cutoff crossing during shear, the omitted weight is already at
this truncation level; exact smoothness of the infinitely long sum is not
claimed for the finite implementation.

The diagonal high-frequency term is the continuum integral of the
**same ellipsoid form factors**, multiplied by
`1-exp(-|k|²/(4 xi²))`. Its radial integrals are analytic. With
`b=2 xi a(n)`, define

```
t_F(b) = erfc(b) + (1-exp(-b²))/(sqrt(pi) b),
t_D(b) = erfc(b) + 3(1-exp(-b²))/(sqrt(pi) b)
                  - 2[1-(1+b²)exp(-b²)]/(sqrt(pi) b³).
```

Then the self tail has zero force/dipole cross block and

```
M_high,FF = integral dOmega P(n) t_F(b) / [16 pi² eta a(n)],
M_high,DD = integral dOmega 3 B(n)^T P(n) B(n) t_D(b)
                        / [16 pi² eta a(n)³].
```

The small-argument series avoids cancellation. Positive angular weights
are used in the principal body frame, and the result is rotated into the
world frame. For a spheroid, 256 Gauss-Legendre polar points and 16 exact
azimuthal polynomial points are the default. Identical principal shapes
share the cached body-frame integral within a geometry update.

Every retained spectral contribution is a positive Gram term. The self
tail is a positive integral. Thus the assembled mobility is symmetric
and positive; this property does not depend on configuration sampling.
It does not prove a finite-distance accuracy bound. The leading force,
rotlet, and stresslet Fourier coefficients remain those of the selected
ellipsoid model, since the filter tends to one at small wavenumber.

## Particle–mesh evaluation (default since 2026-10-22)

The retained Gaussian sum is evaluated by `slurry_periodic_mesh.h`. Let
`L L^T=A`, and integrate over the unit sphere with normalized positive weights.
At a quadrature point `n`, the source position is `x+L*n` and the vector
coefficient is `v=3*L^(-T)*n`. Spread the three-vector `F+B(v)*(T,S)` with the
quadrature weight. This uses mapped uniform unit-sphere measure, not uniform
ellipsoid surface-area measure. Its exact Fourier transform is the same
`Psi` above, including both the monopole and derivative shape factors.

Each particle caches four coefficients at each affected grid node: one
scalar and one three-vector. Gaussian spreading and its exact transpose
gathering enclose the native threaded radix-2/3 FFT. A positive Fourier
deconvolution factor, the same Gaussian physical filter and the Stokes
projector act on the retained physical modes. Thus

```
M_mesh = J^T F^* D F J + blockdiag(M_high,self),  D >= 0.
```

No isolated self term is subtracted from this operator. Reciprocity and
nonnegative power hold independently of the interpolation accuracy. The
same ellipsoid high-frequency tail is retained. Modes and mesh geometry use
the actual triclinic reciprocal lattice; grid-phase and lattice-basis errors
are bounded by the mesh approximation, rather than exact invariance.

Defaults are a 10-node Gaussian window per direction, exponent coefficient
0.6, a `2^p` or `3*2^p` grid exceeding twice each reciprocal-index bound, and
automatic shell quadrature order `max(12,ceil(0.65*kmax*a_max+8))` with twice
as many azimuthal points. These are numerical settings, separate from
`spectral_tolerance`. Three real vector fields are packed into two complex
FFT grids. There is no additional FFT library dependency. The bridge solver
still runs on MPI rank 0 with OpenMP; this is not a distributed MPI FFT.

The `N*K` coefficient array and self-block loop are absent from the mesh path.
For fixed shape, cutoff and density, a mobility product uses cached stencil
work proportional to N and FFT work proportional to `G log G`. Enumeration
of K physical modes remains proportional to K. Positive continuum isolated
self blocks serve only as preconditioners. `--mobility-backend direct`
retains the old sum for reference; unfiltered Fourier API tests explicitly
select this backend. New checks and timings are recorded in
[FFT_CONTACT_UPDATE.md](FFT_CONTACT_UPDATE.md).

## Earlier direct-sum model checks (2026-10-20)

The two previously troublesome AR20 poses were reused in a cubic cell
of side `100c`, with `c=150 nm`. The reference retained the periodic
Fourier modes without Gaussian filtering, up to `kmax*c=4`, and added
the analytic continuum self tail above that cutoff. This is a numerical
reference for the selected Galerkin model, **not** a full Stokes solution.
Increasing its cutoff from `3/c` to `4/c` changed the full middle
resistance by 0.00230% for the 45-degree, 1 nm pose and 0.03288% for the
30 nm slide pose.

The table uses the **same accepted near matrix** on both sides. Values
are differences from that reference, not inferred errors in bulk viscosity.

| Quantity | xi*a=2 | xi*a=4 | xi*a=8 |
|---|---:|---:|---:|
| Tilt: full combined resistance matrix norm difference | 7.03% | 2.51% | 0.724% |
| Tilt: relative-x force response difference | 11.01% | 3.64% | 1.02% |
| Tilt: relative-x torque response difference | 18.07% | 6.57% | 1.92% |
| Tilt: relative-x stresslet response difference | 18.83% | 6.35% | 1.78% |
| Tilt: common-xz-strain stresslet response difference | 5.68% | 2.00% | 0.564% |
| Slide: relative-x stresslet response difference | 0.636% | 0.407% | 0.192% |
| Slide: relative-x work difference | -7.80% | -4.77% | -2.13% |
| Slide: common-xz-strain stresslet response difference | 0.0238% | 0.0106% | 0.00394% |
| Retained half-lattice modes in this cell | 5,389 | 42,735 | 341,842 |

A small full-matrix difference can conceal a larger weak-mode difference;
the slide work row illustrates this. The default is a cost-conscious
finite-distance approximation. `--spectral 4` or `8` provides a direct
resolution/sensitivity comparison, at a substantially higher cost.

For an N108, AR20, phi=0.24 **kernel-only geometry**, `xi*a=2` retained
4,060 half-lattice modes and used 15.28 MB for the cached mobility data.
One serial measurement gave 0.0612 s setup and 0.0125 s per matvec.
This is not the time per LAMMPS timestep: near geometry, contact
iterations, Krylov iterations, communication, and output are additional.
The kernel benchmark positions are synthetic and are not a validated
suspension configuration. The actual integration report records its own
per-step timing and residuals.

## Tests and reproduction

The direct-backend C++ test checks reciprocity, positive eigenvalues of a representative
assembled matrix, image-shift invariance, integer lattice-basis changes,
joint cell/body rotations, parallel reduction, and the analytic sphere
self normalization. The invariance errors are around `1e-15`.

`tests/validate_periodic_mobility.py` independently reconstructs the
unsmoothed Fourier sum with NumPy/SciPy: the C++ relative discrepancy
was `1.16e-15`. It independently integrates the Gaussian and hard-cutoff
radial self factors. The hard-cutoff path is used for this numerical
reference, not selected by the production runner; its finite mode-entry
jumps are unsuitable for the default deformation path.

```
python lammps_slurry/tests/validate_periodic_mobility.py \
  --probe /path/to/slurry_periodic_mobility_test \
  --output periodic_mobility_validation.json
```

`--quick` skips the two expensive reference comparisons. NumPy/SciPy are
needed only for this optional independent validation; the LAMMPS runtime
uses native C++/Eigen. The two fixed near matrices and geometry are in
`tests/data/periodic_fixed_cases.json`; the full measured report is
`tests/data/periodic_mobility_validation.json`.
