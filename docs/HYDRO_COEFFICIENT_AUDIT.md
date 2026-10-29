# AR1 hydrodynamic coefficient audit — 2026-10-27

## Finding and decision

The audited equal-sphere limit has the correct leading squeeze, shear and
pump prefactors. The numerical difference from the Mari sphere reference
comes from explicit finite-gap modifications, not a missing factor in those
prefactors. No unambiguous hydrodynamic implementation error was established,
so this audit does **not** change production hydrodynamics.

This does not validate the composite model against exact Stokes flow. The
active model is a different approximation from Mari's diagonal Stokes drag
plus pair lubrication. Its matching accuracy still needs a separately
specified spherical reference. A solver change cannot establish that accuracy.

## Active path and definitions

`fix_slurry_bridge.cpp` calls `SlurryCompactBridge::evaluate` for every shape,
including AR1, and combines its sparse near resistance with the periodic
11-moment mobility in `slurry_periodic_mobility.h`. The separate Mari sphere
kernel in `slurry_pair_hydro_math.h`, reached by the sphere branch in
`slurry_spheroid_hydro_math.h`, is not this production path.

For equal spheres of radius `a`, viscosity `eta`, physical gap `h`, and
dimensionless regularization `delta`, define

```
x = max(h,0)/a + delta
w = compact_weight(max(h,0)/a)
ell_M = log(1/x)
ell_C = log(1 + 1/x)
```

The coefficient table below uses `a=eta=1`, `delta=0.001`, and `x<1`, where
`ell_M>0`. The separate Mari helper clips its logarithm to zero for `x>=1`.
Both reference pair operators are assigned an outer cutoff `h=a`
for this controlled comparison; this is not a claim that the paper fixes
that numerical outer cutoff. The compact taper is one through `h/a=0.1`
and smoothly reaches zero at `h/a=1`.

## Analytical sphere limit

Mari's Appendix A, Eqs. A7–A8, specifies the leading resistance functions
and their radius-dependent constants [1]. Restoring its force normalization
gives the following equal-sphere modal coefficients:

| Mode | Mari leading-sphere reference | Active compact model |
| --- | --- | --- |
| Normal squeeze | `3*pi*eta*a/(2*x)` | `w*3*pi*eta*a/(2*x*(1+x)^2)` |
| Tangential shear | `pi*eta*a*ell_M` | `w*pi*eta*a*ell_C` |
| Pump, relative transverse rotation | `3*pi*eta*a^3*ell_M/5` | `w*3*pi*eta*a^3*ell_C/5` |

Direct reduction of `detail::coefficients()` gives
`P=P'=Lambda=I/a`, `D=5I/a^2`, zero cubic terms and zero strain coupling
`GS`. Consequently `b=0`: the Schur positive-part completion is **zero
analytically for equal spheres**. It does not explain the AR1 discrepancy.
The completion can be nonzero for other geometries.

Expanding the surface-point levers gives translation–rotation coupling
magnitude `pi*eta*a^2*ell`, rotational diagonal `8*pi*eta*a^3*ell/5`,
and rotational cross coefficient `2*pi*eta*a^3*ell/5`. These agree with
the equal-radius specialization of Mari A8 before the finite-gap changes.
Rotation about the pair normal has no retained singular resistance.

Thus the normal ratio is exactly `w/(1+x)^2`. The logarithmic modes have
ratio `w*log(1+1/x)/log(1/x)`. Both tend to one as `x` tends to zero
inside the untapered region. At fixed `delta=0.001`, contact has the finite
normal ratio `1/(1.001)^2`, approximately `0.998003`; it is not exactly one.

The origin of the discrepancy is therefore identifiable:

1. `normal_scale^2` attenuates squeeze at finite gap.
2. `log1p` replaces the reference logarithm in shear and pump.
3. The physical-gap taper attenuates all near modes away from contact.

These choices are explicitly part of `slurry_compact_bridge.h` and
`COMPACT_BRIDGE_MODEL.md`. They are not required by the cited sphere paper.

## Numerical evidence

The independent audit evaluates production headers directly. Its checks
cover 81 pair cases: AR1/2/5, three regularizations (`1e-6`, `1e-3`,
`0.1`), and nine gaps including overlap, contact and the cutoff region.
The minimum eigenvalue normalized by `max(1,max(abs(eigenvalues)))` is
`-4.95e-16`; there is no negative value beyond roundoff. Common translation
has normalized residual at most `7.41e-17`.

At AR1 contact the 22-dimensional near pair has 17 zero modes. It is
positive **semidefinite**, not strictly positive definite. Independent
normal, shear, translation–rotation, rotational diagonal/cross and twist
formula checks at four gaps have maximum relative error `5.53e-14`.
The computed `b` is zero; the apparent completion is only `8.84e-15`
relative roundoff.

| `h/a` | Compact near normal coefficient | Mari near normal coefficient | Compact / Mari |
| --- | ---: | ---: | ---: |
| 0 | 4702.978321 | 4712.388980 | 0.998003 |
| 0.1 | 38.489753 | 46.657317 | 0.824946 |
| 0.5 | 2.518746 | 9.405966 | 0.267782 |

These are **near-pair coefficients**, not measured viscosity or total drag
errors. In particular, the 17.5% and 73.2% reductions cannot be transferred
to total resistance.

To demonstrate the distinction, the audit also prescribes identical
two-sphere kinematics: centers separated by `2+h` along z, velocities
`U_i=-z/2`, `U_j=z/2`, zero rotations and zero imposed strain. It reports
the quadratic power for unit relative speed. The current operator is
`R_current=M^-1+L_compact`; the reference is diagonal sphere drag plus
Mari leading lubrication. The reference drag contribution is `3*pi`.

The current model uses a periodic cube of side 6, `xi*a=2`, spectral
tolerance `1e-8`, tail quadrature order 256, and 9174 retained half-lattice
modes. The reference drag-plus-lubrication model has no periodic many-body
hydrodynamic contribution. No claim is made that either result is an exact
Stokes benchmark.

| `h/a` | Current middle power | Compact near power | Current total power | Mari drag + near power |
| --- | ---: | ---: | ---: | ---: |
| 0.1 | 31.378763 | 38.489753 | 69.868515 | 56.082095 |
| 0.5 | 27.129844 | 2.518746 | 29.648590 | 18.830744 |

The table uses direct reciprocal evaluation. The mesh implementation
agrees with these total powers within `5.5e-8` relative error. These checks
use the built-in FFT with one thread, not the HPC FFTW/OpenMP execution.
The smaller compact near coefficient does not make the composite total
smaller in this fixture. The two totals belong to different approximations;
their difference is not an error estimate against exact Stokes flow.

## Matching, positivity and limits

Production assembly adds `L_compact` to `M^-1` without an explicit pair
overlap subtraction. That observation alone does not prove double
counting: an additive correction can be a deliberately chosen composite
model. What has not been established is that its attenuation, taper and
omitted high-frequency cross-body mobility combine into an accurate
spherical resistance over finite gaps. The spectral tolerance controls
Fourier truncation, not that modeling error.

The mobility is a positive Fourier Gram operator plus local positive
self tails. Together with PSD near resistance this supports positive
hydrodynamic dissipation. It does not make the mixed saddle system or
frictional contact Jacobian SPD, and does not prove finite-gap accuracy.
The N500 failure at step 62 was not reconstructed by this audit; no
eigenvalue claim is made for that particular state.

If the requested spherical target is Mari LF-DEM, the entire baseline
and lubrication definition must match it. If periodic many-body HI is
intentional, AR1 may differ from Mari but must be checked against a
suitable spherical composite or Stokes reference. Replacing just one
near coefficient because it differs from Mari would not resolve that
choice consistently.

## Reproduction and sources

From the repository root:

```bash
g++ -std=c++17 -O1 -DNDEBUG \
  -I lammps_slurry/vendor/eigen3 -I lammps_slurry/src \
  validation_hydro_20261027/audit.cpp -o /tmp/slurry_hydro_audit
/tmp/slurry_hydro_audit
```

The checked source and recorded output are
`validation_hydro_20261027/audit.cpp` and `results.txt`.

1. Mari, Seto, Morris and Denn, *J. Rheol.* **58**, 1693 (2014),
   [DOI:10.1122/1.4890747](https://doi.org/10.1122/1.4890747),
   [author manuscript v4](https://arxiv.org/abs/1403.6793v4),
   Sec. II A 2 and Appendix A, Eqs. A1–A8. This defines the diagonal
   Stokes-drag plus leading-lubrication reference, not an exact all-range
   Stokes solution.
2. Singh, Ness, Seto, de Pablo and Jaeger, *Phys. Rev. Lett.* **124**,
   248005 (2020),
   [DOI:10.1103/PhysRevLett.124.248005](https://doi.org/10.1103/PhysRevLett.124.248005),
   [manuscript and supplemental material](https://arxiv.org/abs/2002.10996).
   Its Stokes drag and short-range lubrication context does not establish
   the finite-gap compact or periodic-mobility modifications audited here.
