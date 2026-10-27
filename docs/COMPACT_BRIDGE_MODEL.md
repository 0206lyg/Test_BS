# Spheroid bridge implementation contract — 2026-10-27

The active `fix slurry/bridge` uses Mari-style additive lubrication regularization and the Singh sliding/rolling contact law, extended to oblate spheroids. Its compact near hydrodynamics and periodic ellipsoid mobility remain approximations; this is not the complete spherical LF-DEM operator of either paper. See [MARI_SINGH_OVERLAY.md](MARI_SINGH_OVERLAY.md) for provenance, scope and validation.

## Geometry, coordinates and units

The supported shape is an oblate spheroid with AR=a/c between 1 and 5. The nominal diameter D is interpreted as volume-equivalent: a=(D/2) AR^(1/3), c=(D/2) AR^(-2/3). Thus V=pi D^3/6 is independent of AR. Default D=8 micrometers. This diameter interpretation is a modeling choice; it does not infer a three-dimensional size distribution from a projected image.

The fixed normalization length is L=D/2, not the polar semiaxis. The rate scale is s=max(abs(shear_rate),1/s). Each particle has q=(U-u_inf, Omega-Omega_inf, -E_inf[STF5]) in normalized coordinates. The 11 conjugate moments act on the fluid. Force, torque/stresslet and power scales are eta L^2 s, eta L^3 s and eta L^3 s^2. The code checks that every particle has the prescribed volume-equivalent diameter.

The STF basis is orthonormal: (xx-yy)/sqrt(2), (xx+yy-2zz)/sqrt(6), symmetric xy, xz, yz, with each off-diagonal entry 1/sqrt(2). The same basis is used throughout the hydrodynamic solve and stress calculation.

## Regularized near hydrodynamics

Convex support geometry computes the signed surface gap h, contact points, normal and local curvature. Negative h is the shallow penetration used by the finite-stiffness contact penalty. Neither the geometry nor the contact onset is shifted by a roughness layer.

Scalar resistance functions use

```
ell_reg = epsilon * (D/2)
h_eff = max(h,0) + ell_reg
```

Default epsilon=0.001 gives ell_reg=4 nm at D=8 micrometers. Mari's dimensionless gap is 2h/(a_i+a_j); equal reference radii therefore use h/(D/2), not h/D. The additive rule differs from the old hard floor max(h,h_floor), including for positive gaps smaller than ell_reg. During shallow overlap, scalar resistance stays at its contact limit while actual geometry is retained.

In the local 11-dimensional relative coordinates, retained terms are alpha=a0/h_eff, v=b log1p(ell_t/h_eff), C_log=C log1p(ell_t/h_eff). The normal coordinate is separated from the other ten coordinates. Positive completion is

```
Delta = positive_part(v*v^T/alpha - C_log)
Hplus = [[alpha, v^T], [v, C_log+Delta]]
```

This finite completion uses the fixed reference L=D/2. Changing L changes that finite model, so the shift from the previous L=c is an intentional model change as well as a normalization change. The normal-normal subleading logarithm remains omitted.

The contact-centred curvature ellipse fits inside both projected particles. Its depth is ell_n and its short semiaxis is Lp. The log reference is ell_t=min(polar semiaxes, local curvature radii). With D_nn=ell_n/(h_eff+ell_n) and other diagonal entries one,

```
Hbridge = w(max(h,0)/Lp) D Hplus D
Rnear_pair = B^T Hbridge B
```

The quintic switch w is one below 0.1 and zero from 1. Its support uses the actual nonnegative gap, so regularization does not move the outer geometric support. Pair matrices are symmetric positive semidefinite by construction. All periodic images within ai+aj+min(ai,aj), where ai and aj are the largest semiaxes, are enumerated; one representative of each self-image pair is included. This is support of the near correction, not a cutoff of all hydrodynamics.

The near model retains the previous surface-point moment maps, finite-distance completion, curvature patch and taper. Coordinate-rotation covariance is tested; these tests do not prove an exact finite-gap rigid-body null mode or an exact Stokes solution. Periodic middle hydrodynamics is documented in [PERIODIC_MIDDLE.md](PERIODIC_MIDDLE.md).

## Surface sliding and rolling

Contact onset is h=0. For the actual end-of-step gap,

```
Fn = kn * max(-h_endpoint,0)
|Ft| <= mu_s * Fn
```

No positive-gap shell, rough hydrodynamic excess, repulsion barrier, contact dashpot or twisting spring is active. Sliding uses the existing tangential displacement spring and Coulomb return map. Its numerical stiffness kt has units N/m. At noncompressive separation all sliding and rolling history is discarded. An implicitly predicted overlap can create a contact within a step that starts at positive gap; it is the endpoint overlap that supplies the normal load.

Singh's rolling quasi-force spring has stiffness kr in N/m. Let aij=2*a_ref_i*a_ref_j/(a_ref_i+a_ref_j). Here every reference radius is D/2, hence aij=D/2 independently of AR. The angular-history implementation uses

```
K_theta = kr * aij^2
M_trial = K_theta * theta_tangent
|M_roll| <= mu_r * Fn * aij
```

This is equivalent to the quasi-force rolling representation for a frozen contact normal. Relative angular motion parallel to the normal produces no rolling torque. The two particles receive opposite resisting couples. Existing objective history transport follows common rotation and changes of contact normal. Using volume-equivalent aij for nonspherical particles is an explicit extension of the sphere law, not a shape-friction mapping established by the cited papers.

Default kn=100, kt=80, kr=80 N/m; mu_s=mu_r=10. These are trial numerical parameters. They are neither measured graphite coefficients nor demonstrated infinite-friction limits for every AR. Stiffness, timestep and friction saturation require separate convergence tests. The launcher permits mu_s=0 and/or mu_r=0 to disable either friction branch.

## Coupled solve and integration

The system solves M*g_mid=q, q=P*u+qE and P^T(g_mid+Rnear*q)=F_contact. When rolling is enabled, the retained contact unknown is a six-component force/couple wrench. The solver uses semismooth Newton with matrix-free right FGMRES and local contact-block condensation. The production sparse preconditioner uses oneMKL PARDISO; a reference Eigen backend exists for tests. Physical acceptance is checked against the original contact return map and hydrodynamic equations, not only the augmented residual.

A nonlinear endpoint correction updates the contact compression to agree with trial Euler translation, quaternion rotation and lattice shear. Fn and normal energy are evaluated from that actual endpoint gap. Every correction starts from the original committed history. Hydro matrices, normals and sliding maps remain evaluated at the beginning of the first-order step. No position projection or automatic timestep subdivision is introduced. Failure of force/mobility convergence, gap consistency or unlisted new-contact detection aborts the advance rather than silently accepting it.

Each periodic contact keeps its own history. Predicted physical image displacement identifies history after wrapping or a lattice flip. History is committed only on a genuine position advance, not during `run 0`, setup or residual evaluation.

LAMMPS owns periodic geometry via `fix deform 0 xz erate ... remap v`, defined before this fix. Supported runs are constant-volume xz shear, periodic 3D SI ellipsoid atoms and `pair_style zero`. This fix advances positions and orientations itself and must not be combined with another integrator.

## Stress, power and output

```
sigma_H = -sum(S_on_fluid)/V
sigma_C = sum(sym(d_image * F_contact_on_i^T))/V
sigma = 2*eta*E + sigma_H + sigma_C
eta_app = sigma_xz / shear_rate
```

No extra near hydrodynamic virial is added. Contact couples enter rotational balance and contact power. The symmetric shear stress uses the force virial above. `stress_power_W` and `hydro_dissipation_W + contact_power_W` are reported independently; their difference is a consistency diagnostic. Contact power includes elastic storage/release and is not purely plastic dissipation.

The CSV retains normal and sliding elastic energies and adds/retains rolling yield count, rolling elastic energy, rolling plastic dissipation and rolling contact power. Obsolete rough-hydro and twisting columns are removed. Force and mobility residual norms use the stated nondimensional coordinates, not SI newtons. Solver acceptance uses its configured relative norm.

Averages include every committed timestep, excluding step-zero output. Selecting an averaging window does not establish stationarity. `final.data` stores geometry, not friction histories, so it is not a physical history-preserving restart.
