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

Contact onset is h=0. For the current-state gap,

```
Fn = kn * max(-h_current,0)
|Ft| <= mu_s * Fn
```

No positive-gap shell, rough hydrodynamic excess, repulsion barrier, contact dashpot or twisting spring is active. Sliding uses the existing tangential displacement spring and Coulomb return map. Its numerical stiffness kt has units N/m. At noncompressive separation all sliding and rolling history is discarded. A new overlap becomes a contact at the next velocity stage, with initially unstretched sliding and rolling springs. No future overlap enters the current velocity solve.

Singh's rolling quasi-force spring has stiffness kr in N/m. Let aij=2*a_ref_i*a_ref_j/(a_ref_i+a_ref_j). Here every reference radius is D/2, hence aij=D/2 independently of AR. The angular-history implementation uses

```
K_theta = kr * aij^2
M_trial = K_theta * theta_tangent
|M_roll| <= mu_r * Fn * aij
```

This is equivalent to the quasi-force rolling representation for a frozen contact normal. Relative angular motion parallel to the normal produces no rolling torque. The two particles receive opposite resisting couples. Existing objective history transport follows common rotation and changes of contact normal. Using volume-equivalent aij for nonspherical particles is an explicit extension of the sphere law, not a shape-friction mapping established by the cited papers.

Default kn=0.3, kt=0.24, kr=0.24 N/m; mu_s=mu_r=10. These are trial numerical parameters. They are neither measured graphite coefficients nor demonstrated infinite-friction limits for every AR. Stiffness, timestep and friction saturation require separate convergence tests. The launcher permits mu_s=0 and/or mu_r=0 to disable either friction branch.

## Coupled solve and integration

Production uses explicit Heun predictor–corrector. Each stage solves M*g_mid=q, q=P*u+qE and P^T(g_mid+Rnear*q)=F_external+F_contact with known stage geometry and histories. The predictor rebuilds the same compact near operator and periodic FFT mobility. Two matrix-free mixed FGMRES solves replace nonlinear contact/endpoint iterations. The production preconditioner uses MKL-PARDISO; the local reference uses Eigen.

Translation includes the average total (affine plus nonaffine) velocity; quaternion rotation uses the average angular velocity. Corrected sliding/rolling histories start from the original committed history, with predicted rates transported into its contact frame. Only the corrected history is committed. Instantaneous output velocities remain distinct from the prepared average integration velocities. No position projection or hidden physical substeps are introduced.

The default time step is 2.5e-6 s. The numerical spring constants and time step were selected together using overlap, temporal refinement and a matched single-thread timing comparison. See [the stability report](CONTACT_STABILITY_UPDATE.md) for scope and limitations.

## Stress, power and output

```
sigma_H = -sum(S_on_fluid)/V
sigma_C = sum(sym(d_image * F_contact_on_i^T))/V
sigma = 2*eta*E + sigma_H + sigma_C
eta_app = sigma_xz / shear_rate
```

No extra near hydrodynamic virial is added. Contact couples enter rotational balance and contact power. The symmetric shear stress uses the force virial above. `stress_power_W` and `hydro_dissipation_W + contact_power_W` are reported independently; their difference is a consistency diagnostic. Contact power includes elastic storage/release and is not purely plastic dissipation.

The CSV reports current normal, sliding and rolling energies and current contact power at time_s. Endpoint gaps and rolling projection count/work refer to the prepared next step identified by next_step_end_time_s. rolling_plastic_dissipation_W is corrected projection work divided by that step duration; next_step_rolling_projection_work_J reports the work directly. contact_gap_consistency_m is NaN because implicit endpoint-force consistency is not imposed. Linear gap defect is computed from actual corrected geometry. Solver residual diagnostics take the larger of both linear stages. Newton and continuation counters are zero.

Averages include every committed timestep, excluding step-zero output. Selecting an averaging window does not establish stationarity. `final.data` stores geometry, not friction histories, so it is not a physical history-preserving restart.
