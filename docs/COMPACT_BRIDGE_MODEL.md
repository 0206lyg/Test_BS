# Compact bridge v4 implementation contract

`fix slurry/bridge` is a separate LAMMPS integrator. The near model accepted on 2026-10-20 is retained. The periodic middle approximation is documented separately in [PERIODIC_MIDDLE.md](PERIODIC_MIDDLE.md); it is not an exact periodic Stokes solver.

## Coordinates and units

For every particle the numerical variables are
`q=(U-u_inf, Omega-Omega_inf, -E_inf[STF5])` after nondimensionalization by the common polar semiaxis `c` and rate `max(abs(shear_rate),1/s)`. The 11 conjugate moments act **on the fluid**. In SI, force, torque/stresslet scales are respectively `eta*c^2*rate` and `eta*c^3*rate`. Shape semiaxes are `(AR*c,AR*c,c)` in the particle frame, with an arbitrary quaternion. The implementation permits different ARs at a common c.

The STF basis is orthonormal: `(xx-yy)/sqrt(2)`, `(xx+yy-2zz)/sqrt(6)`, symmetric xy, xz, yz, with each off-diagonal entry `1/sqrt(2)`. This basis is shared by the near matrix, periodic mobility, mixed solver, output, and stress calculation.

## Near connection

The exact convex support geometry computes the actual signed gap, contact points, normal, and local curvature. Only scalar gap functions use `he=max(h,hfloor)`, default 1 nm. No surface positions or curvatures are moved to the floor.

In the local 11-dimensional relative coordinates, the retained singular terms are `alpha=a0/he`, `v=b*log1p(ell_t/he)`, `C_log=C*log1p(ell_t/he)`. The normal component is separated from the remaining ten coordinates. The positive completion is

```
Delta = positive_part(v*v^T/alpha - C_log)
Hplus = [[alpha, v^T], [v, C_log+Delta]]
```

The completion is evaluated in the fixed nondimensional coordinates using `c`, not an arbitrary instantaneous contact length. It preserves the retained formal singular coefficients before applying the floor. The normal-normal subleading logarithmic coefficient remains omitted, as in the accepted model; this is not a complete reconstruction of every Claeys–Brady coefficient.

The contact-centred homothetic curvature ellipse is maximized inside both projected particle regions. Its depth is `ell_n`, its short semiaxis `Lp=sqrt(ell_n/lambda_max)`. The log reference remains `ell_t=min(c, all four local curvature radii)`. With `D_nn=ell_n/(he+ell_n)` and all other diagonal entries one,

```
Hbridge = w(he/Lp) D Hplus D
Rnear_pair = B^T Hbridge B
```

`w=1` up to 0.1 and zero from 1; between them the quintic is evaluated stably as `(1-t)^3*(1+3t+6t^2)`, `t=(he/Lp-0.1)/0.9`. This near correction contains no isolated-particle drag. The positive completion, ellipse, and switch are explicit finite-distance model choices. The geometric ellipse cusp and positive-part kink found in the previous audit are retained.

All images within `ai+aj+min(ai,aj)` are enumerated. Equal sizes give the proven near support radius `3a`. For `i<j` every admissible lattice vector is included; for `i=j`, one of ±image is included and its complete 22×22 matrix is assembled onto the same 11 unknowns. This preserves the congruence/PSD property. The center radius is a support of this near model, not a cutoff of physical hydrodynamics.

## Velocity and contact solve

The system solves `M*g_mid=q`, `q=P*u+qE`, and

```
P^T(g_mid + Rnear*q) = F_contact
```

The production solver retains three traction components per periodic contact in addition to the `11N+6N` hydrodynamic variables. It uses a force-variable semismooth Newton method with right FGMRES. This changes the nonlinear solution variables, not the contact law or the accepted near/middle model. There is no nested many-body mobility inverse.

For each contact, write the traction as `-N*n+T`, with tangential `T`, relative normal/tangential velocities `vn,vt`, and committed tangential spring history `xi`. The equivalent proximal equations are

```
N = max(0, N - rho_n*(vn + (gap-onset)/dt + N/(kn*dt)))
T = project_disk(T + rho_t*(vt + xi/dt - T/(kt*dt)), mu*max(N,0))
```

The normal and tangential scales are positive scalars. Each combines isolated-particle relative mobility with spring compliance: `rho_n=1/(n^T W n+1/(kn*dt))` and `rho_t=1/(trace((I-n*n^T)W)/2+1/(kt*dt))`. Zero stiffness or friction is handled explicitly before taking reciprocals. These equations have exactly the same roots as the frozen-geometry backward-Euler normal penalty and elastic/Coulomb return map. Retaining the contact force changes the off-root line search; it avoids forcing the search to locate a narrow sticking interval only in velocity space. It does not change the physical stiffness.

The preconditioner analytically eliminates each local three-component contact block, then factors the sparse `6N` Schur matrix `A+P^T D^-1 P`. For the mesh backend, `D` contains positive isolated continuum mobility blocks; the direct reference uses the discrete self blocks. This approximation is confined to the preconditioner and does not alter the physical mobility. The complete nonsymmetric contact derivative, including the normal-load dependence of friction, is retained. SparseLU handles this nonsymmetric matrix. The global mobility remains matrix-free, with one mobility product per Jacobian application.

The augmented nonlinear merit uses a fixed weighted norm. FGMRES uses the same norm and a bounded inexact-Newton forcing term, at most 0.05, to avoid oversolving early linear systems. Backtracking checks the actual augmented residual and its directional derivative; a Jacobian-transpose Cauchy step is available if the Newton line search fails. Linear hydrodynamic products are reused exactly during backtracking, and the full residual is reevaluated at every accepted iterate.

Acceptance is independently checked using the **original** contact return map and original hydrodynamic equations at the user's tolerance. The retained auxiliary tractions do not supply the reported contact forces or spring histories. A final call to the original formulation verifies or polishes that physical state. Physical residuals may temporarily increase while the augmented merit decreases; only the final physical residual controls acceptance. No continuation parameter or environment variable is required by the production path. The older stiffness-continuation implementation remains a lower-level regression/reference path and is not used silently.

The production default is the same linear contact law with `kn=100 N/m`, `kt=80 N/m`, `mu=0.1`, and `onset=3 nm`. A nonlinear endpoint correction wraps the frozen-geometry solves: trial Euler translation, quaternion rotation and lattice shear produce the actual endpoint gap. The inner contact gap is offset until its predicted compression agrees with that endpoint. Final force and normal energy are evaluated directly as `N=kn*max(onset-h_endpoint,0)` and `Un=N*N/(2*kn)`. Every correction uses the original committed tangential history. Both contactwise force consistency and the complete force/mobility residual must pass before a state is accepted. Hydrodynamics, normals and tangential velocity maps remain evaluated at the beginning of the first-order step. No Hertz law, stress activation barrier or position projection is introduced.

`solver_relative_residual` is the final endpoint-corrected physical-equation residual. Newton/Krylov counts and `mobility_applications` include all geometry corrections and final verification/polish. `SLURRY_BRIDGE_TRACE=1` prints `prox_*` and `polish_*` phases, physical `residual`, augmented `merit`, and timing. It changes no physical parameters. Nonfinite residuals, linear failures, an unlisted new endpoint contact, or failure to converge within 24 geometry corrections stop before committing that pending advance. Automatic timestep subdivision is not implemented.

The contact force acts at a common midpoint. Its velocity uses the actual center displacement and levers, including the affine contribution `E*d`. Each periodic contact has its own history. After a committed step its predicted physical image displacement identifies the history after periodic wrapping or a lattice flip; reversed self-image representatives reverse the normal and elastic vector. History is committed only during a genuine position advance, never during `run 0`, setup, or residual evaluations.

LAMMPS owns the periodic geometry through `fix deform 0 xz erate ... remap v`, defined before this fix. The bridge advances physical velocities and quaternions once; `remap x` is rejected because it would apply affine displacement twice. The current fix supports constant-volume xz shear, periodic 3D, SI ellipsoid atoms, and `pair_style zero`. It cannot be combined with another integrator or slurry force fix. Position/orientation updates are first order; implicit contact does not remove the need to resolve geometric motion with the timestep.

## Stress and power

The reported symmetric stresses are

```
sigma_H = -sum(S_on_fluid)/V
sigma_C = sum(sym(d_image * F_contact_on_i^T))/V
sigma = 2*eta*E + sigma_H + sigma_C
eta_app = sigma_xz / shear_rate
```

No extra hydrodynamic pair virial is added: the near strain-conjugate moment already carries its lever contribution. The code independently reports `stress_power=V*sigma_xz*shear_rate` and `hydro_dissipation + contact_power`. Contact power includes elastic energy change and can have either sign during relaxation; it is not frictional dissipation alone. `hydro_dissipation_W` includes the solvent term. Their difference is a power-balance diagnostic, not a Stokes-accuracy estimate.

Per-particle hydrodynamic force/torque and contact force/torque are output separately. Their sum should agree with the accepted force-balance residual. The raw force and mobility residual columns are norms in the stated nondimensional variables; they are not SI newtons or a relative error. The solver's accepted residual is measured in fixed preconditioned coordinates with the configured tolerance.

Time averages use every committed timestep. Initial output at step zero is not included in the average. `--average-start-strain` selects an averaging window; it does not establish stationarity. `final.data` stores geometry, not tangential contact history; restarting a physical trajectory from that file is not supported as a history-preserving restart in this version.
