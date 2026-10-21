# Compact bridge v4 regression fixture

`compact_bridge_v4.dat` was generated from the previously validated independent
NumPy/SciPy implementation, before evaluating the C++ port. It contains nine
complete 22-by-22 resistance matrices, not just selected scalar entries.
No fixture was replaced by C++ output.

Cases cover the central and tilted AR20 configurations, the previous slide
problem, the actual transition, compact-support termination, an actual 0.2 nm
gap with a 1 nm floor, a signed -0.2 nm overlap with the same floor, physical SI
scaling, and AR2. The overlap fixture checks that support geometry is retained
and only scalar gap functions are floored.

Each record contains its name; eta, reference length and floor; both bodies'
semiaxes, row-major rotation and center; gap, ell_n, ell_t, Lp, w, s; then the
row-major full matrix. The basis is (U, Omega, e=-E) per body with STF order
(diag(1,-1,0)/sqrt2, diag(1,1,-2)/sqrt6, xy, xz, yz). Off-diagonal STF basis
entries are 1/sqrt2. Output generalized moments act on the fluid.

The SI fixture is transformed by
R_SI = eta L^3 T^T R_nd T,
T=diag(1/L,1/L,1/L,1,1,1,1,1,1,1,1), repeated for both bodies.
The Schur positive part is computed before this unit transformation.

The test independently calls the repository's closestEllipsoidGap (rather than
injecting the reference gap or normal), checks the complete matrix, geometry,
body exchange invariance and PSD in nondimensional coordinates. These are
implementation regression and algebra checks, not exact-Stokes accuracy tests.

Reference prototype SHA-256 values from the preceding verification bundle:

- `validation/near/near_model.py`: `16e1a0f030d9fbf4dd97e779fe754e08604411bbe57fbf7234271415cc5441b5`
- `validation/patch_v2/finite_patch.py`: `6ab6a8497cd700c1a3ad8b0117979d543d463f6e87f53c90685392ae8a6f7cb0`
- `validation/patch_v4/compact_bridge.py`: `607e8ed199eb7ae8c68af0b55a02d6aa7b8e6f4aa420f18ffbf23f115d2b6b97`
