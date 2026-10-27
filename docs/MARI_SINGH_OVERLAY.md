# Mari/Singh convention overlay — 2026-10-27

Base: `0206lyg/Test_BS`, `main`, commit `4a0a1ea66e02e7d46d64d1bec9602e8cb141897e`.

This overlay removes the positive-gap mechanical contact layer and rough-surface hydrodynamic addition from the active spheroid runtime. It implements Mari-style additive lubrication regularization and Singh-style sliding/rolling contact. Existing nonspherical near and periodic hydrodynamic approximations remain; the result must not be described as an exact reproduction of either spherical paper.

## Fixed size and regularization

- `NOMINAL_DIAMETER_UM=8`: volume-equivalent diameter D, held fixed over AR.
- `LUBRICATION_EPSILON=0.001`: dimensionless offset relative to reference radius D/2.
- Actual offset is **4 nm**, independently of AR. It is not a physical shell that triggers contact.
- Spheroid semiaxes: a=(D/2) AR^(1/3), c=(D/2) AR^(-2/3); supported AR 1–5.
- Hydro scalar gap: max(actual signed gap,0)+epsilon*(D/2).
- Normal/sliding/rolling contact uses actual surface overlap, onset zero.
- Computational reference length and rolling reference lever are D/2, independently of AR.

Mari (2014) defines dimensionless gap as 2(r-ai-aj)/(ai+aj), and uses an additive delta=10^-3. For equal-radius particles this is h/a. Multiplying the nominal diameter by 10^-3 would instead give a factor-two larger offset. Normalizing nonspherical particles by their fixed volume-equivalent radius is the convention adopted here; it is not a unique consequence of the sphere paper.

## Contact law and editable settings

The batch script retains kn, kt, kr because finite spring compliance remains part of the numerical contact model. All three user-facing stiffnesses have units N/m. The rolling angular stiffness is kr*(D/2)^2; its torque cap is mu_r*Fn*(D/2). Default kn=100, kt=kr=80 and mu_s=mu_r=10 are strong-friction trial settings, not measured graphite properties or an established saturation limit.

Removed active inputs: fixed thickness, absolute gap floor, positive contact gap, rough-hydro enable/height/width/coverage, independent rolling lever and its selection mode, twisting friction/stiffness/lever. Old CLI options and old fix syntax are intentionally rejected. The active shape is spheroid; legacy cylinder/flake code remains only for historical/reference tests and is inaccessible from this launcher. Generic lower-level solver types still contain their legacy optional modes; the active fix supplies none of them.

No new repulsive-force barrier or dashpot is introduced. This matches the always-frictional contact branch relevant to a frictional jamming scan, rather than the entire stress-activated shear-thickening model. Mari (2014) and Mari (2015) differ in their use of dashpots; the latter has none. Singh's stated contact law uses normal, sliding and rolling springs. Singh's main text/SI does not establish a separate explicit numerical value for epsilon; 0.001 here is the verified Mari convention and the requested shared choice.

## Apply and run

The ZIP contains only changed/new paths relative to the base commit, including this documentation and focused regression tests. Apply it at the repository root after preserving any local edits that overlap these files:

```bash
unzip -o /path/to/slurry_mari_singh_overlay_20261027.zip
sbatch build_slurry_cpu.sbatch
```

After the build reports completion, edit the variables at the top of `run_slurry_cpu.sbatch`, then submit:

```bash
sbatch run_slurry_cpu.sbatch
```

Rebuilding is mandatory: the C++ fix syntax and rolling-stiffness meaning changed. Reusing an old binary with the new launcher is unsupported. The existing build still requires production FFTW and static oneMKL; no automatic slow fallback was added. The existing private runtime snapshot mechanism is retained. `parameters.json` records the actual axes, volume, normalization, offset and rolling mapping.

Default run: AR2, phi0.24, N500, 100000 steps, dt1e-6 s, shear100/s, viscosity0.01 Pa s. Initial packing certifies nonnegative physical surface gaps, not a positive roughness clearance.

The AR5 two-particle contact smoke fixture failed to converge at dt=1e-6 s and passed at dt=1e-7 s. For an AR5 trial, explicitly set `TIMESTEP=1e-7`; convergence still needs checking for the actual many-particle state. Set `STEPS=1000000` to retain the default total strain 10 at shear100/s. Keeping 100000 steps after this timestep reduction would only reach strain1, before the default averaging start strain5. No automatic timestep adaptation or hidden retry is performed.

## Validation and remaining physical work

See `validation_mari_singh/REPORT.md` for executed commands/results and backend limits. The new focused tests check additive regularization below the offset, finite shallow-overlap resistance, geometry at AR1/2/2.09/3/5, zero contact onset, sliding/rolling limits, rolling stiffness conversion, signed endpoint compression, volume invariance and rejection of retired options.

A contact-containing short integration test establishes code execution and finite residuals only. It does not determine phi_J, validate eta_r=100, prove strong-friction saturation, validate the absolute rheology of this spheroid hydrodynamic approximation, or establish long-run stability at N500. Next physical runs should compare mu=(10,10) against larger values and jointly check timestep, spring stiffness, overlap relative to both reference radius and local particle thickness, steady-state strain and finite-size effects before fitting phi_J(AR). Values or slopes from the sphere papers are not guaranteed for this extension.

## Primary references

- Mari et al., *J. Rheol.* 58, 1693 (2014), DOI [10.1122/1.4890747](https://doi.org/10.1122/1.4890747); [author manuscript](https://arxiv.org/abs/1403.6793). Dimensionless gap/additive cutoff, sliding springs and stiffness convergence.
- Mari et al., *Phys. Rev. E* 91, 052302 (2015), DOI [10.1103/PhysRevE.91.052302](https://doi.org/10.1103/PhysRevE.91.052302); [author manuscript](https://arxiv.org/abs/1412.4661). Version without contact dashpots.
- Singh et al., *Phys. Rev. Lett.* 124, 248005 (2020), DOI [10.1103/PhysRevLett.124.248005](https://doi.org/10.1103/PhysRevLett.124.248005); [manuscript and SI](https://arxiv.org/abs/2002.10996). Rolling quasi-force spring and reduced radius aij=2ai aj/(ai+aj).
