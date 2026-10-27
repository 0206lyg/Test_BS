# Explicit pair time-step comparison

All six runs completed successfully using the same initial particles for each AR, the O3 linear solver, one thread, kn=0.3 N/m, kt=kr=0.24 N/m, and sliding/rolling friction coefficients 10. Coarse dt=1 microsecond (120 steps) and fine dt=0.5 microsecond (240 steps) cover the same 120-microsecond interval, or strain 0.012 at 100 s^-1. Hydrodynamics and all other inputs are unchanged.

The maximum error below means max_t|coarse-fine| / max_t|fine| over 121 matching physical times. This avoids meaningless pointwise division by zero at initially uncharged springs. The late window is 60–120 microseconds, a transient comparison window rather than a certified steady state.

| AR | Relative viscosity max error | Contact shear stress max error | Sliding energy max error | Rolling energy max error | Late mean contact stress error |
|---|---:|---:|---:|---:|---:|
|1|2.5243e-05%|0.2006%|0.0109%|noise level|-0.1777%|
|2|0.002228%|2.348%|2.751%|3.12%|0.01934%|
|5|0.0013158%|4.352%|3.66%|3.138%|0.322%|

AR5 has a finite rolling response: fine peak rolling energy is 5.115e-22 J, and its late mean rolling-energy discrepancy is 0.000466%. AR1 remains a symmetric normal-compression fixture: its rolling energy is only 1.18e-36 J, so its large fractional rolling discrepancy measures numerical noise rather than a meaningful rolling test. AR2 also exercises finite sliding and rolling.

The largest early contact-stress discrepancy is 4.352% (AR5); the largest finite rolling-energy discrepancy is 3.138% (AR5). These transients are less tightly resolved than total viscosity. The maximum total relative-viscosity discrepancy is 0.002228% (AR2), while the late mean contact-stress discrepancy is at most 0.3220% (AR5).

These are successful short pair stability and temporal-refinement checks. They do not establish stiffness independence, a hard-particle limit, dense-system stability, or long-strain rheology. No model defaults or solver source were changed. Detailed absolute errors, RMS errors, endpoint values, and complete run summaries are in comparison.json.
