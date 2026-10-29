# Predictor-corrector pair time-step validation

All nine runs completed successfully with the PC v2 binary, the same AR1/2/5 initial pair configurations as the previous linear tests, kn=0.3 N/m, kt=kr=0.24 N/m, mu_s=mu_r=10, and one thread. Each trajectory spans 120 microseconds, or strain 0.012. The time steps are 5, 2.5, and 1.25 microseconds. No model or solver source was changed.

**The 5-microsecond step is bounded but does not accurately resolve the early contact transient.** Total viscosity alone conceals this error because the contact stress is a small part of the total in these fixtures. The 2.5-microsecond step gives a substantially better contact-transient comparison. Neither result certifies dense or long-strain behavior.

Maximum errors are max_t|coarse-fine| divided by the corresponding maximum |fine| at matching physical times, not pointwise relative errors at vanishing signals.

| Comparison | AR | Total relative viscosity error | Contact stress error | Sliding energy error | Rolling energy error |
|---|---:|---:|---:|---:|---:|
|5 vs 2.5 microseconds|1|0.0012974%|0.01095%|0.03807%|noise level|
|5 vs 2.5 microseconds|2|0.01077%|11.09%|13.15%|30.51%|
|5 vs 2.5 microseconds|5|0.0098727%|33.52%|27.47%|11%|
|2.5 vs 1.25 microseconds|1|0.00033198%|0.003111%|0.04283%|noise level|
|2.5 vs 1.25 microseconds|2|0.0018079%|1.874%|2.203%|3.833%|
|2.5 vs 1.25 microseconds|5|0.0015025%|5.074%|4.14%|2.344%|

The late 60–120-microsecond mean contact-stress errors for 5 versus 2.5 microseconds are +0.00923%, -0.00582%, and -0.01296% for AR1/2/5. These small late errors do not erase the early transient discrepancies; newly formed contacts in a many-body trajectory can repeatedly excite such modes.

AR2 and AR5 have finite rolling energies: their fine-run peaks are about 1.590e-22 and 5.115e-22 J. AR1 remains the symmetric compression fixture, whose rolling energy is only around 1e-36 J; its fractional rolling error is not physically informative.

Actual current-geometry maximum surface overlaps at the 5-microsecond step are 2.476e-10, 2.078e-10, and 2.305e-10 m for AR1/2/5, respectively, or approximately 0.0062%, 0.0052%, and 0.0058% of the 4-micrometer reference radius. Endpoint overlap diagnostics are also finite, below 2.532e-10 m across these runs. Endpoint columns describe the next-step endpoint; they should not be compared between different dt at the same current-row time without aligning their separate endpoint timestamps. Current overlap columns are correctly aligned here.

All recorded solver relative residuals satisfy the specified 1e-6 tolerance. This is a short stability and temporal-resolution check, not a stiffness-independence or hard-particle-limit test. Compared with the previous Euler 1-microsecond result, the PC 2.5-microsecond maximum AR5 contact-transient error (5.07%) is similar in magnitude to Euler 1 versus 0.5 microseconds (4.35%). A speed preference should use measured work at comparable temporal error.

Raw histories, individual summaries, and comparisons are retained in this directory. comparison.json contains 5 versus 2.5 microseconds; comparison_refined.json contains 2.5 versus 1.25 microseconds.
