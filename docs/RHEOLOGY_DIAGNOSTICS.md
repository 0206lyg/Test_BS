# Normal stresses and nonrattler coordination

The shared `slurry/bridge` fix always records three additional columns in
`history.csv`. They are automatically available through `run_baseline.sbatch`,
`run_pureGr.sbatch`, and `run_cmc.sbatch`; no recording switch or additional run
argument is needed. The existing run settings and numerical acceptance rules
are unchanged.

| Column | Definition | Position in `history.csv` |
| --- | --- | --- |
| `interval_mean_N1_normalized` | Time average of `(Sigma_xx - Sigma_zz)/(eta_m*gamma_dot)` | Immediately after `interval_mean_sigma_xz_Pa` |
| `interval_mean_N2_normalized` | Time average of `(Sigma_zz - Sigma_yy)/(eta_m*gamma_dot)` | Immediately after the new N1 column |
| `interval_mean_Z_nonrattler` | Time average of `2*C_NR/N_NR` after iterative rattler removal | Immediately after `contacts` |

Existing columns retain their names and meanings. Their numerical column
indices shift where the new columns are inserted. The bundled analysis tools
read columns by name. External analysis should do the same.

## Normal stress differences

The actual imposed velocity is `u_x = gamma_dot*z`: flow is x, gradient is z,
and vorticity is y. Thus `N1 = Sigma_xx - Sigma_zz` and
`N2 = Sigma_zz - Sigma_yy`.

The diagonal differences use the same hydrodynamic stresslets, contact force
moments, contact-attraction stress, and background-vdW stress already used by
the shear-stress calculation, with the same stage selection and unit
conversion. A Newtonian solvent in simple shear contributes no diagonal
difference; an isotropic pressure also cancels. No extra hydrodynamic solve or
pair-geometry calculation is performed.

`eta_m` is the medium viscosity actually passed to the run, including the
CMC-dependent value. `gamma_dot` is the physical imposed shear rate, not the
internal computational rate scale. Signed N1 and N2 are stored. Mari's
second-normal-stress plot uses `-N2/(eta_m*gamma_dot)`; apply that minus sign only
when reproducing the plot, not when reading the stored definition.

## Coordination and accepted substeps

The three launchers above use the practical integrator. For each accepted
substep, its corrected endpoint contact responses supply the mechanical
contact graph. Only `active` overlap contacts are included. A nonzero
noncontact attraction, background-vdW force, or lubrication interaction does
not create a mechanical contact. Net force cancellation does not remove an
otherwise active contact.

Particles with fewer than two contacts to the currently retained particles
are removed. Their removal can create new particles with fewer than two
contacts, which are removed in turn until no further removal is possible.
Both the numerator and denominator are then restricted to the retained graph:
`Z_NR = 2*C_NR/N_NR`. This is the iterative nonrattler convention cited by
Singh et al. (2020), through Seto et al. (2019). It is a reproducible contact
network statistic; it does not independently certify mechanical rigidity in
an adhesive, nonspherical suspension.

A queue processes newly removable particles rather than repeatedly scanning
the full graph. The existing contact loop supplies active edges; the graph
calculation requires O(N+C) work and memory for N particles and C active
contacts. Each distinct periodic-image contact remains a separate edge, and
a self-image contact contributes two to degree. Reusable graph buffers avoid
repeated per-particle allocations. The calculation is performed on the CSV
writer rank, after acceptance, once per accepted substep. It is not inside
linear/Newton iterations and is not performed for rejected trials. The graph
is a separate diagnostic object and never modifies solver contact lists or
contact histories.

## Time averaging and undefined values

All three values describe the same completed output interval bounded by
`interval_start_time_s` and `interval_end_time_s`. Accepted-substep samples
are weighted by their actual durations, and the prepared advance is added to
the output integral only when it is committed. Setup or a `run 0` does not
add unadvanced time. Average Z itself; do not replace this with the ratio of
averaged contact and particle counts.

The existing `contacts` column is a maximum over the prepared step's accepted
substeps. The new Z column is a completed-interval mean, as its name states;
`2*contacts/N` is not the new Z definition.

An empty output interval has NaN in all three columns. Zero shear produces
NaN in the normalized normal stresses. A contact graph with no retained
nonrattlers has undefined Z (`0/0`) and produces NaN. If any positive-duration
Z sample in an interval is undefined, that interval's Z is NaN; undefined
time is not silently dropped from the averaging denominator.

The definitions are also recorded under `history_rheology` in each newly
prepared run's `parameters.json`. For averaging several output intervals,
use their actual durations and select the desired steady-state strain window;
the diagnostics do not choose a steady state automatically.

## Applying an overlay

Extract the overlay at the repository root so its `lammps_slurry/`, `docs/`,
and `release.json` paths replace the matching files. Rebuild using the existing
`sbatch build_slurry_cpu.sbatch` and wait for successful completion before
launching a new run. Then submit the same run script as before. The source
and build guards deliberately reject an old binary after source changes.
An already running job uses its existing private runtime snapshot and does
not acquire new columns retroactively. Old history files do not contain the
diagonal stresses needed to reconstruct these N1 and N2 values exactly.

## Definition sources

- Mari et al., *Shear thickening, frictionless and frictional rheologies in
  non-Brownian suspensions*, J. Rheol. 58, 1693–1724 (2014), stress definition
  and Figs. 6–7: <https://arxiv.org/abs/1403.6793>.
- Singh et al., *Shear Thickening and Jamming of Dense Suspensions: The “Roll”
  of Friction*, Phys. Rev. Lett. 124, 248005 (2020), Fig. 1(c) and reference 21:
  <https://doi.org/10.1103/PhysRevLett.124.248005>.
- Seto et al., *Shear jamming and fragility in dense suspensions*, Granular
  Matter 21, 82 (2019), iterative nonrattler definition:
  <https://arxiv.org/abs/1902.04361>.
