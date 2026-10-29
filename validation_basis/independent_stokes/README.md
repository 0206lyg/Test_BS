> New AR2/5/10/20 distance scan: see `../../RUN_DISTANCE_SCAN_KO.md`.
> The material below documents earlier reference/table work. Use the new root guide for this campaign.

# Independent Stokes reference and provisional table workflow

The user authorized using the existing 0.753223% boundary-residual result for
end-to-end development before further refinement. The new
[provisional workflow](PROVISIONAL_PIPELINE.md) has imported the actual AR20
matrix, generated a 222-gap-node exact-pose table over1–150 nm, and checked
force, torque and intrinsic stresslet through the runtime. The gap rule is
`max(actual smooth-surface gap,1 nm)`. This is numerical transport validation,
not narrow-gap Stokes or rheology qualification. A new explicitly selected
dissipative-increment closure replaces the indefinite constant-residual
candidate in this trial; its finite-gap physical accuracy remains unknown.

Generic AR, size and orientation arguments and a complete-grid input path
are implemented. No new other-AR Stokes solutions or actual complete angular
table have been produced. The following preserves the stricter reference
convergence record and direct-solver reproduction instructions.


Current Korean development/validation status: [AR20_INDEPENDENT_STATUS.md](../AR20_INDEPENDENT_STATUS.md). The corrected 9963-node calculation at 150 nm completed in 1224.97 seconds, but remains unqualified: maximum independent no-slip error is 0.753223% globally and 0.229996% on the gap patch, above the unchanged 0.1% criterion. See the [completed report](results/pair_h150_p8_anisotropic_chunked.json), [validation summary](results/pair_h150_p8_anisotropic_chunked_validation_summary.json), and [preserved initial OOM attempt](results/pair_h150_p8_anisotropic_patched.json).

Its small linear residual (`1.44e-13`), reciprocity errors, and integrated
matrix changes do not override the failed boundary check. The actual
153-value encoding roundtrip passes at `2.66e-12` relative energy error;
this remains an algebra-only result. A tighter-quadrature probe at 36
collocation sites gives at most `2.16e-9` scaled error, while off-grid errors
persist. Inadequate between-node density resolution is therefore a plausible
next refinement target, not a proved global error explanation.

The same-pose p8 refinement has 159 panels and 12879 nodes. The out-of-core
v2 attempt completed the entire assembly in 932.935845 seconds, then stopped
before LU: 1.211 GB headroom was below the estimated 2.279 GB requirement.
The v3 retry, with cache-bounded hashing, an early maximum-stage memory guard
and 512/256 LU blocks, stopped at entry: 0.668 GB was available versus
0.830 GB required. Neither attempt produced a new physical density, matrix
response or boundary result. The current roughly 9 GB free work disk is
also below the approximately 12.44 GB requirement including reserve.
See the [v2 failure](results/pair_h150_p8_159_ooc_buffered_retry.json),
[v3 failure](results/pair_h150_p8_159_ooc_v3_resource_retry.json), and
[storage implementation/reproduction](OUT_OF_CORE_SOLVER.md).
The [2026-10-19 attempt summary](results/resumed_ar20_attempt_20261019.json)
links the attempts and their source hashes. The memory guard credits only
clean inactive file cache; its refusal is not proof that an 8 GiB allocation
cannot hold the bounded working arrays. Cache advice is not guaranteed eviction.

At a small actual 300-node mesh of this same pose, v3 preserves v2's original
A bitwise; all 17 density responses differ by `1.09449e-14`, raw17 by
`2.72680e-14`, and the recomputed original-operator residual is `4.08190e-14`.
These are [storage-equivalence checks](results/ooc_v3_assembly_equivalence.json),
not physical convergence. Latest Python validation ran 49 tests: 48 passed
and one optional benchmark was skipped. Earlier p7/COW and OOM failures
remain preserved, including the [10176-node retry](results/pair_h150_p7_159_cow.json).
No matching anchor or 1 nm response is qualified.

A saved-density coarsening screen preserved all 24 high-error regions. With
a `2e-4` jump-change indicator limit, only 159 to 157 panels was supported,
still requiring an 11.644 GB matrix. Reaching at most 138 panels required
about `0.008506` in this candidate set and was not adopted. This indicator
is not a velocity-error bound or a new solve; see the
[coarsening audit](results/pair_h150_p8_coarsened_hotspots_preserved_projection_audit.json).
A separate FMM `+48` far-order probe at 300 nodes measured `2.82263e-7`
operator-action error on the first saved RHS, then stopped after unexpected
native RSS near 1.77 GiB. It did not check all 17 RHS or solve a new system;
the 512 MB light-probe estimate was inadequate, but this does not establish
that the large FMM case is impossible. See the
[partial action probe](results/saved_n300_fmm48_action_comparison.json).

The production operator implements all 16 independent leading resistance
entries actually listed in Janoschek Appendix A; a source/formula audit and
entrywise tests at the recorded AR20 pose pass. The paper itself intentionally
omits the K33 subleading logarithm, so this is a limit of the adopted
approximation rather than an omitted Appendix formula. It is not an
all-divergent-terms expansion, and a constant anchor residual does not recover
every omitted or finite-gap term. The extension from the paper's local six
velocity coordinates to full17 strain/stresslet coordinates has algebraic
energy consistency tests, but still requires independent physical validation.
Neither these checks nor 153-value storage establish 1 nm accuracy.
See the [Appendix audit](../JANOSCHEK_APPENDIX_A_AUDIT.md).

This directory develops an independent boundary-integral reference for the
single AR20 configuration that failed the existing near-gap source checks.
The purpose is to qualify the finite-gap resistance used to match the analytic
lubrication model. A successful executable, a small linear-system residual, or
a positive matrix does not establish physical accuracy.

The direct-reference driver does not generate tables. A separate explicitly
provisional import/table/runtime workflow now consumes completed reports; see
PROVISIONAL_PIPELINE.md. Strict reference convergence remains distinct from
this development workflow, and old production checkpoints are not reused.

The independent path uses analytic spheroid surface panels, singular and
near-singular quadrature from fmm3dbie, and a combined single/double-layer
representation. Its density is not surface traction: force, torque, and
intrinsic stresslet must be recovered with the representation's moment
identities and checked against isolated spheroids. Dense assembly supports
small reference experiments; `matrix_free.py` provides a near-corrected FMM
operator for larger refinements. Agreement between these implementations
checks their assembly, while independent surface residuals and refinement
checks determine whether the pair response is accurate. No pair convergence
or matching-anchor qualification is asserted by this README.

## Fixed configuration and units

| Quantity | Value |
| --- | --- |
| Aspect ratio | 20, oblate spheroids |
| Polar semiaxis | 150 nm |
| Equatorial semiaxis | 3000 nm |
| Internal length unit | 150 nm; hence `c=1`, `a=20` |
| First director angle | `0.0049245303592113559` radians |
| Second director angle | `0.041010932014513654` radians |
| Relative azimuth | `1.7016960206944713` radians |

The lower closest point is `(0,0,0)` and the upper closest point is `(0,0,h)`
in the canonical frame. Centers follow the exact spheroid support geometry;
the center-to-center vector is not the contact normal.

Reference gaps are actual smooth-surface separations. The reference solver
does not add 0.15 nm, subtract the 3 nm mechanical contact onset, or silently
apply the production 1 nm lubrication floor. The historical failed case used
75 nm (`h/c=0.5`). Every candidate reference gap and refinement must be selected
explicitly at this same pose; no particular gap is presumed to be a valid
matching anchor.

The separate [exact-gap geometry diagnostic](results/ar20_exact_gap_geometry.json)
does not solve Stokes flow. At 75 nm, its local quadratic contour
`0.5*x^T*H*x=h` extends outside at least one projected particle silhouette for
304 of 720 sampled angles. A vertical gap between both surfaces is undefined
at those samples. This is a limitation of that geometric footprint; it is
not a force-error estimate, a hydrodynamic convergence result, or evidence
that a particular matching anchor is acceptable.

The full resistance uses `[U_i, a*Omega_i, U_j, a*Omega_j, a*E_5]`: two
three-component translations, two three-component angular velocities scaled
by `a`, and five orthonormal strain components scaled by `a`.
Keep all 17 modes, including common translation and
rotation. An eleven-coordinate reduction cannot retain the measured
common-motion shielding. Force, torque, and intrinsic stresslet conventions
must match the comparator before accepting a matrix difference.

## Dependency pinning

The external boundary-integral implementation is
[fmm3dbie](https://github.com/fastalgorithms/fmm3dbie), with its FMM3D submodule:

| Source | Pinned commit |
| --- | --- |
| fmm3dbie | `be6b2b3700b77e9cbf20c512ff4f0922aaee45d9` |
| FMM3D | `d2b5e6e983e1ec915d15330ce74df44221a9ac42` |

Keep these dependencies outside this repository. Use a recursive checkout and
verify both commits before building. The direct Fortran interface requires a
Fortran compiler, C/C++ compiler, `make`, BLAS/LAPACK, and OpenMP; no upstream
Python wrapper is required. Compiler and library paths belong in each run's
provenance. The reference driver itself uses Python, NumPy and SciPy.

From the repository root, build into an external directory:

```bash
module purge
module load anaconda/2024.02-py311
module load gcc/12.2.0
export REFERENCE_DEPS="/scratch/negishi/$USER/stokes_reference_deps"
bash validation_basis/independent_stokes/build_reference_backend.sh \
  --prefix "$REFERENCE_DEPS" --jobs 4
```

`FC`, `CC`, `CXX`, and `PYTHON` override the build tools. `REFERENCE_BLAS`
overrides the default `-lblas -llapack` linker flags when the cluster requires
an explicit BLAS/LAPACK installation. The build produces
`$REFERENCE_DEPS/fmm3dbie/lib/libfmm3dbie.so` and
`$REFERENCE_DEPS/reference_backend_manifest.json`. Keep the manifest with the
reference results. Compilation and load/symbol smoke checks qualify the
installation only.

Use a fresh dependency prefix when another reference calculation is active.
The build deliberately rebuilds library objects; do not replace an active
installation underneath a running experiment.

The development build also ran upstream examples. The coarse exterior
example completed with a small GMRES residual but about 1.6% off-surface
error. This is installation evidence, not AR20 accuracy evidence.

The documented high-order quadrature patch has a separate builder under
`patches/build_quadparams_library.py`. It checks the pinned upstream source,
copies the existing archive into a new prefix, replaces one specified object,
and writes `patch_manifest.json`. It leaves the original source and active
library unchanged. The patch manifest records the patch/source/library hashes
and supported panel-order range; it is implementation provenance, not an
accuracy qualification. Select its library explicitly when testing that
backend, and retain the patch manifest with the run.

After the pinned upstream build above, create the separate patched library
from the repository root:

```bash
export PATCHED_REFERENCE_DEPS="/scratch/negishi/$USER/stokes_quadparams_highorder_v1"
python3 validation_basis/independent_stokes/patches/build_quadparams_library.py \
  --upstream "$REFERENCE_DEPS/fmm3dbie" \
  --output-prefix "$PATCHED_REFERENCE_DEPS" \
  --fc "$(command -v gfortran)"
```

The output prefix must not already exist. The selected library is then
`$PATCHED_REFERENCE_DEPS/lib/libfmm3dbie.so`, with the adjacent-prefix
`patch_manifest.json`. Its driver permits panel order 4 or greater. Read
[`patches/README.md`](patches/README.md) for the precise selector change,
precision bins, limitations, and policy probe. Do not treat two requested
tolerances within one backend precision bin as an independent quadrature
refinement.

## Run one gap on CPU

Submit from the repository root. The runner uses the existing Negishi account
and CPU partition defaults; override Slurm resource/account flags as required
by the allocation. The following is an initial two-resolution experiment,
not a recommendation that these resolutions are sufficient:

```bash
sbatch --cpus-per-task=8 --mem=32G \
  validation_basis/independent_stokes/run_reference_cpu.sbatch \
  --library "$REFERENCE_DEPS/fmm3dbie/lib/libfmm3dbie.so" \
  --gap-nm 75 --profiles 4:0,4:1 \
  --quad-tol 1e-7 --max-memory-gb 6 \
  --output-dir "/scratch/negishi/$USER/ar20_reference_75nm_trial1"
```

`--profiles` contains comma-separated `order:level` pairs. Every profile uses
the same recorded AR20 pose and the same supplied gap. If omitted, `4:0` is a
candidate setup only. The runner never advances to another gap or AR and never
generates a lubrication table. It stops after the first failed process and
preserves the complete run directory. A new or empty output directory is
required so that an earlier attempt cannot be overwritten.

For the exact isolated geometry of either body at this same AR, add
`--isolated-body 0` or `--isolated-body 1` and use a separate output directory.
This is an isolated-body diagnostic, not a second aspect-ratio campaign.

The equivalent direct invocation for one profile is:

```bash
python3 validation_basis/independent_stokes/run_reference.py \
  --library "$REFERENCE_DEPS/fmm3dbie/lib/libfmm3dbie.so" \
  --gap-nm 75 --order 4 --level 0 --quad-tol 1e-7 \
  --max-memory-gb 6 --output /path/to/new/reference_75nm_o4_l0.json
```

Additional controls are `--check-points-per-patch` (default 3), `--alpha`
(default 1), `--beta` (default 1), and `--focus` (default 0). Local mesh
refinement uses `--rim-levels` with `--rim-width` (default 0 and 0.05), and
`--contact-levels` with `--contact-radius` (default 0 and 0.25). Rim refinement
requires `--focus 0`. The runner forwards these mesh controls to every profile
at the same gap. Vary them deliberately when separating global, rim, and
contact-patch resolution errors.

The dense implementation checks its memory budget; increasing Slurm memory alone does not change
`--max-memory-gb`. Raising a resolution is not a substitute for verifying
that the density and near-singular quadrature have both converged.

`--solver dense` is the default. `--solver mmap-dense` uses exact quadrature
in bounded target blocks, file-backed dense assembly, and LU to reduce
simultaneous resident copies, with the same physical operator. The block
assembly avoids the full native quadratic quadrature workspace that caused
the initial 9963-node attempt to exceed the cgroup limit. Agreement at a
smaller mesh checks this implementation change, not pair physical accuracy.
The default `--mmap-factor-mode shared` stores LU in a second file. Optional
`--mmap-factor-mode cow` factors a private copy-on-write mapping of the original
operator, avoiding that second file and its repeated disk writes. The private
LU must fit RAM; the cgroup guard still applies immediately before factorization.
The original operator is hash-checked before and after COW and used for the
true residual. COW factors are not persisted by `keep_matrices`. The two modes
agree in the recorded 17-RHS algebra checks; this is not a physical accuracy
or large-matrix speed claim.
`--solver all-near` applies the corrected
quadrature to every target/source panel interaction and solves iteratively;
it still retains a quadratic-sized weight array. `--solver fmm` uses the
near-corrected FMM operator. Neither iterative route is qualified merely by
converging GMRES. The memory guard covers the arrays it accounts for; native
temporary arrays and iterative workspace also need Slurm memory headroom.

For a larger direct comparison using the patched backend and a selected
same-pose mesh:

```bash
sbatch --cpus-per-task=8 --mem=32G \
  validation_basis/independent_stokes/run_reference_cpu.sbatch \
  --library "$PATCHED_REFERENCE_DEPS/lib/libfmm3dbie.so" \
  --gap-nm "$SELECTED_GAP_NM" --profiles 8:1 \
  --mesh-from /path/to/same_pose_mesh_or_report.json \
  --solver mmap-dense --max-memory-gb 9 --quad-tol 1e-7 \
  --matrix-workdir "/scratch/negishi/$USER/ar20_matrix_work" \
  --output-dir /path/to/new/mmap_dense_experiment
```

`--matrix-workdir` is optional; its default is `matrix_work` beside the report.
Supply an absolute path when overriding it through the runner. The solver
creates two temporary matrix files and removes them after the solve, including
failed solves handled by its cleanup path. At 9963 surface nodes the current
estimate is about 7.5 GB of working memory and 14.8 GB of free disk including
reserve. Actual operating-system page residency is not a hard process-memory
bound. Keep enough job memory and disk space, and retain the small work record
and run logs; temporary matrix work files are not source snapshots.

The iterative controls are `--gmres-rtol 1e-9`, `--gmres-restart 80`, and
`--gmres-maxiter 20` by default. The last value counts restart cycles per
right-hand side. `--preconditioner` accepts `patch` (default), `body`, or
`none`. FMM refinement controls include `--far-order-increment 0` and
`--near-radius-factor 1`. Keep these numerical settings with the report and
check the unpreconditioned residual and independent boundary error.

For an explicit all-near experiment on a previously selected same-pose mesh,
set `SELECTED_GAP_NM` to that mesh's gap and run:

```bash
sbatch --cpus-per-task=8 --mem=32G \
  validation_basis/independent_stokes/run_reference_cpu.sbatch \
  --library "$REFERENCE_DEPS/fmm3dbie/lib/libfmm3dbie.so" \
  --gap-nm "$SELECTED_GAP_NM" --profiles 6:1 \
  --mesh-from /path/to/same_pose_mesh_or_report.json \
  --solver all-near --preconditioner patch \
  --gmres-rtol 1e-9 --gmres-restart 80 --gmres-maxiter 20 \
  --quad-tol 1e-7 --max-memory-gb 6 \
  --output-dir /path/to/new/all_near_experiment
```

The optional compiled near-application helper changes only how already
computed all-near weights are applied. Build it explicitly at a fresh path:

```bash
python3 validation_basis/independent_stokes/near_apply.py build \
  --output /path/to/fresh/libnear_apply.so
```

Add `--near-apply-library /path/to/fresh/libnear_apply.so` to the all-near
invocation to use it. The adjacent `libnear_apply.so.json` is required. The
runner captures both files together, preserves their adjacent filenames,
checks their library/source hashes against the captured `near_apply.cpp`, and
rewrites the driver argument to the captured library. The wrapper also checks
the native ABI and build provenance when loading. Do not replace a helper
while an experiment is using it.

The direct driver also accepts `--resume` to reuse a matching saved density
and rerun boundary checks. Use it with the original output path and the
captured source/library when resuming an existing calculation. The Slurm
runner requires a new or empty output directory and does not reopen a prior
job directory.

The runner sets `OMP_NUM_THREADS` and `OPENBLAS_NUM_THREADS` to
`SLURM_CPUS_PER_TASK` and records both. It snapshots the Python/source files,
the requested shared library, and resolved non-system shared libraries. Its
`launch_manifest.json` records SHA-256 hashes, the Python environment, library
linkage, exact commands, and exit statuses. Each profile has a JSON report and
a live `.log` file. Process completion leaves
`physical_accuracy_qualified=false`; inspect the scientific reports separately.
The source snapshot includes top-level Python modules such as `matrix_free.py`
and the source/patch files below `patches/`. Native
Fortran diagnostics are preserved in sibling `.assembly.log` and
`.boundary.log` files when those stages run; an empty file is retained as
evidence that no diagnostic was emitted. Keep these logs, the `.solution.npz`
density, and the report together. The runner does not filter or delete failed
or clean native logs.

For analytic isolated-response checks, `--operator-export /absolute/path/to/operator_export`
passes the separate operator-export executable to the driver. Pair experiments
can attach each already checked body's report with repeated
`--isolated-validation /absolute/path/to/report.json` arguments. The runner
snapshots these inputs. It also captures an adjacent standard dependency build
manifest, or one explicitly selected with `--dependency-manifest`, and checks
that the selected library's hash occurs in that manifest.
For a patched backend, `--backend-manifest /path/to/patch_manifest.json`
captures and forwards the selected-library manifest after checking its
top-level `library_sha256`. If omitted, the runner looks for
`patch_manifest.json` one directory above the original library's directory.
This format is distinct from the broader dependency build manifest; the
runner does not substitute one for the other.

For residual-driven mesh refinement, `--adapt-from /path/to/parent.json`
selects a previous report for the same pose and gap. The runner captures the
parent JSON as `inputs/adaptive_parent.json`, records its hash, and points the
driver at that captured file. Each requested profile uses that same explicit
parent; the runner does not automatically chain one profile into the next.
`--mesh-from /path/to/proposal_or_report.json` instead uses the exact patch
descriptors from that same-pose input, without residual-based refinement. The
runner captures it as `inputs/mesh_parent.json`. These two parent options
cannot be combined. Working matrices are not copied into source snapshots.

Useful environment overrides:

| Variable | Meaning |
| --- | --- |
| `SLURRY_ROOT` | Explicit repository root, overriding the submission directory |
| `SLURRY_REFERENCE_PYTHON` | Python executable used for the runner and solver |
| `SLURRY_REFERENCE_LOAD_MODULES=0` | Preserve an already configured environment |
| `SLURRY_PYTHON_MODULE`, `SLURRY_GCC_MODULE` | Site-specific module names |
| `REFERENCE_LD_LIBRARY_PATH` | Additional directories for shared dependencies |
| `REFERENCE_LIBRARY`, `GAP_NM` | Alternatives to `--library` and `--gap-nm` |
| `REFERENCE_PROFILES`, `REFERENCE_OUTPUT_DIR` | Default profiles and output directory |
| `SLURRY_RESULTS_ROOT` | Parent of the automatically named output directory |

To run without Slurm, invoke the same script with `bash`, specify an output
directory, and configure the Python/compiler environment first. No MPI or
LAMMPS installation is needed by this reference workflow.

## Compare references and test the matching model

Build the adapter that exports the production isolated and leading operators,
and exercises the full17 table encoding. From the repository root:

```bash
g++ -O1 -std=c++17 \
  -I lammps_slurry/src -I lammps_slurry/vendor/eigen3 \
  validation_basis/independent_stokes/operator_export.cpp \
  -o /path/to/operator_export
```

This executable is an operator adapter, not an independent Stokes solver.
Pass it to isolated reference runs with `--operator-export` to check the
combined-layer force, torque, and intrinsic-stresslet extraction against the
isolated spheroid response.

First compare at least three actual refinements at one fixed gap:

```bash
python3 validation_basis/independent_stokes/compare_reference.py \
  /path/to/ref_coarse.json /path/to/ref_medium.json /path/to/ref_fine.json \
  --output /path/to/reference_comparison.json
```

Without `--anchor-gap-over-c`, this qualifies only the supplied reference
matrices. Each gap needs distinct `resolution_level` values and distinct
discretizations; changing the label alone cannot qualify a refinement. The
driver defaults the label to `100*level+order`. For a deliberate refinement
sequence involving other mesh controls, assign increasing labels with
`--resolution-level` on separate runs.

The latest three reference sources must also identify the **same numerical
operator**. For panel-BIE reports, the comparator requires valid, identical
`library_sha256`, `source_hashes["backend.py"]`, and
`source_hashes["geometry.py"]`, together with matching finite `alpha`, `beta`,
and the full17 `convention_id`. Missing identities or mixed libraries/source
implementations fail qualification even when all matrix changes are small.
Driver and storage hashes and solver-contract labels are not identity keys;
changing output handling alone need not invalidate an unchanged operator.
Earlier inputs outside the latest-three window remain diagnostic and do not
override the identity of the three sources used for qualification.

An external independent solver can provide the same reference17 data without
the panel-specific library fields. It must explicitly declare its operator:

```json
{
  "convention_id": "slurry_full17_c_eta_1_v1",
  "reference_operator_identity": {
    "solver_family": "external_solver_name",
    "implementation_sha256": "REPLACE_WITH_ACTUAL_64_HEX_IMPLEMENTATION_DIGEST",
    "geometry_sha256": "REPLACE_WITH_ACTUAL_64_HEX_GEOMETRY_DIGEST",
    "parameters": {"viscosity": 1.0}
  }
}
```

This is an identity-field template, not a complete reference report. Replace
both digest placeholders with actual SHA256 values, retain all other required
reference, geometry, moment and boundary evidence, and include any operator
parameters in `parameters`. The complete identity must match across the
latest three sources. If an external report supplies a `library_sha256`, that
digest is also checked and included in its identity. A generic declaration
cannot override missing panel-BIE identity fields, and synthetic fixtures
remain ineligible for physical qualification. Identity failures leave the
numerical comparison metrics available as diagnostics.

After an anchor and held-out gaps have independent reference evidence, set
`ANCHOR_H_OVER_C` to the chosen dimensionless gap and compare the model:

```bash
python3 validation_basis/independent_stokes/compare_reference.py \
  /path/to/anchor_ref_*.json /path/to/heldout_ref_*.json \
  --anchor-gap-over-c "$ANCHOR_H_OVER_C" \
  --operator-helper /path/to/operator_export \
  --output /path/to/matching_comparison.json
```

There is no default anchor at 75 nm or any other gap. The anchor must occur
in the input reports. The default model gate uses held-out gaps below the
anchor; exact reconstruction at the anchor is not an independent model test.
`--model-domain all` also counts explicitly supplied gaps above it.

The default reference criterion requires the last two changes across at least
three distinct refinements to be at most 0.2% in full17 energy and the
force/torque/stresslet block metrics. The source checks also require isolated
moment validation, independent boundary error and surface-position error
relative to the gap at most `1e-3`, raw and energy-scaled reciprocity error at
most `1e-3`, and linear relative residual at most `1e-8`.

For matching, the 1% budget includes model discrepancy, observed query-reference
refinement change, and observed anchor refinement change propagated into the
query's metric. These are refinement-based uncertainty estimates, not rigorous
error bounds. The comparator also checks the 153-value encode/decode roundtrip
at `1e-9` energy tolerance; that roundtrip validates storage, not physical
accuracy. Synthetic inputs cannot obtain physical qualification. Exit status
`2` means unqualified, with reasons retained in the comparison JSON.

## Acceptance evidence

### Prepared server retry for this same pose

The saved `pair_h150_p8_hpc_refinement_proposal.json` specifies the same
159-panel mesh attempted in the resource-limited p8 runs above; no p8
solution on that mesh has completed. First retry that exact p8 mesh on a
stable CPU environment with at least **32 GiB allocated RAM and 32 GB free
work disk**. This is a resource recommendation, not an accuracy guarantee.
The prepared command below uses the existing mmap/COW route and has not
been submitted here. The bounded v3 alternative with 512/256 blocks is in
[OUT_OF_CORE_SOLVER.md](OUT_OF_CORE_SOLVER.md). Use the newly built, hash-matching
backend and passing isolated reports from that installation. Define
`ISOLATED0_REPORT`, `ISOLATED1_REPORT`, `OPERATOR_EXPORT`,
`AR20_REFERENCE_OUTPUT` (a new/empty directory), and `AR20_MATRIX_WORK`
as absolute paths before running from the repository root.

```bash
sbatch --cpus-per-task=8 --mem=32G --time=04:00:00 \
  validation_basis/independent_stokes/run_reference_cpu.sbatch \
  --library "$PATCHED_REFERENCE_DEPS/lib/libfmm3dbie.so" \
  --gap-nm 150 --profiles 8:1 \
  --mesh-from validation_basis/independent_stokes/results/pair_h150_p8_hpc_refinement_proposal.json \
  --solver mmap-dense --mmap-factor-mode cow --max-memory-gb 30 \
  --check-points-per-patch 9 --quad-tol 1e-7 \
  --isolated-validation "$ISOLATED0_REPORT" \
  --isolated-validation "$ISOLATED1_REPORT" \
  --operator-export "$OPERATOR_EXPORT" \
  --output-dir "$AR20_REFERENCE_OUTPUT" --matrix-workdir "$AR20_MATRIX_WORK"

# After this process finishes, inspect its source gates; one mesh cannot qualify.
python3 validation_basis/independent_stokes/compare_reference.py \
  "$AR20_REFERENCE_OUTPUT/pair_o8_l1.json" \
  --output "$AR20_REFERENCE_OUTPUT/p8_source_diagnostic.json"
```

The one-input diagnostic intentionally returns exit status 2 because three
same-operator spatial refinements are required. Select the further refinements
after inspecting the completed p8 boundary and response evidence; increasing
orders to 9 and 10 would use 15900 and 19239 nodes and require a separate
resource assessment (the largest matrix is about 26.7 GB). These refinements
have not been run. This does not select 150 nm as a matching anchor. A failed
reference gate still blocks model qualification and production-table generation.
Quadrature refinement remains a separate check even if spatial profiles pass.

For this same pose and gap, retain the raw, unsymmetrized full matrix and check:

1. Geometry and surface quadrature, including closest-point neighborhoods.
2. Boundary conditions at independent points, with separate global and gap
   patch errors. Solver collocation residuals alone are insufficient.
3. Force, torque, intrinsic stresslet, raw reciprocity, and physical
   dissipation. Keep signs, origins, and `a` scaling explicit.
4. Refinement of the density resolution and near-singular quadrature as
   separate controls; compare full17 response and power, not only selected
   normal forces.
5. Stability under justified changes of discretization and quadrature before
   assigning a matching anchor or activating a production table.

Preserve failed attempts and their numerical settings. A report must say when
boundary checks, refinement, or force/torque/stresslet extraction remain
unqualified; do not turn completion of a job into a passing source gate.
