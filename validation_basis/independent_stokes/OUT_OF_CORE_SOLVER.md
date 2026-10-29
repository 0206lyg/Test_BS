# Exact out-of-core Stokes solve

The 12,879-node p8/159-panel attempts at the recorded AR20 pose and 150 nm have **stopped at resource guards**. V2 completed full assembly in 932.935845 seconds, then rejected LU because the guard reported 1.211 GB headroom versus 2.279 GB required. V3, using 512/256 LU blocks, rejected the retry at entry: 0.668 GB headroom versus 0.830 GB required. Neither produced a new density or physical response. See the [v2 report](results/pair_h150_p8_159_ooc_buffered_retry.json) and [v3 report](results/pair_h150_p8_159_ooc_v3_resource_retry.json).

The best-resolved completed pair remains the 9963-node result, with maximum independent boundary error 0.753223% against the unchanged 0.1% criterion. No matching anchor, 1 nm response or production table is physically qualified. This document describes storage implementation and reproduction, not a passing lubrication result.

## What changes

`out_of_core_dense.py` v3 uses the same exterior combined-layer operator, quadrature and all 17 boundary-velocity RHS as the existing dense solver. It does not use FMM approximations or alter a physical tolerance. Its source SHA256 is `112e816727006d8dee92e331a59e8bf035c4481dab64739fd2f2876cf6d99f43`.

1. Assemble 128 target nodes at a time into a bounded RAM row buffer. Copy the buffer into the Fortran-order matrix file in contiguous column stripes. `MADV_RANDOM` is requested for this assembly mapping to avoid unnecessary file read-ahead.
2. Hash the original operator in 64 MiB chunks, advising release only of this job's already-read file ranges. Cache eviction is not guaranteed. Factor the single file **A → LU** using `out_of_core_lu.py`. Global partial pivoting includes previously computed L columns and trailing columns. Solve all 17 RHS using the recorded row pivots and streamed triangular blocks.
3. Reassemble the original operator in target chunks and compute each RHS's true residual `A_original x − b`. The adapter rejects a maximum relative residual above `1e-8`. This check does not apply the overwritten LU file as though it were A.
4. Run the driver's independent surface checks and extract force, torque and intrinsic stresslet from the combined-layer representation. Small linear residuals and successful storage tests do not qualify these physical quantities.

V3 checks the maximum estimated assembly, LU and residual working set before creating the matrix, then repeats the unchanged stage guards. The temporary matrix is removed by default, including on handled failure. Source archives, reports and diagnostics remain. Partial LU is destructive and has **no supported resume**; restart a failed factorization with a fresh output name and reconstruct A. The driver's `--resume` is only for an already completed, identity-checked density checkpoint followed by repeated boundary checks.

## Resources

For this mesh the matrix has 38,637 rows and columns and occupies **11,942,542,152 bytes** on disk. Only one matrix file is required. The disk guard additionally reserves 0.5 GB, for a minimum of **12,442,542,152 free bytes**. Observed free space was approximately 9 GB and subsequently declined further; it was insufficient. Allow space for reports, checkpoints and unrelated activity too.

With the attempted v2 LU panels/stripes of 2048/512 columns, the estimated extra factor working set was **2.279 GB**. With v3's proposed 512/256 blocks, assembly/residual work sets the maximum estimate at **0.830492 GB**. These estimates are not a hard RSS cap. The guard applies a 0.25 GB reserve and credits only clean inactive file cache, conservatively counting active file pages as occupied. At the v2 failure, anonymous memory was about 0.166 GB, file memory 8.228 GB and active file memory 6.884 GB. This guard failure is **not proof that an 8 GiB machine cannot hold the bounded working arrays**, nor proof that all charged file cache belonged to this job's hash pass. V3's cache advice improves handling of its own file but does not guarantee eviction.

For a stable retry of this same p8 batch, allocate **at least 32 GiB RAM and 32 GB free work disk** on a CPU system. This is a resource recommendation, not guaranteed Stokes accuracy. Use either the existing mmap/COW route in the README or the v3 512/256 route below; retain all actual resource and scientific gates. The command uses eight CPU threads and a 4 GB extra-working-memory budget.

## Reproduce this one pose

Run from the repository root after the [pinned backend and high-order patch](README.md) have been built. Set `PATCHED_REFERENCE_DEPS`, `OPERATOR_EXPORT`, `ISOLATED0_REPORT`, `ISOLATED1_REPORT` and `AR20_OOC_WORK` to absolute paths. The exporter is the compiled `operator_export.cpp` helper. Both isolated reports must pass validation and match the selected library and backend/geometry source hashes; existing reports from a different build cannot be substituted.

Use a fresh output filename. This command solves only the saved pose and gap; it does not create a table or select a matching anchor.

```bash
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
python3 validation_basis/independent_stokes/run_reference.py \
  --library "$PATCHED_REFERENCE_DEPS/lib/libfmm3dbie.so" \
  --backend-manifest "$PATCHED_REFERENCE_DEPS/patch_manifest.json" \
  --gap-nm 150 --order 8 --level 1 --resolution-level 14 \
  --mesh-from validation_basis/independent_stokes/results/pair_h150_p8_hpc_refinement_proposal.json \
  --solver out-of-core-dense --lu-panel-columns 512 --lu-stripe-columns 256 \
  --max-memory-gb 4 --matrix-workdir "$AR20_OOC_WORK" \
  --quad-tol 1e-7 --check-points-per-patch 9 \
  --isolated-validation "$ISOLATED0_REPORT" \
  --isolated-validation "$ISOLATED1_REPORT" \
  --operator-export "$OPERATOR_EXPORT" \
  --output validation_basis/independent_stokes/results/pair_h150_p8_159_ooc_v3_reproduction_001.json
```

The `--mesh-from` descriptors determine the 159 panels; `--level 1` does not refine that mesh again. `--resolution-level 14` is an ordering label for comparison, not evidence of refinement or accuracy. Do not modify the source files or selected shared library while a run is active. The driver records their identities and captures native diagnostics.

## Completed implementation evidence

- Latest Python validation ran **49 tests: 48 passed and one optional moderate benchmark was skipped**. Earlier 45-test and 36-test runs remain preserved as history. Three separately executed synthetic benchmarks at dimensions 3000 and 5000 passed known-solution, standard-LU and original-residual comparisons. Their exact source hashes and cleanup-only source difference are retained; these tests do not establish Stokes convergence.
- At the same AR20 pose and 150 nm, the **300-node, all-17-RHS** tiled solve agreed with the prior COW solve: density relative difference `8.24e-15`, raw17 difference `9.48e-15`.
- Switching assembly from v1 to the buffered v2 produced a **bitwise identical original A**, zero observed density difference, and maximum true linear residual `3.73e-14` at that same small mesh. This is storage equivalence, not a boundary-accuracy pass.
- Switching from v2 to v3 at the same actual **300-node, all-17-RHS** Stokes problem preserves original A bitwise, gives density difference `1.0944930340633586e-14`, raw17 difference `2.726800138690093e-14`, and original-operator residual `4.0819039994881485e-14`. Different LU block sizes may change floating-point rounding. This does not complete the large mesh.

See [implementation_validation.json](results/implementation_validation.json), [latest Python log](results/independent_python_tests_ooc_v3.log), [tiled-LU comparison](results/ooc_stokes_storage_equivalence.json), [buffered-assembly comparison](results/ooc_buffered_assembly_equivalence.json), and [v3 comparison](results/ooc_v3_assembly_equivalence.json). A large completed solve would still require independent boundary accuracy, same-operator spatial/quadrature convergence and separate matching comparisons before physical qualification.

Saved-density coarsening with 24 high-error regions protected and jump-change indicator at most `2e-4` supports only 157 panels, still an 11.644 GB matrix. Reaching at most 138 panels needs about `0.008506` in the tested candidate set and was not adopted. This is a projection diagnostic, not a new solve or full velocity-error bound. A separate FMM `+48` action probe at 300 nodes checked only the first RHS (`2.82263e-7` error) before unexpectedly high native RSS near 1.77 GiB exceeded its light-probe envelope. That invalidates the 512 MB probe estimate; it does not prove the large FMM case impossible or establish all-17-RHS accuracy. See the [coarsening audit](results/pair_h150_p8_coarsened_hotspots_preserved_projection_audit.json) and [partial FMM probe](results/saved_n300_fmm48_action_comparison.json).
