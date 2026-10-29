"""One resident combined-layer matrix and in-place LAPACK LU for all RHS.

The quadrature, exterior trace, force/torque/stresslet convention and original
operator residual are unchanged. Only storage and linear algebra differ from
the file-backed reference solvers. No matrix-sized files are created, and no
automatic out-of-core fallback is permitted when the RAM budget is exceeded.
"""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import time
import warnings

import numpy as np
from scipy import __version__ as scipy_version
from scipy.linalg import LinAlgWarning, lu_factor, lu_solve

from memory_bounded_dense import (
    MMapDenseError, _assemble_target_chunks, _check_cgroup_headroom,
    _process_io, _release_unused_heap, _write_record,
)
from out_of_core_dense import reassembled_residual


class RamDenseError(MMapDenseError):
    """Failure with an auditable stage record; never a physical qualification."""


def working_memory_estimate(node_count, patch_count, rhs_count=17, target_chunk=128):
    """Conservative *additional* stage allocations, in bytes (decimal budgets).

    Existing interpreter/backend/geometry allocations are charged by the live
    cgroup guard. The native quadrature estimate follows the bounded assembler;
    512 MiB slack covers LAPACK/BLAS workspaces, allocator arenas and small arrays.
    This is an admission estimate, not a hard RSS limiter.
    """
    n, p, k = int(node_count), int(patch_count), int(rhs_count)
    if min(n, p, k) < 1 or int(target_chunk) != target_chunk or target_chunk < 1:
        raise ValueError('Positive dimensions and integer target chunk required')
    chunk = min(int(target_chunk), n)
    d = 3*n
    matrix = 8*d*d
    vectors = 8*d*k
    slack = 512*1024**2
    quadrature = 256*n*chunk + 16*chunk*p
    assembly = matrix + quadrature + 8*vectors + slack
    factor = matrix + 8*vectors + 8*d + slack
    # LU is released before the residual assembler allocates its row buffer.
    residual = quadrature + 8*3*chunk*d + 8*vectors + slack
    return {'matrix_bytes': matrix, 'assembly_bytes': assembly,
            'factor_bytes': factor, 'residual_bytes': residual,
            'peak_additional_bytes': max(assembly, factor, residual),
            'native_and_blas_slack_bytes': slack, 'assembly_target_chunk': chunk}


def _threadpools():
    try:
        from threadpoolctl import threadpool_info
        return threadpool_info()
    except ImportError:
        return []


def solve_ram_dense(backend, mesh, rhs, eps, workdir, alpha=1., beta=1.,
                    max_memory_gb=56., assembly_target_chunk=128,
                    cgroup_reserve_gb=1., progress=None):
    """Return ``(solution, diagnostics)`` for a (3*N,k) array or (3*N,) RHS.

    All k RHS share one factorization. Caller RHS remains unchanged. The original
    matrix is overwritten in RAM, released, and exactly reassembled in bounded
    row chunks for an independent A*x-b check. A record alone survives in
    ``workdir``. Set BLAS thread environment variables before Python starts.
    """
    n = int(mesh['srcvals'].shape[1])
    d = 3*n
    rhs = np.asarray(rhs, dtype=np.float64)
    was_vector = rhs.ndim == 1
    if was_vector:
        rhs = rhs[:, None]
    if (rhs.ndim != 2 or rhs.shape[0] != d or rhs.shape[1] < 1
            or not np.isfinite(rhs).all()):
        raise ValueError('rhs must be finite with shape (3*N,nrhs)')
    if not (n > 0 and 0 < eps < 1 and np.isfinite(alpha) and alpha > 0
            and np.isfinite(beta) and beta > 0
            and np.isfinite(max_memory_gb) and max_memory_gb > 0
            and np.isfinite(cgroup_reserve_gb) and cgroup_reserve_gb >= 0):
        raise ValueError('Positive finite operator parameters and memory budget required')
    estimate = working_memory_estimate(n, len(mesh['norders']), rhs.shape[1],
                                      assembly_target_chunk)
    if estimate['peak_additional_bytes'] > max_memory_gb*1e9:
        raise MemoryError(f'RAM dense estimated peak additional memory '
                          f'{estimate["peak_additional_bytes"]/1e9:.3f} GB exceeds '
                          f'budget {max_memory_gb:g} GB; no out-of-core fallback')
    reserve = int(cgroup_reserve_gb*1e9)
    preflight = _check_cgroup_headroom(estimate['peak_additional_bytes'], reserve)
    folder = Path(workdir).expanduser().resolve()
    folder.mkdir(parents=True, exist_ok=True)
    record_path = Path(tempfile.mkdtemp(prefix='ram_dense_', dir=folder))/'diagnostics.json'
    record = dict(schema='independent_stokes_ram_dense_v1', status='preparing',
                  node_count=n, dof_count=d, rhs_count=rhs.shape[1],
                  alpha=float(alpha), beta=float(beta), quadrature_tolerance=float(eps),
                  matrix_dtype='float64', matrix_order='F', matrix_storage='anonymous_ram',
                  backend_library=str(getattr(backend, 'path', 'unspecified')),
                  numpy_version=np.__version__, scipy_version=scipy_version,
                  threadpools=_threadpools(), diagnostics_path=str(record_path),
                  matrix_bytes_each=estimate['matrix_bytes'], memory_estimate=estimate,
                  memory_budget_bytes=int(max_memory_gb*1e9), cgroup_preflight=preflight,
                  cgroup_reserve_bytes=reserve, matrix_files_created=[],
                  assembly_method='target_chunked_exact_near_quadrature',
                  assembly_target_chunk=estimate['assembly_target_chunk'],
                  assembly_completed_targets=0, residual_completed_targets=0,
                  factorization_method='LAPACK_GETRF_in_place_RAM',
                  all_rhs_solved_together=True, lu_aliases_original_allocation=None,
                  original_operator_residual_checked=False,
                  operator_residual_method='fresh_exact_target_chunk_reassembly_after_destructive_LU',
                  physical_accuracy_qualified=False, timings_seconds={})
    started = time.monotonic()
    matrix = lu = pivots = solution = None
    initial_io = _process_io()

    def stage(name):
        record['status'] = name
        record['elapsed_seconds'] = time.monotonic()-started
        _write_record(record_path, record)
        if progress is not None:
            progress(name, json.loads(json.dumps(record)))

    try:
        stage('assembling')
        matrix = np.empty((d,d), dtype=np.float64, order='F')
        t = time.monotonic()

        def assembled(first, last, rfac0):
            record['assembly_completed_targets'] = int(last)
            record['assembly_rfac0'] = float(rfac0)
            stage('assembling')

        _assemble_target_chunks(backend, mesh, eps, alpha, beta, matrix,
                                estimate['assembly_target_chunk'], assembled)
        record['timings_seconds']['assembly'] = time.monotonic()-t
        # Avoid a second quadratic boolean allocation from check_finite=True.
        for first in range(0, d, 3*estimate['assembly_target_chunk']):
            if not np.isfinite(matrix[:,first:first+3*estimate['assembly_target_chunk']]).all():
                raise FloatingPointError('Non-finite assembled RAM matrix')
        record['heap_release_after_assembly'] = _release_unused_heap()
        # The resident matrix is already charged in memory.current here.
        record['cgroup_before_factor'] = _check_cgroup_headroom(
            estimate['factor_bytes']-estimate['matrix_bytes'], reserve)
        stage('factorizing')
        t = time.monotonic()
        with warnings.catch_warnings():
            warnings.simplefilter('error', LinAlgWarning)
            lu, pivots = lu_factor(matrix, overwrite_a=True, check_finite=False)
        record['timings_seconds']['factor'] = time.monotonic()-t
        record['lu_aliases_original_allocation'] = bool(np.shares_memory(lu, matrix))
        if not record['lu_aliases_original_allocation']:
            raise RuntimeError('LAPACK unexpectedly copied a Fortran-order RAM matrix')
        stage('solving')
        t = time.monotonic()
        solution = lu_solve((lu,pivots), np.array(rhs, dtype=np.float64, order='F', copy=True),
                            overwrite_b=True, check_finite=False)
        if not np.isfinite(solution).all():
            raise FloatingPointError('Non-finite RAM LU solution')
        record['timings_seconds']['solve'] = time.monotonic()-t
        matrix = lu = pivots = None
        record['heap_release_before_residual'] = _release_unused_heap()
        record['cgroup_before_residual'] = _check_cgroup_headroom(
            estimate['residual_bytes'], reserve)
        stage('checking_reassembled_original_residual')
        t = time.monotonic()

        def checked(first, last, rfac0):
            record['residual_completed_targets'] = int(last)
            stage('checking_reassembled_original_residual')

        absolute, relative = reassembled_residual(backend, mesh, rhs, solution, eps,
            alpha, beta, estimate['assembly_target_chunk'], checked)
        record['timings_seconds']['original_operator_residual'] = time.monotonic()-t
        record['absolute_residual_by_rhs'] = absolute.tolist()
        record['relative_residual_by_rhs'] = relative.tolist()
        record['relative_residual_max'] = float(relative.max())
        record['original_operator_residual_checked'] = True
        if record['relative_residual_max'] > 1e-8:
            raise ArithmeticError('RAM LU original-operator residual exceeds 1e-8')
        stage('complete')
        return (solution[:,0] if was_vector else solution), record
    except BaseException as exc:
        record.update(failure_type=type(exc).__name__, failure_message=str(exc))
        if hasattr(exc, 'cgroup_snapshot'):
            record['cgroup_guard_failure'] = exc.cgroup_snapshot
        stage('failed')
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        raise RamDenseError(str(exc), record) from exc
    finally:
        matrix = lu = pivots = None
        record['heap_release_at_exit'] = _release_unused_heap()
        final_io = _process_io()
        record['process_io_delta'] = {key:final_io[key]-value for key,value in initial_io.items()
                                      if key in final_io}
        record['elapsed_seconds'] = time.monotonic()-started
        _write_record(record_path, record)
