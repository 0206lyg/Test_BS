"""Exact partial-pivot LU with bounded panels over one file-backed matrix.

The input float64 Fortran-order file is overwritten with the standard LAPACK
L/U factors. This is a storage algorithm, not a new Stokes approximation.
Global row pivots include the previously computed L columns. The caller must
retain or reassemble the original operator to check its true residual.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy import __version__ as scipy_version
from scipy.linalg.blas import dgemm, dtrsm
from scipy.linalg.lapack import dgetrf

from memory_bounded_dense import (_check_cgroup_headroom, _drop_clean_pages,
                                  _process_io, _write_record)


class OutOfCoreLUError(RuntimeError):
    def __init__(self, message, diagnostics):
        self.diagnostics = diagnostics
        super().__init__(message)


def working_memory_bytes(dimension, rhs_count, panel_columns=1024,
                         stripe_columns=512):
    """Conservative extra working set including mapped windows and BLAS slack.

    No full dense anonymous matrix is allocated. The estimate includes two
    full panels during panel extraction, lower-panel plus stripe copies during
    updates, dirty mapped windows, vector work, and 256 MiB native slack.
    """
    d, nr = int(dimension), int(rhs_count)
    b, c = min(int(panel_columns), d), min(int(stripe_columns), d)
    return int(8*max(3*d*b+2*b*b, d*b+4*d*c+2*b*b)
               + 64*d*nr + 64*d + 256*1024**2)


def _copy_f(value):
    return np.array(value, dtype=np.float64, order='F', copy=True)


def _alias_blas(result, target, name):
    if not np.shares_memory(result, target):
        raise RuntimeError(f'{name} copied its output; bounded-buffer contract failed')
    return result


def _swap_rows(values, pivots, offset=0):
    for local, pivot in enumerate(pivots):
        first, second = offset+local, int(pivot)
        if first != second:
            saved = values[first].copy()
            values[first] = values[second]
            values[second] = saved


def factor_solve(matrix_path, dimension, rhs, panel_columns=1024,
                 stripe_columns=512, max_memory_gb=3., cgroup_reserve_gb=.25,
                 progress=None, diagnostics_path=None):
    """Overwrite one matrix file with exact LU; return all RHS solutions.

    ``matrix_path`` must contain exactly ``dimension**2`` native float64
    entries in Fortran order. ``rhs`` is finite (dimension,nrhs) or a vector.
    ``progress(stage, record_copy)`` reports each completed factor/solve panel.
    The source matrix is destructive even on failure; no automatic resume is
    supported. Failure metadata records the last completed panel. No true
    original-operator residual or physical accuracy is claimed here.
    """
    if int(dimension) != dimension or dimension < 1:
        raise ValueError('dimension must be a positive integer')
    d = int(dimension)
    for value, name in [(panel_columns, 'panel_columns'),
                        (stripe_columns, 'stripe_columns')]:
        if int(value) != value or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    if not np.isfinite(max_memory_gb) or max_memory_gb <= 0:
        raise ValueError('max_memory_gb must be positive and finite')
    if not np.isfinite(cgroup_reserve_gb) or cgroup_reserve_gb < 0:
        raise ValueError('cgroup_reserve_gb must be finite and nonnegative')
    b, c = min(int(panel_columns), d), min(int(stripe_columns), d)
    supplied = np.asarray(rhs, dtype=np.float64)
    was_vector = supplied.ndim == 1
    if was_vector:
        supplied = supplied[:, None]
    if (supplied.ndim != 2 or supplied.shape[0] != d or supplied.shape[1] < 1
            or not np.isfinite(supplied).all()):
        raise ValueError('rhs must be finite with shape (dimension,nrhs)')
    path = Path(matrix_path).expanduser().resolve()
    if not path.is_file() or path.stat().st_size != 8*d*d:
        raise ValueError('matrix file size must be exactly 8*dimension**2 bytes')
    needed = working_memory_bytes(d, supplied.shape[1], b, c)
    if needed > max_memory_gb*1e9:
        raise MemoryError(f'Blocked LU working estimate {needed/1e9:.3f} GB exceeds '
                          f'budget {max_memory_gb:g} GB')
    preflight = _check_cgroup_headroom(needed, int(cgroup_reserve_gb*1e9))
    log_path = (Path(diagnostics_path).expanduser().resolve() if diagnostics_path
                else path.with_name(path.name+'.lu.json'))
    if log_path == path:
        raise ValueError('diagnostics_path must differ from matrix_path')
    log_path.parent.mkdir(parents=True, exist_ok=True)
    record = {'schema':'independent_stokes_out_of_core_lu_v1',
              'status':'preparing', 'matrix_path':str(path),
              'matrix_dimension':d, 'rhs_count':supplied.shape[1],
              'matrix_dtype':'float64', 'matrix_order':'F',
              'matrix_bytes':8*d*d, 'destructive_in_place':True,
              'algorithm':'right_looking_blocked_LU_global_partial_pivoting',
              'panel_columns':b, 'stripe_columns':c,
              'factor_completed_columns':0, 'forward_completed_columns':0,
              'backward_completed_columns':0,
              'estimated_working_bytes':needed,
              'memory_budget_bytes':int(max_memory_gb*1e9),
              'cgroup_reserve_bytes':int(cgroup_reserve_gb*1e9),
              'cgroup_preflight':preflight,
              'maximum_explicit_buffer_bytes':0,
              'blas_output_alias_checks':0, 'pivot_swap_count':0,
              'maximum_pivot_distance':0,
              'original_operator_residual_checked':False,
              'physical_accuracy_qualified':False,
              'numpy_version':np.__version__, 'scipy_version':scipy_version,
              'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'timings_seconds':{}, 'diagnostics_path':str(log_path)}
    begun = time.monotonic()
    mapping = None
    pivots = np.empty(d, dtype=np.int64)
    solution = _copy_f(supplied)

    def stage(name):
        record.update(status=name, elapsed_seconds=time.monotonic()-begun)
        _write_record(log_path, record)
        if progress is not None:
            progress(name, json.loads(json.dumps(record)))

    def buffers(*values):
        count = sum(value.nbytes for value in values) + solution.nbytes+pivots.nbytes
        record['maximum_explicit_buffer_bytes'] = max(
            record['maximum_explicit_buffer_bytes'],int(count))
        if count > needed:
            raise MemoryError('Explicit buffers exceeded the admitted working estimate')

    def alias(result, target, name):
        result = _alias_blas(result, target, name)
        record['blas_output_alias_checks'] += 1
        return result

    try:
        mapping = np.memmap(path, dtype=np.float64, mode='r+', shape=(d,d), order='F')
        stage('factorizing')
        factor_start, io_start = time.monotonic(), _process_io()
        for k in range(0,d,b):
            end = min(k+b,d); width = end-k; remaining = d-k
            record['cgroup_before_panel'] = _check_cgroup_headroom(
                needed, int(cgroup_reserve_gb*1e9))
            panel = _copy_f(mapping[k:,k:end])
            _drop_clean_pages(mapping,path,record)
            if not np.isfinite(panel).all():
                raise FloatingPointError('Non-finite matrix panel')
            factored, local_pivots, info = dgetrf(panel, overwrite_a=True)
            alias(factored,panel,'dgetrf')
            if info < 0:
                raise ValueError(f'dgetrf rejected argument {-info}')
            if info > 0:
                record['singular_pivot_index_zero_based'] = k+int(info)-1
                raise np.linalg.LinAlgError(f'Singular LU pivot at zero-based index {k+int(info)-1}')
            pivots[k:end] = local_pivots+k
            distances = local_pivots-np.arange(width)
            record['pivot_swap_count'] += int(np.count_nonzero(distances))
            record['maximum_pivot_distance'] = max(record['maximum_pivot_distance'],
                                                    int(np.max(distances,initial=0)))
            mapping[k:,k:end] = panel
            _drop_clean_pages(mapping,path,record)
            diagonal = _copy_f(panel[:width,:])
            lower = _copy_f(panel[width:,:])
            buffers(panel, diagonal, lower, local_pivots)
            del factored, panel

            # Applying the panel's ordered row swaps to old L is essential.
            # Gather only changed rows; a no-swap panel performs no old-L I/O.
            permutation = np.arange(remaining, dtype=np.int64)
            for i,pivot in enumerate(local_pivots):
                permutation[i],permutation[pivot] = permutation[pivot],permutation[i]
            changed = np.flatnonzero(permutation != np.arange(remaining))
            if changed.size and k:
                for first in range(0,k,c):
                    last = min(first+c,k)
                    previous = _copy_f(mapping[k+permutation[changed],first:last])
                    buffers(diagonal,lower,previous,permutation,changed)
                    mapping[k+changed,first:last] = previous
                    del previous
                    _drop_clean_pages(mapping,path,record)
            del changed, permutation

            for first in range(end,d,c):
                last = min(first+c,d)
                stripe = _copy_f(mapping[k:,first:last])
                _drop_clean_pages(mapping,path,record)
                _swap_rows(stripe,local_pivots)
                upper = _copy_f(stripe[:width,:])
                upper = alias(dtrsm(1.,diagonal,upper,side=0,lower=1,diag=1,
                                    overwrite_b=1),upper,'dtrsm')
                tail = _copy_f(stripe[width:,:])
                buffers(diagonal,lower,stripe,upper,tail,local_pivots)
                del stripe
                tail = alias(dgemm(-1.,lower,upper,beta=1.,c=tail,
                                   overwrite_c=1),tail,'dgemm')
                if not np.isfinite(tail).all() or not np.isfinite(upper).all():
                    raise FloatingPointError('Non-finite LU Schur update')
                mapping[k:end,first:last] = upper
                mapping[end:,first:last] = tail
                del upper,tail
                _drop_clean_pages(mapping,path,record)
            del diagonal,lower,local_pivots
            record['factor_completed_columns'] = end
            stage('factorizing')
        record['timings_seconds']['factorization'] = time.monotonic()-factor_start
        io_end = _process_io()
        record['factorization_io_delta'] = {key:io_end[key]-io_start[key]
                                            for key in io_start.keys() & io_end.keys()}
        record['pivot_sha256'] = hashlib.sha256(pivots.tobytes()).hexdigest()
        # Applying all chronological pivots once produces P*b for standard LU.
        _swap_rows(solution,pivots)
        solve_start = time.monotonic()
        stage('forward_solving')
        for k in range(0,d,b):
            end = min(k+b,d)
            diagonal = _copy_f(mapping[k:end,k:end])
            top = _copy_f(solution[k:end])
            top = alias(dtrsm(1.,diagonal,top,side=0,lower=1,diag=1,
                              overwrite_b=1),top,'dtrsm_forward')
            solution[k:end] = top
            for first in range(end,d,c):
                last = min(first+c,d)
                tile = _copy_f(mapping[first:last,k:end])
                target = _copy_f(solution[first:last])
                buffers(diagonal,top,tile,target)
                target = alias(dgemm(-1.,tile,top,beta=1.,c=target,
                                     overwrite_c=1),target,'dgemm_forward')
                solution[first:last] = target
                del tile,target
                _drop_clean_pages(mapping,path,record)
            del diagonal,top
            _drop_clean_pages(mapping,path,record)
            record['forward_completed_columns'] = end
            stage('forward_solving')
        stage('backward_solving')
        for k in reversed(range(0,d,b)):
            end = min(k+b,d)
            diagonal = _copy_f(mapping[k:end,k:end])
            top = _copy_f(solution[k:end])
            top = alias(dtrsm(1.,diagonal,top,side=0,lower=0,diag=0,
                              overwrite_b=1),top,'dtrsm_backward')
            solution[k:end] = top
            for first in range(0,k,c):
                last = min(first+c,k)
                tile = _copy_f(mapping[first:last,k:end])
                target = _copy_f(solution[first:last])
                buffers(diagonal,top,tile,target)
                target = alias(dgemm(-1.,tile,top,beta=1.,c=target,
                                     overwrite_c=1),target,'dgemm_backward')
                solution[first:last] = target
                del tile,target
                _drop_clean_pages(mapping,path,record)
            del diagonal,top
            _drop_clean_pages(mapping,path,record)
            record['backward_completed_columns'] = d-k
            stage('backward_solving')
        record['timings_seconds']['solve'] = time.monotonic()-solve_start
        if not np.isfinite(solution).all():
            raise FloatingPointError('Non-finite blocked triangular solution')
        stage('completed')
        return (solution[:,0] if was_vector else solution), record
    except BaseException as exc:
        record.update(status='failed', exception_type=type(exc).__name__,
                      exception_message=str(exc), elapsed_seconds=time.monotonic()-begun)
        if hasattr(exc,'cgroup_snapshot'):
            record['cgroup_guard_failure'] = exc.cgroup_snapshot
        _write_record(log_path,record)
        if isinstance(exc,(KeyboardInterrupt,SystemExit)):
            raise
        raise OutOfCoreLUError(str(exc),record) from exc
    finally:
        if mapping is not None:
            # Every successful write is flushed in its owning step. Retrying
            # a failed flush here could mask the recorded exception (e.g. an
            # exhausted filesystem). A failed in-place file is never resumed.
            mapping._mmap.close()
