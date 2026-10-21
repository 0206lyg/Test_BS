"""File-backed dense reference solve without a second resident dense matrix.

This changes storage only. The operator is the original fmm3dbie exterior
combined-layer matrix, assembled from the same near-quadrature kernels in
bounded target chunks. The full native N-by-N quadrature workspace is avoided.
The compatible shared mode uses separate original/LU files. Optional COW mode
keeps LU in private anonymous pages over the original matrix's backing file.
Matrix files are removed by default, including after failure; after preflight checks
pass, a JSON stage record is retained. Disk-backed storage does not qualify
the physical solution.
"""
from __future__ import annotations

import json
import ctypes
import gc
import hashlib
import mmap
import os
from pathlib import Path
import shutil
import tempfile
import time
import warnings

import numpy as np
from scipy import __version__ as scipy_version
from scipy.linalg import LinAlgWarning, lu_factor, lu_solve


class MMapDenseError(RuntimeError):
    def __init__(self, message, diagnostics):
        self.diagnostics = diagnostics
        super().__init__(message)


def _write_record(path, record):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _process_io():
    try:
        return {key.rstrip(':'):int(value) for key,value in
                (line.split() for line in Path('/proc/self/io').read_text().splitlines())}
    except (OSError, ValueError):
        return {}


def _drop_clean_pages(mapping, path, record):
    """Best-effort cache advice; correctness never depends on OS eviction."""
    mapping.flush()
    try:
        mapping._mmap.madvise(mmap.MADV_DONTNEED)
        record['madvise_dontneed_supported'] = True
    except (AttributeError, OSError, ValueError):
        record['madvise_dontneed_supported'] = False
    if hasattr(os, 'posix_fadvise'):
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            record['posix_fadvise_dontneed_supported'] = True
        except OSError:
            record['posix_fadvise_dontneed_supported'] = False
        finally:
            os.close(fd)


def _cgroup_memory_snapshot():
    """Read the effective finite cgroup memory ceiling when Linux exposes it."""
    root = Path('/sys/fs/cgroup')
    directories = {root}
    try:
        for line in Path('/proc/self/cgroup').read_text().splitlines():
            hierarchy, controllers, relative = line.split(':', 2)
            if hierarchy == '0' and controllers == '':
                candidate = root/relative.lstrip('/')
                directories.update([candidate, *[p for p in candidate.parents if p == root or root in p.parents]])
    except (OSError, ValueError):
        pass
    limits = []
    for directory in directories:
        try:
            limit = (directory/'memory.max').read_text().strip()
            if limit != 'max':
                item = {'path': str(directory), 'limit_bytes': int(limit),
                        'current_bytes': int((directory/'memory.current').read_text())}
                try:
                    stats = {key:int(value) for key,value in
                             (line.split() for line in (directory/'memory.stat').read_text().splitlines())}
                    clean_file = max(0, stats.get('file',0)-stats.get('file_dirty',0)
                                     -stats.get('file_writeback',0)-stats.get('shmem',0))
                    # Global clean-file bytes do not identify which inactive
                    # pages are clean. Charge all dirty/writeback/shmem bytes
                    # against inactive_file before granting any cache credit.
                    inactive_clean_lower_bound = max(0, stats.get('inactive_file',0)
                        -stats.get('file_dirty',0)-stats.get('file_writeback',0)-stats.get('shmem',0))
                    reclaimable = min(inactive_clean_lower_bound, clean_file)
                    item['memory_stat'] = {key:stats.get(key,0) for key in
                        ['anon','file','kernel','inactive_file','active_file','file_dirty','file_writeback','shmem']}
                except (OSError, ValueError):
                    reclaimable = 0
                item['reclaimable_clean_inactive_file_bytes'] = max(0,reclaimable)
                item['effective_current_bytes'] = max(0,item['current_bytes']-max(0,reclaimable))
                limits.append(item)
        except (OSError, ValueError):
            pass
    if not limits:
        try:
            directory = root/'memory'
            limit = int((directory/'memory.limit_in_bytes').read_text())
            if limit < (1 << 60):
                limits.append({'path': str(directory), 'limit_bytes': limit,
                               'current_bytes': int((directory/'memory.usage_in_bytes').read_text())})
        except (OSError, ValueError):
            pass
    # A parent with a higher nominal ceiling can still be the binding limit
    # when siblings consume memory. Enforce the smallest remaining headroom.
    if not limits:
        return None
    selected = min(limits, key=lambda item: item['limit_bytes']-
                   item.get('effective_current_bytes',item['current_bytes']))
    selected['minimum_hierarchy_limit_bytes'] = min(item['limit_bytes'] for item in limits)
    return selected


def _check_cgroup_capacity(required_bytes, reserve_bytes):
    """Reject intrinsically oversized LU before assembly, not transient usage."""
    snapshot = _cgroup_memory_snapshot()
    if snapshot is not None:
        capacity = snapshot.get('minimum_hierarchy_limit_bytes',snapshot['limit_bytes'])
        snapshot.update(admission_scope='nominal_factor_capacity_only',
                        reserve_bytes=int(reserve_bytes),
                        required_factor_bytes=int(required_bytes),
                        nominal_capacity_bytes=int(capacity),
                        nominal_capacity_after_reserve_bytes=int(capacity-reserve_bytes),
                        actual_factor_headroom_check_required=True)
        if required_bytes+reserve_bytes > capacity:
            raise MemoryError(f'Factor working set plus reserve {(required_bytes+reserve_bytes)/1e9:.3f} GB '
                              f'exceeds nominal cgroup capacity {capacity/1e9:.3f} GB')
    return snapshot


def _check_cgroup_headroom(required_bytes, reserve_bytes):
    snapshot = _cgroup_memory_snapshot()
    if snapshot is not None:
        snapshot['reserve_bytes'] = int(reserve_bytes)
        snapshot['required_additional_bytes'] = int(required_bytes)
        baseline = snapshot.get('effective_current_bytes',snapshot['current_bytes'])
        available = snapshot['limit_bytes']-baseline-reserve_bytes
        snapshot['available_after_reserve_bytes'] = int(available)
        if required_bytes > available:
            error = MemoryError(f'Cgroup has {available/1e9:.3f} GB headroom after reserve; '
                                f'this stage needs an estimated {required_bytes/1e9:.3f} GB '
                                f'(limit {snapshot["limit_bytes"]/1e9:.3f} GB)')
            error.cgroup_snapshot = snapshot
            raise error
    return snapshot


def _release_unused_heap():
    """Return freed native arenas before budgeting a resident LU, if supported."""
    before = _cgroup_memory_snapshot()
    collected = gc.collect()
    trimmed = None
    try:
        trim = ctypes.CDLL(None).malloc_trim
        trim.argtypes = [ctypes.c_size_t]
        trim.restype = ctypes.c_int
        trimmed = int(trim(0))
    except AttributeError:
        pass
    return {'cgroup_before': before, 'cgroup_after': _cgroup_memory_snapshot(),
            'gc_collected_objects': collected, 'glibc_malloc_trim_result': trimmed}


def _assemble_target_chunks(backend, mesh, eps, alpha, beta, output,
                            target_chunk, progress):
    """Exact matgen component order and self trace, with bounded near weights."""
    n = mesh['srcvals'].shape[1]
    p = len(mesh['norders'])
    patch_ids = np.empty(n, dtype=np.int64)
    patch_uv = np.empty((2,n), order='F')
    backend.call('get_patch_id_uvs', p, mesh['norders'], mesh['ixyzs'],
                 mesh['iptype'], n, patch_ids, patch_uv)
    rfac, rfac0 = np.zeros((), dtype=float), np.zeros((), dtype=float)
    backend.call('get_rfacs', int(np.floor(np.mean(mesh['norders']))),
                 int(np.floor(np.mean(mesh['iptype']))), rfac, rfac0)
    strengths = np.array([alpha, beta], dtype=np.float64)
    widths = np.diff(mesh['ixyzs'])
    for first in range(0, n, target_chunk):
        last = min(first+target_chunk, n)
        nt = last-first
        nnz = nt*p
        row = np.arange(nt+1, dtype=np.int64)*p+1
        col = np.tile(np.arange(1,p+1,dtype=np.int64), nt)
        iq = np.r_[1,1+np.cumsum(widths[col-1])].astype(np.int64)
        nquad = int(iq[-1]-1)
        weights = np.zeros((6,nquad), order='F')
        backend.call('getnearquad_stok_comb_vel_eval', p, mesh['norders'],
                     mesh['ixyzs'], mesh['iptype'], n, mesh['srccoefs'],
                     mesh['srcvals'], 3, nt,
                     np.asfortranarray(mesh['srcvals'][:3,first:last]),
                     np.asfortranarray(patch_ids[first:last]),
                     np.asfortranarray(patch_uv[:,first:last]), float(eps),
                     strengths, 1, nnz, row, col, iq, float(rfac0), nquad, weights)
        if not np.isfinite(weights).all():
            raise FloatingPointError('Non-finite assembled target-chunk quadrature')
        for k,(i,j) in enumerate([(0,0),(0,1),(0,2),(1,1),(1,2),(2,2)]):
            kernel = weights[k].reshape(nt,n)
            output[3*first+i:3*last+i:3,j::3] = kernel
            if i != j:
                output[3*first+j:3*last+j:3,i::3] = kernel
        diagonal = np.arange(3*first,3*last)
        output[diagonal,diagonal] += beta/2
        del kernel, weights, row, col, iq
        progress(first,last,float(rfac0))


def solve_mmap_dense(backend, mesh, rhs, eps, workdir, alpha=1., beta=1.,
                     max_memory_gb=9., keep_matrices=False, chunk_columns=256,
                     disk_reserve_gb=.5, progress=None, assembly_target_chunk=128,
                     cgroup_reserve_gb=.25, factor_mode='shared'):
    """Return ``(solution, diagnostics)`` using a Fortran-order mmap operator.

    ``rhs`` has shape (3*N,nrhs), or (3*N,). ``progress(stage, record_copy)``
    receives preparation/assembly/factor/solve/residual status updates. Native
    output/warnings should be captured by the caller around this function.

    Memory guards cover the near-quadrature heap estimate, one resident LU,
    and bounded copying/verification buffers. File pages remain OS-managed;
    these estimates are not a hard RSS cap or a guarantee against OOM from
    other processes. No call to Backend.matrix's in-memory allocator is made.
    ``factor_mode='cow'`` maps the original A privately and skips the LU file.
    Its full LU must fit private memory. Live private factors are never flushed
    or discarded, and original A's SHA256 is checked before/after the solve.
    ``keep_matrices`` in COW mode preserves original A, not the private LU.
    Admission checks nominal LU capacity and actual assembly headroom. Actual
    LU headroom is checked again immediately before factorization, after
    assembly buffers and clean mapped pages have been released.
    """
    n = int(mesh['srcvals'].shape[1])
    d = 3*n
    rhs = np.asarray(rhs, dtype=np.float64)
    was_vector = rhs.ndim == 1
    if was_vector:
        rhs = rhs[:, None]
    if rhs.ndim != 2 or rhs.shape[0] != d or rhs.shape[1] < 1 or not np.isfinite(rhs).all():
        raise ValueError('rhs must be finite with shape (3*N,nrhs)')
    if not (n > 0 and 0 < eps < 1 and np.isfinite(alpha) and np.isfinite(beta)
            and alpha > 0 and beta > 0 and np.isfinite(max_memory_gb) and max_memory_gb > 0):
        raise ValueError('Positive finite exterior operator parameters and memory budget required')
    if int(chunk_columns) != chunk_columns or chunk_columns < 1:
        raise ValueError('chunk_columns must be a positive integer')
    if int(assembly_target_chunk) != assembly_target_chunk or assembly_target_chunk < 1:
        raise ValueError('assembly_target_chunk must be a positive integer')
    if not np.isfinite(disk_reserve_gb) or disk_reserve_gb < 0:
        raise ValueError('disk_reserve_gb must be finite and nonnegative')
    if not np.isfinite(cgroup_reserve_gb) or cgroup_reserve_gb < 0:
        raise ValueError('cgroup_reserve_gb must be finite and nonnegative')
    if factor_mode not in ('shared','cow'):
        raise ValueError('factor_mode must be shared or cow')
    chunk_columns = min(int(chunk_columns), d)
    assembly_target_chunk = min(int(assembly_target_chunk), n)
    matrix_bytes = 8*d*d
    vector_bytes = rhs.nbytes
    # Target chunks bound the six kernels, native quadrature work and mapped
    # row write window. Unlike full matgen, assembly is linear in n*chunk.
    slack = 256*1024**2
    assembly_heap = 256*n*assembly_target_chunk+16*assembly_target_chunk*len(mesh['norders'])+slack
    factor_working = matrix_bytes+8*d*chunk_columns+8*vector_bytes+slack
    memory_needed = max(assembly_heap, factor_working)
    if memory_needed > max_memory_gb*1e9:
        raise MemoryError(f'Mmap dense estimated working memory {memory_needed/1e9:.3f} GB '
                          f'exceeds budget {max_memory_gb:g} GB; file pages are OS-managed')
    cgroup_capacity_preflight = _check_cgroup_capacity(factor_working, int(cgroup_reserve_gb*1e9))
    cgroup_preflight = _check_cgroup_headroom(assembly_heap, int(cgroup_reserve_gb*1e9))
    workdir = Path(workdir).expanduser().resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    disk_free = shutil.disk_usage(workdir).free
    disk_needed = (2 if factor_mode == 'shared' else 1)*matrix_bytes+int(disk_reserve_gb*1e9)
    if disk_free < disk_needed:
        raise OSError(f'Mmap dense needs {disk_needed/1e9:.3f} GB free disk including reserve; '
                      f'only {disk_free/1e9:.3f} GB available')
    folder = Path(tempfile.mkdtemp(prefix='mmap_dense_', dir=workdir))
    operator_path, lu_path = folder/'operator.f64', folder/'lu.f64'
    record_path = folder/'diagnostics.json'
    record = {'schema': 'independent_stokes_mmap_dense_v3', 'status': 'preparing',
              'node_count': n, 'dof_count': d, 'rhs_count': rhs.shape[1],
              'alpha': float(alpha), 'beta': float(beta), 'quadrature_tolerance': float(eps),
              'operator_path': str(operator_path),
              'lu_path': str(lu_path) if factor_mode == 'shared' else None,
              'factor_mode': factor_mode,
              'factor_memory_kind': 'private_anonymous' if factor_mode == 'cow' else 'shared_file_backed',
              'private_lu_persisted': False,
              'matrix_dtype': 'float64', 'matrix_order': 'F',
              'backend_library': str(getattr(backend, 'path', 'unspecified')),
              'numpy_version': np.__version__, 'scipy_version': scipy_version,
              'diagnostics_path': str(record_path), 'matrix_bytes_each': matrix_bytes,
              'estimated_assembly_heap_bytes': assembly_heap,
              'assembly_method': 'target_chunked_exact_near_quadrature',
              'assembly_target_chunk': assembly_target_chunk,
              'assembly_completed_targets': 0,
              'estimated_factor_working_bytes': factor_working,
              'memory_budget_bytes': int(max_memory_gb*1e9),
              'cgroup_preflight': cgroup_preflight,
              'cgroup_capacity_preflight': cgroup_capacity_preflight,
              'cgroup_admission_policy': 'nominal_factor_capacity_and_actual_assembly_then_actual_factor_headroom',
              'cgroup_reserve_bytes': int(cgroup_reserve_gb*1e9),
              'disk_free_at_start_bytes': disk_free, 'disk_needed_bytes': disk_needed,
              'chunk_columns': chunk_columns, 'keep_matrices': bool(keep_matrices),
              'lu_aliases_memmap': None, 'physical_accuracy_qualified': False,
              'timings_seconds': {}}
    started = time.monotonic()
    operator_map = lu_map = lu = None

    def stage(name):
        record['status'] = name
        record['elapsed_seconds'] = time.monotonic()-started
        _write_record(record_path, record)
        if progress is not None:
            progress(name, json.loads(json.dumps(record)))

    try:
        stage('assembling')
        operator_map = np.memmap(operator_path, dtype=np.float64, mode='w+', shape=(d,d), order='F')
        t = time.monotonic()
        def assembly_progress(first,last,rfac0):
            _drop_clean_pages(operator_map, operator_path, record)
            record['assembly_completed_targets'] = last
            record['assembly_rfac0'] = rfac0
            stage('assembling')
        _assemble_target_chunks(backend, mesh, eps, alpha, beta, operator_map,
                                assembly_target_chunk, assembly_progress)
        record['timings_seconds']['assembly'] = time.monotonic()-t
        _drop_clean_pages(operator_map, operator_path, record)
        record['heap_release_after_assembly'] = _release_unused_heap()
        if factor_mode == 'shared':
            stage('copying_for_lu')
            lu_map = np.memmap(lu_path, dtype=np.float64, mode='w+', shape=(d,d), order='F')
            t = time.monotonic()
            for first in range(0, d, chunk_columns):
                last = min(first+chunk_columns, d)
                block = operator_map[:, first:last]
                if not np.isfinite(block).all():
                    raise FloatingPointError('Non-finite assembled matrix')
                lu_map[:, first:last] = block
                del block
                _drop_clean_pages(operator_map, operator_path, record)
                _drop_clean_pages(lu_map, lu_path, record)
            record['timings_seconds']['copy_and_finite_check'] = time.monotonic()-t
        else:
            stage('verifying_original_before_cow')
            t = time.monotonic()
            record['operator_sha256_before_factorization'] = _file_sha256(operator_path)
            for first in range(0, d, chunk_columns):
                if not np.isfinite(operator_map[:,first:min(first+chunk_columns,d)]).all():
                    raise FloatingPointError('Non-finite assembled matrix')
            _drop_clean_pages(operator_map, operator_path, record)
            record['timings_seconds']['cow_input_verification'] = time.monotonic()-t
        operator_map._mmap.close()
        operator_map = None
        if factor_mode == 'shared':
            _drop_clean_pages(lu_map, lu_path, record)
        else:
            lu_map = np.memmap(operator_path, dtype=np.float64, mode='c', shape=(d,d), order='F')
            # NEVER call _drop_clean_pages on this private mapping while LU
            # is needed: MADV_DONTNEED would restore original A's file pages.
        record['heap_release_before_factorization'] = _release_unused_heap()
        record['cgroup_before_factorization'] = _check_cgroup_headroom(
            factor_working, int(cgroup_reserve_gb*1e9))
        stage('factorizing')
        t = time.monotonic()
        factor_io_before = _process_io()
        # Native float64 + F-contiguous ndarray + overwrite_a=True is LAPACK's
        # in-place path. Check aliasing explicitly instead of assuming it.
        with warnings.catch_warnings():
            warnings.simplefilter('error', LinAlgWarning)
            lu, pivots = lu_factor(np.asarray(lu_map), overwrite_a=True, check_finite=False)
        record['lu_aliases_memmap'] = bool(np.shares_memory(lu, lu_map))
        if not record['lu_aliases_memmap']:
            raise RuntimeError('SciPy copied the LU matrix; in-place storage contract failed')
        record['timings_seconds']['factorization'] = time.monotonic()-t
        factor_io_after = _process_io()
        record['factorization_io_delta'] = {key:factor_io_after[key]-factor_io_before[key]
            for key in factor_io_before.keys() & factor_io_after.keys()}
        stage('solving')
        t = time.monotonic()
        solution = lu_solve((lu,pivots), rhs, check_finite=False)
        if not np.isfinite(solution).all():
            raise FloatingPointError('Non-finite dense solution')
        record['timings_seconds']['solve'] = time.monotonic()-t
        del lu
        lu = None
        if factor_mode == 'shared':
            _drop_clean_pages(lu_map, lu_path, record)
        lu_map._mmap.close()
        lu_map = None
        if factor_mode == 'cow':
            stage('verifying_original_after_cow')
            t = time.monotonic()
            record['operator_sha256_after_solve'] = _file_sha256(operator_path)
            record['original_operator_unchanged'] = (
                record['operator_sha256_after_solve'] == record['operator_sha256_before_factorization'])
            if not record['original_operator_unchanged']:
                raise RuntimeError('Original operator changed during COW solve')
            record['timings_seconds']['cow_original_verification'] = time.monotonic()-t
        stage('checking_residual')
        t = time.monotonic()
        operator_map = np.memmap(operator_path, dtype=np.float64, mode='r', shape=(d,d), order='F')
        residual = -rhs.copy()
        for first in range(0, d, chunk_columns):
            last = min(first+chunk_columns, d)
            residual += operator_map[:, first:last] @ solution[first:last]
            _drop_clean_pages(operator_map, operator_path, record)
        if not np.isfinite(residual).all():
            raise FloatingPointError('Non-finite original-operator residual')
        norms = np.linalg.norm(rhs, axis=0)
        absolute = np.linalg.norm(residual, axis=0)
        relative = np.divide(absolute, norms, out=absolute.copy(), where=norms > 0)
        if not (np.isfinite(norms).all() and np.isfinite(absolute).all()
                and np.isfinite(relative).all()):
            raise FloatingPointError('Non-finite original-operator residual norm')
        record['absolute_residual_by_rhs'] = absolute.tolist()
        record['relative_residual_by_rhs'] = relative.tolist()
        record['relative_residual_max'] = float(np.max(relative))
        record['timings_seconds']['residual'] = time.monotonic()-t
        operator_map._mmap.close()
        operator_map = None
        stage('completed')
        return (solution[:,0] if was_vector else solution), record
    except BaseException as exc:
        record['status'] = 'failed'
        record['exception_type'] = type(exc).__name__
        record['exception_message'] = str(exc)
        if hasattr(exc,'cgroup_snapshot'):
            record['cgroup_guard_failure'] = exc.cgroup_snapshot
        record['elapsed_seconds'] = time.monotonic()-started
        _write_record(record_path, record)
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        raise MMapDenseError(str(exc), record) from exc
    finally:
        lu = None
        for mapping in [operator_map, lu_map]:
            if mapping is not None:
                try:
                    mapping._mmap.close()
                except (BufferError, OSError):
                    pass
        if not keep_matrices:
            for path in [operator_path, lu_path]:
                path.unlink(missing_ok=True)
        record['matrix_files_retained'] = {str(path): path.exists() for path in [operator_path,lu_path]}
        _write_record(record_path, record)
