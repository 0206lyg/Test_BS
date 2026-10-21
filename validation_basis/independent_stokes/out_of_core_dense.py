"""Exact Stokes assembly and tiled LU with one on-disk matrix.

Only storage and linear algebra change. The original operator residual is
reassembled from the same quadrature after destructive factorization, so no
second quadratic-sized file or resident matrix is required.
"""
from __future__ import annotations

import hashlib
import json
import mmap
import os
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np

from memory_bounded_dense import (
    MMapDenseError, _assemble_target_chunks, _check_cgroup_headroom,
    _drop_clean_pages, _release_unused_heap, _write_record,
)


def _hash_file_cache_bounded(path, details, chunk_bytes=64*1024**2):
    """Hash one owned file, advising away its already-read clean cache pages.

    Advice is best effort and never grants memory credit. The unchanged
    strict cgroup guard still decides whether factorization may proceed.
    A digest is published only after a complete, unchanged-file read.
    """
    if int(chunk_bytes) != chunk_bytes or chunk_bytes < 1:
        raise ValueError('hash chunk_bytes must be a positive integer')
    path = Path(path)
    supported = hasattr(os,'posix_fadvise') and hasattr(os,'POSIX_FADV_DONTNEED')
    details.update(status='reading', algorithm='sha256', chunk_bytes=int(chunk_bytes),
                   bytes_read=0, digest_valid=False,
                   cache_advice_scope='only_this_operator_file_already_read_ranges_then_entire_file',
                   posix_fadvise_supported=supported,
                   advice_calls_accepted=0, advice_calls_failed=0,
                   cache_eviction_guaranteed=False)
    digest = hashlib.sha256()

    def advise(fd, offset, length):
        if not supported:
            return
        try:
            os.posix_fadvise(fd,int(offset),int(length),os.POSIX_FADV_DONTNEED)
            details['advice_calls_accepted'] += 1
        except OSError as exc:
            details['advice_calls_failed'] += 1
            errors = details.setdefault('advice_errors',[])
            if len(errors) < 8:
                errors.append({'offset':int(offset),'length':int(length),
                               'errno':exc.errno,'message':str(exc)})

    try:
        with path.open('rb',buffering=0) as stream:
            initial = os.fstat(stream.fileno())
            details['expected_bytes'] = initial.st_size
            while True:
                block = stream.read(int(chunk_bytes))
                if not block:
                    break
                offset, length = details['bytes_read'],len(block)
                digest.update(block)
                details['bytes_read'] += length
                del block
                advise(stream.fileno(),offset,length)
            final = os.fstat(stream.fileno())
            unchanged = ((initial.st_dev,initial.st_ino,initial.st_size,initial.st_mtime_ns)
                         == (final.st_dev,final.st_ino,final.st_size,final.st_mtime_ns))
            details['file_unchanged_during_hash'] = unchanged
            if not unchanged or details['bytes_read'] != initial.st_size:
                raise RuntimeError('Operator changed or was incompletely read while hashing')
            advise(stream.fileno(),0,0)
        result = digest.hexdigest()
        details.update(status='completed',digest_valid=True,sha256=result)
        return result
    except BaseException as exc:
        details.update(status='failed',digest_valid=False,
                       exception_type=type(exc).__name__,exception_message=str(exc))
        details.pop('sha256',None)
        raise


class TargetChunkBuffer:
    """Translate the exact assembler's global row indices into one row block."""

    def __init__(self, node_count, target_chunk):
        self.dimension = 3 * int(node_count)
        self.target_chunk = int(target_chunk)
        self.first_row = 0
        self.values = np.empty((min(3 * target_chunk, self.dimension), self.dimension))

    def _local_key(self, key):
        row, column = key
        if isinstance(row, slice):
            start, stop, step = row.indices(self.dimension)
            last = start + ((stop - start - 1) // step) * step
            if step < 1 or start < self.first_row or last >= self.first_row + len(self.values):
                raise IndexError('Assembler attempted a row outside the active target block')
            local = slice(start - self.first_row, stop - self.first_row, step)
        else:
            local = np.asarray(row) - self.first_row
            if np.any(local < 0) or np.any(local >= len(self.values)):
                raise IndexError('Assembler diagonal outside active target block')
        return local, column

    def __getitem__(self, key):
        return self.values[self._local_key(key)]

    def __setitem__(self, key, value):
        self.values[self._local_key(key)] = value

    def advance(self, completed_nodes):
        self.first_row = 3 * int(completed_nodes)
        remaining = self.dimension - self.first_row
        if remaining and remaining < len(self.values):
            self.values = self.values[:remaining]


def assemble_buffered(backend, mesh, eps, alpha, beta, mapping,
                      target_chunk=128, stripe_columns=512, progress=None):
    """Write each target block in F-contiguous column stripes.

    The native assembler's component views are C-ordered and would otherwise
    scatter individual components across the complete F-order matrix file.
    A bounded RAM buffer prevents repeated broad page-fault sweeps.
    """
    n = int(mesh['srcvals'].shape[1])
    sink = TargetChunkBuffer(n, min(target_chunk,n))
    copy_seconds = 0.

    def block_done(first, last, rfac0):
        nonlocal copy_seconds
        started = time.monotonic()
        for column in range(0,3*n,stripe_columns):
            end = min(column+stripe_columns,3*n)
            stripe = np.asfortranarray(sink.values[:,column:end])
            mapping[3*first:3*last,column:end] = stripe
            del stripe
        if progress is not None:
            progress(first,last,rfac0)
        copy_seconds += time.monotonic()-started
        sink.advance(last)

    _assemble_target_chunks(backend,mesh,eps,alpha,beta,sink,
                            min(target_chunk,n),block_done)
    return copy_seconds


def reassembled_residual(backend, mesh, rhs, solution, eps, alpha=1., beta=1.,
                         target_chunk=128, progress=None):
    """Apply original A to all RHS through the unchanged exact assembler."""
    n = int(mesh['srcvals'].shape[1])
    sink = TargetChunkBuffer(n, min(target_chunk, n))
    absolute_squared = np.zeros(rhs.shape[1])

    def block_done(first, last, rfac0):
        block = sink.values
        if not np.isfinite(block).all():
            raise FloatingPointError('Non-finite reassembled original operator')
        residual = block @ solution - rhs[3 * first:3 * last]
        if not np.isfinite(residual).all():
            raise FloatingPointError('Non-finite reassembled residual')
        absolute_squared[:] += np.einsum('ij,ij->j', residual, residual)
        if progress is not None:
            progress(first, last, rfac0)
        sink.advance(last)

    _assemble_target_chunks(backend, mesh, eps, alpha, beta, sink,
                            min(target_chunk, n), block_done)
    absolute = np.sqrt(absolute_squared)
    norms = np.linalg.norm(rhs, axis=0)
    relative = np.divide(absolute, norms, out=absolute.copy(), where=norms > 0)
    if not np.isfinite(relative).all():
        raise FloatingPointError('Non-finite relative residual')
    return absolute, relative


def solve_out_of_core_dense(backend, mesh, rhs, eps, workdir, alpha=1., beta=1.,
                            max_memory_gb=4., panel_columns=1024,
                            stripe_columns=512, assembly_target_chunk=128,
                            cgroup_reserve_gb=.25, disk_reserve_gb=.5,
                            progress=None, keep_matrices=False):
    """Assemble, factor/solve, then independently reassemble A for residuals."""
    from out_of_core_lu import factor_solve, working_memory_bytes

    n = int(mesh['srcvals'].shape[1])
    dimension = 3 * n
    rhs = np.asarray(rhs, dtype=np.float64)
    if (rhs.ndim != 2 or rhs.shape[0] != dimension or rhs.shape[1] < 1 or
            not np.isfinite(rhs).all()):
        raise ValueError('Finite matrix of RHS with 3*N rows is required')
    if not (n > 0 and 0 < eps < 1 and alpha > 0 and beta > 0 and
            np.isfinite(alpha) and np.isfinite(beta) and
            np.isfinite(max_memory_gb) and max_memory_gb > 0):
        raise ValueError('Invalid operator parameters or memory budget')
    for value in (panel_columns, stripe_columns, assembly_target_chunk):
        if int(value) != value or value < 1:
            raise ValueError('Block dimensions must be positive integers')
    if not all(np.isfinite(value) and value >= 0 for value in
               (cgroup_reserve_gb, disk_reserve_gb)):
        raise ValueError('Resource reserves must be finite and nonnegative')
    target_chunk = min(int(assembly_target_chunk), n)
    matrix_bytes = 8 * dimension * dimension
    # The residual row buffer is in addition to native near-quadrature work.
    assembly_working = (256 * n * target_chunk +
                        16 * target_chunk * len(mesh['norders']) +
                        8 * dimension * 3 * target_chunk + 256 * 1024**2 +
                        4 * rhs.nbytes)
    factor_working = working_memory_bytes(dimension,rhs.shape[1],
                                         panel_columns,stripe_columns)
    maximum_working = max(assembly_working,factor_working)
    if maximum_working > max_memory_gb * 1e9:
        raise MemoryError(f'Exact assembly/factor working estimate {maximum_working/1e9:.3f} GB '
                          f'exceeds memory budget {max_memory_gb:g} GB')
    # Reject an already unavailable LU before spending time assembling A.
    # Factor_solve still repeats the unchanged actual guard at factorization.
    initial_guard = _check_cgroup_headroom(maximum_working, int(cgroup_reserve_gb * 1e9))
    workdir = Path(workdir).expanduser().resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    disk_free = shutil.disk_usage(workdir).free
    if disk_free < matrix_bytes + disk_reserve_gb * 1e9:
        raise OSError('Insufficient disk for one exact matrix and reserve')
    folder = Path(tempfile.mkdtemp(prefix='ooc_dense_', dir=workdir))
    matrix_path = folder / 'operator_then_lu.f64'
    diagnostic_path = folder / 'diagnostics.json'
    record = dict(schema='independent_stokes_out_of_core_dense_v3', status='preparing',
                  node_count=n, dof_count=dimension, rhs_count=rhs.shape[1],
                  alpha=float(alpha), beta=float(beta), quadrature_tolerance=float(eps),
                  matrix_path=str(matrix_path), matrix_bytes=matrix_bytes,
                  matrix_order='F', matrix_dtype='float64',
                  estimated_assembly_and_residual_working_bytes=assembly_working,
                  estimated_factor_working_bytes=factor_working,
                  estimated_maximum_working_bytes=maximum_working,
                  admission_policy='actual_maximum_stage_headroom_before_matrix_creation_then_unchanged_stage_guards',
                  cgroup_preflight=initial_guard, memory_budget_bytes=int(max_memory_gb * 1e9),
                  disk_free_at_start_bytes=disk_free,
                  panel_columns=int(panel_columns), stripe_columns=int(stripe_columns),
                  assembly_target_chunk=target_chunk, assembly_completed_targets=0,
                  residual_completed_targets=0, original_operator_residual_checked=False,
                  operator_residual_method='fresh_exact_target_chunk_reassembly_after_destructive_LU',
                  assembly_output_method='rolling_target_buffer_F_contiguous_column_stripes',
                  physical_accuracy_qualified=False, timings_seconds={})
    started = time.monotonic()
    mapping = None

    def stage(name):
        record['status'] = name
        record['elapsed_seconds'] = time.monotonic() - started
        _write_record(diagnostic_path, record)
        if progress is not None:
            progress(name, json.loads(json.dumps(record)))

    try:
        stage('assembling')
        mapping = np.memmap(matrix_path, dtype=np.float64, mode='w+',
                            shape=(dimension, dimension), order='F')
        try:
            mapping._mmap.madvise(mmap.MADV_RANDOM)
            record['assembly_madvise_random'] = True
        except (AttributeError,OSError,ValueError):
            record['assembly_madvise_random'] = False

        def assembly_progress(first, last, rfac0):
            _drop_clean_pages(mapping, matrix_path, record)
            record['assembly_completed_targets'] = last
            stage('assembling')

        t = time.monotonic()
        record['assembly_output_seconds'] = assemble_buffered(
            backend,mesh,eps,alpha,beta,mapping,target_chunk,stripe_columns,assembly_progress)
        record['timings_seconds']['assembly'] = time.monotonic() - t
        _drop_clean_pages(mapping, matrix_path, record)
        mapping._mmap.close()
        mapping = None
        stage('hashing_original_operator')
        t = time.monotonic()
        record['original_operator_hash'] = {}
        record['original_operator_sha256_before_factorization'] = _hash_file_cache_bounded(
            matrix_path,record['original_operator_hash'])
        record['timings_seconds']['original_operator_hash'] = time.monotonic()-t
        _release_unused_heap()

        def factor_progress(name, detail):
            record['tiled_lu'] = detail
            stage('tiled_lu_' + name)

        t = time.monotonic()
        solution, detail = factor_solve(matrix_path, dimension, rhs,
            panel_columns=panel_columns, stripe_columns=stripe_columns,
            max_memory_gb=max_memory_gb, cgroup_reserve_gb=cgroup_reserve_gb,
            progress=factor_progress, diagnostics_path=folder / 'lu_diagnostics.json')
        record['tiled_lu'] = detail
        record['timings_seconds']['factor_and_solve'] = time.monotonic() - t
        if not np.isfinite(solution).all():
            raise FloatingPointError('Non-finite tiled-LU solution')
        if not keep_matrices:
            matrix_path.unlink()
        _release_unused_heap()
        record['cgroup_before_residual'] = _check_cgroup_headroom(
            assembly_working, int(cgroup_reserve_gb * 1e9))

        def residual_progress(first, last, rfac0):
            record['residual_completed_targets'] = last
            stage('checking_reassembled_original_residual')

        t = time.monotonic()
        absolute, relative = reassembled_residual(backend, mesh, rhs, solution, eps,
            alpha, beta, target_chunk, residual_progress)
        record['timings_seconds']['original_operator_residual'] = time.monotonic() - t
        record['absolute_residual_by_rhs'] = absolute.tolist()
        record['relative_residual_by_rhs'] = relative.tolist()
        record['relative_residual_max'] = float(relative.max())
        record['original_operator_residual_checked'] = True
        if record['relative_residual_max'] > 1e-8:
            raise ArithmeticError('Tiled-LU original-operator residual exceeds 1e-8')
        record['matrix_file_retained'] = matrix_path.exists()
        stage('completed')
        return solution, record
    except BaseException as exc:
        record['exception_type'] = type(exc).__name__
        record['exception_message'] = str(exc)
        if hasattr(exc, 'diagnostics'):
            record['failed_tiled_lu_diagnostics'] = exc.diagnostics
        if hasattr(exc, 'cgroup_snapshot'):
            record['cgroup_guard_failure'] = exc.cgroup_snapshot
        stage('failed')
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        raise MMapDenseError(str(exc), record) from exc
    finally:
        if mapping is not None:
            mapping._mmap.close()
        if not keep_matrices:
            matrix_path.unlink(missing_ok=True)
        record['matrix_file_retained'] = matrix_path.exists()
        _write_record(diagnostic_path, record)
