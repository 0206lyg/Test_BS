#!/usr/bin/env python3
"""Independent panel-BIE reference, with the recorded AR20 pose as default.

Explicit AR/size/orientation options support new reference calculations. This
driver never loads a fitted scattering basis, creates a production table, or
labels a pair physically qualified from one solve. Generic records use a separate
schema and cannot silently pass the strict recorded-AR20 comparator.
"""
from __future__ import annotations

import os
# libgfortran may be loaded by NumPy/SciPy, before the reference backend.
os.environ.setdefault('GFORTRAN_UNBUFFERED_ALL', '1')

import argparse
import hashlib
import json
import ctypes
import sys
from pathlib import Path
import resource
import subprocess
import tempfile
import time
import zipfile
import numpy as np
from scipy.linalg import lu_factor, lu_solve

from backend import Backend, representation_moments
from geometry import (build_mesh, recorded_pose, spheroid_pose, strain_basis,
                      velocity_modes, refine_descriptors, C_NM, THETA_I, THETA_J, PHI)

SCHEMA = 'independent_ar20_panel_bie_v1'
GENERIC_SCHEMA = 'independent_spheroid_panel_bie_v1'
CONVENTION = 'slurry_full17_c_eta_1_v1'
SOLVER_CONTRACT = 'dense_combined_dirichlet_full17_v1'
CHECKPOINT_ARRAYS = ('srcvals', 'weights', 'body_ids', 'norders', 'ixyzs', 'iptype')


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_record(path):
    payload = Path(path).read_bytes()
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError(f'Expected a JSON object: {path}')
    return value, hashlib.sha256(payload).hexdigest(), payload


def array_sha256(value):
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256(str((array.shape, array.dtype.str)).encode('ascii'))
    digest.update(array.tobytes())
    return digest.hexdigest()


def finite_error(value, limit):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and np.isfinite(value) and 0 <= value <= limit)


def validate_pose(data, expected, include_gap=True):
    """Reject adaptation/resume/operator evidence with a different physical pose."""
    keys = ('aspect_ratio', 'theta_i', 'theta_j', 'phi', 'c_nm') + (('h_over_c',) if include_gap else ())
    for key in keys:
        actual = data.get(key)
        if (not isinstance(actual, (int, float)) or isinstance(actual, bool)
                or not np.isfinite(actual) or abs(actual-expected[key]) > 1e-13):
            raise ValueError(f'Wrong reference geometry: {key}')


def flush_native(library):
    """Flush the actual backend's gfortran units as well as C stdio."""
    if library is not None:
        try:
            flush = library._gfortran_flush_i4
        except AttributeError as exc:
            raise RuntimeError('Cannot flush the pinned gfortran backend diagnostics') from exc
        flush.argtypes = [ctypes.POINTER(ctypes.c_int32)]
        flush.restype = None
        for number in (6, 0):  # gfortran preconnected stdout and stderr units.
            unit = ctypes.c_int32(number)
            flush(ctypes.byref(unit))
    ctypes.CDLL(None).fflush(None)


def logged_fortran(path, operation, library=None):
    """Capture native diagnostics; an invalid quadrature is a failed check."""
    sys.stdout.flush()
    sys.stderr.flush()
    flush_native(library)
    saved = [os.dup(1), os.dup(2)]
    with open(path,'wb') as log:
        try:
            os.dup2(log.fileno(),1)
            os.dup2(log.fileno(),2)
            result = operation()
        finally:
            try:
                flush_native(library)
            finally:
                for destination, original in zip((1, 2), saved):
                    os.dup2(original, destination)
                    os.close(original)
    diagnostic = path.read_text(errors='replace')
    if 'Failed to ' in diagnostic or 'Returning without' in diagnostic:
        raise RuntimeError(f'Native quadrature failed; see {path}')
    return result


def write_json(path, value):
    payload = (json.dumps(value, indent=2, allow_nan=False)+'\n').encode()
    atomic_bytes(path, payload)


def atomic_bytes(path, payload):
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name+'.', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_checkpoint(path, mesh, density, rhs):
    """Publish a complete NPZ atomically; a failed write preserves its predecessor."""
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name+'.', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, density=density, rhs=rhs,
                                **{key: mesh[key] for key in CHECKPOINT_ARRAYS})
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return file_sha256(path)


def load_checkpoint(report_path, previous, current, mesh, rhs, pose):
    """Bind saved density to the complete mesh and physical RHS before reuse.

    Legacy NPZs lack a saved SHA/RHS. They remain usable only when their full
    mesh and solver/geometry identities match and their density reproduces the
    previously recorded raw moments. A SHA is established for subsequent use.
    """
    validate_pose(previous, current)
    if previous.get('schema') != current.get('schema', SCHEMA) or previous.get('convention_id') != CONVENTION:
        raise ValueError('Resume schema/convention mismatch')
    if previous.get('status') not in ('boundary_check', 'complete'):
        raise ValueError('Resume requires a completed linear solve checkpoint')
    for key in ('order','level','focus','alpha','beta','isolated_body',
                'quadrature_tolerance','node_count','library_sha256','discretization_sha256'):
        if previous.get(key) != current[key]:
            raise ValueError(f'Resume mismatch: {key}')
    for key, default in (('rim_levels',0),('rim_width',.05),('contact_levels',0),('contact_radius',.25)):
        if previous.get(key,default) != current[key]:
            raise ValueError(f'Resume mismatch: {key}')
    for key, default in (('geometry_refine_tolerance',0.),('geometry_max_refinements',3)):
        if previous.get(key,default) != current.get(key,default):
            raise ValueError(f'Resume geometry preflight mismatch: {key}')
    if current.get('geometry_refine_tolerance',0.) > 0:
        if previous.get('source_hashes',{}).get('geometry_preflight.py') != current['source_hashes'].get('geometry_preflight.py'):
            raise ValueError('Resume geometry preflight implementation mismatch')
    for hashes in (previous.get('source_hashes', {}),
                   previous.get('resumed_solver_source_hashes', previous.get('source_hashes', {}))):
        for key in ('backend.py', 'geometry.py'):
            if hashes.get(key) != current['source_hashes'][key]:
                raise ValueError(f'Resume implementation mismatch: {key}')
    if previous.get('solver_contract', SOLVER_CONTRACT) != current.get('solver_contract', SOLVER_CONTRACT):
        raise ValueError('Resume solver contract mismatch')
    if current.get('solver', 'dense') != 'dense':
        for key in ('matrix_free.py', 'near_apply.py', 'memory_bounded_dense.py'):
            if previous.get('source_hashes', {}).get(key) != current['source_hashes'].get(key):
                raise ValueError(f'Resume iterative implementation mismatch: {key}')
        for key in ('far_order_increment', 'near_radius_factor', 'near_apply_library_sha256'):
            if previous.get(key) != current.get(key):
                raise ValueError(f'Resume iterative operator mismatch: {key}')
    if current.get('solver') == 'mmap-dense' and previous.get('mmap_factor_mode', 'shared') != current.get('mmap_factor_mode', 'shared'):
        raise ValueError('Resume file-backed factorization mode mismatch')
    if current.get('solver') == 'out-of-core-dense':
        for key in ('out_of_core_dense.py', 'out_of_core_lu.py'):
            if previous.get('source_hashes', {}).get(key) != current['source_hashes'].get(key):
                raise ValueError(f'Resume out-of-core implementation mismatch: {key}')
        for key in ('lu_panel_columns', 'lu_stripe_columns'):
            if previous.get(key) != current.get(key):
                raise ValueError(f'Resume out-of-core blocking mismatch: {key}')
    if current.get('solver') == 'ram-dense':
        for key in ('ram_dense.py', 'out_of_core_dense.py'):
            if previous.get('source_hashes', {}).get(key) != current['source_hashes'].get(key):
                raise ValueError(f'Resume RAM-dense implementation mismatch: {key}')
    if not finite_error(previous.get('linear_residual'), float('inf')) or not previous.get('solution_file'):
        raise ValueError('Resume requires a finite recorded linear residual and density file')
    path = Path(previous['solution_file'])
    if not path.is_absolute():
        path = Path(report_path).parent/path
    digest = file_sha256(path)
    if previous.get('solution_sha256') is not None and previous['solution_sha256'] != digest:
        raise ValueError('Resume density SHA mismatch')
    if previous.get('rhs_sha256') is not None and previous['rhs_sha256'] != array_sha256(rhs):
        raise ValueError('Resume RHS hash mismatch')
    with np.load(path, allow_pickle=False) as saved:
        for key in CHECKPOINT_ARRAYS:
            if key not in saved or not np.array_equal(saved[key], mesh[key]):
                raise ValueError(f'Resume checkpoint arrays differ: {key}')
        if 'rhs' in saved and not np.array_equal(saved['rhs'], rhs):
            raise ValueError('Resume checkpoint RHS differs')
        density = saved['density'].copy()
    if file_sha256(path) != digest:
        raise ValueError('Resume density file changed while loading')
    if density.shape != (len(mesh['body_ids']),3,17) or not np.isfinite(density).all():
        raise ValueError('Invalid checkpoint density')
    raw, _ = representation_moments(mesh['srcvals'][:3].T, mesh['srcvals'][9:12].T,
        mesh['weights'], density, mesh['body_ids'], mesh['centers'], strain_basis(),
        current['alpha'], current['beta'], pose.a)
    old_raw = np.asarray(previous.get('raw_full17'), dtype=float)
    if (old_raw.shape != (17,17) or not np.isfinite(old_raw).all() or
            not np.allclose(raw, old_raw, rtol=1e-12, atol=1e-12*max(1.,np.abs(old_raw).max()))):
        raise ValueError('Checkpoint density does not reproduce its recorded raw moments')
    return density, {'solution_file': str(path.resolve()), 'solution_sha256': digest,
                     'rhs_sha256': array_sha256(rhs),
                     'checkpoint_identity_verification': ('sha_and_complete_arrays' if previous.get('solution_sha256')
                         else 'legacy_complete_arrays_source_identity_and_raw_moments; SHA_established_now')}


def isolated_evidence(path, current):
    data, digest, _ = read_record(path)
    failures = []
    try:
        validate_pose(data, current, include_gap=False)
    except ValueError as exc:
        failures.append(str(exc))
    body = data.get('isolated_body')
    if (data.get('schema') != current.get('schema', SCHEMA) or data.get('convention_id') != CONVENTION or
            data.get('status') != 'complete' or data.get('independent_reference') is not True or
            data.get('shares_production_solver') is not False or data.get('synthetic') is True or
            type(body) is not int or body not in (0,1)):
        failures.append('invalid_isolated_reference_identity')
    for key in ('library_sha256','alpha','beta'):
        if data.get(key) != current[key]:
            failures.append('isolated_identity_mismatch:'+key)
    for key in ('backend.py','geometry.py'):
        if data.get('source_hashes',{}).get(key) != current['source_hashes'][key]:
            failures.append('isolated_source_mismatch:'+key)
    validation = data.get('validation', {})
    if validation.get('off_grid_boundary') is not True or validation.get('boundary_evaluation_valid') is not True:
        failures.append('isolated_boundary_evaluation_not_verified')
    for key, limit in (('boundary_max_scaled_error',.001),('exact_surface_position_max_error_over_gap',.001),
                       ('linear_relative_residual',1e-8)):
        if not finite_error(validation.get(key),limit):
            failures.append('isolated_validation_missing_or_failed:'+key)
    exact = data.get('isolated_exact_validation', {})
    if exact.get('passed') is not True or any(not finite_error(exact.get('block_relative_errors',{}).get(k),.002)
                                             for k in ('force','torque','stresslet')):
        failures.append('isolated_moment_errors_missing_or_failed')
    raw = np.asarray(data.get('raw_full17'),dtype=float)
    if raw.shape != (17,17) or not np.isfinite(raw).all() or np.linalg.norm(raw) == 0:
        failures.append('invalid_isolated_raw_matrix')
    elif not finite_error(float(np.linalg.norm(raw-raw.T)/np.linalg.norm(raw)),.001):
        failures.append('isolated_raw_reciprocity_failed')
    if not finite_error(data.get('raw_reciprocity'),.001):
        failures.append('isolated_reported_reciprocity_failed')
    return {'path': str(Path(path).resolve()), 'sha256': digest, 'isolated_body': body,
            'passed': not failures, 'failures': failures}


def block_errors(actual, exact):
    result = {}
    for name, idx in [('force',[0,1,2,6,7,8]),
                      ('torque',[3,4,5,9,10,11]), ('stresslet',list(range(12,17)))]:
        denominator = np.linalg.norm(exact[idx])
        result[name] = float(np.linalg.norm(actual[idx]-exact[idx])/max(denominator, 1e-100))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--library', required=True)
    p.add_argument('--backend-manifest', type=Path, help='Hash-bound optional build/patch manifest')
    p.add_argument('--gap-nm', type=float, required=True)
    p.add_argument('--aspect-ratio', type=float, default=20., help='a/c >= 1, equal oblate particles')
    p.add_argument('--c-nm', type=float, default=C_NM, help='Polar semiaxis in nm; computational c=1')
    p.add_argument('--theta-i', type=float, default=THETA_I, help='Body i polar angle in radians')
    p.add_argument('--theta-j', type=float, default=THETA_J, help='Body j polar angle in radians')
    p.add_argument('--phi', type=float, default=PHI, help='Relative azimuth in radians')
    p.add_argument('--order', type=int, default=4)
    p.add_argument('--level', type=int, default=0)
    p.add_argument('--focus', type=float, default=0.)
    p.add_argument('--rim-levels', type=int, default=0)
    p.add_argument('--rim-width', type=float, default=.05)
    p.add_argument('--contact-levels', type=int, default=0)
    p.add_argument('--contact-radius', type=float, default=.25)
    p.add_argument('--resume', action='store_true', help='Reuse a matching saved density and repeat all boundary checks')
    p.add_argument('--adapt-from', type=Path, help='Refine high-residual panels of this same-pose result')
    p.add_argument('--adapt-fraction', type=float, default=.2)
    p.add_argument('--mesh-from', type=Path, help='Use exact patch descriptors from a same-pose mesh proposal or reference')
    p.add_argument('--solver', choices=['dense','ram-dense','mmap-dense','out-of-core-dense','all-near','fmm'], default='dense')
    p.add_argument('--matrix-workdir', type=Path, help='Temporary file-backed matrix directory; default beside output')
    p.add_argument('--mmap-factor-mode', choices=['shared','cow'], default='shared',
                   help='Shared file-backed LU or private copy-on-write LU; cow requires the factors to fit RAM')
    p.add_argument('--lu-panel-columns', type=int, default=1024,
                   help='Panel width of exact out-of-core LU')
    p.add_argument('--lu-stripe-columns', type=int, default=512,
                   help='Streaming stripe width of exact out-of-core LU')
    p.add_argument('--gmres-rtol', type=float, default=1e-9)
    p.add_argument('--gmres-restart', type=int, default=80)
    p.add_argument('--gmres-maxiter', type=int, default=20, help='Maximum restart cycles per RHS')
    p.add_argument('--preconditioner', choices=['body','patch','none'], default='patch')
    p.add_argument('--far-order-increment', type=int, default=0)
    p.add_argument('--near-radius-factor', type=float, default=1.)
    p.add_argument('--near-apply-library', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--quad-tol', type=float, default=1e-7)
    p.add_argument('--max-memory-gb', type=float, default=6.)
    p.add_argument('--check-points-per-patch', type=int, default=3)
    p.add_argument('--geometry-refine-tolerance', type=float, default=0.,
                   help='Sampled polynomial position error/gap target before RAM LU; 0 disables')
    p.add_argument('--geometry-max-refinements', type=int, default=3,
                   help='Maximum cheap selective geometry refinements before RAM LU')
    p.add_argument('--isolated-body', type=int, choices=[0,1])
    p.add_argument('--alpha', type=float, default=1.)
    p.add_argument('--beta', type=float, default=1.)
    p.add_argument('--operator-export', type=Path)
    p.add_argument('--isolated-validation', type=Path, action='append', default=[])
    p.add_argument('--resolution-level', type=int)
    args = p.parse_args()
    if not (args.gap_nm > 0 and args.order >= 2 and args.level >= 0 and
            0 < args.quad_tol < .01 and args.max_memory_gb > 0 and
            args.alpha > 0 and args.beta > 0 and args.check_points_per_patch >= 1):
        p.error('Positive gap, valid resolution/tolerances and positive combined strengths required')
    if args.mesh_from and args.adapt_from:
        p.error('Choose mesh-from or adapt-from, not both')
    if not (0 < args.gmres_rtol < 1 and args.gmres_restart > 0 and args.gmres_maxiter > 0):
        p.error('Invalid GMRES controls')
    if not (np.isfinite(args.geometry_refine_tolerance) and 0 <= args.geometry_refine_tolerance < 1
            and args.geometry_max_refinements >= 0):
        p.error('Invalid geometry preflight controls')
    if args.geometry_refine_tolerance > 0 and args.solver != 'ram-dense':
        p.error('Geometry preflight is supported for the new ram-dense workflow')
    try:
        pose = spheroid_pose(args.gap_nm, aspect_ratio=args.aspect_ratio, c_nm=args.c_nm,
                             theta_i=args.theta_i, theta_j=args.theta_j, phi=args.phi)
    except ValueError as exc:
        p.error(str(exc))
    schema = SCHEMA if pose.is_recorded else GENERIC_SCHEMA
    start = time.monotonic()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    b = Backend(args.library)
    manifest_path = args.backend_manifest
    if manifest_path is None:
        candidate = Path(args.library).resolve().parent.parent/'patch_manifest.json'
        if candidate.exists():
            manifest_path = candidate
    build_manifest = None
    if manifest_path is not None:
        build_manifest, build_manifest_sha, _ = read_record(manifest_path)
        if build_manifest.get('library_sha256') != file_sha256(args.library):
            raise ValueError('Backend manifest does not identify the selected library')
        if args.order < build_manifest.get('supported_panel_order_minimum', 2):
            raise ValueError('Panel order is below the documented patched-backend range')
    uv = b.discretization(args.order)[0]
    adapted = None
    adaptive_parent = None
    if args.mesh_from:
        parent, parent_sha, _ = read_record(args.mesh_from)
        validate_pose(parent, pose.metadata())
        if (parent.get('schema') not in (schema, ('independent_ar20_mesh_proposal_v1' if pose.is_recorded
                                                    else 'independent_spheroid_mesh_proposal_v1')) or
                parent.get('isolated_body') != args.isolated_body or parent.get('focus', 0.) != args.focus):
            raise ValueError('Mesh proposal must use the same pose, bodies and parameterization')
        adapted = parent['patch_descriptors']
        adaptive_parent = {'path':str(args.mesh_from.resolve()), 'sha256':parent_sha,
                           'selection':'explicit same-pose patch descriptors'}
    if args.adapt_from:
        parent, parent_sha, _ = read_record(args.adapt_from)
        validate_pose(parent, pose.metadata())
        if parent.get('schema') != schema or parent.get('convention_id') != CONVENTION:
            raise ValueError('Adaptive parent schema/convention mismatch')
        if parent['h_over_c'] != pose.gap_over_c or parent['isolated_body'] != args.isolated_body:
            raise ValueError('Adaptive parent must have the same gap and bodies')
        if not 0 < args.adapt_fraction <= 1:
            raise ValueError('adapt-fraction must lie in (0,1]')
        scores = np.asarray(parent['validation']['boundary_per_patch_max_scaled_error'])
        if (scores.shape != (len(parent['patch_descriptors']),) or not len(scores) or
                not np.isfinite(scores).all() or np.any(scores < 0)):
            raise ValueError('Adaptive parent needs finite nonnegative scores for every patch')
        selected = np.argsort(scores)[-max(1,int(np.ceil(len(scores)*args.adapt_fraction))):]
        adapted = refine_descriptors(parent['patch_descriptors'],selected)
        adaptive_parent = {'path':str(args.adapt_from.resolve()),
                           'sha256':parent_sha,
                           'selected_patches':selected.tolist(),'selection':'largest sampled no-slip residual'}
    mesh = build_mesh(args.gap_nm, args.order, args.level, uv,
                      isolated_body=args.isolated_body, focus=args.focus,
                      rim_levels=args.rim_levels, rim_width=args.rim_width,
                      contact_levels=args.contact_levels, contact_radius=args.contact_radius,
                      patch_descriptors=adapted, pose=pose)
    geometry_preflight = None
    if args.geometry_refine_tolerance > 0:
        from geometry_preflight import refine_to_tolerance, GeometryPreflightError
        def rebuild_geometry(descriptors):
            return build_mesh(args.gap_nm,args.order,args.level,uv,
                isolated_body=args.isolated_body,focus=args.focus,
                rim_levels=args.rim_levels,rim_width=args.rim_width,
                contact_levels=args.contact_levels,contact_radius=args.contact_radius,
                patch_descriptors=descriptors,pose=pose)
        try:
            mesh,geometry_preflight = refine_to_tolerance(mesh,uv,pose,rebuild_geometry,
                tolerance=args.geometry_refine_tolerance,max_refinements=args.geometry_max_refinements,
                max_memory_gb=args.max_memory_gb,check_points_per_patch=args.check_points_per_patch,
                progress=lambda result: write_json(args.output.with_suffix('.geometry_progress.json'),result))
        except GeometryPreflightError as exc:
            failure = dict(pose.metadata(),schema=schema,status='geometry_preflight_unresolved',
                independent_reference=True,shares_production_solver=False,
                physical_accuracy_qualified=False,physical_pair_gate_passed=False,
                failure_type=type(exc).__name__,failure_message=str(exc),geometry_preflight=exc.diagnostics)
            destination = args.output if not args.output.exists() else args.output.with_name(
                args.output.stem+'.geometry_failed.'+str(time.time_ns())+'.json')
            write_json(destination,failure)
            raise
    mesh = b.coefficients(mesh)
    xyz = mesh['srcvals'][:3].T
    normals = mesh['srcvals'][9:12].T
    n = len(xyz)
    rhs = velocity_modes(xyz, mesh['body_ids'], pose).reshape(3*n,17)
    record = dict(pose.metadata())
    record.update(schema=schema,
                  convention_id=CONVENTION,
                  solver_contract=(SOLVER_CONTRACT if args.solver == 'dense' else args.solver+'_combined_dirichlet_full17_v1'),
                  solver=args.solver, far_order_increment=args.far_order_increment,
                  geometry_refine_tolerance=args.geometry_refine_tolerance,
                  geometry_max_refinements=args.geometry_max_refinements,
                  geometry_preflight=geometry_preflight,
                  mmap_factor_mode=args.mmap_factor_mode,
                  lu_panel_columns=args.lu_panel_columns,
                  lu_stripe_columns=args.lu_stripe_columns,
                  near_radius_factor=args.near_radius_factor,
                  near_apply_library_sha256=(file_sha256(args.near_apply_library) if args.near_apply_library else None),
                  rhs_sha256=array_sha256(rhs),
                  independent_reference=True,
                  shares_production_solver=False,
                  physical_accuracy_qualified=False,
                  physical_pair_gate_passed=False,
                  status='assembling', effective_gap_nm=args.gap_nm,
                  isolated_body=args.isolated_body,
                  order=args.order, level=args.level, focus=args.focus,
                  rim_levels=args.rim_levels, rim_width=args.rim_width,
                  contact_levels=args.contact_levels, contact_radius=args.contact_radius,
                  resolution_level=(args.resolution_level if args.resolution_level is not None
                                    else args.level*100+args.order),
                  node_count=n, patch_count=len(mesh['norders']),
                  patch_descriptors=[(int(body),vertices.tolist()) for body,vertices in mesh['patches']],
                  adaptive_parent=adaptive_parent,
                  alpha=args.alpha, beta=args.beta, quadrature_tolerance=args.quad_tol,
                  backend='fmm3dbie_combined_exterior_panel_dirichlet',
                  backend_commit='be6b2b3700b77e9cbf20c512ff4f0922aaee45d9',
                  fmm_commit='d2b5e6e983e1ec915d15330ce74df44221a9ac42',
                  library_sha256=hashlib.sha256(Path(args.library).read_bytes()).hexdigest(),
                  discretization_sha256=hashlib.sha256(b''.join(
                      np.asarray(mesh[key]).tobytes(order='F') for key in
                      ['srcvals','weights','ixyzs','norders'])).hexdigest(),
                  source_hashes={f:hashlib.sha256(Path(__file__).with_name(f).read_bytes()).hexdigest()
                                 for f in ['backend.py','geometry.py','run_reference.py']})
    if args.solver != 'dense':
        for name in ('matrix_free.py', 'near_apply.py', 'memory_bounded_dense.py'):
            source = Path(__file__).with_name(name)
            if source.exists():
                record['source_hashes'][name] = file_sha256(source)
    if args.solver == 'out-of-core-dense':
        for name in ('out_of_core_dense.py', 'out_of_core_lu.py'):
            record['source_hashes'][name] = file_sha256(Path(__file__).with_name(name))
    if args.solver == 'ram-dense':
        for name in ('ram_dense.py', 'out_of_core_dense.py'):
            record['source_hashes'][name] = file_sha256(Path(__file__).with_name(name))
    if args.geometry_refine_tolerance > 0:
        record['source_hashes']['geometry_preflight.py'] = file_sha256(Path(__file__).with_name('geometry_preflight.py'))
        record['source_hashes']['contact_check_sites.py'] = file_sha256(Path(__file__).with_name('contact_check_sites.py'))
    if build_manifest is not None:
        record['backend_build_manifest'] = {'path':str(manifest_path.resolve()),
            'sha256':build_manifest_sha, 'patch_policy':build_manifest.get('patch_policy'),
            'patch_sha256':build_manifest.get('patch_sha256')}
    previous = None
    checkpoint_density = None
    checkpoint_identity = None
    report_path = args.output
    if args.resume and args.output.exists():
        previous, previous_sha, previous_bytes = read_record(args.output)
        checkpoint_density, checkpoint_identity = load_checkpoint(args.output, previous, record, mesh, rhs, pose)
        attempt = str(time.time_ns())
        report_path = args.output.with_name(args.output.stem+'.resume.'+attempt+'.json')
        preserved = args.output.with_name(args.output.stem+'.before_resume.'+attempt+'.json')
        atomic_bytes(preserved, previous_bytes)
        record.update(checkpoint_identity)
        record['resume_previous_report'] = {'path':str(preserved.resolve()),'sha256':previous_sha}
        record['resume_attempt_report'] = str(report_path.resolve())
    elif args.output.exists():
        raise FileExistsError(f'{args.output} exists; choose another output or --resume')
    elif args.resume:
        raise FileNotFoundError(f'Resume report does not exist: {args.output}')
    elif args.output.with_suffix('.solution.npz').exists():
        raise FileExistsError('An existing density checkpoint would be overwritten; choose another output')
    source_archive = report_path.with_suffix('.sources.zip')
    with zipfile.ZipFile(source_archive, 'x', zipfile.ZIP_DEFLATED) as archive:
        for name, digest in record['source_hashes'].items():
            payload = Path(__file__).with_name(name).read_bytes()
            if hashlib.sha256(payload).hexdigest() != digest:
                raise RuntimeError(f'Source changed during launch: {name}')
            archive.writestr(name, payload)
        archive.writestr('source_hashes.json', json.dumps(record['source_hashes'], indent=2))
        if build_manifest is not None:
            archive.writestr('backend_manifest.json', json.dumps(build_manifest, indent=2))
    record['source_archive'] = {'path':str(source_archive.resolve()),'sha256':file_sha256(source_archive)}
    write_json(report_path, record)
    print(json.dumps({'stage':'assembly','nodes':n,'patches':len(mesh['norders']),
                      'gap_nm':args.gap_nm,'isolated_body':args.isolated_body}), flush=True)
    t = time.monotonic()
    if previous is None and args.solver == 'dense':
        matrix = logged_fortran(report_path.with_suffix('.assembly.log'),lambda:
            b.matrix(mesh, args.quad_tol, args.alpha, args.beta, args.max_memory_gb),b.lib)
        record['assembly_seconds'] = time.monotonic()-t
        print(json.dumps({'stage':'factorization','assembly_seconds':record['assembly_seconds']}),flush=True)
        t = time.monotonic()
        lu, piv = lu_factor(matrix, overwrite_a=False, check_finite=True)
        solution = lu_solve((lu,piv), rhs, check_finite=True)
        relative_residual = np.linalg.norm(matrix@solution-rhs)/np.linalg.norm(rhs)
        record['solve_seconds'] = time.monotonic()-t
        del matrix, lu, piv
        density = solution.reshape(n,3,17)
    elif previous is None and args.solver in ('ram-dense','mmap-dense','out-of-core-dense'):
        from memory_bounded_dense import solve_mmap_dense, MMapDenseError
        workdir = args.matrix_workdir or args.output.parent/'matrix_work'
        def mmap_progress(stage, diagnostic):
            write_json(report_path.with_suffix('.storage_progress.json'), diagnostic)
        try:
            if args.solver == 'ram-dense':
                from ram_dense import solve_ram_dense
                solve_storage = lambda: solve_ram_dense(b, mesh, rhs, args.quad_tol, workdir,
                    alpha=args.alpha, beta=args.beta, max_memory_gb=args.max_memory_gb,
                    progress=mmap_progress)
            elif args.solver == 'out-of-core-dense':
                from out_of_core_dense import solve_out_of_core_dense
                solve_storage = lambda: solve_out_of_core_dense(b, mesh, rhs, args.quad_tol, workdir,
                    alpha=args.alpha, beta=args.beta, max_memory_gb=args.max_memory_gb,
                    panel_columns=args.lu_panel_columns, stripe_columns=args.lu_stripe_columns,
                    progress=mmap_progress)
            else:
                solve_storage = lambda: solve_mmap_dense(b, mesh, rhs, args.quad_tol, workdir,
                    alpha=args.alpha, beta=args.beta, max_memory_gb=args.max_memory_gb,
                    progress=mmap_progress, factor_mode=args.mmap_factor_mode)
            solution, diagnostic = logged_fortran(report_path.with_suffix('.assembly.log'), lambda:
                solve_storage(), b.lib)
        except Exception as exc:
            record.update(status='linear_solve_failed', storage_solver=getattr(exc,'diagnostics',None),
                          failure_type=type(exc).__name__, failure_message=str(exc))
            write_json(report_path, record)
            raise
        record['storage_solver'] = diagnostic
        record['assembly_seconds'] = diagnostic['timings_seconds']['assembly']
        record['solve_seconds'] = sum(value for key,value in diagnostic['timings_seconds'].items() if key != 'assembly')
        relative_residual = diagnostic['relative_residual_max']
        density = solution.reshape(n,3,17)
        del solution
    elif previous is None:
        from matrix_free import MatrixFreeContext, GMRESConvergenceError
        context = logged_fortran(report_path.with_suffix('.assembly.log'), lambda:
            MatrixFreeContext(b, mesh, args.quad_tol, args.alpha, args.beta,
                max_memory_gb=args.max_memory_gb,
                far_order_increment=args.far_order_increment,
                near_radius_factor=args.near_radius_factor,
                all_patches_near=args.solver == 'all-near',
                near_apply_library=args.near_apply_library), b.lib)
        record['assembly_seconds'] = time.monotonic()-t
        record['status'] = 'iterative_solve'
        write_json(report_path, record)
        print(json.dumps({'stage':'iterative_solve','assembly_seconds':record['assembly_seconds'],
                          'solver':args.solver,'preconditioner':args.preconditioner}),flush=True)
        progress_records = []
        def progress(index, prefix, diagnostic):
            progress_records.append(diagnostic)
            write_json(report_path.with_suffix('.progress.json'), {
                'physical_accuracy_qualified':False,'completed_rhs':index+1,
                'right_hand_sides':progress_records})
            # Preserve all completed RHS on interruption; a partial solve is
            # explicitly not accepted by the normal reference resume path.
            with tempfile.NamedTemporaryFile(dir=report_path.parent, suffix='.npz', delete=False) as stream:
                partial_path = Path(stream.name)
                np.savez_compressed(stream, solution=prefix)
            partial_path.replace(report_path.with_suffix('.partial.npz'))
        t = time.monotonic()
        try:
            solution, iterative = logged_fortran(report_path.with_suffix('.solve.log'), lambda:
                context.solve(rhs, rtol=args.gmres_rtol, restart=args.gmres_restart,
                    maxiter=args.gmres_maxiter, preconditioner=args.preconditioner,
                    progress=progress), b.lib)
        except GMRESConvergenceError as exc:
            record.update(status='linear_solve_failed', iterative_solver=exc.diagnostics)
            write_json(report_path, record)
            raise
        record['solve_seconds'] = time.monotonic()-t
        record['iterative_solver'] = iterative
        relative_residual = max(x['relative_residual'] for x in iterative['right_hand_sides'])
        density = solution.reshape(n,3,17)
        del context, solution
    else:
        density = checkpoint_density
        relative_residual = previous['linear_residual']
        record['assembly_seconds'] = previous['assembly_seconds']
        record['solve_seconds'] = previous['solve_seconds']
        if 'iterative_solver' in previous:
            record['iterative_solver'] = previous['iterative_solver']
        if 'storage_solver' in previous:
            record['storage_solver'] = previous['storage_solver']
        record['resumed_solver_source_hashes'] = previous.get('resumed_solver_source_hashes',previous['source_hashes'])
    raw, flux = representation_moments(xyz, normals, mesh['weights'], density,
                                     mesh['body_ids'], mesh['centers'], strain_basis(),
                                     args.alpha, args.beta, pose.a)
    record.update(status='boundary_check', raw_full17=raw.tolist(),
                  symmetric_full17=((raw+raw.T)*.5).tolist(),
                  linear_residual=float(relative_residual),
                  raw_reciprocity=float(np.linalg.norm(raw-raw.T)/np.linalg.norm(raw)),
                  density_normal_flux=flux,
                  moments={'method':'combined_layer_farfield_moments',
                           'isolated_spheroid_validation_passed':False,
                           'density_is_traction':False})
    flux_scaled = []
    for body in range(2):
        mask = mesh['body_ids']==body
        denominator = np.einsum('n,nr->r',mesh['weights'][mask],
                                 np.linalg.norm(density[mask],axis=1))
        flux_scaled.extend((np.abs(flux[body])/np.maximum(denominator,1e-100)).tolist())
    record['moments']['max_scaled_flux_error'] = float(max(flux_scaled))
    if previous is None:
        solution_path = args.output.with_suffix('.solution.npz')
        record['solution_sha256'] = write_checkpoint(solution_path, mesh, density, rhs)
        record['solution_file'] = str(solution_path.resolve())
    # A resume never rewrites its unchanged, already verified density file.
    write_json(report_path, record)
    print(json.dumps({'stage':'boundary_check','raw_reciprocity':record['raw_reciprocity'],
                      'linear_residual':float(relative_residual)}), flush=True)
    if args.geometry_refine_tolerance > 0:
        from geometry_preflight import boundary_sites
        patches,checks,extra_count = boundary_sites(mesh,pose,args.check_points_per_patch,broad_contact=True)
        extra_patches = [None]*extra_count
    else:
        # Deterministic staggered sites, never Gauss collocation nodes. The irrational
        # rotations also allow increasing the number of checks without coincidence.
        count = args.check_points_per_patch
        k = np.arange(count)
        check_uv = np.array([2*((.314159265358979+k*.618033988749895)%1)-1,
                             2*((.271828182845905+k*.414213562373095)%1)-1])*.91
        patches = np.repeat(np.arange(len(mesh['norders']),dtype=np.int64), count)
        checks = np.asfortranarray(np.tile(check_uv, (1,len(mesh['norders']))))
        extra_patches, extra_uv = [], []
        # Exact closest point plus two rings around it on the unit parameter sphere.
        # This supplements every-panel checks and explicitly probes the narrow gap.
        for item in mesh['contact_patch_id_uv']:
            body = item['body']
            direction = np.asarray(item['direction'])
            tangent = np.cross(direction,[1.,0.,0.])
            tangent /= np.linalg.norm(tangent)
            other = np.cross(direction,tangent)
            directions = [direction]
            angular_width = np.sqrt(pose.gap_over_c/pose.a)
            for radius in [.25*angular_width,angular_width]:
                for angle in np.arange(8)*np.pi/4+.173:
                    d = direction+radius*(np.cos(angle)*tangent+np.sin(angle)*other)
                    directions.append(d/np.linalg.norm(d))
            for d in directions:
                pid, u = mesh['locate_direction'](body,d)
                extra_patches.append(pid)
                # Upstream self GGQ does not support a target exactly on a shared
                # panel edge. Sample just inside the selected panel instead; these
                # are near-contact checks, not evaluations of the exact contact.
                extra_uv.append(np.clip(u,-1+1e-3,1-1e-3))
        if extra_patches:
            patches = np.r_[patches,np.asarray(extra_patches,dtype=np.int64)]
            checks = np.asfortranarray(np.c_[checks,np.asarray(extra_uv).T])
    t = time.monotonic()
    points, velocity = logged_fortran(report_path.with_suffix('.boundary.log'),lambda:
        b.boundary_values(mesh, density, patches, checks,args.quad_tol,args.alpha,args.beta),b.lib)
    body_check = mesh['body_ids'][mesh['ixyzs'][patches]-1]
    expected = velocity_modes(points, body_check, pose)
    scale = np.maximum(np.max(np.linalg.norm(rhs.reshape(n,3,17),axis=1),axis=0),1e-30)
    error = np.linalg.norm(velocity-expected,axis=1)/scale
    active = np.max(np.abs(rhs),axis=0)>0
    per_patch_error = np.zeros(len(mesh['norders']))
    for pid in range(len(per_patch_error)):
        per_patch_error[pid] = np.max(error[patches==pid][:,active])
    worst_point,worst_mode = np.unravel_index(np.argmax(error),error.shape)
    geometry_error = []
    for j,pid in enumerate(patches):
        exact = mesh['exact_patch_values'](int(pid),checks[:,j:j+1])[:3,0]
        geometry_error.append(np.linalg.norm(points[j]-exact))
    area_error = []
    volume_error = []
    for body in sorted(set(mesh['body_ids'].tolist())):
        mask = mesh['body_ids']==body
        area_error.append(abs(mesh['weights'][mask].sum()/pose.exact_area-1))
        volume = np.einsum('n,ni,ni->',mesh['weights'][mask],
                           xyz[mask]-pose.centers[body],normals[mask])/3
        volume_error.append(abs(volume/pose.exact_volume-1))
    record['validation'] = {
        'off_grid_boundary':True,
        'boundary_evaluation_valid':True,
        'boundary_scope':'sampled polynomial-panel surface trace; not global supremum bound',
        'contact_site_sampling':mesh.get('_contact_site_sampling'),
        'boundary_sample_count':len(points),
        'boundary_max_scaled_error':float(error[:,active].max()),
        'boundary_rms_scaled_error':float(np.sqrt(np.mean(error[:,active]**2))),
        'boundary_per_mode_max_scaled_error':error.max(axis=0).tolist(),
        'boundary_per_patch_max_scaled_error':per_patch_error.tolist(),
        'boundary_worst_point':points[worst_point].tolist(),
        'boundary_worst_patch':int(patches[worst_point]),
        'boundary_worst_mode':int(worst_mode),
        'boundary_worst_body':int(body_check[worst_point]),
        'gap_patch_sample_count':len(extra_patches),
        'gap_patch_edge_avoidance_uv':1e-3,
        'gap_patch_max_scaled_error':float(error[-len(extra_patches):,active].max()) if extra_patches else None,
        'linear_relative_residual':float(relative_residual),
        'exact_surface_position_max_error_c':float(max(geometry_error)),
        'exact_surface_position_max_error_over_gap':float(max(geometry_error)/pose.gap_over_c),
        'area_relative_errors':area_error,'volume_relative_errors':volume_error,
    }
    record['boundary_check_seconds'] = time.monotonic()-t
    if args.operator_export:
        helper_sha = file_sha256(args.operator_export)
        operators = json.loads(subprocess.check_output(
            [str(args.operator_export.resolve()), '--gap-over-c', str(pose.gap_over_c),
             '--aspect-ratio', str(pose.aspect_ratio), '--c-nm', str(pose.length_unit_nm),
             '--theta-i', str(pose.theta_i), '--theta-j', str(pose.theta_j), '--phi', str(pose.phi)],text=True))
        if operators.get('convention_id') != CONVENTION or operators.get('independent_reference') is not False:
            raise ValueError('Operator export convention/scope mismatch')
        validate_pose(operators, record)
        if file_sha256(args.operator_export) != helper_sha:
            raise ValueError('Operator export changed during evaluation')
        record['operator_export'] = {'path':str(args.operator_export.resolve()),'sha256':helper_sha}
        record['isolated_full17'] = operators['isolated17']
        if args.isolated_body is not None:
            exact = np.asarray(operators['isolated_bodies17'][args.isolated_body])
            errors = block_errors(raw,exact)
            passed = (max(errors.values())<=.002 and
                      record['validation']['boundary_max_scaled_error']<=.001 and
                      record['raw_reciprocity']<=.001 and
                      finite_error(relative_residual,1e-8) and
                      max(geometry_error)/pose.gap_over_c<=.001)
            record['isolated_exact_validation'] = {'block_relative_errors':errors,
                                                   'passed':bool(passed)}
            record['moments']['isolated_spheroid_validation_passed'] = bool(passed)
    if args.isolated_validation and args.isolated_body is None:
        evidence = [isolated_evidence(path, record) for path in args.isolated_validation]
        bodies = {item['isolated_body'] for item in evidence}
        record['moments']['isolated_spheroid_validation_passed'] = all(item['passed'] for item in evidence) and bodies=={0,1}
        record['moments']['isolated_validation_files'] = [str(x.resolve()) for x in args.isolated_validation]
        record['moments']['isolated_validation_evidence'] = evidence
        for item in evidence:
            if file_sha256(item['path']) != item['sha256']:
                raise ValueError('Isolated validation input changed during evaluation')
    if file_sha256(record['solution_file']) != record['solution_sha256']:
        raise ValueError('Density checkpoint changed during validation')
    for source, digest in record['source_hashes'].items():
        if file_sha256(Path(__file__).with_name(source)) != digest:
            raise ValueError('Reference source changed during validation: '+source)
    record.update(status='complete',wall_seconds=time.monotonic()-start,
                  peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    write_json(report_path, record)
    if previous is not None:
        # Publish only after all checks complete. Earlier reports and failed
        # attempt logs remain available alongside the unchanged density.
        write_json(args.output, record)
    print(json.dumps({'output':str(args.output),'nodes':n,
                      'boundary_max':record['validation']['boundary_max_scaled_error'],
                      'reciprocity':record['raw_reciprocity'],
                      'isolated':record.get('isolated_exact_validation'),
                      'seconds':record['wall_seconds']}),flush=True)


if __name__ == '__main__':
    main()
