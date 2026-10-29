#!/usr/bin/env python3
"""Controlled quadrature-only check of one saved density; no solve or qualification."""
import os
os.environ.setdefault('GFORTRAN_UNBUFFERED_ALL', '1')
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')

import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

from backend import Backend
from geometry import build_mesh, recorded_pose, velocity_modes


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def flush_native(library):
    flush = library._gfortran_flush_i4
    flush.argtypes = [ctypes.POINTER(ctypes.c_int32)]
    flush.restype = None
    for number in (6,0):
        unit = ctypes.c_int32(number)
        flush(ctypes.byref(unit))
    ctypes.CDLL(None).fflush(None)


def logged(path, operation, library):
    sys.stdout.flush()
    sys.stderr.flush()
    flush_native(library)
    originals = [os.dup(1),os.dup(2)]
    with path.open('wb') as output:
        try:
            os.dup2(output.fileno(),1)
            os.dup2(output.fileno(),2)
            result = operation()
        finally:
            try:
                flush_native(library)
            finally:
                for target, original in zip((1,2), originals):
                    os.dup2(original,target)
                    os.close(original)
    diagnostic = path.read_text(errors='replace')
    if 'Failed to ' in diagnostic or 'Returning without' in diagnostic:
        raise RuntimeError(f'Invalid native quadrature diagnostic in {path}')
    return result


def main(args):
    source = json.loads(args.reference.read_text())
    if source.get('status') != 'complete':
        raise ValueError('Wait for the saved run to complete before probing its density')
    if source['h_over_c'] != 1. or source['aspect_ratio'] != 20:
        raise ValueError('This diagnostic is restricted to the recorded AR20 pose at150nm')
    library_matches_density = sha(args.library) == source['library_sha256']
    if not library_matches_density and not args.cross_library_diagnostic:
        raise ValueError('Saved density and quadrature library hashes differ')
    backend = Backend(args.library)
    pose = recorded_pose(150.)
    for key in ('theta_i','theta_j','phi'):
        if abs(source[key]-pose.metadata()[key]) > 1e-13:
            raise ValueError('Wrong recorded pose')
    uv = backend.discretization(source['order'])[0]
    mesh = backend.coefficients(build_mesh(150.,source['order'],source['level'],uv,
        focus=source['focus'],patch_descriptors=source['patch_descriptors']))
    checkpoint = Path(source['solution_file'])
    saved = np.load(checkpoint)
    differences = {}
    for key in ('srcvals','weights','body_ids','norders','ixyzs','iptype'):
        differences[key] = float(np.max(abs(mesh[key]-saved[key])))
        if differences[key] > 1e-12:
            raise ValueError(f'Rebuilt mesh differs from saved density mesh: {key}')
    density = saved['density']
    rhs = velocity_modes(mesh['srcvals'][:3].T,mesh['body_ids'],pose)
    scale = np.max(np.linalg.norm(rhs,axis=1),axis=0)
    count = 9
    if args.top_panels is not None:
        scores = np.asarray(source['validation']['boundary_per_patch_max_scaled_error'])
        panels = np.argsort(scores)[-args.top_panels:][::-1].tolist()
    else:
        panels = args.panels if args.panels is not None else [35,25,52,41]
    if len(panels)*count > 36:
        raise ValueError('This lightweight probe allows at most36 total targets')
    if not panels or len(set(panels)) != len(panels) or any(p < 0 or p >= len(mesh['norders']) for p in panels):
        raise ValueError('Require distinct panel indices belonging to the saved mesh')
    local_nodes = None
    if args.target_kind == 'collocation':
        # Select actual backend GL nodes nearest a spread3x3 reference grid.
        # These are interior collocation nodes, never panel-edge targets.
        nominal = np.array(np.meshgrid([-.75,0.,.75],[-.75,0.,.75],indexing='ij')).reshape(2,-1)
        local_nodes = []
        for target in nominal.T:
            ordered = np.argsort(np.sum((uv-target[:,None])**2,axis=0))
            index = next(int(i) for i in ordered if int(i) not in local_nodes and np.max(abs(uv[:,i])) < 1-1e-14)
            local_nodes.append(index)
        check_uv = uv[:,local_nodes]
    else:
        k = np.arange(count)
        check_uv = np.array([2*((.314159265358979+k*.618033988749895)%1)-1,
                             2*((.271828182845905+k*.414213562373095)%1)-1])*.91
    patch_ids = np.repeat(np.asarray(panels,dtype=np.int64),count)
    checks = np.asfortranarray(np.tile(check_uv,(1,len(panels))))
    bodies = mesh['body_ids'][mesh['ixyzs'][patch_ids]-1]
    global_nodes = None if local_nodes is None else mesh['ixyzs'][patch_ids]-1+np.tile(local_nodes,len(panels))
    report = dict(schema='saved_density_boundary_quadrature_probe_v1',physical_accuracy_qualified=False,
        source_report=str(args.reference.resolve()),source_sha256=sha(args.reference),
        checkpoint=str(checkpoint.resolve()),checkpoint_sha256=sha(checkpoint),
        library_sha256=sha(args.library),rebuilt_mesh_max_absolute_differences=differences,
        saved_density_library_sha256=source['library_sha256'],
        library_matches_saved_density=library_matches_density,
        cross_library_diagnostic=bool(args.cross_library_diagnostic),
        source_hashes={name:sha(Path(__file__).with_name(name)) for name in
                       ('backend.py','geometry.py','probe_saved_boundary_quadrature.py')},
        panels=panels,points_per_panel=count,check_uv=check_uv.tolist(),
        target_kind=args.target_kind,
        local_collocation_node_indices=local_nodes,
        global_collocation_node_indices=None if global_nodes is None else global_nodes.tolist(),
        saved_linear_relative_residual=source.get('linear_residual'),
        source_offgrid_boundary_max_scaled_error=source.get('validation',{}).get('boundary_max_scaled_error'),
        source_selected_panel_offgrid_errors={str(p):source['validation']['boundary_per_patch_max_scaled_error'][p] for p in panels},
        scope=f'Same saved density and same{len(patch_ids)} {args.target_kind} sites; only boundary quadrature tolerance changes',
        evaluations=[])
    velocities = []
    for eps in args.epsilons:
        started = time.monotonic()
        log = args.output.with_name(args.output.stem+f'_eps{eps:.0e}.log')
        points, velocity = logged(log,lambda:backend.boundary_values(mesh,density,patch_ids,checks,eps,
                                      source['alpha'],source['beta']),backend.lib)
        expected = velocity_modes(points,bodies,pose) if global_nodes is None else rhs[global_nodes]
        errors = np.linalg.norm(velocity-expected,axis=1)/scale
        per_patch = {str(p):float(np.max(errors[patch_ids==p])) for p in panels}
        record = dict(eps=eps,boundary_evaluation_valid=True,max_scaled_error=float(errors.max()),
                      per_panel_max_scaled_error=per_patch,per_mode_max_scaled_error=errors.max(axis=0).tolist(),
                      log=str(log),log_sha256=sha(log),seconds=time.monotonic()-started)
        if global_nodes is not None:
            record['target_position_vs_saved_node_max_absolute_error'] = float(np.max(abs(points-mesh['srcvals'][:3,global_nodes].T)))
        report['evaluations'].append(record)
        velocities.append(velocity)
        print(json.dumps(record),flush=True)
    report['changes_relative_to_first'] = []
    for index in range(1,len(velocities)):
        difference = np.linalg.norm(velocities[index]-velocities[0],axis=1)/scale
        report['changes_relative_to_first'].append(dict(
            first_eps=args.epsilons[0],other_eps=args.epsilons[index],
            max_scaled_change=float(difference.max()),
            per_mode_max_scaled_change=difference.max(axis=0).tolist(),
            per_panel_max_scaled_change={str(p):float(np.max(difference[patch_ids==p])) for p in panels}))
    report['interpretation'] = 'Boundary quadrature sensitivity at sampled points only; density, geometry, assembly quadrature, and full physical reference remain unqualified. Upstream1e-7 and1e-9 requests use the same iprec3 bin; a request below5e-10 reaches iprec4. Self GGQ does not take eps, so this does not independently refine the singular self rule.'
    if global_nodes is not None:
        report['collocation_interpretation'] = 'Same-tolerance collocation trace checks assembly/evaluation consistency. Tight-tolerance collocation trace tests nonself assembly quadrature sensitivity on the saved density. Small collocation errors together with large off-grid errors support unresolved interpolation between nodes as a plausible cause, but do not exclude fixed self-GGQ or geometry errors and do not establish a global bound.'
    arrays = args.output.with_suffix('.velocities.npz')
    np.savez_compressed(arrays,epsilons=np.asarray(args.epsilons),velocities=np.asarray(velocities),
                        points=points,patch_ids=patch_ids,uv=checks,scale=scale,expected_velocity=expected)
    report['velocity_checkpoint'] = str(arrays)
    report['velocity_checkpoint_sha256'] = sha(arrays)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--library',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--panels',type=int,nargs='+')
    selection.add_argument('--top-panels',type=int)
    parser.add_argument('--target-kind',choices=['offgrid','collocation'],default='offgrid')
    parser.add_argument('--epsilons',type=float,nargs='+',default=[1e-7,1e-9])
    parser.add_argument('--cross-library-diagnostic',action='store_true',
                        help='Explicitly evaluate old density with a different library; never qualifies a reference')
    arguments = parser.parse_args()
    if arguments.output.resolve() == arguments.reference.resolve():
        parser.error('Diagnostic output must not replace the source reference report')
    if not all(np.isfinite(eps) and 0 < eps < .01 for eps in arguments.epsilons):
        parser.error('Quadrature tolerances must be finite and in (0,.01)')
    arguments.output.parent.mkdir(parents=True,exist_ok=True)
    try:
        main(arguments)
    except Exception as error:
        arguments.output.write_text(json.dumps(dict(physical_accuracy_qualified=False,failure=str(error)),indent=2)+'\n')
        raise
