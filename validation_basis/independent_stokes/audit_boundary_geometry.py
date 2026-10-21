#!/usr/bin/env python3
"""Lightweight geometry/density diagnosis of a saved same-pose BIE solution.

No kernel assembly, boundary quadrature, or Stokes solve is performed. Optional
saved velocity samples identify actual residual maxima; otherwise the driver's
deterministic check sites and its recorded worst point are reconstructed.
Small geometric errors do not exclude an error in the assembled operator or
the singular self quadrature.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import legder,legval,legvander

from backend import Backend
from geometry import build_mesh,evaluate_quad_panels,recorded_pose,velocity_modes
from mesh_adapt import density_tails


def driver_sites(mesh,record,pose):
    validation = record['validation']
    count,remainder = divmod(validation['boundary_sample_count']-validation['gap_patch_sample_count'],len(mesh['norders']))
    if remainder or count < 1:
        raise ValueError('Cannot reconstruct the driver site count')
    k = np.arange(count)
    base = np.array([2*((.314159265358979+k*.618033988749895)%1)-1,
                     2*((.271828182845905+k*.414213562373095)%1)-1])*.91
    patches = np.repeat(np.arange(len(mesh['norders']),dtype=np.int64),count)
    uv = np.tile(base,(1,len(mesh['norders'])))
    extra_patches,extra_uv = [],[]
    for item in mesh['contact_patch_id_uv']:
        direction = np.asarray(item['direction'])
        tangent = np.cross(direction,[1.,0.,0.]);tangent /= np.linalg.norm(tangent)
        other = np.cross(direction,tangent)
        directions = [direction]
        width = np.sqrt(pose.gap_over_c/pose.a)
        for radius in (.25*width,width):
            for angle in np.arange(8)*np.pi/4+.173:
                d = direction+radius*(np.cos(angle)*tangent+np.sin(angle)*other)
                directions.append(d/np.linalg.norm(d))
        for d in directions:
            pid,u = mesh['locate_direction'](item['body'],d)
            extra_patches.append(pid);extra_uv.append(np.clip(u,-.999,.999))
    return np.r_[patches,extra_patches].astype(np.int64),np.c_[uv,np.asarray(extra_uv).T]


def audit(report_path,samples_path,library,comparison_paths,top):
    record = json.loads(report_path.read_text())
    if record.get('aspect_ratio') != 20 or record.get('isolated_body') is not None:
        raise ValueError('This diagnostic is restricted to the recorded AR20 pair')
    pose = recorded_pose(record['h_nm'])
    backend = Backend(library)
    order = record['order'];n = order+1;nq = n*n
    uv = backend.discretization(order)[0]
    mesh = build_mesh(record['h_nm'],order,record.get('level',1),uv,
                      focus=record.get('focus',0),patch_descriptors=record['patch_descriptors'])
    with np.load(record['solution_file']) as saved:
        values,density = saved['srcvals'],saved['density']
    if not np.array_equal(values,mesh['srcvals']):
        raise ValueError('Rebuilt exact geometry differs from checkpoint')
    errors = None
    if samples_path:
        with np.load(samples_path) as samples:
            pids,targets,points = samples['patch_ids'],samples['uv'],samples['points']
            velocity,scale = samples['velocities'][-1],samples['scale']
        bodies = np.array([mesh['patches'][int(i)][0] for i in pids])
        errors = np.linalg.norm(velocity-velocity_modes(points,bodies,pose),axis=1)/scale
    else:
        pids,targets = driver_sites(mesh,record,pose)
        points = backend.interpolate(mesh,values[:3],pids,targets).T
    source_scores = np.asarray(record['validation']['boundary_per_patch_max_scaled_error'])
    available = set(int(i) for i in pids)
    selected = [int(i) for i in np.argsort(source_scores)[::-1] if int(i) in available][:top]
    x,iu = np.unique(uv[0],return_inverse=True)
    y,iv = np.unique(uv[1],return_inverse=True)
    A,B = np.linalg.inv(legvander(x,order)),np.linalg.inv(legvander(y,order))
    tails = density_tails(density,uv,len(mesh['patches']))
    comparators = []
    for path in comparison_paths:
        proposal = json.loads(path.read_text());porder = proposal['order']
        other = build_mesh(record['h_nm'],porder,1,backend.discretization(porder)[0],
                           patch_descriptors=proposal['patch_descriptors'])
        comparators.append((str(path),proposal,other))
    patches = []
    for pid in selected:
        body,vertices = mesh['patches'][pid]
        where = np.flatnonzero(pids == pid);ts = targets[:,where]
        exact = evaluate_quad_panels(pose,body,vertices[None],ts)
        grid = np.empty((n,n,12));grid[iu,iv] = values[:,pid*nq:(pid+1)*nq].T
        coefficients = np.einsum('iu,jv,uvc->ijc',A,B,grid)
        P,Q = legvander(ts[0],order),legvander(ts[1],order)
        derivative = lambda z:np.stack([legval(z,legder(np.eye(n)[k])) for k in range(n)],axis=1)
        dP,dQ = derivative(ts[0]),derivative(ts[1])
        field = np.einsum('ki,kj,ijc->kc',P,Q,coefficients)
        xu = np.einsum('ki,kj,ijc->kc',dP,Q,coefficients[:,:,:3])
        xv = np.einsum('ki,kj,ijc->kc',P,dQ,coefficients[:,:,:3])
        normal = np.cross(field[:,3:6],field[:,6:9]);normal /= np.linalg.norm(normal,axis=1)[:,None]
        relative = lambda a,b:float(np.max(np.linalg.norm(a-b,axis=1)/np.linalg.norm(b,axis=1)))
        result = {'patch':pid,'body':int(body),'check_site_count':len(where),
                  'recorded_patch_boundary_error':float(source_scores[pid]),
                  'max_position_error_c':float(np.max(np.linalg.norm(field[:,:3]-exact.points[0],axis=1))),
                  'max_normal_error':float(np.max(np.linalg.norm(normal-exact.normals[0],axis=1))),
                  'max_interpolated_analytic_derivative_relative_error':max(relative(field[:,3:6],exact.du[0]),relative(field[:,6:9],exact.dv[0])),
                  'max_position_derivative_consistency_relative_error':max(relative(xu,field[:,3:6]),relative(xv,field[:,6:9])),
                  'density_directional_tail_indicators':tails[pid].tolist()}
        worst_index = None
        if errors is not None:
            local,mode = np.unravel_index(errors[where].argmax(),errors[where].shape)
            worst_index = int(where[local]);result['sampled_worst_mode'] = int(mode)
            result['sampled_max_boundary_error'] = float(errors[worst_index,mode])
        elif pid == record['validation']['boundary_worst_patch']:
            distance = np.linalg.norm(points[where]-record['validation']['boundary_worst_point'],axis=1)
            local = int(np.argmin(distance))
            if distance[local] > 1e-10:
                raise ValueError('Reconstructed check sites do not contain recorded worst point')
            worst_index = int(where[local]);result['sampled_worst_mode'] = record['validation']['boundary_worst_mode']
        if worst_index is not None:
            point = points[worst_index]
            result['worst_uv'] = targets[:,worst_index].tolist()
            result['worst_point_nm'] = (point*150).tolist()
            result['opposite_body_projected_ellipse_level'] = float((point[:2]-pose.centers[1-body,:2])@
                np.linalg.inv(pose.shape_tensors[1-body,:2,:2])@(point[:2]-pose.centers[1-body,:2]))
            direction = pose.rotations[body].T@(point-pose.centers[body])/np.array([20,20,1])
            direction /= np.linalg.norm(direction)
            alternatives = []
            for path,proposal,other in comparators:
                index,local_uv = other['locate_direction'](int(body),direction)
                sv = other['exact_patch_values'](index,local_uv[:,None])
                length = 2*np.linalg.norm(sv[3:9,0].reshape(2,3),axis=1)*150
                alternatives.append({'proposal':path,'order':proposal['order'],'patch':index,
                                     'cube_coordinate_spans':np.ptp(other['patches'][index][1],axis=0).tolist(),
                                     'local_tangent_panel_scales_nm':length.tolist(),
                                     'tangent_scale_divided_by_nodes_per_axis_nm':(length/(proposal['order']+1)).tolist()})
            result['alternative_mesh_resolution_indicators'] = alternatives
        patches.append(result)
    return {'schema':'recorded_ar20_saved_boundary_geometry_audit_v1',
            'physical_accuracy_qualified':False,'contains_new_stokes_solve':False,
            'excludes_operator_or_self_quadrature_error':False,
            'scope':'Geometry interpolation and density-tail indicators at saved/reconstructed off-grid sites; no kernel evaluation',
            'report':str(report_path.resolve()),'report_sha256':hashlib.sha256(report_path.read_bytes()).hexdigest(),
            'sample_velocity_file':str(samples_path) if samples_path else None,
            'sample_velocity_sha256':hashlib.sha256(samples_path.read_bytes()).hexdigest() if samples_path else None,
            'patches':patches}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report',type=Path,required=True)
    p.add_argument('--samples',type=Path)
    p.add_argument('--library',required=True)
    p.add_argument('--compare-mesh',type=Path,action='append',default=[])
    p.add_argument('--top',type=int,default=12)
    p.add_argument('--output',type=Path,required=True)
    a = p.parse_args()
    result = audit(a.report,a.samples,a.library,a.compare_mesh,a.top)
    result['source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if a.output.exists():
        raise FileExistsError(a.output)
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'output':str(a.output),'patches':len(result['patches']),
                      'physical_accuracy_qualified':False}))


if __name__ == '__main__':
    main()
