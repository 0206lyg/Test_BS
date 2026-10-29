#!/usr/bin/env python3
"""Prepare one same-pose p6 retry from a completed p8 reference diagnostic.

No new Stokes calculation is performed. Fresh no-slip residuals determine the
first refinements. Remaining nodes preserve the saved p8 density when moving
to p6: on independent parameter sites we compare that polynomial to each p6
child interpolant. Half their density difference is the change in the explicit
double-layer jump term ONLY, not the complete velocity error or an error bound.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss,legvander

from backend import Backend
from geometry import recorded_pose,refine_descriptors,velocity_modes,evaluate_quad_panels
from mesh_adapt import density_tails

ACTIONS = {
    'none':[(-1.,1.,-1.,1.)],
    'u':[(-1.,0.,-1.,1.),(0.,1.,-1.,1.)],
    'v':[(-1.,1.,-1.,0.),(-1.,1.,0.,1.)],
    'both':[(-1.,0.,-1.,0.),(0.,1.,-1.,0.),(0.,1.,0.,1.),(-1.,0.,0.,1.)],
}
COST = {k:len(v)-1 for k,v in ACTIONS.items()}


def evaluate(coefficients,u,v):
    return np.einsum('ki,kj,ijcm->kcm',legvander(np.asarray(u),coefficients.shape[0]-1),
                     legvander(np.asarray(v),coefficients.shape[1]-1),coefficients)


def candidate_proxies(coefficients,scale,pose,body,vertices,beta,output_order=6):
    width = output_order+1
    nodes,_ = leggauss(width)
    u,v = np.meshgrid(nodes,nodes,indexing='ij')
    inverse = np.linalg.inv(legvander(nodes,output_order))
    # Includes edges: this is polynomial comparison, not singular quadrature.
    checks = np.cos(np.arange(17)*np.pi/16)
    x,y = np.meshgrid(checks,checks,indexing='ij')
    proxies = {}
    for action,children in ACTIONS.items():
        max_jump,max_geometry = 0.,0.
        for u0,u1,v0,v1 in children:
            map_u = lambda z:(u0+u1)/2+(u1-u0)*z/2
            map_v = lambda z:(v0+v1)/2+(v1-v0)*z/2
            sampled = evaluate(coefficients,map_u(u.ravel()),map_v(v.ravel())).reshape(width,width,3,17)
            child_coefficients = np.einsum('iu,jv,uvcm->ijcm',inverse,inverse,sampled)
            reference = evaluate(coefficients,map_u(x.ravel()),map_v(y.ravel()))
            projected = evaluate(child_coefficients,x.ravel(),y.ravel())
            max_jump = max(max_jump,float(np.max(abs(beta)*.5*np.linalg.norm(projected-reference,axis=1)/scale)))
            sample_uv = np.stack((map_u(u.ravel()),map_v(v.ravel())))
            exact_nodes = evaluate_quad_panels(pose,body,vertices[None],sample_uv).points[0].reshape(width,width,3)
            geometry_coefficients = np.einsum('iu,jv,uvc->ijc',inverse,inverse,exact_nodes)
            interpolated = np.einsum('ki,kj,ijc->kc',legvander(x.ravel(),output_order),legvander(y.ravel(),output_order),geometry_coefficients)
            exact = evaluate_quad_panels(pose,body,vertices[None],np.stack((map_u(x.ravel()),map_v(y.ravel())))).points[0]
            max_geometry = max(max_geometry,float(np.max(np.linalg.norm(interpolated-exact,axis=1))/pose.gap_over_c))
        proxies[action] = {'saved_density_jump_change_scaled':max_jump,
                           'exact_position_interpolation_error_over_gap':max_geometry}
    return proxies


def prepare(parent_path,library,max_nodes=9947,target=.001,output_order=6):
    parent = json.loads(parent_path.read_text())
    if parent.get('status') != 'complete' or parent.get('order') != 8 or parent.get('focus',0) != 0:
        raise ValueError('Requires a completed unfocused p8 parent')
    pose = recorded_pose(parent['h_nm'])
    for key in ('aspect_ratio','theta_i','theta_j','phi'):
        if parent[key] != pose.metadata()[key]:
            raise ValueError('Only the recorded AR20 pose is supported')
    descriptors = parent['patch_descriptors'];npatch = len(descriptors)
    if output_order not in (6,7):
        raise ValueError('Supported retry degrees are6 and7')
    points_per_patch = (output_order+1)**2
    budget = max_nodes//points_per_patch-npatch
    if budget < 0:
        raise ValueError('Parent geometry at p6 exceeds budget')
    with np.load(parent['solution_file']) as data:
        density,values,ids = data['density'],data['srcvals'],data['body_ids']
    rhs = velocity_modes(values[:3].T,ids,pose)
    scale = np.maximum(np.max(np.linalg.norm(rhs,axis=1),axis=0),1e-30)
    uv = Backend(library).discretization(8)[0]
    x,iu = np.unique(uv[0],return_inverse=True);y,iv = np.unique(uv[1],return_inverse=True)
    A,B = np.linalg.inv(legvander(x,8)),np.linalg.inv(legvander(y,8))
    tails = density_tails(density,uv,npatch)
    scores = np.asarray(parent['validation']['boundary_per_patch_max_scaled_error'])
    proxy = []
    for i,(body,vertices) in enumerate(descriptors):
        grid = np.empty((9,9,3,17));grid[iu,iv] = density[i*81:(i+1)*81]
        coefficients = np.einsum('iu,jv,uvcm->ijcm',A,B,grid)
        proxy.append(candidate_proxies(coefficients,scale,pose,body,np.array(vertices),parent['beta'],output_order))
    allowed = [];mandatory = []
    # Repair every sampled failing panel before allocating density-preservation
    # refinements to other panels. Tail ratios choose direction, never a pass.
    for i in range(npatch):
        lo,hi = tails[i].min(),tails[i].max()
        desired = ('u' if tails[i,0] > tails[i,1] else 'v') if hi > 3*max(lo,1e-100) else 'both'
        mandatory.append(desired if scores[i] > target else 'none')
        allowed.append(([desired,'both'] if desired != 'both' else ['both']) if scores[i]>target else list(ACTIONS))
    initial_cost = sum(COST[a] for a in mandatory)
    if initial_cost > budget:
        raise ValueError('Budget cannot refine every sampled failing panel')
    levels = sorted(set(proxy[i][a]['saved_density_jump_change_scaled'] for i in range(npatch) for a in allowed[i]))
    choices = None
    for level in levels:
        trial = []
        for i in range(npatch):
            eligible = [a for a in allowed[i] if proxy[i][a]['saved_density_jump_change_scaled'] <= level and
                        proxy[i][a]['exact_position_interpolation_error_over_gap'] <= .001]
            if not eligible:
                break
            trial.append(min(eligible,key=lambda a:(COST[a],proxy[i][a]['saved_density_jump_change_scaled'])))
        if len(trial) == npatch and sum(COST[a] for a in trial) <= budget:
            choices = trial;break
    if choices is None:
        raise ValueError('No candidate satisfies the mesh budget and geometric interpolation check')
    remaining = budget-sum(COST[a] for a in choices)
    reasons = [('fresh sampled boundary failure; directional tail ratio selects split' if scores[i]>target
                else 'preserve saved density under polynomial order reduction') for i in range(npatch)]
    # Exact discrete minimax allocation above selects the smallest achievable
    # maximum jump proxy under the budget and mandatory boundary refinements.
    # Spend any leftover capacity on the largest remaining polynomial loss.
    while remaining:
        ranked = sorted(range(npatch),key=lambda i:proxy[i][choices[i]]['saved_density_jump_change_scaled'],reverse=True)
        changed = False
        for i in ranked:
            old = choices[i]
            allowed = ['u','v','both'] if old == 'none' else ['both'] if old in ('u','v') else []
            allowed = [a for a in allowed if COST[a]-COST[old] <= remaining]
            if not allowed:
                continue
            best = min(allowed,key=lambda a:proxy[i][a]['saved_density_jump_change_scaled'])
            if proxy[i][best]['saved_density_jump_change_scaled'] >= proxy[i][old]['saved_density_jump_change_scaled']:
                continue
            remaining -= COST[best]-COST[old];choices[i] = best
            reasons[i] += '; preserve saved density under p6 projection'
            changed = True
            break
        if not changed:
            break
    children = [];lineage=[]
    for i,(descriptor,action) in enumerate(zip(descriptors,choices)):
        refined = refine_descriptors([descriptor],[] if action == 'none' else [0],
                                     axis={'none':None,'u':0,'v':1,'both':None}[action])
        children.extend(refined);lineage.extend([i]*len(refined))
    def area(ds):
        total = np.zeros((2,3,2))
        for body,vertices in ds:
            q = np.asarray(vertices);axis = int(np.argmin(np.ptp(q,axis=0)))
            total[body,axis,int(q[0,axis]>0)] += np.linalg.norm(np.cross(q[1]-q[0],q[3]-q[0]))
        return total
    coverage_error = float(np.max(np.abs(area(children)-area(descriptors))))
    if coverage_error != 0 or len(children)*points_per_patch > max_nodes:
        raise ValueError('Invalid coverage or budget')
    chosen_jump = [proxy[i][choices[i]]['saved_density_jump_change_scaled'] for i in range(npatch)]
    chosen_geometry = [proxy[i][choices[i]]['exact_position_interpolation_error_over_gap'] for i in range(npatch)]
    result = pose.metadata()
    result.update(schema='independent_ar20_mesh_proposal_v1',physical_accuracy_qualified=False,
        contains_new_stokes_solve=False,selection_is_error_bound=False,focus=0.,isolated_body=None,
        order=output_order,parent_order=8,parent_path=str(parent_path.resolve()),
        parent_sha256=hashlib.sha256(parent_path.read_bytes()).hexdigest(),
        parent_solution_sha256=hashlib.sha256(Path(parent['solution_file']).read_bytes()).hexdigest(),
        strategy='Refine every fresh sampled boundary failure; discrete minimax allocation of the remaining nodes preserves the saved density under polynomial order reduction',
        parent_patch_count=npatch,patch_count=len(children),node_count=len(children)*points_per_patch,max_nodes=max_nodes,
        initial_boundary_repair_added_patches=initial_cost,density_preservation_added_patches=budget-remaining-initial_cost,
        target_sampled_boundary_error=target,remaining_added_patch_budget=remaining,
        projection_metric_definition='max(beta/2*norm(q_retry_interpolant-q_saved_p8)/boundary_mode_scale) on17x17 Chebyshev-Lobatto sites per child; jump term ONLY, not full velocity error or bound',
        max_saved_density_jump_change_without_refinement=max(p['none']['saved_density_jump_change_scaled'] for p in proxy),
        max_saved_density_jump_change_after_refinement=max(chosen_jump),
        max_exact_position_interpolation_error_over_gap=max(chosen_geometry),
        exact_cube_face_coverage_error=coverage_error,
        unrefined_failing_parent_patches=[i for i in range(npatch) if scores[i]>target and choices[i]=='none'],
        choices=[dict(parent_patch=i,action=choices[i],reason=reasons[i],boundary_score=float(scores[i]),
                      directional_density_tail=tails[i].tolist(),candidate_projection_indicators=proxy[i]) for i in range(npatch)],
        new_patch_parent_ids=lineage,patch_descriptors=[[int(body),vertices.tolist()] for body,vertices in children])
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent',type=Path,required=True)
    p.add_argument('--library',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-nodes',type=int,default=9947)
    p.add_argument('--order',type=int,choices=[6,7],default=6)
    a = p.parse_args()
    result = prepare(a.parent,a.library,a.max_nodes,output_order=a.order)
    result['source_sha256'] = {name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                              for name in ('retry_mesh_from_fine.py','geometry.py','mesh_adapt.py')}
    if a.output.exists():
        raise FileExistsError(a.output)
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('node_count','patch_count','max_saved_density_jump_change_without_refinement',
                                          'max_saved_density_jump_change_after_refinement','max_exact_position_interpolation_error_over_gap')}))


if __name__ == '__main__':
    main()
