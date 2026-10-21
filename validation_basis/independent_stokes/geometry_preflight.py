"""Cheap sampled polynomial-geometry checks before any Stokes quadrature/LU.

These checks establish a sampled geometry tolerance only. They do not qualify
boundary densities or replace the independent post-solve geometry/flow checks.
"""
from __future__ import annotations

import numpy as np
from numpy.polynomial.legendre import leggauss, legvander

from geometry import build_mesh, refine_descriptors, spheroid_pose
from ram_dense import working_memory_estimate


class GeometryPreflightError(RuntimeError):
    def __init__(self, message, diagnostics):
        self.diagnostics = diagnostics
        super().__init__(message)


def gauss_uv(order):
    x = leggauss(int(order)+1)[0]
    u, v = np.meshgrid(x, x, indexing='ij')
    return np.asfortranarray(np.stack((u.ravel(), v.ravel())))


def boundary_sites(mesh, pose, count, broad_contact=False):
    """Driver's staggered off-grid sites plus closest-point/ring checks.

    New distance runs retain the legacy rings and add exact gap-Hessian
    ellipses, covering the different physical scales of face and rim contacts.
    """
    k = np.arange(count)
    check_uv = np.array([2*((.314159265358979+k*.618033988749895)%1)-1,
                         2*((.271828182845905+k*.414213562373095)%1)-1])*.91
    patches = np.repeat(np.arange(len(mesh['norders']),dtype=np.int64),count)
    checks = np.asfortranarray(np.tile(check_uv,(1,len(mesh['norders']))))
    if broad_contact:
        from contact_check_sites import build_contact_check_sites
        extra_patches, extra_uv, metadata = build_contact_check_sites(mesh, pose)
        mesh['_contact_site_sampling'] = metadata
        return (np.r_[patches, extra_patches],
                np.asfortranarray(np.c_[checks, extra_uv]), len(extra_patches))
    extra_patches, extra_uv = [], []
    for item in mesh['contact_patch_id_uv']:
        body, direction = item['body'], np.asarray(item['direction'])
        # Choose a safe axis even for an exactly x-oriented contact normal.
        basis = np.eye(3)[int(np.argmin(np.abs(direction)))]
        tangent = np.cross(direction,basis); tangent /= np.linalg.norm(tangent)
        other = np.cross(direction,tangent)
        directions = [direction]
        narrow = np.sqrt(pose.gap_over_c/pose.a)
        radii = [.25*narrow,narrow]
        if broad_contact:
            wider = min(.5,np.sqrt(pose.gap_over_c))
            radii.extend([.25*wider,wider])
        for radius in sorted(set(radii)):
            for angle in np.arange(8)*np.pi/4+.173:
                d = direction+radius*(np.cos(angle)*tangent+np.sin(angle)*other)
                directions.append(d/np.linalg.norm(d))
        for direction in directions:
            pid, uv = mesh['locate_direction'](body,direction)
            extra_patches.append(pid)
            extra_uv.append(np.clip(uv,-1+1e-3,1-1e-3))
    if extra_patches:
        patches = np.r_[patches,np.asarray(extra_patches,dtype=np.int64)]
        checks = np.asfortranarray(np.c_[checks,np.asarray(extra_uv).T])
    return patches, checks, len(extra_patches)


def sample_geometry(mesh, uv, pose, check_points_per_patch=3):
    """Tensor interpolation error on dense off-grid sites and boundary sites.

    A polynomial interpolant is unique, so this NumPy evaluation matches the
    backend interpolant without calling native kernels or assuming node order.
    Positions and discrepancies are in c units; errors are normalized by h/c.
    """
    uv = np.asarray(uv,dtype=float)
    order = int(mesh['order'])
    size = order+1
    xu, iu = np.unique(uv[0],return_inverse=True)
    xv, iv = np.unique(uv[1],return_inverse=True)
    if (uv.shape != (2,size*size) or len(xu) != size or len(xv) != size
            or len(set(zip(iu.tolist(),iv.tolist()))) != size*size):
        raise ValueError('Geometry preflight requires tensor Gauss nodes of a uniform order')
    if np.any(mesh['norders'] != order) or np.any(mesh['iptype'] != 11):
        raise ValueError('Geometry preflight requires uniform quadrangular patches')
    inverse_u = np.linalg.inv(legvander(xu,order))
    inverse_v = np.linalg.inv(legvander(xv,order))
    y = np.linspace(-.999,.999,max(11,2*order+3))
    yy, zz = np.meshgrid(y,y,indexing='ij')
    dense_uv = np.stack((yy.ravel(),zz.ravel()))
    dense_u = legvander(y,order)@inverse_u
    dense_v = legvander(y,order)@inverse_v
    boundary_patches,boundary_uv,extra_count = boundary_sites(
        mesh,pose,check_points_per_patch,broad_contact=True)
    errors = []
    sample_count = 0
    for pid in range(len(mesh['norders'])):
        begin,end = mesh['ixyzs'][pid:pid+2]-1
        values = np.empty((size,size,3))
        values[iu,iv] = mesh['srcvals'][:3,begin:end].T
        polynomial = np.einsum('iu,jv,uvc->ijc',dense_u,dense_v,values).reshape(-1,3)
        exact = mesh['exact_patch_values'](pid,dense_uv)[:3].T
        maximum = float(np.max(np.linalg.norm(polynomial-exact,axis=1)))
        extra = boundary_uv[:,boundary_patches==pid]
        if extra.size:
            left = legvander(extra[0],order)@inverse_u
            right = legvander(extra[1],order)@inverse_v
            polynomial = np.einsum('ki,kj,ijc->kc',left,right,values)
            exact = mesh['exact_patch_values'](pid,extra)[:3].T
            maximum = max(maximum,float(np.max(np.linalg.norm(polynomial-exact,axis=1))))
        errors.append(maximum/pose.gap_over_c)
        sample_count += len(dense_uv.T)+len(extra.T)
    if not np.isfinite(errors).all():
        raise FloatingPointError('Non-finite sampled polynomial geometry')
    return {'scope':'sampled polynomial geometry only; not a supremum or flow accuracy bound',
            'order':order,'node_count':int(mesh['srcvals'].shape[1]),
            'patch_count':len(mesh['norders']),'sample_count':sample_count,
            'dense_grid_points_per_axis':len(y),'broad_contact_sample_count':extra_count,
            'max_position_error_over_gap':float(max(errors)),
            'max_position_error_nm':float(max(errors)*pose.gap_nm),
            'per_patch_position_error_over_gap':errors,'physical_accuracy_qualified':False}


def choose_orders(physical, resolution, tolerance=.0005, coarse_candidates=(6,8,10),
                  max_memory_gb=56., check_points_per_patch=3):
    """Choose fixed planned p and p+2 before costly pair work, using geometry.

    If the largest candidate still fails, bounded h refinement remains necessary.
    Order selection never silently qualifies a reference or exceeds a RAM guard.
    """
    pose = spheroid_pose(physical['gap_nm'], **{key:physical[key] for key in
                         ('aspect_ratio','c_nm','theta_i','theta_j','phi') if key in physical})
    settings = {key:value for key,value in resolution.items() if key in
                ('level','focus','rim_levels','rim_width','contact_levels','contact_radius')}
    settings.setdefault('level',1)
    trials = []
    selected = int(coarse_candidates[-1])
    passed = False
    for order in coarse_candidates:
        uv = gauss_uv(order)
        mesh = build_mesh(pose.gap_nm,int(order),uv=uv,pose=pose,**settings)
        result = sample_geometry(mesh,uv,pose,check_points_per_patch)
        result['estimated_ram_bytes'] = working_memory_estimate(
            result['node_count'],result['patch_count'])['peak_additional_bytes']
        result['within_ram_budget'] = result['estimated_ram_bytes'] <= max_memory_gb*1e9
        result['sampled_geometry_passed'] = result['max_position_error_over_gap'] <= tolerance
        trials.append(result)
        if result['sampled_geometry_passed'] and result['within_ram_budget']:
            selected,passed = int(order),True
            break
    return {'coarse_order':selected,'fine_order':selected+2,'tolerance_over_gap':float(tolerance),
            'requires_h_refinement':not passed,'candidates':trials,
            'selection_scope':'geometry preflight only; independent flow convergence still required'}


def refine_to_tolerance(mesh, uv, pose, rebuild, tolerance=.0005, max_refinements=3,
                        max_memory_gb=56., check_points_per_patch=3, progress=None):
    """Refine only failing exact patches, rejecting oversized next meshes early."""
    if not (np.isfinite(tolerance) and 0<tolerance<1 and
            int(max_refinements)==max_refinements and max_refinements>=0):
        raise ValueError('Valid geometry tolerance and nonnegative refinement count required')
    record = {'schema':'independent_stokes_geometry_preflight_v1',
              'status':'checking','tolerance_over_gap':float(tolerance),
              'max_refinements':int(max_refinements),'attempts':[],
              'physical_accuracy_qualified':False}

    def fail(status,message):
        record.update(status=status,failure=message)
        if progress is not None: progress(record)
        raise GeometryPreflightError(message,record)

    for index in range(int(max_refinements)+1):
        nodes,patches = int(mesh['srcvals'].shape[1]),len(mesh['norders'])
        estimate = working_memory_estimate(nodes,patches)
        if estimate['peak_additional_bytes'] > max_memory_gb*1e9:
            record['rejected_memory_estimate'] = estimate
            fail('memory_budget_exceeded','Geometry mesh would exceed RAM dense budget before LU')
        sampled = sample_geometry(mesh,uv,pose,check_points_per_patch)
        sampled.update(refinement=index,memory_estimate=estimate)
        record['attempts'].append(sampled)
        bad = np.flatnonzero(np.asarray(sampled['per_patch_position_error_over_gap'])>tolerance)
        sampled['selected_patches'] = bad.tolist()
        if not len(bad):
            record.update(status='passed',refinements=index)
            if progress is not None: progress(record)
            return mesh,record
        if progress is not None: progress(record)
        if index == max_refinements:
            fail('refinement_limit','Sampled geometry error exceeds tolerance after bounded refinements')
        next_patches = patches+3*len(bad)
        next_nodes = next_patches*(int(mesh['order'])+1)**2
        estimate = working_memory_estimate(next_nodes,next_patches)
        if estimate['peak_additional_bytes'] > max_memory_gb*1e9:
            record['rejected_memory_estimate'] = estimate
            fail('memory_budget_exceeded','Required geometry refinement exceeds RAM dense budget before LU')
        mesh = rebuild(refine_descriptors(mesh['patches'],bad))
    raise AssertionError('Unreachable geometry refinement state')
