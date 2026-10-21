#!/usr/bin/env python3
"""Off-grid no-slip check of the actual stored pair solution, exact failed pose.

Uses analytic-ring angular integrals rather than the runtime's sampled source
quadrature. This checks the same MFS representation, not an independent Stokes
formulation. The global grid and gap-patch points are separate from solve nodes.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import argparse
import json
import time
import resource
from pathlib import Path
import numpy as np
from scipy.special import roots_legendre, lpmv, gammaln
import analytic_ring_reference as ring
from full17_reference import rotation, strain_basis


def samples(record, global_rings, global_phi, patch_rings, patch_phi):
    ar = record['aspect_ratio']
    gap = record['h_over_c']
    axes = np.asarray(record['axes']).reshape(2, 3)
    centers = np.asarray(record['centers']).reshape(2, 3)
    shapes = [np.eye(3)/ar**2 + (1-1/ar**2)*np.outer(n, n) for n in axes]
    tensors = [ar**2*np.eye(3)+(1-ar**2)*np.outer(n, n) for n in axes]
    curvature = sum(q[:2, :2]*np.sqrt(m[2, 2]) for q, m in zip(shapes, tensors))
    values, vectors = np.linalg.eigh(curvature)
    lengths = np.sqrt(2*gap/values)
    mu, _ = roots_legendre(global_rings)
    phi = (np.arange(global_phi)+.371)*2*np.pi/global_phi
    points, body_ids, labels = [], [], []
    for body in range(2):
        rho = ar*np.sqrt(1-mu*mu)
        local = np.stack((rho[:, None]*np.cos(phi), rho[:, None]*np.sin(phi),
                          np.broadcast_to(mu[:, None], (global_rings, global_phi))), axis=-1)
        world = local.reshape(-1, 3) @ rotation(axes[body]).T + centers[body]
        points.extend(world)
        body_ids.extend([body]*len(world))
        labels.extend(['global']*len(world))
        # Radius through four characteristic gap widths, including exact closest
        # points. Staggered angles avoid solve-grid coincidence.
        xy = [np.zeros(2)]
        for radius in np.geomspace(.08, 4., patch_rings):
            for angle in (np.arange(patch_phi)+.217)*2*np.pi/patch_phi:
                xy.append(vectors @ (lengths*radius*np.array([np.cos(angle), np.sin(angle)])))
        q, center = shapes[body], centers[body]
        for x, y in xy:
            dx, dy = x-center[0], y-center[1]
            linear = q[2, 0]*dx+q[2, 1]*dy
            disc = linear**2-q[2, 2]*(q[0, 0]*dx*dx+2*q[0, 1]*dx*dy+q[1, 1]*dy*dy-1)
            if disc <= 0:
                continue
            z = center[2]+(-linear+(1 if body == 0 else -1)*np.sqrt(disc))/q[2, 2]
            points.append([x, y, z]);body_ids.append(body);labels.append('gap_patch')
    return np.asarray(points), np.asarray(body_ids), np.asarray(labels), axes, centers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('solution')
    parser.add_argument('--output', required=True)
    parser.add_argument('--global-rings', type=int, default=11)
    parser.add_argument('--global-phi', type=int, default=23)
    parser.add_argument('--patch-rings', type=int, default=8)
    parser.add_argument('--patch-phi', type=int, default=16)
    args = parser.parse_args();started = time.time()
    record = json.loads(Path(args.solution).read_text())
    coefficients = np.fromfile(record['coefficients_file'], dtype=np.float64).reshape(record['coefficients_rows'], 17)
    points, body_ids, labels, axes, centers = samples(record, args.global_rings, args.global_phi, args.patch_rings, args.patch_phi)
    expected = np.zeros((len(points), 3, 17));es = strain_basis();ar = record['aspect_ratio']
    for body in range(2):
        mask = body_ids == body;relative = points[mask]-centers[body]
        for k in range(3):
            expected[mask, k, 6*body+k] = 1.
            expected[mask, :, 6*body+3+k] = np.cross(np.eye(3)[k], relative)/ar
        for k in range(5):
            expected[mask, :, 12+k] = -relative @ es[k].T/ar
    velocity = np.empty_like(expected)
    own_velocity = np.empty_like(expected)
    harmonic_velocity = None
    if 'harmonics_file' in record:
        hc = np.fromfile(record['harmonics_file'], dtype=np.float64).reshape(2, -1, 3, 17)
        harmonic_velocity = np.zeros_like(expected)
        for body in range(2):
            mask = body_ids == body;rot = rotation(axes[body])
            local = (points[mask]-centers[body]) @ rot
            mu, phi = np.clip(local[:, 2], -1., 1.), np.arctan2(local[:, 1], local[:, 0])
            y = np.zeros((mask.sum(), (record['degree']+1)**2))
            for l in range(record['degree']+1):
                for m in range(l+1):
                    p = lpmv(m, l, mu)*np.sqrt((2*l+1)/(4*np.pi)*np.exp(gammaln(l-m+1)-gammaln(l+m+1)))
                    if m == 0:y[:, l*l] = p
                    else:
                        y[:, l*l+2*m-1] = np.sqrt(2)*p*np.cos(m*phi)
                        y[:, l*l+2*m] = np.sqrt(2)*p*np.sin(m*phi)
            harmonic_velocity[mask] = np.einsum('pa,acj,bc->pbj', y, hc[body], rot)
    bodies = list(zip(axes, centers))
    for begin in range(0, len(points), 16):
        end = min(begin+16, len(points))
        kernel = ring.matrix(points[begin:end], record['source_rings'], record['source_order'], record['source_depth'], bodies, a=ar)
        velocity[begin:end] = (kernel@coefficients).reshape(end-begin, 3, 17)
        half = len(coefficients)//2
        for body in range(2):
            local_mask = body_ids[begin:end] == body
            value = (kernel[:, body*half:(body+1)*half]@coefficients[body*half:(body+1)*half]).reshape(end-begin, 3, 17)
            own_velocity[begin:end][local_mask] = value[local_mask]
    error = np.linalg.norm(velocity-expected, axis=1)
    scale = np.max(np.linalg.norm(expected, axis=1), axis=0)
    result = {'schema': 'ar20_offgrid_noslip_v1', 'solution': str(Path(args.solution).resolve()),
              'h_over_c': record['h_over_c'], 'degree': record['degree'],
              'scope': 'Actual pair MFS velocity, independent analytic angular integration; same representation, not independent Stokes solver',
              'global_rings': args.global_rings, 'global_phi': args.global_phi,
              'patch_rings': args.patch_rings, 'patch_phi': args.patch_phi}
    for label in ['global', 'gap_patch']:
        mask = labels == label
        relative_max = np.max(error[mask], axis=0)/scale
        result[label] = {'count': int(mask.sum()), 'per_rhs_max_scaled_error': relative_max.tolist(),
                         'max_scaled_error': float(relative_max.max()),
                         'rms_scaled_error': float(np.sqrt(np.mean((error[mask]/scale)**2))),
                         'per_rhs_max_absolute_error': np.max(error[mask], axis=0).tolist(),
                         'per_rhs_rms_scaled_error': np.sqrt(np.mean((error[mask]/scale)**2, axis=0)).tolist()}
        result[label]['blocks'] = {}
        for name, cols in [('translation', [0,1,2,6,7,8]), ('rotation', [3,4,5,9,10,11]), ('strain', list(range(12,17)))]:
            result[label]['blocks'][name] = {'max_scaled_error': float(relative_max[cols].max()), 'rms_scaled_error': float(np.sqrt(np.mean((error[mask][:,cols]/scale[cols])**2)))}
        if harmonic_velocity is not None:
            self_error = np.linalg.norm(own_velocity-harmonic_velocity, axis=1)
            cross_error = np.linalg.norm(velocity-own_velocity-(expected-harmonic_velocity), axis=1)
            for name, values in [('self_fit_defect', self_error), ('cross_remainder', cross_error)]:
                scaled = values[mask]/scale
                result[label][name] = {'max_scaled_error': float(scaled.max()),
                                      'per_rhs_max_scaled_error': scaled.max(axis=0).tolist(),
                                      'rms_scaled_error': float(np.sqrt(np.mean(scaled**2)))}
            result['decomposition'] = 'self_fit=GselfD*x-H*x; cross_remainder=GcrossD*x-(prescribed-H*x), equals pure cross projection defect only with identity diagonal'

    result.update(wall_seconds=time.time()-started, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    Path(args.output).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ['degree', 'h_over_c', 'global', 'gap_patch', 'wall_seconds']}))


if __name__ == '__main__':
    main()
