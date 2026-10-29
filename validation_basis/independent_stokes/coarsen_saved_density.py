#!/usr/bin/env python3
"""Screen p8 sibling merges using the saved, unqualified AR20 h150 density.

No Stokes kernel is evaluated and no physical solve occurs.  The reported
change is only beta/2 times the difference between two density interpolants.
It is not an error bound for the complete boundary velocity or the moments.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss, legvander
from scipy.optimize import Bounds, LinearConstraint, milp

from backend import Backend
from geometry import evaluate_quad_panels, recorded_pose


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve_solution_file(parent, parent_path):
    """Resolve an archived solution without changing its physical report.

    An existing recorded file always wins, including when its hash is wrong.
    Only a missing recorded path permits the deterministic adjacent fallback.
    """
    solution = Path(parent['solution_file'])
    if not solution.exists():
        solution = Path(parent_path).with_suffix('.solution.npz')
    if sha(solution) != parent['solution_sha256']:
        raise ValueError('Saved density hash differs from its report')
    return solution


def cube_points(vertices, uv):
    vertices, uv = np.asarray(vertices), np.asarray(uv)
    return (vertices[0] + (uv[0, :, None]+1)*.5*(vertices[1]-vertices[0])
            + (uv[1, :, None]+1)*.5*(vertices[3]-vertices[0]))


def cube_uv(vertices, points):
    vertices = np.asarray(vertices)
    directions = np.stack((vertices[1]-vertices[0], vertices[3]-vertices[0]))
    affine = (np.asarray(points)-vertices[0]) @ directions.T
    return (2*affine/np.sum(directions*directions, axis=1)-1).T


def evaluate(coefficients, uv):
    return np.einsum('ki,kj,ijcm->kcm', legvander(uv[0], 8),
                     legvander(uv[1], 8), coefficients)


def face_rectangle(descriptor):
    body, vertices = descriptor
    vertices = np.asarray(vertices)
    low, high = vertices.min(0), vertices.max(0)
    fixed = int(np.argmin(high-low))
    free = [k for k in range(3) if k != fixed]
    return (int(body), fixed, int(round(low[fixed]))), low[free], high[free]


def check_coverage(descriptors):
    """Check complete, nonoverlapping coverage of each exact cube face."""
    faces = {}
    for descriptor in descriptors:
        face, low, high = face_rectangle(descriptor)
        faces.setdefault(face, []).append((low, high))
    if len(faces) != 12:
        raise ValueError('Expected all twelve body/cube faces')
    max_area_error, max_overlap = 0., 0.
    for rectangles in faces.values():
        area = sum(float(np.prod(hi-lo)) for lo, hi in rectangles)
        max_area_error = max(max_area_error, abs(area-4.))
        for i, (low, high) in enumerate(rectangles):
            if np.any(low < -1) or np.any(high > 1) or np.any(high <= low):
                raise ValueError('Invalid cube rectangle')
            for other_low, other_high in rectangles[i+1:]:
                overlap = np.maximum(np.minimum(high, other_high)-np.maximum(low, other_low), 0.)
                max_overlap = max(max_overlap, float(np.prod(overlap)))
    if max_area_error > 1e-13 or max_overlap > 1e-13:
        raise ValueError('Coarsened descriptors overlap or leave a hole')
    return {'maximum_cube_face_area_error': max_area_error,
            'maximum_pair_overlap_area': max_overlap}


def choose_merges(records, budget, panel_count):
    eligible = [i for i, row in enumerate(records)
                if row.get('projection_screen_available') and
                row['maximum_scaled_jump_change'] <= budget]
    if not eligible:
        return []
    incidence = np.zeros((panel_count, len(eligible)))
    for column, i in enumerate(eligible):
        incidence[records[i]['fine159_patch_indices'], column] = 1.
    # A single extra merge dominates the entire secondary penalty.  Thus this
    # maximizes cardinality first and minimizes summed indicators second.
    penalties = np.array([records[i]['maximum_scaled_jump_change']/budget for i in eligible])
    result = milp(-np.ones(len(eligible)) + 1e-4*penalties,
                  integrality=np.ones(len(eligible)), bounds=Bounds(0., 1.),
                  constraints=LinearConstraint(incidence, 0., 1.),
                  options={'time_limit': 30.})
    if not result.success or result.x is None:
        raise RuntimeError(f'Sibling matching did not certify optimum: {result.message}')
    return [eligible[j] for j in np.flatnonzero(result.x > .5)]


def run(parent_path, topology_path, screen_path, library, output_prefix, budgets):
    parent = json.loads(parent_path.read_text())
    topology = json.loads(topology_path.read_text())
    screen = json.loads(screen_path.read_text())
    pose = recorded_pose(parent['h_nm'])
    if parent.get('order') != 8 or parent.get('status') != 'complete':
        raise ValueError('A completed p8 parent is required')
    for record in (parent, topology):
        if record.get('h_nm') != 150 or record.get('focus', 0) != 0:
            raise ValueError('Only the recorded unfocused AR20 h150 pose is supported')
        for key in ('aspect_ratio', 'theta_i', 'theta_j', 'phi'):
            if record[key] != pose.metadata()[key]:
                raise ValueError(f'Pose mismatch: {key}')
    if sha(library) != parent['library_sha256']:
        raise ValueError('Use the same native library as the saved density')
    solution = resolve_solution_file(parent, parent_path)
    with np.load(solution) as data:
        density = data['density']
        rhs = data['rhs'].reshape(-1, 3, 17)
        if not np.all(data['norders'] == 8) or not np.all(np.diff(data['ixyzs']) == 81):
            raise ValueError('Expected uniform p8 saved density')
    scale = np.max(np.linalg.norm(rhs, axis=1), axis=0)
    if np.any(scale <= 0):
        raise ValueError('Every prescribed mode must have a nonzero velocity scale')
    uv = Backend(library).discretization(8)[0]
    gu, iu = np.unique(uv[0], return_inverse=True)
    gv, iv = np.unique(uv[1], return_inverse=True)
    invu, invv = np.linalg.inv(legvander(gu, 8)), np.linalg.inv(legvander(gv, 8))
    coefficients = []
    reconstruction_error = 0.
    for i in range(parent['patch_count']):
        grid = np.empty((9, 9, 3, 17))
        grid[iu, iv] = density[81*i:81*(i+1)]
        coefficient = np.einsum('iu,jv,uvcm->ijcm', invu, invv, grid)
        coefficients.append(coefficient)
        reconstruction_error = max(reconstruction_error, float(np.max(
            np.linalg.norm(evaluate(coefficient, uv)-density[81*i:81*(i+1)], axis=1)/scale)))
    # Use the same tensor layout as uv for fitting a merged parent.  The
    # independent check sites include each side of every former interface.
    checks = np.cos(np.arange(33)*np.pi/32)
    cu, cv = np.meshgrid(checks, checks, indexing='ij')
    check_uv = np.array([cu.ravel(), cv.ravel()])
    hotspots = set(map(int, np.flatnonzero(np.asarray(parent['validation']['boundary_per_patch_max_scaled_error']) > .0008)))
    records = []
    for original in screen['candidate_records']:
        row = dict(original)
        row['projection_screen_available'] = False
        if not row['geometry_screen_passed']:
            row['exclusion'] = 'analytic_geometry_screen_failed'
        elif hotspots.intersection(row['inherited123_parent_indices']):
            row['exclusion'] = 'preserve_targeted_hotspot_refinements'
        else:
            body, vertices = row['merged_descriptor']
            sample_points = cube_points(vertices, uv)
            samples = np.zeros((81, 3, 17))
            counts = np.zeros(81, dtype=int)
            for p in sorted(set(row['inherited123_parent_indices'])):
                local = cube_uv(parent['patch_descriptors'][p][1], sample_points)
                inside = np.max(np.abs(local), axis=0) <= 1+1e-12
                samples[inside] += evaluate(coefficients[p], local[:, inside])
                counts[inside] += 1
            if np.any(counts == 0):
                raise ValueError('Candidate parent extends beyond saved density coverage')
            samples /= counts[:, None, None]
            grid = np.empty((9, 9, 3, 17))
            grid[iu, iv] = samples
            merged_coefficients = np.einsum('iu,jv,uvcm->ijcm', invu, invv, grid)
            maximum, squares, count, worst = 0., 0., 0, None
            per_mode = np.zeros(17)
            for fine_id, old_id in zip(row['fine159_patch_indices'], row['inherited123_parent_indices']):
                points = cube_points(topology['patch_descriptors'][fine_id][1], check_uv)
                old_uv = cube_uv(parent['patch_descriptors'][old_id][1], points)
                merged_uv = cube_uv(vertices, points)
                reference = evaluate(coefficients[old_id], old_uv)
                projected = evaluate(merged_coefficients, merged_uv)
                difference = abs(parent['beta'])*.5*np.linalg.norm(projected-reference, axis=1)/scale
                per_mode = np.maximum(per_mode, difference.max(axis=0))
                where = np.unravel_index(np.argmax(difference), difference.shape)
                value = float(difference[where])
                if value > maximum:
                    maximum = value
                    worst = {'fine159_patch': fine_id, 'saved123_patch': old_id,
                             'mode': int(where[1]), 'child_uv': check_uv[:, where[0]].tolist()}
                squares += float(np.sum(difference**2))
                count += difference.size
            # Geometry is rechecked from the exact analytic surface, not merely
            # copied from the preceding diagnostic.
            nodes = evaluate_quad_panels(pose, body, np.asarray(vertices)[None], uv).points[0]
            node_grid = np.empty((9, 9, 3)); node_grid[iu, iv] = nodes
            geom = np.einsum('iu,jv,uvc->ijc', invu, invv, node_grid)
            predicted = np.einsum('ki,kj,ijc->kc', legvander(check_uv[0], 8), legvander(check_uv[1], 8), geom)
            exact = evaluate_quad_panels(pose, body, np.asarray(vertices)[None], check_uv).points[0]
            geometry_error = float(np.linalg.norm(predicted-exact, axis=1).max()/pose.gap_over_c)
            if geometry_error > .001:
                raise ValueError('A candidate passed an inconsistent geometry screen')
            row.update(projection_screen_available=True, maximum_scaled_jump_change=maximum,
                       rms_scaled_jump_change=(squares/count)**.5,
                       per_mode_maximum_scaled_jump_change=per_mode.tolist(), worst_projection_sample=worst,
                       merged_p8_exact_position_error_over_gap=geometry_error,
                       averaged_interface_fit_node_count=int(np.sum(counts > 1)))
        records.append(row)
    dependencies = {str(path.resolve()): sha(path) for path in
                    (parent_path, topology_path, screen_path, solution, Path(__file__),
                     Path(__file__).with_name('geometry.py'), Path(library))}
    audit = {'schema': 'ar20_saved_density_sibling_projection_audit_v1',
             'contains_new_stokes_solve': False, 'physical_accuracy_qualified': False,
             'uses_new159_solution': False, 'source_files': dependencies,
             'preserved_hotspot123_indices': sorted(hotspots),
             'boundary_mode_velocity_scales': scale.tolist(),
             'saved_node_polynomial_reconstruction_max_scaled_error': reconstruction_error,
             'metric_definition': 'max |beta|/2*norm(q_merged_p8-q_saved_piecewise_p8)/original_prescribed_boundary_mode_scale',
             'metric_is_full_velocity_error_bound': False,
             'sampling': '33x33 Chebyshev-Lobatto sites on EACH constituent child, including both former interface traces',
             'interface_fit_policy': 'At merged Gauss nodes on a saved interface, average the saved polynomial traces; independently check against both traces.',
             'candidate_records': records, 'selections': []}
    # This is a finite candidate-set feasibility diagnostic, not permission to
    # increase a screening budget.  It explains an infeasible requested cap.
    levels = sorted({r['maximum_scaled_jump_change'] for r in records
                     if r.get('projection_screen_available')})
    audit['resource_cap_projection_thresholds_not_adopted'] = []
    for cap in (138, 135):
        needed = topology['patch_count']-cap
        result = None
        for level in levels:
            selected = choose_merges(records, level, topology['patch_count'])
            if len(selected) >= needed:
                result = {'target_patch_cap': cap, 'required_merges': needed,
                          'minimum_maximum_jump_proxy_in_this_candidate_set': level,
                          'available_nonoverlapping_merges': len(selected),
                          'adopted_as_acceptable': False}
                break
        audit['resource_cap_projection_thresholds_not_adopted'].append(result or
            {'target_patch_cap': cap, 'required_merges': needed,
             'feasible_even_without_projection_limit': False})
    for budget in budgets:
        indices = choose_merges(records, budget, topology['patch_count'])
        selected = [records[i] for i in indices]
        consumed = set()
        replacements = {}
        for row in selected:
            first, second = row['fine159_patch_indices']
            if consumed.intersection((first, second)):
                raise ValueError('Selected merges overlap')
            consumed.update((first, second)); replacements[first] = row['merged_descriptor']
        descriptors = []
        for i, descriptor in enumerate(topology['patch_descriptors']):
            if i in replacements:
                descriptors.append(replacements[i])
            elif i not in consumed:
                descriptors.append(descriptor)
        coverage = check_coverage(descriptors)
        nodes = len(descriptors)*81
        proposal = pose.metadata()
        proposal.update(schema='independent_ar20_mesh_proposal_v1', contains_new_stokes_solve=False,
                        physical_accuracy_qualified=False, selection_is_error_bound=False,
                        order=8, focus=0., isolated_body=None,
                        strategy='Preserve all24 targeted hotspot regions; maximum-cardinality nonoverlapping sibling merges screened using the OLD123-panel density.',
                        parent_path=str(parent_path.resolve()), parent_sha256=sha(parent_path),
                        topology_path=str(topology_path.resolve()), topology_sha256=sha(topology_path),
                        parent_solution_sha256=sha(solution), patch_count=len(descriptors), node_count=nodes,
                        matrix_bytes_each=72*nodes*nodes, projection_screen_budget=budget,
                        maximum_selected_scaled_jump_change=max((r['maximum_scaled_jump_change'] for r in selected), default=0.),
                        maximum_selected_geometry_error_over_gap=max((r['merged_p8_exact_position_error_over_gap'] for r in selected), default=0.),
                        preserved_hotspot123_indices=sorted(hotspots), unpreserved_hotspot123_indices=[],
                        selected_merges=[r['fine159_patch_indices'] for r in selected],
                        merge_count=len(selected), exact_coverage=coverage, patch_descriptors=descriptors,
                        distinct_from_159_topology=bool(selected),
                        requested_panel_cap_138_met=len(descriptors) <= 138,
                        desired_panel_cap_135_met=len(descriptors) <= 135,
                        source_files=dependencies,
                        limitations=['Density projection is only a screening indicator, not a complete Stokes boundary error or moment uncertainty.',
                                     'All candidate densities come from the old unqualified123-panel solution; the159-panel solve produced no density.',
                                     'A fresh physical solve and unchanged0.1% independent boundary gate remain necessary.'])
        suffix = f'{budget:.0e}'.replace('-', 'm').replace('+', 'p')
        path = Path(str(output_prefix)+f'_budget_{suffix}_proposal.json')
        path.write_text(json.dumps(proposal, indent=2, allow_nan=False)+'\n')
        audit['selections'].append({'budget': budget, 'path': str(path.resolve()), 'sha256': sha(path),
                                    'merge_count': len(selected), 'patch_count': len(descriptors),
                                    'node_count': nodes, 'matrix_bytes_each': 72*nodes*nodes,
                                    'maximum_selected_scaled_jump_change': proposal['maximum_selected_scaled_jump_change']})
    audit_path = Path(str(output_prefix)+'_projection_audit.json')
    audit_path.write_text(json.dumps(audit, indent=2, allow_nan=False)+'\n')
    return audit['selections'], audit_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--topology', type=Path, required=True)
    parser.add_argument('--geometry-screen', type=Path, required=True)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--output-prefix', type=Path, required=True)
    parser.add_argument('--budgets', type=float, nargs='+', default=[1e-4, 2e-4])
    args = parser.parse_args()
    if not all(np.isfinite(b) and b > 0 for b in args.budgets):
        parser.error('Projection budgets must be finite and positive')
    selections, path = run(args.parent, args.topology, args.geometry_screen,
                           args.library, args.output_prefix, args.budgets)
    print(json.dumps({'audit': str(path), 'selections': selections}, indent=2))


if __name__ == '__main__':
    main()
