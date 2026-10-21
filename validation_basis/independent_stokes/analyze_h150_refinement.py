#!/usr/bin/env python3
"""Read-only diagnostics of the recorded h=150nm AR20 refinement attempts.

This analyzes saved matrices and sampled boundary errors. It launches no
Stokes solves and never qualifies a physical reference.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.linalg import eigh

from compare_reference import energy_scaled_reciprocity, response_metrics, symmetric
from geometry import canonical_geometry, evaluate_quad_panels


NAMES = ['pair_h150_p6_l1', 'pair_h150_p8_l1', 'pair_h150_p6_adapt1',
         'pair_h150_p6_adapt2', 'pair_h150_p8_adapt1']
PAIRS = [(0, 1), (0, 2), (2, 3), (2, 4), (3, 4), (1, 4)]


def analyze(directory):
    pose = canonical_geometry(1.)
    inputs = {}
    runs = []
    for name in NAMES:
        path = directory/(name+'.json')
        record = json.loads(path.read_text())
        for field in ('h_over_c', 'aspect_ratio', 'theta_i', 'theta_j', 'phi'):
            if abs(record[field]-pose.metadata()[field]) > 1e-13:
                raise ValueError('Diagnostic scope is the recorded AR20 pose at150nm only')
        inputs[name] = record
        validation = record['validation']
        raw = np.asarray(record['raw_full17'])
        scores = np.asarray(validation['boundary_per_patch_max_scaled_error'])
        mode_errors = np.asarray(validation['boundary_per_mode_max_scaled_error'])
        patches = []
        for index in np.argsort(scores)[::-1]:
            body, corners = record['patch_descriptors'][index]
            corners = np.asarray(corners)
            surface = evaluate_quad_panels(pose, body, corners[None], np.zeros((2,1)))
            point = surface.points.reshape(-1,3)[0]
            normal = surface.normals.reshape(-1,3)[0]
            local_direction = corners.mean(axis=0)
            local_direction /= np.linalg.norm(local_direction)
            patches.append(dict(patch=int(index), body=int(body), max_scaled_error=float(scores[index]),
                exact_center_c=point.tolist(), exact_normal=normal.tolist(),
                local_sphere_center=local_direction.tolist(),
                touches_equatorial_parameter_rim=bool(np.min(abs(corners[:,2])) < 1e-14),
                normal_faces_contact_plane=bool(normal[2]*(1 if body==0 else -1) > 0)))
        top8 = [item['patch'] for item in patches[:8]]
        top13 = [item['patch'] for item in patches[:13]]
        runs.append(dict(name=name, path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            order=record['order'], node_count=record['node_count'], patch_count=record['patch_count'],
            boundary_evaluation_valid=validation.get('boundary_evaluation_valid'),
            boundary_max_scaled_error=validation['boundary_max_scaled_error'],
            boundary_rms_scaled_error=validation['boundary_rms_scaled_error'],
            gap_patch_max_scaled_error=validation['gap_patch_max_scaled_error'],
            boundary_sample_count=validation['boundary_sample_count'],
            patches_above_point1_percent=int((scores>.001).sum()),
            patches_above_one_percent=int((scores>.01).sum()),
            raw_reciprocity=record['raw_reciprocity'],
            energy_scaled_raw_reciprocity=energy_scaled_reciprocity(raw,symmetric(raw)),
            exact_surface_position_max_error_over_gap=validation['exact_surface_position_max_error_over_gap'],
            max_scaled_flux_error=record['moments'].get('max_scaled_flux_error'),
            dominant_boundary_modes=[dict(index=int(i),name=record['mode_names'][i],max_scaled_error=float(mode_errors[i]))
                                     for i in np.argsort(mode_errors)[::-1]],
            ranked_panels_top20=patches[:20],
            candidate_refine_top8=top8,
            top8_fourway_resulting_panels=record['patch_count']+3*len(top8),
            top8_fourway_resulting_nodes=(record['patch_count']+3*len(top8))*(record['order']+1)**2,
            candidate_refine_top13=top13,
            top13_fourway_resulting_panels=record['patch_count']+3*len(top13),
            top13_fourway_resulting_nodes=(record['patch_count']+3*len(top13))*(record['order']+1)**2))
    changes = []
    for i,j in PAIRS:
        previous, current = inputs[NAMES[i]], inputs[NAMES[j]]
        before = symmetric(np.asarray(previous['raw_full17']))
        after = symmetric(np.asarray(current['raw_full17']))
        metrics = response_metrics(after,before,1.)
        values,vectors = eigh(after-before,before)
        index = np.argmax(abs(values))
        direction = vectors[:,index]
        direction /= np.linalg.norm(direction)
        metrics.update(previous=NAMES[i],current=NAMES[j],
            scope='Observed numerical change only, not an error bound or physical qualification',
            worst_energy_signed_relative_change=float(values[index]),
            worst_energy_input_euclidean_normalized=direction.tolist(),
            worst_energy_input_largest_components=[dict(index=int(k),name=previous['mode_names'][k],value=float(direction[k]))
                                                  for k in np.argsort(abs(direction))[-6:][::-1]])
        changes.append(metrics)
    return dict(schema='ar20_h150_saved_refinement_diagnostics_v1',physical_accuracy_qualified=False,
        scope='Exact recorded AR20 pose at h=150nm only; read-only saved-output diagnostics',
        runs=runs,observed_matrix_changes=changes,
        interpretation=[
            'At72 panels, increasing p6 to p8 changes integrated full17 energy by about0.043%, but maximum boundary error remains2.59%.',
            'At p6, adapt1 to adapt2 changes energy by about0.045%, but maximum boundary error remains3.40%. Small integral changes do not establish no-slip convergence.',
            'Boundary maxima across different meshes use different sample sets; their ratios are diagnostic, not rigorous convergence orders.',
            'Ui_z and Uj_z dominate sampled no-slip errors, followed by x rotations and yz strain. Per-panel worst mode is not stored, so no panel-specific mode attribution is inferred.',
            'p8 adapt1 top panels35 and25 face the gap, but significant residuals also occur on outer-facing panels. Refining only the closest point will not address all observed errors.',
            'Candidate top8 and top13 lists refer to each named source mesh; do not transfer numeric panel IDs between adaptive generations.',
            'Use existing matrix-free work before launching the indicated larger candidates; this script neither solves nor recommends a dense allocation.',
            'All current runs remain physically unqualified. Raw matrices, clean boundary execution, isolated moments, independent spatial and quadrature refinement must all pass before matching.'
        ])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path(__file__).with_name('results'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or args.results/'h150_refinement_diagnostics.json'
    report = analyze(args.results)
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(output),physical_accuracy_qualified=False,runs=len(report['runs']))))
