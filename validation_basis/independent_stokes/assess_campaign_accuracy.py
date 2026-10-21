#!/usr/bin/env python3
"""Empirical, held-out full17 table screening against independent Stokes reports.

Two resolutions support a *sampled empirical comparison*, not a continuum error
bound or global physical qualification. A boundary residual is never converted
to a force/torque/stresslet error estimate. Missing or underresolved references
produce UNRESOLVED, retaining provisional comparisons whenever possible.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from compare_reference import (block_errors, discretization_signature,
                               energy_delta, energy_scaled_reciprocity,
                               operator_identity, symmetric)
from prepare_reference_source import (CONVENTION, PANEL_SCHEMAS, checkpoint,
                                      isolated_evidence, matrix17, metadata,
                                      read_json)

SCHEMA = 'reference17_accuracy_campaign_v1'
DEFAULTS = {'model': .05, 'refinement': .01, 'boundary': .01,
            'linear': 1e-8, 'reciprocity': .001, 'surface': .001}
BLOCK_NAMES = {'force': 'force', 'torque': 'torque_over_a',
               'stresslet': 'intrinsic_stresslet_over_a'}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def fnv_file(path):
    h = 14695981039346656037
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            for value in block:
                h = ((h ^ value) * 1099511628211) & ((1 << 64) - 1)
    return f'{h:016x}'


def number(value):
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return float(value)
    return None


def resolve(value, parent):
    if not isinstance(value, str) or not value:
        raise ValueError('A nonempty evidence path is required')
    path = Path(value)
    return path if path.is_absolute() else parent / path


def close(a, b):
    return math.isclose(float(a), float(b), rel_tol=1e-10, abs_tol=1e-13)


def axes(geometry):
    ti, tj, phi = (geometry[key] for key in ('theta_i', 'theta_j', 'phi'))
    return np.array([[math.sin(ti), 0., math.cos(ti)],
                     [math.sin(tj)*math.cos(phi), math.sin(tj)*math.sin(phi), math.cos(tj)]])


def same_pose(a, b):
    """Compare director lines modulo a common rotation around the gap normal."""
    if not close(a['aspect_ratio'], b['aspect_ratio']):
        return False
    aa, bb = axes(a), axes(b)
    pivots = [k for k in range(2) if np.linalg.norm(aa[k, :2]) > 1e-12]
    angle = 0.
    if pivots:
        k = pivots[0]
        if np.linalg.norm(bb[k, :2]) <= 1e-12:
            return False
        angle = math.atan2(bb[k, 1], bb[k, 0]) - math.atan2(aa[k, 1], aa[k, 0])
    for branch in (0., math.pi):
        c, s = math.cos(angle+branch), math.sin(angle+branch)
        rotation = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
        rotated = aa @ rotation.T
        if all(min(np.linalg.norm(rotated[k]-bb[k]), np.linalg.norm(rotated[k]+bb[k])) < 1e-10 for k in range(2)):
            return True
    return False


def geometry_matches(reference, query):
    # Matrices are compared in their supplied canonical Cartesian frame. A
    # physically equivalent rotated pose is NOT silently compared unrotated.
    for key in ('aspect_ratio', 'theta_i', 'theta_j', 'h_over_c', 'c_m'):
        if not close(reference[key], query[key]):
            return False
    return abs(math.remainder(reference['phi']-query['phi'], 2*math.pi)) <= 1e-10


def response_metrics(predicted, raw_reference):
    reference = symmetric(raw_reference)
    if np.linalg.eigvalsh(reference)[0] <= 0:
        raise ValueError('Reference symmetric part is not SPD; clipping is disabled')
    delta = predicted - raw_reference
    blocks = block_errors(delta, reference)
    return {
        'full17_relative_energy_error': energy_delta(delta, reference),
        'blocks': blocks,
        'raw_response_relative_frobenius_error': float(np.linalg.norm(delta) / np.linalg.norm(raw_reference)),
        'reference_raw_reciprocity': float(np.linalg.norm(raw_reference-raw_reference.T)/np.linalg.norm(raw_reference)),
        'reference_energy_scaled_reciprocity': energy_scaled_reciprocity(raw_reference, reference),
        'reference_minimum_symmetric_eigenvalue': float(np.linalg.eigvalsh(reference)[0]),
        'predicted_minimum_symmetric_eigenvalue': float(np.linalg.eigvalsh(symmetric(predicted))[0]),
        'energy_uses_symmetric_part': True,
        'force_torque_stresslet_errors_retain_raw_antisymmetric_response': True,
    }


def screening_values(metrics):
    return {'energy': metrics['full17_relative_energy_error'],
            **{name: metrics['blocks'][key]['energy_input_scaled_operator_relative']
               for name, key in BLOCK_NAMES.items()}}


def inspect_reference(path, query, policy):
    result = {'path': str(path), 'issues': [], 'raw': None, 'identity': None}
    issues = result['issues']
    if not path.is_file():
        issues.append('reference_report_missing')
        return result
    try:
        data = read_json(path)
        result.update(sha256=digest(path), data=data)
        stored_geometry = data.get('geometry_metadata', data)
        geometry = metadata(data, None if 'length_unit_nm' in stored_geometry else query['c_m']*1e9)
        result['geometry'] = geometry
        result['comparison_geometry_compatible'] = geometry_matches(geometry, query)
        if not result['comparison_geometry_compatible']:
            issues.append('reference_query_geometry_or_units_mismatch')
        raw = matrix17(data.get('raw_full17'), 'raw_full17')
        result['raw'] = raw
        total = symmetric(raw)
        minimum = float(np.linalg.eigvalsh(total)[0])
        result['minimum_symmetric_eigenvalue'] = minimum
        if minimum <= 0:
            issues.append('reference_not_spd')
        norm = np.linalg.norm(raw)
        reciprocity = float(np.linalg.norm(raw-raw.T)/norm) if norm > 0 else None
        scaled_reciprocity = energy_scaled_reciprocity(raw, total) if minimum > 0 else None
        result.update(raw_reciprocity=reciprocity, energy_scaled_reciprocity=scaled_reciprocity,
                      discretization_signature=discretization_signature(data), node_count=data.get('node_count'))
        for label, value in [('raw_reciprocity', reciprocity), ('energy_scaled_reciprocity', scaled_reciprocity)]:
            if value is None or value > policy['reciprocity']:
                issues.append(label+'_missing_or_failed')
        if data.get('status') != 'complete':
            issues.append('reference_not_complete')
        if data.get('independent_reference') is not True or data.get('shares_production_solver') is not False:
            issues.append('reference_independence_not_declared')
        if data.get('synthetic', False) is not False:
            issues.append('synthetic_reference_is_not_physical_evidence')
        if data.get('isolated_body') is not None:
            issues.append('isolated_report_is_not_a_pair_reference')
        identity_input = dict(data, schema='independent_ar20_panel_bie_v1') if data.get('schema') in PANEL_SCHEMAS else data
        identity, identity_issues = operator_identity(identity_input)
        result['identity'] = identity
        result['moment_method'] = data.get('moments', {}).get('method')
        issues.extend(identity_issues)
        validation = data.get('validation', {})
        checks = {'boundary_max_scaled_error': policy['boundary'], 'gap_patch_max_scaled_error': policy['boundary'],
                  'linear_relative_residual': policy['linear'], 'exact_surface_position_max_error_over_gap': policy['surface']}
        measurements = {}
        for name, limit in checks.items():
            value = number(validation.get(name, data.get('linear_residual') if name == 'linear_relative_residual' else None))
            measurements[name] = value
            if value is None or value < 0 or value > limit:
                issues.append(name+'_missing_or_failed')
        for name in ('off_grid_boundary', 'boundary_evaluation_valid'):
            if validation.get(name) is not True:
                issues.append(name+'_not_established')
        for name in ('boundary_sample_count', 'gap_patch_sample_count'):
            if type(validation.get(name)) is not int or validation[name] <= 0:
                issues.append(name+'_missing_or_nonpositive')
        result['measurements'] = measurements
        result['strict_0_1_percent_boundary_diagnostic'] = all(
            measurements[k] is not None and measurements[k] <= .001
            for k in ('boundary_max_scaled_error', 'gap_patch_max_scaled_error'))
        # Reuse the established moment/checkpoint evidence contract without
        # altering its isolated exact-spheroid gates or trusting a pass boolean.
        try:
            isolated_evidence(data, path, geometry, identity)
            result['isolated_moment_evidence_verified'] = True
        except (ValueError, OSError, KeyError, TypeError) as error:
            issues.append('isolated_moment_evidence_unverified: '+str(error))
        try:
            result['checkpoint'] = checkpoint(data, path)
        except (ValueError, OSError, KeyError, TypeError) as error:
            issues.append('solution_checkpoint_unverified: '+str(error))
    except (ValueError, OSError, KeyError, TypeError, np.linalg.LinAlgError) as error:
        issues.append('invalid_reference_report: '+str(error))
    return result


def public_reference(reference):
    return {key: value for key, value in reference.items() if key not in ('raw', 'data')}


def heldout_check(query, training, report_hashes, kind):
    poses = [row['geometry'] for row in training]
    matching_pose = [pose for pose in poses if same_pose(pose, query)]
    same_configuration = any(close(pose['h_over_c'], query['h_over_c']) for pose in matching_pose)
    same_gap = any(close(pose['h_over_c'], query['h_over_c']) for pose in poses)
    reused = any(value in {row.get('source_json_sha256') for row in training} for value in report_hashes)
    measured = 'training' if same_configuration else ('gap' if matching_pose else ('angle' if same_gap else 'both'))
    passed = not reused and not same_configuration
    if kind == 'gap':
        passed &= bool(matching_pose)
    elif kind == 'angle':
        passed &= not matching_pose
    elif kind == 'both':
        passed &= not matching_pose and not same_gap
    else:
        passed = False
    return {'passed': bool(passed), 'requested_kind': kind, 'measured_kind': measured,
            'training_pose_match': bool(matching_pose), 'training_gap_match': same_gap,
            'training_configuration_match': same_configuration, 'training_report_hash_reused': reused}


def assess_case(case, query, training, parent, policy, context_issues=()):
    result = {'id': str(case.get('id', case.get('query_id', ''))), 'query_id': str(case.get('query_id', '')),
              'geometry': query['geometry'], 'status': 'UNRESOLVED', 'physical_accuracy_qualified': False,
              'reasons': list(context_issues), 'table_vs_independent_reference': None, 'paired_refinement': None,
              'surrogate_vs_table_at_this_query': {'available': False, 'reason': 'No independent exact-surrogate matrix was supplied; angular interpolation is part of the queried table.'}}
    references = []
    paths = case.get('references', [])
    for value in paths:
        try:
            references.append(inspect_reference(resolve(value, parent), query['geometry'], policy))
        except (ValueError, OSError) as error:
            references.append({'path': str(value), 'issues': ['invalid_reference_path: '+str(error)], 'raw': None, 'identity': None})
    result['references'] = [public_reference(reference) for reference in references]
    result['strict_three_resolution_qualification_established'] = False
    heldout = heldout_check(query['geometry'], training, [r.get('sha256') for r in references], case.get('heldout_kind'))
    result['heldout'] = heldout
    if not heldout['passed']:
        result['reasons'].append('genuine_requested_heldout_configuration_not_established')
    if query['geometry']['h_over_c'] < query['effective_h_over_c']*(1-1e-12):
        result['reasons'].append('subfloor_regularization_is_not_the_same_physical_gap_BVP')
    comparable = [(index, reference) for index, reference in enumerate(references)
                  if reference.get('raw') is not None and reference.get('comparison_geometry_compatible')]
    if comparable:
        comparison_index, fine = comparable[-1]
        result['comparison_reference_index'] = comparison_index
        result['comparison_uses_requested_finest_reference'] = comparison_index == len(references)-1
        try:
            result['table_vs_independent_reference'] = response_metrics(query['total'], fine['raw'])
            if query.get('surrogate') is not None:
                result['surrogate_vs_table_at_this_query'] = {'available': True,
                    'metrics': response_metrics(query['total'], query['surrogate']),
                    'scope': 'Numerical transport of supplied surrogate, not physical reference accuracy'}
        except (ValueError, np.linalg.LinAlgError) as error:
            result['reasons'].append('comparison_unavailable: '+str(error))
    if len(references) < 2:
        result['reasons'].append('fewer_than_two_resolution_reports')
    else:
        coarse, fine = references[-2:]
        for role, reference in [('coarse', coarse), ('fine', fine)]:
            result['reasons'].extend(role+':'+issue for issue in reference['issues'])
        compatible_operator = (coarse.get('identity') is not None and coarse.get('identity') == fine.get('identity')
                               and coarse.get('moment_method') == fine.get('moment_method'))
        if not compatible_operator:
            result['reasons'].append('coarse_fine_numerical_operators_incompatible')
        if not coarse.get('discretization_signature') or coarse.get('discretization_signature') == fine.get('discretization_signature'):
            result['reasons'].append('distinct_spatial_discretizations_not_established')
        if coarse.get('sha256') == fine.get('sha256'):
            result['reasons'].append('coarse_fine_report_reused')
        if (coarse.get('raw') is not None and fine.get('raw') is not None
                and coarse.get('comparison_geometry_compatible') and fine.get('comparison_geometry_compatible')):
            try:
                measured_change = response_metrics(coarse['raw'], fine['raw'])
                if compatible_operator:
                    result['paired_refinement'] = measured_change
                else:
                    result['uncontrolled_operator_change'] = measured_change
            except (ValueError, np.linalg.LinAlgError) as error:
                result['reasons'].append('refinement_comparison_unavailable: '+str(error))
    model_metrics, refinement = result['table_vs_independent_reference'], result['paired_refinement']
    if refinement is not None:
        values = screening_values(refinement)
        result['paired_refinement_screen_passed'] = all(value <= policy['refinement'] for value in values.values())
        if not result['paired_refinement_screen_passed']:
            result['reasons'].append('paired_resolution_change_exceeds_screening_tolerance')
    if model_metrics is not None:
        values = screening_values(model_metrics)
        result['observed_table_error_screen_passed'] = all(value <= policy['model'] for value in values.values())
    if not result['reasons'] and model_metrics is not None and refinement is not None:
        result['status'] = 'PASS' if result['observed_table_error_screen_passed'] else 'FAIL'
        if result['status'] == 'FAIL':
            result['reasons'].append('observed_table_response_exceeds_model_screening_tolerance')
    result['next_action'] = {
        'PASS': 'Retain this sampled result; extend independent angle/gap coverage before assigning an accuracy range.',
        'FAIL': 'Inspect matching and angular/gap interpolation at this configuration; the measured table response exceeds the selected tolerance.',
        'UNRESOLVED': 'Resolve the listed reference, refinement, geometry or provenance issues at the same configuration before judging the model.'}[result['status']]
    result['judgement_scope'] = 'Provisional empirical screening at this sampled configuration using two checked resolutions; no continuum error bound or unsampled-range guarantee.'
    result['resolution_change_is_error_bound'] = False
    result['boundary_residual_is_force_error_bound'] = False
    return result


def context(group, parent):
    issues = []
    query_path = resolve(group['query_result'], parent)
    queried = read_json(query_path)
    if (queried.get('schema') != 'reference17_query_v1' or queried.get('convention') != CONVENTION
            or queried.get('operator') != 'symmetric_full17_total'):
        raise ValueError('Query result schema/full17 convention mismatch')
    source_path = resolve(group['training_source'], parent)
    manifest_path = resolve(group['training_manifest'], parent)
    training = read_json(manifest_path)
    if training.get('source_text_sha256') != digest(source_path):
        issues.append('training_source_manifest_hash_mismatch')
    sources = training.get('sources')
    if not isinstance(sources, list) or not sources:
        raise ValueError('Nonempty training geometry/provenance records are required')
    for source in sources:
        geometry = source.get('geometry', {})
        for key in ('aspect_ratio', 'c_m', 'theta_i', 'theta_j', 'phi', 'h_over_c'):
            if number(geometry.get(key)) is None:
                raise ValueError('Incomplete training geometry: '+key)
        if not close(geometry['aspect_ratio'], training['aspect_ratio']) or not close(geometry['c_m'], training['c_m']):
            raise ValueError('Training row units differ from its manifest')
        value = source.get('source_json_sha256')
        if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdefABCDEF' for c in value):
            raise ValueError('Training reference JSON identity is missing')
    if training.get('synthetic', False) is not False:
        issues.append('synthetic_training_source_is_not_physical_evidence')
    table_path = resolve(group['table_path'], parent)
    table_hash = digest(table_path)
    if queried.get('table_fnv1a64') != fnv_file(table_path):
        issues.append('query_table_hash_mismatch')
    if queried.get('physical_accuracy_qualified', False) is not False:
        issues.append('query_tool_cannot_claim_physical_qualification')
    generation_path = resolve(group['generation_report'], parent)
    generation = read_json(generation_path)
    if generation.get('source_fnv1a64') != fnv_file(source_path):
        issues.append('generator_training_source_hash_mismatch')
    if queried.get('source_identity') and queried['source_identity'] != 'fnv1a64:'+fnv_file(source_path):
        issues.append('query_training_source_identity_mismatch')
    if generation.get('model_revision') != queried.get('model_revision'):
        issues.append('generator_query_model_revision_mismatch')
    if generation.get('status') != 'provisional_numerical_transport_passed':
        issues.append('generator_numerical_transport_not_passed')
    measured_transport = number(generation.get('maximum_gap_interpolation_relative_energy_error'))
    transport_tolerance = number(generation.get('interpolation_tolerance'))
    if (measured_transport is None or measured_transport < 0 or transport_tolerance is None
            or transport_tolerance <= 0 or measured_transport > transport_tolerance):
        issues.append('generator_transport_measurement_missing_or_failed')
    if (generation.get('numerical_source_fingerprint')
            and generation['numerical_source_fingerprint'] != queried.get('numerical_source_fingerprint')):
        issues.append('generator_query_numerical_source_fingerprints_differ')
    if 'pipeline_path' in group:
        pipeline_path = resolve(group['pipeline_path'], parent)
        pipeline = read_json(pipeline_path)
        if pipeline.get('status') != 'complete' or pipeline.get('workflow_completed') is not True:
            issues.append('table_pipeline_not_complete')
        for path in (source_path, manifest_path, generation_path, table_path):
            stored = pipeline.get('outputs', {}).get(path.name, {}).get('sha256')
            if stored != digest(path):
                issues.append('pipeline_output_hash_mismatch:'+path.name)
    else:
        issues.append('complete_table_pipeline_manifest_missing')
    for key, expected in [('ar', training.get('aspect_ratio')), ('c_m', training.get('c_m')), ('floor_m', training.get('floor_m'))]:
        if number(queried.get(key)) is None or number(expected) is None or not close(queried[key], expected):
            issues.append('query_training_units_mismatch:'+key)
    rows = {}
    for row in queried.get('rows', []):
        key = str(row.get('id', ''))
        if not key or key in rows:
            raise ValueError('Missing/duplicate query id')
        geometry = {'aspect_ratio': queried['ar'], 'c_m': queried['c_m'],
                    **{name: row[name] for name in ('theta_i', 'theta_j', 'phi', 'h_over_c')}}
        if any(number(value) is None for value in geometry.values()):
            raise ValueError('Query geometry contains nonfinite values')
        if not close(row['gap_nm']*1e-9/queried['c_m'], row['h_over_c']):
            raise ValueError('Query physical gap and normalized gap disagree')
        effective = max(row['h_over_c'], queried['floor_m']/queried['c_m'])
        if not close(effective, row['effective_h_over_c']):
            raise ValueError('Query does not obey max(actual_gap, floor)')
        total = matrix17(row['total_full17'], 'query total_full17')
        if np.linalg.norm(total-total.T) > 1e-10*max(np.linalg.norm(total), 1e-300) or np.linalg.eigvalsh(symmetric(total))[0] <= 0:
            raise ValueError('Queried table total is not reciprocal SPD')
        if 'correction_full17' in row and 'isolated_full17' in row:
            restored = matrix17(row['correction_full17'], 'correction') + matrix17(row['isolated_full17'], 'isolated')
            if not np.allclose(restored, total, rtol=1e-11, atol=1e-11):
                raise ValueError('Query correction+isolated differs from total')
        rows[key] = {'geometry': geometry, 'effective_h_over_c': effective, 'total': total,
                     'surrogate': matrix17(row['surrogate_total_full17'], 'surrogate') if 'surrogate_total_full17' in row else None}
    transport = {key: generation.get(key) for key in ('maximum_roundtrip_relative_energy_error',
        'maximum_gap_interpolation_relative_energy_error', 'interpolation_tolerance', 'interior_samples_per_interval')}
    transport.update(scope='Generator sampled surrogate-to-table numerical transport at training poses; not a held-out Stokes error bound.', physical_accuracy_qualified=False)
    return rows, sources, issues, transport, {
        'query_result_sha256': digest(query_path), 'table_sha256': table_hash,
        'training_source_sha256': digest(source_path), 'training_manifest_sha256': digest(manifest_path),
        'generation_report_sha256': digest(generation_path), 'model_revision': queried.get('model_revision'),
        'numerical_source_fingerprint': queried.get('numerical_source_fingerprint')}


def assess_campaign(config, parent, policy=None):
    if config.get('schema') != SCHEMA:
        raise ValueError('Expected '+SCHEMA)
    policy = dict(DEFAULTS if policy is None else policy)
    for key, value in policy.items():
        if number(value) is None or value <= 0:
            raise ValueError('Invalid positive tolerance: '+key)
    groups = []
    for group in config.get('groups', []):
        record = {'id': str(group.get('id', '')), 'physical_accuracy_qualified': False, 'cases': []}
        try:
            rows, training, issues, transport, provenance = context(group, parent)
            record.update(surrogate_to_table_transport=transport, provenance=provenance)
            for case in group.get('cases', []):
                key = str(case.get('query_id', ''))
                if key not in rows:
                    record['cases'].append({'id': str(case.get('id', key)), 'query_id': key, 'status': 'UNRESOLVED',
                        'geometry': case.get('geometry', {}),
                        'physical_accuracy_qualified': False, 'reasons': ['query_row_missing']})
                else:
                    record['cases'].append(assess_case(case, rows[key], training, parent, policy, issues))
        except (ValueError, OSError, KeyError, TypeError, np.linalg.LinAlgError) as error:
            record['context_error'] = str(error)
            record['cases'] = [{'id': str(case.get('id', case.get('query_id', ''))), 'status': 'UNRESOLVED',
                'geometry': case.get('geometry', {}),
                'physical_accuracy_qualified': False, 'reasons': ['campaign_context_unavailable: '+str(error)]}
                for case in group.get('cases', [])]
        record['counts'] = {status: sum(case['status'] == status for case in record['cases']) for status in ('PASS', 'FAIL', 'UNRESOLVED')}
        groups.append(record)
    return {'schema': 'reference17_accuracy_assessment_v1', 'physical_accuracy_qualified': False,
            'scope': 'Sampled empirical independent-reference screening; untested ARs, gaps and angles are not certified.',
            'screening_tolerances': policy, 'boundary_residual_is_force_error_bound': False,
            'two_resolution_change_is_continuum_error_bound': False,
            'strict_reference_qualification_assessed': False,
            'strict_previous_three_resolution_and_0_1_percent_gates_modified': False,
            'groups': groups, 'assessor_sha256': digest(__file__)}


def flat_rows(report):
    for group in report['groups']:
        for case in group['cases']:
            g = case.get('geometry', {})
            row = {'group': group['id'], 'case': case['id'], 'status': case['status'],
                   'AR': g.get('aspect_ratio'), 'gap_nm': g.get('h_over_c', 0)*g.get('c_m', 0)*1e9 if g else None,
                   **{key: g.get(key) for key in ('theta_i', 'theta_j', 'phi')},
                   'heldout_kind': case.get('heldout', {}).get('measured_kind'),
                   'physical_accuracy_qualified': False, 'reasons': '; '.join(case.get('reasons', []))}
            for name, key in [('model', 'table_vs_independent_reference'), ('refinement', 'paired_refinement')]:
                metrics = case.get(key)
                row.update({name+'_'+field: value for field, value in (screening_values(metrics).items() if metrics else [(k, None) for k in ('energy', 'force', 'torque', 'stresslet')])})
            refs = case.get('references', [])
            row['fine_boundary_scaled_error'] = refs[-1].get('measurements', {}).get('boundary_max_scaled_error') if refs else None
            yield row


def save_plot(report, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = list(flat_rows(report));ars = sorted({row['AR'] for row in rows if row['AR'] is not None})
    if not ars:
        ars = [None]
    figure, panels = plt.subplots(2, len(ars), figsize=(max(7, 4*len(ars)), 7), squeeze=False, sharey='row')
    colors = {'PASS': '#277a4b', 'FAIL': '#b3261e', 'UNRESOLVED': '#8a6a1e'}
    for column, ar in enumerate(ars):
        for index, metric in enumerate(('model', 'refinement')):
            panel = panels[index, column]
            plotted = 0
            for row in rows:
                if row['AR'] != ar or not row['gap_nm'] or row[metric+'_energy'] is None:
                    continue
                panel.scatter(row['gap_nm'], max(row[metric+'_energy'], 1e-12), color=colors[row['status']], marker='o', s=25)
                block = max(row[metric+'_'+key] for key in ('force', 'torque', 'stresslet'))
                panel.scatter(row['gap_nm'], max(block, 1e-12), color=colors[row['status']], marker='x', s=25)
                plotted += 1
            panel.axhline(report['screening_tolerances'][metric], color='black', linestyle='--', linewidth=.8)
            panel.set_xscale('log');panel.set_yscale('log');panel.grid(alpha=.2)
            panel.set_title('AR '+str(ar) if ar is not None else 'No comparable references')
            panel.set_xlabel('Surface gap (nm)');panel.set_ylabel('Table/reference error' if metric == 'model' else 'Coarse/fine change')
            if not plotted:
                unresolved = sum(row['AR'] == ar and row['status'] == 'UNRESOLVED' for row in rows)
                panel.text(.5, .5, f'No comparable reference\n{unresolved} unresolved cases',
                           transform=panel.transAxes, ha='center', va='center', fontsize=10)
    figure.suptitle('Provisional sampled screening\nCircles: energy; crosses: largest F/T/S block error\n'
                     'Green: PASS; red: FAIL; amber: UNRESOLVED\n'
                     'Dashed: screening tolerances. No unsampled-range claim.', fontsize=10)
    figure.tight_layout(rect=(0, 0, 1, .87));figure.savefig(path, dpi=160);plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--csv', type=Path)
    parser.add_argument('--plot', type=Path)
    for name in ('model', 'refinement', 'boundary'):
        parser.add_argument('--'+name+'-tolerance', type=float, default=DEFAULTS[name])
    args = parser.parse_args()
    policy = dict(DEFAULTS, model=args.model_tolerance, refinement=args.refinement_tolerance, boundary=args.boundary_tolerance)
    report = assess_campaign(read_json(args.manifest), args.manifest.resolve().parent, policy)
    report['input_manifest_sha256'] = digest(args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    csv_path = args.csv or args.output.with_suffix('.csv');csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(flat_rows(report))
    with csv_path.open('w', newline='') as stream:
        if rows:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    if args.plot:
        args.plot.parent.mkdir(parents=True, exist_ok=True);save_plot(report, args.plot)
    counts = {status: sum(group['counts'][status] for group in report['groups']) for status in ('PASS', 'FAIL', 'UNRESOLVED')}
    print(json.dumps({'report': str(args.output), 'counts': counts, 'physical_accuracy_qualified': False}))
    # A completed assessment is successful even when physics is unresolved.
    # Its explicit per-case statuses are the workflow result, not a crash.
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
