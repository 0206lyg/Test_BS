#!/usr/bin/env python3
"""Distance-first, sampled full17 screening; never a global accuracy certificate.

A solved linear system is not by itself an accurate Stokes reference. This
module verifies the existing SHA-backed isolated/checkpoint evidence contract,
spatial refinement, geometry, off-grid boundary diagnostics, and reciprocity.
It compares energy and separately scaled force/torque/stresslet response modes.
No matrices are clipped. A boundary residual is not a force-error bound.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import subprocess

import numpy as np
from scipy.linalg import solve_triangular

from assess_campaign_accuracy import (digest, geometry_matches, inspect_reference,
    public_reference, response_metrics, screening_values)
from compare_reference import symmetric, matrix as numerical_matrix
from prepare_reference_source import CONVENTION, matrix17
from reference_table_campaign import read, save, task_report, identity

SCHEMA = 'distance_regime_assessment_v1'
DEFAULTS = dict(model=.01, refinement=.002, boundary=.001,
                linear=1e-8, reciprocity=.001, surface=.001)
SCOPE = ('Empirical screening only at the supplied AR, orientation and gap samples. '
         'Two spatial resolutions are not a continuum error bound; four orientations '
         'do not establish angular generalization. No unsampled interval is certified.')
MODEL_NAMES = dict(near='isolated_plus_raw_janoschek_leading_v1',
                  far='independent_isolated_spheroids_exact_self_v1',
                  matched_near='raw_janoschek_plus_constant_full17_anchor_residual_v1')


def policy_values(policy=None):
    out = dict(DEFAULTS)
    if policy:
        out.update(policy)
    for name, value in out.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError('Finite positive tolerance required: '+name)
    return out


def query_geometry(point):
    return dict(aspect_ratio=float(point['aspect_ratio']), c_m=float(point['c_nm'])*1e-9,
                theta_i=float(point['theta_i']), theta_j=float(point['theta_j']),
                phi=float(point['phi']), h_over_c=float(point['gap_nm'])/float(point['c_nm']))


def balanced_metrics(predicted, raw_reference):
    result = response_metrics(predicted, raw_reference)
    # Unlike a Frobenius norm, this also sees weak modes and antisymmetric
    # perturbations in the same energy-normalized input/output coordinates.
    chol = np.linalg.cholesky(symmetric(raw_reference))
    left = solve_triangular(chol, predicted-raw_reference, lower=True)
    normalized = solve_triangular(chol, left.T, lower=True).T
    result['energy_normalized_full_response_operator_error'] = float(np.linalg.norm(normalized, 2))
    return result


def values(metrics):
    return dict(screening_values(metrics), full_response=metrics['energy_normalized_full_response_operator_error'])


def model_comparison(predicted, reference, tolerance, model_name):
    result = dict(model=model_name, status='UNRESOLVED', passed=False, reasons=[])
    try:
        predicted = numerical_matrix(predicted)
        result['metrics'] = balanced_metrics(predicted, reference)
        result['screening_values'] = values(result['metrics'])
        result['maximum_screened_error'] = max(result['screening_values'].values())
        result['positive_dissipation'] = result['metrics']['predicted_minimum_symmetric_eigenvalue'] > 0
        result['passed'] = result['positive_dissipation'] and result['maximum_screened_error'] <= tolerance
        result['status'] = 'PASS' if result['passed'] else 'FAIL'
        if not result['positive_dissipation']:
            result['reasons'].append('candidate_total_is_not_positive_definite_no_clipping_applied')
        if result['maximum_screened_error'] > tolerance:
            result['reasons'].append('candidate_error_exceeds_model_tolerance')
    except (ValueError, TypeError, np.linalg.LinAlgError) as error:
        result['reasons'].append(str(error))
    return result


def classify(point):
    if not point.get('reference_resolved'):
        return 'UNRESOLVED'
    models = point['models']
    near = models.get('selected_matched_near', models.get('near', {}))
    far = models.get('far', {})
    # A point where both approximations pass is still eligible for the far
    # regime. OVERLAP is assigned by the distance sweep only when multiple
    # adjacent near-model/independent-Stokes comparisons pass.
    if far.get('passed'):
        return 'FAR'
    if near.get('status') == 'UNRESOLVED' or far.get('status') == 'UNRESOLVED':
        return 'UNRESOLVED'
    if near.get('passed'):
        return 'NEAR'
    return 'INTERMEDIATE'


def assess_point(point, reference_paths, operators, policy=None):
    """Assess one frozen geometry. reference_paths ordered coarse then fine."""
    policy = policy_values(policy)
    geometry = query_geometry(point)
    result = dict(id=point['id'], orientation=point['orientation'], geometry=geometry,
                  gap_nm=float(point['gap_nm']), aspect_ratio=float(point['aspect_ratio']),
                  reference_resolved=False, status='UNRESOLVED', reasons=[], models={},
                  physical_accuracy_qualified=False, judgement_scope=SCOPE,
                  boundary_residual_is_force_error_bound=False,
                  resolution_change_is_error_bound=False)
    references = [inspect_reference(Path(path), geometry, policy) for path in reference_paths]
    result['references'] = [public_reference(item) for item in references]
    if len(references) != 2:
        result['reasons'].append('exactly_two_ordered_spatial_reference_reports_required')
    else:
        coarse, fine = references
        for role, reference in zip(('coarse', 'fine'), references):
            result['reasons'].extend(role+':'+issue for issue in reference['issues'])
        compatible = (coarse.get('identity') is not None and coarse.get('identity') == fine.get('identity')
                      and coarse.get('moment_method') == fine.get('moment_method'))
        if not compatible:
            result['reasons'].append('coarse_fine_numerical_operators_incompatible')
        if not coarse.get('discretization_signature') or coarse.get('discretization_signature') == fine.get('discretization_signature'):
            result['reasons'].append('distinct_spatial_discretizations_not_established')
        coarse_order = coarse.get('data', {}).get('order')
        fine_order = fine.get('data', {}).get('order')
        node_increase = (isinstance(coarse.get('node_count'), int) and isinstance(fine.get('node_count'), int)
                         and fine['node_count'] > coarse['node_count'])
        order_increase = (isinstance(coarse_order, int) and isinstance(fine_order, int) and fine_order > coarse_order)
        result['spatial_refinement_design'] = dict(coarse_order=coarse_order, fine_order=fine_order,
            coarse_nodes=coarse.get('node_count'), fine_nodes=fine.get('node_count'),
            polynomial_order_increased=order_increase, total_node_count_increased=node_increase,
            independently_adapted_meshes_may_have_nonmonotonic_node_counts=True)
        if not (node_increase or order_increase):
            result['reasons'].append('higher_polynomial_order_or_larger_spatial_discretization_not_established')
        if coarse.get('sha256') == fine.get('sha256'):
            result['reasons'].append('coarse_fine_report_reused')
        if compatible and all(x.get('raw') is not None and x.get('comparison_geometry_compatible') for x in references):
            try:
                result['paired_refinement'] = balanced_metrics(coarse['raw'], fine['raw'])
                result['maximum_refinement_change'] = max(values(result['paired_refinement']).values())
                if result['maximum_refinement_change'] > policy['refinement']:
                    result['reasons'].append('paired_spatial_resolution_change_exceeds_tolerance')
            except (ValueError, np.linalg.LinAlgError) as error:
                result['reasons'].append('refinement_comparison_failed: '+str(error))
        if 'paired_refinement' not in result:
            result['reasons'].append('paired_spatial_refinement_not_established')
        result['reference_resolved'] = not result['reasons']
    comparable = [item for item in references if item.get('raw') is not None and item.get('comparison_geometry_compatible')]
    if comparable:
        fine = comparable[-1]
        result['comparison_reference_path'] = fine['path']
        result['comparison_uses_requested_finest'] = len(references) == 2 and fine is references[-1]
        try:
            if isinstance(operators, (str, Path)):
                operators = read(operators)
            op_geometry = dict(aspect_ratio=operators['aspect_ratio'], c_m=operators['c_nm']*1e-9,
                               **{name: operators[name] for name in ('theta_i','theta_j','phi','h_over_c')})
            if operators.get('convention_id') != CONVENTION or not geometry_matches(geometry, op_geometry):
                raise ValueError('operator_export_geometry_or_convention_mismatch')
            if operators.get('independent_reference') is not False:
                raise ValueError('production_operator_must_not_claim_to_be_independent_reference')
            matrix17(operators.get('leading17'), 'leading17')
            for name, matrix_key in (('near', 'near_total17'), ('far', 'far_isolated17')):
                result['models'][name] = model_comparison(operators[matrix_key], fine['raw'], policy['model'], MODEL_NAMES[name])
            if 'far_scattering17' in operators:
                scattering = model_comparison(operators['far_scattering17'], fine['raw'], policy['model'], 'production_scattering_basis_full17')
                diagnostics = operators.get('far_scattering_diagnostics', {})
                scattering['source_diagnostics'] = diagnostics
                for name, limit in (('raw_reciprocity', policy['reciprocity']), ('linear_relative_residual', policy['linear'])):
                    value = diagnostics.get(name)
                    if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or value > limit:
                        scattering['passed'] = False
                        scattering['status'] = 'UNRESOLVED'
                        scattering['reasons'].append('scattering_'+name+'_missing_or_failed')
                result['models']['production_scattering'] = scattering
        except (OSError, KeyError, TypeError, ValueError) as error:
            result['reasons'].append('operator_comparison_failed: '+str(error))
            result['models'] = {name: dict(status='UNRESOLVED', passed=False) for name in ('near', 'far')}
    result['status'] = classify(result)
    return result


def consecutive_groups(rows, predicate):
    groups, current = [], []
    for row in rows:
        if predicate(row):
            current.append(row['gap_nm'])
        elif current:
            groups.append(current); current = []
    if current:
        groups.append(current)
    return groups


def matched_anchor_trials(assessed, raw_by_id, operators_by_id, policy=None, selected_gap_nm=None):
    """Fit one constant residual, test strictly smaller sampled gaps.

    Each anchor is a diagnostic candidate, never a held-out test of itself.
    Selecting a candidate from these data is exploratory, not a fresh independent
    validation. Two consecutive lower samples are required for sampled support.
    """
    policy = policy_values(policy)
    trials = []
    if selected_gap_nm is not None:
        if not any(math.isclose(p['gap_nm'], selected_gap_nm, rel_tol=1e-12, abs_tol=1e-12) for p in assessed):
            raise ValueError('Selected near anchor must be one of the frozen sampled gaps')
        for row in assessed:
            if row['gap_nm'] < selected_gap_nm:
                row['models']['selected_matched_near'] = dict(status='UNRESOLVED', passed=False,
                    anchor_gap_nm=selected_gap_nm, reasons=['selected_anchor_reference_or_operator_unresolved'])
                row['status'] = classify(row)
    orientations = sorted({(p['aspect_ratio'], p['orientation']) for p in assessed})
    for ar, orientation in orientations:
        rows = sorted((p for p in assessed if p['aspect_ratio'] == ar and p['orientation'] == orientation), key=lambda p:p['gap_nm'])
        for anchor in rows:
            if not anchor['reference_resolved'] or anchor['id'] not in raw_by_id or anchor['id'] not in operators_by_id:
                continue
            lower = [p for p in rows if p['gap_nm'] < anchor['gap_nm']]
            if not lower:
                continue
            residual = symmetric(raw_by_id[anchor['id']])-matrix17(operators_by_id[anchor['id']]['leading17'], 'anchor leading17')
            trial = dict(aspect_ratio=ar, orientation=orientation, anchor_id=anchor['id'],
                         anchor_gap_nm=anchor['gap_nm'], model=MODEL_NAMES['matched_near'],
                         anchor_excluded_from_validation=True, heldout=[],
                         physical_accuracy_qualified=False,
                         scope='Exploratory single-anchor constant-residual fit; tested only at other smaller gaps; omitted subleading terms are not supplied by this fit.')
            for row in lower:
                comparison = dict(id=row['id'], gap_nm=row['gap_nm'], status='UNRESOLVED', passed=False,
                                  reasons=['heldout_reference_or_operator_unresolved'])
                same_geometry = ('geometry' in row and 'geometry' in anchor and geometry_matches(
                    dict(row['geometry'], h_over_c=anchor['geometry']['h_over_c']), anchor['geometry']))
                if row['reference_resolved'] and same_geometry and row['id'] in raw_by_id and row['id'] in operators_by_id:
                    predicted = matrix17(operators_by_id[row['id']]['leading17'], 'heldout leading17')+residual
                    comparison.update(model_comparison(predicted, raw_by_id[row['id']], policy['model'], MODEL_NAMES['matched_near']))
                trial['heldout'].append(comparison)
            groups = consecutive_groups(trial['heldout'], lambda row:row['passed'])
            trial['consecutive_passing_gap_groups_nm'] = groups
            trial['sampled_overlap_supported'] = any(len(group) >= 2 for group in groups)
            trials.append(trial)
            if selected_gap_nm is not None and math.isclose(anchor['gap_nm'], selected_gap_nm, rel_tol=1e-12, abs_tol=1e-12):
                supported = {gap for group in groups if len(group) >= 2 for gap in group}
                for row, comparison in zip(lower, trial['heldout']):
                    selected = dict(comparison, anchor_id=anchor['id'], anchor_gap_nm=anchor['gap_nm'],
                                    requires_two_consecutive_heldout_samples=True)
                    if selected.get('passed') and row['gap_nm'] not in supported:
                        selected.update(passed=False, status='UNRESOLVED')
                        selected['reasons'] = selected.get('reasons', [])+['fewer_than_two_consecutive_heldout_passing_samples']
                    row['models']['selected_matched_near'] = selected
                    row['status'] = classify(row)
    return trials


def assess_points(points, expected_orientation_ids=None, expected_gaps_nm=None, policy=None):
    """Summarize contiguous *sample groups*, never continuous validity bands."""
    orientations = expected_orientation_ids or sorted({p['orientation'] for p in points})
    gaps = sorted(expected_gaps_nm or {p['gap_nm'] for p in points})
    for point in points:
        point['status'] = classify(point)
        point['near_stokes_overlap_sample_supported'] = False
    def near_pass(row):
        models = row.get('models', {})
        return row.get('reference_resolved') and models.get('selected_matched_near', models.get('near', {})).get('passed', False)
    summaries = []
    for ar in sorted({p['aspect_ratio'] for p in points}):
        by_pose = {}
        for pose in orientations:
            found = {p['gap_nm']:p for p in points if p['aspect_ratio'] == ar and p['orientation'] == pose}
            rows = [found.get(gap, dict(gap_nm=gap, status='UNRESOLVED', reference_resolved=False)) for gap in gaps]
            overlap_groups = [group for group in consecutive_groups(rows, near_pass) if len(group) >= 2]
            overlap_gaps = {gap for group in overlap_groups for gap in group}
            for row in rows:
                row['near_stokes_overlap_sample_supported'] = row['gap_nm'] in overlap_gaps
                if row['status'] == 'NEAR' and row['gap_nm'] in overlap_gaps:
                    row['status'] = 'OVERLAP'
            groups = {name:consecutive_groups(rows, lambda row:row['status']==name) for name in ('NEAR','OVERLAP','INTERMEDIATE','FAR','UNRESOLVED')}
            # Conservative outer-end coverage; one isolated pass is no cutoff.
            far_suffix = []
            for row in reversed(rows):
                if row['status'] == 'FAR':
                    far_suffix.append(row['gap_nm'])
                else:
                    break
            by_pose[pose] = dict(sample_groups_nm=groups,
                near_candidate_agreement_samples_nm=[row['gap_nm'] for row in rows if near_pass(row)],
                near_stokes_overlap_sample_groups_nm=overlap_groups,
                near_stokes_overlap_is_continuous_certified_interval=False,
                far_outer_sample_suffix_nm=list(reversed(far_suffix)),
                far_onset_candidate_nm=min(far_suffix) if len(far_suffix)>=2 else None,
                far_onset_is_continuous_cutoff=False)
        common = []
        for gap in gaps:
            rows = [p for p in points if p['aspect_ratio']==ar and p['gap_nm']==gap]
            complete = len(rows)==len(orientations) and {p['orientation'] for p in rows}==set(orientations)
            common.append(dict(gap_nm=gap, all_requested_orientations_present=complete,
                all_references_resolved=complete and all(p['reference_resolved'] for p in rows),
                all_near_pass=complete and all(near_pass(p) for p in rows),
                all_near_stokes_overlap_supported=complete and all(p['near_stokes_overlap_sample_supported'] for p in rows),
                all_far_pass=complete and all(p['status']=='FAR' for p in rows),
                all_orientations_in_continuum_qualified=False))
        summaries.append(dict(aspect_ratio=ar, orientations=by_pose, common_sample_coverage=common))
    return dict(schema=SCHEMA, policy=policy_values(policy), physical_accuracy_qualified=False,
        model_error_is_against_numerical_reference=True,
        model_tolerance_plus_refinement_tolerance_is_not_a_rigorous_error_bound=True,
        scope=SCOPE, counts={name:sum(p['status']==name for p in points) for name in ('NEAR','OVERLAP','INTERMEDIATE','FAR','UNRESOLVED')},
        reference_resolved_count=sum(p['reference_resolved'] for p in points), point_count=len(points),
        status_meaning=dict(NEAR='Single sampled near-candidate agreement with resolved independent Stokes; a multi-gap overlap is not yet established.',
            OVERLAP='At least two consecutive sampled gaps have near-candidate agreement with resolved independent Stokes; not an unsampled-interval guarantee.',
            INTERMEDIATE='Resolved independent Stokes reference; both available asymptotic candidates fail.',
            FAR='Independent-body Stokes approximation agrees; this label takes priority if the near candidate also passes.',
            UNRESOLVED='Reference or model evidence incomplete; no regime conclusion.'),
        aspects=summaries, points=points)


def write_readable_report(path, result):
    """Make decisions and missing evidence visible without opening raw JSON."""
    def gaps(values):
        return ', '.join(format(value, '.8g') for value in values) or '—'
    def groups(values):
        return '; '.join('['+gaps(group)+']' for group in values) or '—'
    policy = result['policy']
    lines = ['# 거리별 Stokes 영역 판정', '',
        f"기준해 수렴·진단 통과: {result['reference_resolved_count']}/{result['point_count']}개 표본.", '',
        f"모델 비교 허용치 {100*policy['model']:.4g}%, 두 공간해상도 간 변화 허용치 {100*policy['refinement']:.4g}%, "
        f"경계 진단 허용치 {100*policy['boundary']:.4g}%. 힘·토크·stresslet과 에너지 정규화 전체 응답을 각각 검사한다.", '',
        '모든 거리는 실제 최소 표면 간격 h이며 단위는 nm다. 아래 숫자는 계산한 표본이며, 그 사이 모든 거리나 미계산 방향의 정확도를 보증하지 않는다. '
        '두 해상도의 차이는 엄밀한 오차상한이 아니다. 근접해의 단일 표본 일치와 다중 거리의 근접해–독립 Stokes 중첩을 구별한다.', '',
        '| AR | 방향 | 근접 후보 일치 표본 | 근접–Stokes 중첩 표본군 (2개 이상) | 중간영역 표본군 | 원거리 시작 후보 | 미판정 표본군 |',
        '|---|---|---|---|---|---|---|']
    for aspect in result['aspects']:
        for name, pose in aspect['orientations'].items():
            onset = pose['far_onset_candidate_nm']
            lines.append('| '+ ' | '.join((format(aspect['aspect_ratio'],'.8g'), name,
                gaps(pose['near_candidate_agreement_samples_nm']), groups(pose['near_stokes_overlap_sample_groups_nm']),
                groups(pose['sample_groups_nm']['INTERMEDIATE']), '미확정' if onset is None else format(onset,'.8g'),
                groups(pose['sample_groups_nm']['UNRESOLVED'])))+' |')
    lines += ['', '원거리 시작 후보는 계산범위의 가장 바깥쪽까지 연속하여 최소 두 표본에서 독립 단일입자 Stokes 합이 통과한 경우만 표시한다. '
              '연속적인 물리적 cutoff로 확정한 값은 아니다. 별도의 production scattering 모델 통과 여부는 JSON의 production_scattering 항목을 확인한다.', '',
              '기본 근접 후보는 정확한 단일입자 저항 + 기존 Janoschek leading 항이다. 생략된 유한항·subleading 항이 있으므로 근접 후보 실패만으로 실제 근접 점근해의 부재를 단정하지 않는다.', '',
              '## 상수 잔차를 맞춘 근접 후보', '',
              '각 anchor에서 상수 full17 잔차를 한 번 맞추고 더 작은 다른 거리에서 비교했다. anchor 자체는 검증 표본에서 제외했다. '
              '아래 후보를 같은 데이터로 선택하는 것은 탐색적 선택이며, 새 독립 검증을 대신하지 않는다. 행렬의 음의 고유값을 잘라내지 않는다.', '',
              '| AR | 방향 | anchor nm | 통과한 중첩 표본군 (2개 이상) |', '|---|---|---|---|']
    supported = [trial for trial in result.get('matched_near_anchor_trials', []) if trial['sampled_overlap_supported']]
    for trial in supported:
        lines.append('| '+ ' | '.join((format(trial['aspect_ratio'],'.8g'), trial['orientation'],
            format(trial['anchor_gap_nm'],'.8g'), groups([g for g in trial['consecutive_passing_gap_groups_nm'] if len(g)>=2])))+' |')
    if not supported:
        lines += ['| — | 아직 통과한 다중 거리 anchor 후보 없음 | — | — |']
    lines += ['', '미판정 원인과 각 힘·토크·stresslet 오차는 같은 이름의 CSV/JSON에 기록되어 있다. '
              '기준해 자체가 미수렴하거나 isolated 검증이 실패한 경우 모델 실패로 판정하지 않는다.', '']
    Path(path).write_text('\n'.join(lines))


def export_cached(point, executable, folder, scattering_model=None):
    key = dict(exporter_sha256=digest(executable), geometry=query_geometry(point),
               c_nm=point['c_nm'], scattering_sha256=digest(scattering_model) if scattering_model else None)
    path = folder/(point['id']+'.operators.json')
    if path.is_file():
        cached = read(path)
        if cached.get('_cache_identity') == key:
            return cached
    args = [str(executable), '--gap-over-c', str(point['gap_nm']/point['c_nm']),
            '--aspect-ratio', str(point['aspect_ratio']), '--c-nm', str(point['c_nm'])]
    for name in ('theta_i','theta_j','phi'):
        args.extend(['--'+name.replace('_','-'), str(point[name])])
    if scattering_model:
        args.extend(['--scattering-model', str(scattering_model)])
    process = subprocess.run(args, capture_output=True, text=True, timeout=300)
    if process.returncode:
        raise ValueError('operator exporter failed: '+process.stderr[-2000:])
    result = json.loads(process.stdout)
    result['_cache_identity'] = key
    save(path, result)
    return result


def collect_and_assess(campaign_path, operator_export, output, near_anchor_gap_nm=None, scattering_model=None):
    campaign_path, output = Path(campaign_path).resolve(), Path(output).resolve()
    campaign = read(campaign_path)
    root = campaign_path.parent
    tasks = {task['task_id']:task for task in campaign['tasks']}
    settings = campaign.get('settings', {})
    policy = dict(DEFAULTS)
    for source, target in (('model_tolerance','model'), ('refinement_tolerance','refinement'), ('development_boundary_limit','boundary')):
        if source in settings:
            policy[target] = settings[source]
    policy.update(settings.get('assessment_policy', {}))
    policy = policy_values(policy)
    points, raw_by_id, operators_by_id = [], {}, {}
    for point in campaign['points']:
        records = [task_report(root, tasks[task_id], campaign_path) for task_id in point['reference_task_ids']]
        reports = [Path(record['path']) for record in records if record['state']=='available']
        operators, failure = {}, None
        if reports:
            try:
                operators = export_cached(point, Path(operator_export), root/'operators', scattering_model)
                operators_by_id[point['id']] = operators
            except (ValueError, OSError, subprocess.TimeoutExpired) as error:
                failure = str(error)
        assessed = assess_point(point, reports, operators, policy)
        if any(assessed['models'].get(name, {}).get('status') not in ('PASS','FAIL') for name in ('near','far')):
            operators_by_id.pop(point['id'], None)
        assessed['worker_records'] = records
        if len(reports)!=2:
            assessed['reasons'].append('coarse_or_fine_worker_report_unavailable')
        if failure:
            assessed['reasons'].append(failure)
        if assessed['reference_resolved']:
            raw_by_id[point['id']] = matrix17(read(reports[-1])['raw_full17'], 'fine raw_full17')
        points.append(assessed)
    trials = matched_anchor_trials(points, raw_by_id, operators_by_id, policy, near_anchor_gap_nm)
    result = assess_points(points, policy=policy)
    result.update(campaign_path=str(campaign_path), campaign_sha256=digest(campaign_path),
                  selected_near_anchor_gap_nm=near_anchor_gap_nm, matched_near_anchor_trials=trials,
                  production_far_scattering_model=str(scattering_model) if scattering_model else None,
                  production_far_scattering_qualified_by_isolated_far_test=False)
    save(output, result)
    write_readable_report(output.with_suffix('.md'), result)
    csv_path = output.with_suffix('.csv')
    with csv_path.open('w', newline='') as stream:
        model_columns = ('near','far','selected_matched_near')
        metric_columns = ('status','error','energy_error','force_error','torque_error','stresslet_error')
        names = ['id','AR','orientation','gap_nm','status','reference_resolved','near_stokes_overlap_sample_supported',
                 'refinement_error','coarse_reference','fine_reference']
        names += [model+'_'+metric for model in model_columns for metric in metric_columns]+['reasons']
        writer = csv.DictWriter(stream, fieldnames=names); writer.writeheader()
        for point in points:
            row = dict(id=point['id'], AR=point['aspect_ratio'], orientation=point['orientation'],
                gap_nm=point['gap_nm'], status=point['status'], reference_resolved=point['reference_resolved'],
                near_stokes_overlap_sample_supported=point['near_stokes_overlap_sample_supported'],
                refinement_error=point.get('maximum_refinement_change'),
                coarse_reference=point['references'][0]['path'] if point['references'] else None,
                fine_reference=point['references'][1]['path'] if len(point['references'])>1 else None,
                reasons='; '.join(point['reasons']))
            for model in model_columns:
                comparison = point['models'].get(model, {})
                metrics = comparison.get('screening_values', {})
                row[model+'_status'] = comparison.get('status')
                row[model+'_error'] = comparison.get('maximum_screened_error')
                for metric in ('energy','force','torque','stresslet'):
                    row[model+'_'+metric+'_error'] = metrics.get(metric)
            writer.writerow(row)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', required=True, type=Path)
    parser.add_argument('--operator-export', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--near-anchor-gap-nm', type=float)
    parser.add_argument('--scattering-model', type=Path)
    args = parser.parse_args(argv)
    if args.near_anchor_gap_nm is not None and (not math.isfinite(args.near_anchor_gap_nm) or args.near_anchor_gap_nm <= 0):
        parser.error('--near-anchor-gap-nm must be finite and positive')
    result = collect_and_assess(args.campaign, args.operator_export, args.output, args.near_anchor_gap_nm, args.scattering_model)
    print(json.dumps(dict(output=str(args.output), counts=result['counts'], reference_resolved_count=result['reference_resolved_count'],
                          point_count=result['point_count'], physical_accuracy_qualified=False)), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
