#!/usr/bin/env python3
"""Distance-first, restartable independent Stokes scans for one aspect ratio.

Each plan has a frozen number of pair-resolution slots. Completion counts are
work progress, never elapsed-time forecasts or physical qualification. Failed
slots count as processed and remain explicitly unresolved. Each slot may use
up to the manifest's bounded number of mesh-refinement attempts.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import run_campaign_reference as worker
from reference_table_campaign import task_report

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PLAN_KIND = 'distance_regime_scan_v1'
RUN_SCHEMA = 'distance_regime_scan_run_v1'
ORIENTATIONS = (
    ('face_face', 0., 0., 0.),
    ('face_rim', 0., math.pi/2, 0.),
    ('rim_cross', math.pi/2, math.pi/2, math.pi/2),
    ('tilted', math.pi/5, 2*math.pi/5, math.pi/3),
)


def utc():
    return datetime.now(timezone.utc).isoformat()


def emit(event, **values):
    print(json.dumps(dict(event=event, utc=utc(), **values), allow_nan=False), flush=True)


def finite_positive(value, name):
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(name+' must be finite and positive')
    return float(value)


def make_plan(args):
    ar = finite_positive(args.ar, 'ar')
    if ar not in (2., 5., 10., 20.):
        raise ValueError('Supported scan aspect ratios are 2, 5, 10 and 20')
    c = finite_positive(args.c_nm, 'c_nm')
    memory = finite_positive(args.max_memory_gb, 'max_memory_gb')
    if memory > 58:
        raise ValueError('Keep the explicit solver memory budget <=58 decimal GB inside the 64 GiB Slurm allocation')
    if args.coarse_order not in (4, 6, 8, 10) or args.fine_order != args.coarse_order+2:
        raise ValueError('Require coarse-order in (4,6,8,10) and fine-order = coarse-order + 2')
    if min(args.surface_level, args.rim_levels, args.contact_levels, args.max_refinements) < 0 or args.max_refinements > 4:
        raise ValueError('Mesh levels must be nonnegative and max-refinements <=4')
    for value, name in ((args.boundary_limit, 'boundary-limit'),
                        (args.model_tolerance, 'model-tolerance'),
                        (args.refinement_tolerance, 'refinement-tolerance')):
        if not 0 < finite_positive(value, name) < 1:
            raise ValueError(name+' must be smaller than 1')
    if args.gaps_nm:
        gaps = [finite_positive(float(x), 'gap_nm') for x in args.gaps_nm.split(',')]
    else:
        gaps = [1., 3., 10., 30., 100., 150., 300.] + [ar*c*x for x in (1., 3., 10., 30., 100.)]
    gaps = sorted(set(gaps))
    if len(gaps) < 2:
        raise ValueError('A distance scan requires at least two distinct gaps')
    # An anchor-sized point, far field, then near points before filling the grid.
    # These are execution priorities, not fitted boundaries or physical cutoffs.
    priority = []
    for h in (150., gaps[-1], 30., gaps[0]):
        if h in gaps and h not in priority:
            priority.append(h)
    remaining = [h for h in reversed(gaps) if h not in priority]
    while remaining:
        priority.append(remaining.pop(0))
        if remaining:
            priority.append(remaining.pop())
    output = args.output_dir.expanduser().resolve()
    defaults = dict(solver='ram-dense', max_memory_gb=memory, quad_tol=1e-8,
                    check_points_per_patch=3, alpha=1., beta=1., max_refinements=args.max_refinements,
                    adapt_fraction=.2, geometry_refine_tolerance=.0005, geometry_max_refinements=3,
                    isolated_max_refinements=2, isolated_require_pass=True)
    settings = dict(aspect_ratio=ar, c_nm=c, semimajor_axis_nm=ar*c,
                    gaps_nm=gaps, gap_execution_order_nm=priority,
                    gap_definition='actual minimum surface separation; no replacement by center distance',
                    regularization_floor_nm=1.,
                    sampling_policy='prespecified pilot distances, not physical regime boundaries',
                    orientations=[dict(id=name, theta_i=ti, theta_j=tj, phi=phi)
                                  for name, ti, tj, phi in ORIENTATIONS],
                    coarse_order=args.coarse_order, fine_order=args.fine_order,
                    order_policy='Select p>=coarse_order by sampled exact geometry, then p+2; fixed before solving',
                    geometry_tolerance_over_gap=.0005,
                    surface_level=args.surface_level, rim_levels=args.rim_levels,
                    contact_levels=args.contact_levels, isolated_order=args.fine_order,
                    development_boundary_limit=args.boundary_limit,
                    model_tolerance=args.model_tolerance, refinement_tolerance=args.refinement_tolerance,
                    defaults=defaults,
                    scope='Four sampled orientations only; unsampled angles and gaps are not certified')
    path = output/'campaign.json'
    if path.exists():
        plan = read_plan(path)
        if (plan.get('settings') != settings or plan.get('defaults') != defaults
                or plan.get('development_boundary_limit') != args.boundary_limit):
            raise ValueError('Existing campaign has different settings; use a new output directory')
        return path, plan
    if output.exists() and any(output.iterdir()):
        raise ValueError('New distance plan requires an empty output directory')
    from geometry_preflight import choose_orders
    base_resolution = dict(worker.PAIR_RESOLUTION, level=args.surface_level,
                           rim_levels=args.rim_levels, contact_levels=args.contact_levels)
    tasks, points = [], []
    for gap in priority:
        for orientation, ti, tj, phi in ORIENTATIONS:
            point_id = f'AR{ar:g}_{orientation}_h{gap:.12g}'
            point = dict(id=point_id, orientation=orientation, aspect_ratio=ar, c_nm=c,
                         gap_nm=gap, theta_i=ti, theta_j=tj, phi=phi, reference_task_ids=[])
            physical = {key:point[key] for key in ('aspect_ratio','c_nm','gap_nm','theta_i','theta_j','phi')}
            selection = choose_orders(physical, base_resolution, tolerance=.0005,
                coarse_candidates=tuple(range(args.coarse_order,11,2)), max_memory_gb=memory)
            point['geometry_order_selection'] = selection
            for label, order in (('coarse', selection['coarse_order']), ('fine', selection['fine_order'])):
                task_id = len(tasks)
                point['reference_task_ids'].append(task_id)
                tasks.append(dict(task_id=task_id, aspect_ratio=ar, c_nm=c, theta_i=ti, theta_j=tj,
                    phi=phi, gap_nm=gap, orientation=orientation, role='validation', case_id=point_id,
                    query_id=point_id, resolution_label=label, output_subdir=f'tasks/task_{task_id:06d}',
                    max_refinements=args.max_refinements,
                    resolution=dict(order=order, level=args.surface_level, rim_levels=args.rim_levels,
                                    contact_levels=args.contact_levels, resolution_level=100*order+args.surface_level),
                    isolated_resolution=dict(order=args.fine_order, level=args.surface_level,
                        rim_levels=args.rim_levels, contact_levels=0,
                        resolution_level=100*args.fine_order+args.surface_level)))
            points.append(point)
    identity = dict(settings=settings, defaults=defaults, tasks=tasks, points=points,
                    development_boundary_limit=args.boundary_limit)
    plan = dict(schema=worker.SCHEMA, campaign_kind=PLAN_KIND, created_utc=utc(), output_dir=str(output),
                source_root=str(ROOT), configuration_sha256=worker.canonical_digest(identity),
                task_count=len(tasks), point_count=len(points), **identity,
                physical_accuracy_qualified=False,
                progress_definition='processed pair-resolution slots / frozen planned slots; not elapsed-time fraction',
                failure_policy='Preserve failures and continue; retry only with --retry-failed; interrupted work resumes automatically')
    worker.save(path, plan)
    return path, plan


def read_plan(path):
    plan = worker.read_json(path)
    worker.require(plan.get('schema') == worker.SCHEMA and plan.get('campaign_kind') == PLAN_KIND,
                   'Not a distance-scan campaign')
    bound = {key: plan[key] for key in ('settings', 'defaults', 'tasks', 'points', 'development_boundary_limit')}
    worker.require(plan.get('configuration_sha256') == worker.canonical_digest(bound), 'Campaign configuration digest mismatch')
    worker.require(plan.get('task_count') == len(plan['tasks']), 'Campaign task count mismatch')
    worker.require([t['task_id'] for t in plan['tasks']] == list(range(plan['task_count'])), 'Campaign task IDs must be consecutive')
    for point in plan['points']:
        worker.require(len(point['reference_task_ids']) == 2, 'Every point requires two resolution slots')
        for index in point['reference_task_ids']:
            task = plan['tasks'][index]
            worker.require(task['query_id'] == point['id'], 'Point/task identity mismatch')
    return plan


def verify_build(path, expected_sha=None):
    path = Path(path).expanduser().resolve(strict=True)
    actual = worker.digest(path)
    worker.require(expected_sha is None or actual == expected_sha.lower(), 'Build manifest changed since submission')
    manifest = worker.read_json(path)
    worker.require(manifest.get('schema') in ('reference_table_build_v1', 'distance_scan_build_v1')
                   and manifest.get('status') == 'ready', 'Build manifest is not ready')
    worker.require(Path(manifest['source_root']).resolve() == ROOT, 'Build belongs to a different source checkout')
    paths = {}
    for key in ('python', 'reference_library', 'backend_manifest', 'operator_export'):
        selected = Path(manifest['paths'][key]).expanduser().resolve(strict=True)
        record = manifest['artifacts'][key]
        worker.require(selected.is_file() and Path(record['path']).resolve() == selected
                       and worker.digest(selected) == record['sha256'], 'Build artifact changed: '+key)
        paths[key] = str(selected)
    sources = manifest.get('source_sha256', {})
    mandatory = ['distance_scan.py', 'run_campaign_reference.py', 'run_reference.py', 'distance_regime_assessment.py',
                 'ram_dense.py', 'geometry_preflight.py', 'contact_check_sites.py']
    for name in mandatory:
        worker.require('validation_basis/independent_stokes/'+name in sources,
                       'Build manifest lacks new scan source: '+name+'; run build_distance_scan_cpu.sbatch')
    for name, expected in sources.items():
        selected = (ROOT/name).resolve()
        worker.require(ROOT in selected.parents and selected.is_file() and worker.digest(selected) == expected,
                       'Source changed since build: '+name+'; rebuild in a new source snapshot')
    backend = worker.read_json(paths['backend_manifest'])
    worker.require(backend.get('library_sha256') == worker.digest(paths['reference_library'])
                   and backend.get('patch_policy') == 'get_quadparams_adap_high_order_p_ge_4_v1',
                   'Patched backend identity/policy mismatch')
    runtime = manifest.get('blas_configuration', {}).get('runtime_environment', {})
    worker.require(isinstance(runtime, dict) and all(isinstance(k, str) and isinstance(v, str)
                   for k, v in runtime.items()), 'Invalid BLAS runtime environment')
    # The build's verified ABI policy must survive into every child process.
    os.environ.update(runtime)
    return dict(path=str(path), sha256=actual, paths=paths)


def verify_finished_task(root, task, campaign_path):
    """Validate result, solution and archived-source bytes before skipping work."""
    status = task_report(root, task, campaign_path)
    if status['state'] != 'available':
        return status
    state = worker.read_json(root/task['output_subdir']/'task.json')
    identity = state['identity']
    config = identity['configuration']
    worker.unchanged(config)
    data, selected = worker.validate_report(Path(status['path']), config, identity['physical'],
                                           identity['pair_resolution'], None)
    worker.require(selected['sha256'] == status['sha256'], 'Selected pair report changed')
    for isolated in state.get('isolated_reports', []):
        worker.require(worker.digest(isolated['path']) == isolated['sha256'], 'Isolated report changed')
        worker.require(worker.digest(isolated['solution_path']) == isolated['solution_sha256'], 'Isolated checkpoint changed')
    return status


def summarize(plan, root, state=None):
    state = state or {}
    records = state.get('tasks', {})
    counts = dict(planned=plan['task_count'], processed=0, pair_completed=0, development_accepted=0,
                  failed=0, interrupted=0, pending=0, reference_points_planned=plan['point_count'],
                  reference_points_resolved=0, reference_points_resolved_percent=0.)
    gap_summary = []
    for gap in plan['settings']['gap_execution_order_nm']:
        local = dict(gap_nm=gap, planned=0, processed=0, pair_completed=0, failed=0, last_task_updated_utc='')
        for task in plan['tasks']:
            if task['gap_nm'] != gap:
                continue
            local['planned'] += 1
            record = records.get(str(task['task_id']), {})
            status = record.get('status', 'pending')
            local['last_task_updated_utc'] = max(local['last_task_updated_utc'], record.get('updated_utc', ''))
            if status in ('complete', 'completed_unaccepted', 'failed'):
                local['processed'] += 1
                counts['processed'] += 1
            if status in ('complete', 'completed_unaccepted'):
                local['pair_completed'] += 1
                counts['pair_completed'] += 1
                counts['development_accepted'] += status == 'complete'
            elif status == 'failed':
                counts['failed'] += 1
                local['failed'] += 1
            elif status == 'interrupted':
                counts['interrupted'] += 1
            else:
                counts['pending'] += 1
        local['finished'] = local['processed'] == local['planned']
        gap_summary.append(local)
    counts['processed_percent'] = 100*counts['processed']/counts['planned']
    counts['pair_completed_percent'] = 100*counts['pair_completed']/counts['planned']
    result = dict(schema='distance_regime_scan_progress_v1', updated_utc=utc(),
                  aspect_ratio=plan['settings']['aspect_ratio'], status=state.get('status', 'planned'),
                  current_task=state.get('current_task'), counts=counts, gaps=gap_summary,
                  progress_definition=plan['progress_definition'],
                  physical_accuracy_qualified=False)
    assessment = Path(root)/'regimes.json'
    if assessment.is_file():
        try:
            report = worker.read_json(assessment)
            worker.require(report.get('campaign_sha256') == worker.digest(Path(root)/'campaign.json'),
                           'Regime report belongs to another campaign')
            resolved = report.get('reference_resolved_count', 0)
            counts['reference_points_resolved'] = resolved
            counts['reference_points_resolved_percent'] = 100*resolved/plan['point_count']
            result['assessment'] = dict(path=str(assessment),
                reference_resolved_count=resolved, point_count=report.get('point_count'),
                counts=report.get('counts', {}))
            for gap in gap_summary:
                rows = [p for p in report.get('points', []) if p.get('gap_nm') == gap['gap_nm']]
                gap['reference_points_resolved'] = sum(p.get('reference_resolved') is True for p in rows)
                gap['reference_points_planned'] = len(plan['settings']['orientations'])
                gap['orientation_status'] = {p['orientation']:p.get('status', 'UNRESOLVED') for p in rows}
        except (ValueError, OSError):
            result['assessment_error'] = 'Cannot read regimes.json'
    if state.get('assessment_error'):
        result['assessment_error'] = state['assessment_error']
    return result


def slurm_comment(progress):
    job = os.environ.get('SLURM_JOB_ID')
    if not job or not job.isdigit():
        return
    c = progress['counts']
    current = progress.get('current_task') or {}
    current_h = format(current['gap_nm'], '.8g')+'nm' if 'gap_nm' in current else '-'
    finished = [gap for gap in progress['gaps'] if gap['finished']]
    latest = max(finished, key=lambda gap:gap.get('last_task_updated_utc', '')) if finished else None
    last_h = format(latest['gap_nm'], '.8g')+'nm' if latest else '-'
    comment = (f"AR{progress['aspect_ratio']:g} {c['processed']}/{c['planned']} {c['processed_percent']:.1f}% "
               f"pair={c['pair_completed']} failed={c['failed']} refs={c['reference_points_resolved']}/{c['reference_points_planned']} "
               f"h={current_h} last_h={last_h}")
    try:
        subprocess.run(['scontrol', 'update', 'JobId='+job, 'Comment='+comment],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=8, check=False)
    except (OSError, subprocess.TimeoutExpired):
        pass  # Some clusters deny user comments; JSON/Slurm stdout remain authoritative.


def checkpoint(plan, root, state, comment=True):
    state['updated_utc'] = utc()
    worker.save(root/'run_state.json', state)
    progress = summarize(plan, root, state)
    worker.save(root/'progress.json', progress)
    if comment:
        slurm_comment(progress)
    return progress


class StopControl:
    def __init__(self, deadline):
        self.deadline = deadline
        self.stop_requested = False
        self.immediate = False
        self.reason = None
        self.previous = {}

    def handle(self, signum, frame):
        self.stop_requested = True
        self.reason = signal.Signals(signum).name
        # USR1 is Slurm's advance warning: finish the stage if possible, while
        # reserving a minute for a controlled stop before the allocation ends.
        if signum == signal.SIGUSR1:
            self.deadline = min(self.deadline, time.monotonic()+540)
        else:
            self.immediate = True

    def __enter__(self):
        for sig in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
            self.previous[sig] = signal.signal(sig, self.handle)
        return self

    def __exit__(self, *unused):
        for sig, previous in self.previous.items():
            signal.signal(sig, previous)


def execute_child(command, stdout_path, stderr_path, control):
    """Own a process group so a Slurm stop also stops its numerical children."""
    with Path(stdout_path).open('a') as out, Path(stderr_path).open('a') as err:
        process = subprocess.Popen(command, stdout=out, stderr=err, start_new_session=True)
        try:
            while True:
                code = process.poll()
                if code is not None:
                    interrupted = control.immediate or code in (-signal.SIGTERM, -signal.SIGINT, -signal.SIGUSR1)
                    if interrupted:
                        control.stop_requested = True
                        control.reason = control.reason or 'worker_signal_'+str(-code)
                    return code, interrupted
                if control.immediate or time.monotonic() >= control.deadline:
                    control.stop_requested = True
                    control.reason = control.reason or 'wall_budget'
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        process.wait()
                    return process.returncode, True
                time.sleep(.2)
        except BaseException:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()
            raise


def assess(campaign_path, build, root, state):
    command = [build['paths']['python'], '-B', str(HERE/'distance_regime_assessment.py'),
               '--campaign', str(campaign_path), '--operator-export', build['paths']['operator_export'],
               '--output', str(root/'regimes.json')]
    try:
        with (root/'assessment.stdout.log').open('a') as out, (root/'assessment.stderr.log').open('a') as err:
            completed = subprocess.run(command, stdout=out, stderr=err, timeout=180, check=False)
        if completed.returncode:
            raise ValueError('Assessment exited '+str(completed.returncode)+'; see assessment.stderr.log')
        state.pop('assessment_error', None)
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        state['assessment_error'] = str(exc)
        emit('assessment_failed', error=str(exc))


def run(args):
    campaign_path = args.campaign.expanduser().resolve(strict=True)
    plan = read_plan(campaign_path)
    campaign_hash = worker.digest(campaign_path)
    worker.require(args.expected_campaign_sha256 is None or campaign_hash == args.expected_campaign_sha256.lower(),
                   'Campaign manifest changed since submission')
    build = verify_build(args.build_manifest, args.expected_build_sha256)
    root = Path(plan['output_dir']).resolve()
    worker.require(campaign_path == root/'campaign.json', 'Campaign was moved; use its original output directory')
    finite_positive(args.wall_hours, 'wall-hours')
    run_identity = dict(campaign_sha256=campaign_hash, build_manifest_sha256=build['sha256'],
                        build_manifest=build['path'])
    with worker.locked(root/'scan.lock', blocking=False):
        state_path = root/'run_state.json'
        state = worker.read_json(state_path) if state_path.is_file() else dict(schema=RUN_SCHEMA, identity=run_identity,
                    created_utc=utc(), tasks={}, attempts=[], physical_accuracy_qualified=False)
        worker.require(state.get('schema') == RUN_SCHEMA and state.get('identity') == run_identity,
                       'Run source/build/campaign identity changed; use a new output directory')
        state.update(status='running', current_task=None, slurm_job_id=os.environ.get('SLURM_JOB_ID'))
        isolated_retry_token = utc()+'-pid'+str(os.getpid()) if args.retry_failed else None
        state.setdefault('run_invocations', []).append(dict(started_utc=utc(), retry_failed=args.retry_failed,
            isolated_retry_token=isolated_retry_token, slurm_job_id=os.environ.get('SLURM_JOB_ID')))
        deadline = time.monotonic()+args.wall_hours*3600
        previous_gap = None
        checkpoint(plan, root, state)
        with StopControl(deadline) as control:
            for task in plan['tasks']:
                task_id = task['task_id']; key = str(task_id)
                old = state['tasks'].get(key, {})
                finished = verify_finished_task(root, task, campaign_path)
                if finished['state'] == 'invalid':
                    raise ValueError('Refusing invalid saved result: '+finished.get('error', str(task_id)))
                if finished['state'] == 'available' and finished.get('worker_status') in ('complete', 'completed_unaccepted'):
                    state['tasks'][key] = dict(old, status='complete' if finished['accepted'] else 'completed_unaccepted',
                                              report=finished['path'], report_sha256=finished['sha256'])
                    continue
                if old.get('status') == 'failed' and not args.retry_failed:
                    continue
                if control.stop_requested or time.monotonic() >= control.deadline:
                    control.stop_requested = True
                    control.reason = control.reason or 'wall_budget'
                    break
                if previous_gap is not None and previous_gap != task['gap_nm']:
                    assess(campaign_path, build, root, state)
                    progress = checkpoint(plan, root, state)
                    gap = next(g for g in progress['gaps'] if g['gap_nm'] == previous_gap)
                    emit('distance_finished', aspect_ratio=plan['settings']['aspect_ratio'], **gap,
                         overall=progress['counts'])
                previous_gap = task['gap_nm']
                state['current_task'] = dict(task_id=task_id, gap_nm=task['gap_nm'], orientation=task['orientation'],
                                              resolution=task['resolution_label'])
                directory = root/task['output_subdir']; directory.mkdir(parents=True, exist_ok=True)
                attempt = dict(task_id=task_id, started_utc=utc(), status='running', slurm_job_id=os.environ.get('SLURM_JOB_ID'))
                state['attempts'].append(attempt)
                checkpoint(plan, root, state)
                emit('distance_task_started', aspect_ratio=plan['settings']['aspect_ratio'], **state['current_task'])
                command = [build['paths']['python'], '-B', str(HERE/'run_campaign_reference.py'),
                           '--campaign', str(campaign_path), '--task-id', str(task_id),
                           '--reference-library', build['paths']['reference_library'],
                           '--operator-export', build['paths']['operator_export'],
                           '--backend-manifest', build['paths']['backend_manifest']]
                if isolated_retry_token is not None:
                    command += ['--retry-failed-isolated', '--isolated-retry-token', isolated_retry_token]
                started = time.monotonic()
                code, interrupted = execute_child(command, directory/'worker.stdout.log', directory/'worker.stderr.log', control)
                attempt.update(finished_utc=utc(), elapsed_seconds=time.monotonic()-started, returncode=code)
                finished = verify_finished_task(root, task, campaign_path)
                if finished['state'] == 'available' and finished.get('worker_status') in ('complete', 'completed_unaccepted'):
                    record = dict(status='complete' if finished['accepted'] else 'completed_unaccepted',
                                  report=finished['path'], report_sha256=finished['sha256'])
                elif interrupted:
                    record = dict(status='interrupted', reason=control.reason)
                else:
                    record = dict(status='failed', returncode=code, failure=finished)
                    task_state = directory/'task.json'
                    if task_state.exists():
                        record['worker_failure'] = worker.read_json(task_state).get('failure')
                record['updated_utc'] = utc()
                attempt['status'] = record['status']; state['tasks'][key] = record
                state['current_task'] = None
                progress = checkpoint(plan, root, state)
                emit('distance_task_finished', task_id=task_id, gap_nm=task['gap_nm'], status=record['status'],
                     overall=progress['counts'])
                if interrupted:
                    break
            if control.stop_requested:
                state.update(status='paused', stop_reason=control.reason, current_task=None)
            else:
                state.update(status='finished', current_task=None)
            assess(campaign_path, build, root, state)
            progress = checkpoint(plan, root, state)
            for gap in progress['gaps']:
                if gap['finished']:
                    emit('distance_finished', aspect_ratio=plan['settings']['aspect_ratio'], **gap,
                         overall=progress['counts'])
            emit('distance_scan_finished', **{k:progress[k] for k in ('status', 'counts', 'assessment_error') if k in progress})
            # A wall-budget pause is restartable and intentionally not a success.
            if state['status'] == 'paused':
                return 75
            return 2 if progress['counts']['failed'] or state.get('assessment_error') else 0


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='mode', required=True)
    plan = sub.add_parser('plan', help='Write or verify an immutable plan without solving')
    plan.add_argument('--ar', type=float, required=True, choices=[2., 5., 10., 20.])
    plan.add_argument('--output-dir', type=Path, required=True)
    plan.add_argument('--c-nm', type=float, default=150.)
    plan.add_argument('--gaps-nm', help='Comma-separated positive physical gaps; replace default pilot grid')
    plan.add_argument('--max-memory-gb', type=float, default=56.)
    plan.add_argument('--coarse-order', type=int, default=6)
    plan.add_argument('--fine-order', type=int, default=8)
    plan.add_argument('--surface-level', type=int, default=1)
    plan.add_argument('--rim-levels', type=int, default=1)
    plan.add_argument('--contact-levels', type=int, default=1)
    plan.add_argument('--max-refinements', type=int, default=2)
    plan.add_argument('--boundary-limit', type=float, default=.001)
    plan.add_argument('--model-tolerance', type=float, default=.01)
    plan.add_argument('--refinement-tolerance', type=float, default=.005)
    runp = sub.add_parser('run', help='Run or resume one AR; stop preserves completed stages')
    runp.add_argument('--campaign', type=Path, required=True)
    runp.add_argument('--build-manifest', type=Path, required=True)
    runp.add_argument('--expected-build-sha256')
    runp.add_argument('--expected-campaign-sha256')
    runp.add_argument('--wall-hours', type=float, default=335.)
    runp.add_argument('--retry-failed', action='store_true')
    status = sub.add_parser('status', help='Print current counts without running a solver')
    status.add_argument('--campaign', type=Path, required=True)
    verify = sub.add_parser('verify-build', help='Verify compiler artifacts and exact source snapshot')
    verify.add_argument('--build-manifest', type=Path, required=True)
    verify.add_argument('--expected-build-sha256')
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.mode == 'plan':
            path, plan = make_plan(args)
            emit('distance_plan_ready', campaign=str(path), campaign_sha256=worker.digest(path),
                 aspect_ratio=plan['settings']['aspect_ratio'], task_count=plan['task_count'],
                 point_count=plan['point_count'], gaps_nm=plan['settings']['gaps_nm'],
                 gap_execution_order_nm=plan['settings']['gap_execution_order_nm'])
            return 0
        if args.mode == 'run':
            return run(args)
        if args.mode == 'verify-build':
            emit('distance_build_verified', **verify_build(args.build_manifest, args.expected_build_sha256))
            return 0
        plan = read_plan(args.campaign)
        root = Path(plan['output_dir'])
        state_path = root/'run_state.json'
        state = worker.read_json(state_path) if state_path.exists() else {}
        print(json.dumps(summarize(plan, root, state), indent=2, allow_nan=False), flush=True)
        return 0
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        emit('distance_scan_error', error=str(exc))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
