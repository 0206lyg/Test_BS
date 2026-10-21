#!/usr/bin/env python3
"""Plan, resume and collect independent-reference tables for multiple ARs.

Planning does not solve Stokes or submit jobs. Finalization handles each AR
independently, retaining failed attempts and reporting missing references.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

SCHEMA = 'independent_reference_campaign_v1'
HERE = Path(__file__).resolve().parent


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp.'+str(os.getpid()))
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def read(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError('Expected JSON object: '+str(path))
    return value


def label(ar):
    return 'AR'+format(ar, '.12g').replace('.', 'p')


def make_plan(args):
    ars = [float(x) for x in args.ars.split(',')]
    if (not ars or len(ars) != len(set(ars)) or any(not math.isfinite(x) or x < 1 for x in ars)
            or any(not math.isfinite(x) or x<=0 for x in (args.c_nm,args.floor_nm,args.anchor_over_c,args.max_memory_gb,args.quad_tol))
            or not 4 <= args.coarse_order < args.fine_order <= 12 or not 4 <= args.grid_order <= 12
            or min(args.surface_level,args.rim_levels,args.contact_levels,args.max_refinements)<0
            or args.max_refinements>8 or not 129 <= args.max_gap_nodes <= 4096
            or args.theta_nodes < 4 or args.phi_nodes < 4 or args.phi_nodes % 2
            or args.theta_nodes > 16 or args.phi_nodes > 32
            or args.validation_gap_count < 3 or args.validation_gap_count > 32
            or not 0 < args.floor_nm < args.c_nm*args.anchor_over_c
            or not 0 < args.boundary_limit <= 1 or not 0 < args.model_tolerance < 1
            or not 0 < args.refinement_tolerance < 1):
        raise ValueError('Invalid AR, units, grid, gap range or empirical screening tolerances')
    theta = [math.pi/4*(1-math.cos(math.pi*k/(args.theta_nodes-1))) for k in range(args.theta_nodes)]
    phi = [-math.pi+2*math.pi*k/args.phi_nodes for k in range(args.phi_nodes)]
    gap_max = args.c_nm*args.anchor_over_c
    gaps = [args.floor_nm*math.exp(math.log(gap_max/args.floor_nm)*k/(args.validation_gap_count-1))
            for k in range(args.validation_gap_count)]
    gaps[0], gaps[-1] = args.floor_nm, gap_max
    mid = [(theta[k]+theta[k+1])/2 for k in range(len(theta)-1)]
    # Critical almost-face/face pose plus interior and rim-dominated held-out poses.
    validation_poses = [(0.004924530359211356, 0.041010932014513654, 1.7016960206944713),
                        (mid[0], mid[len(mid)//2], phi[0]+math.pi/args.phi_nodes),
                        (mid[len(mid)//2], mid[-1], phi[args.phi_nodes//2]+math.pi/args.phi_nodes),
                        (mid[-1], mid[-1], phi[-2]+math.pi/args.phi_nodes)]
    output = args.output_dir.expanduser().resolve()
    defaults = dict(solver=args.solver, max_memory_gb=args.max_memory_gb, quad_tol=args.quad_tol,
                    check_points_per_patch=3, alpha=1., beta=1., lu_panel_columns=512,
                    lu_stripe_columns=256, max_refinements=args.max_refinements, adapt_fraction=.2)
    settings = dict(aspect_ratios=ars, c_nm=args.c_nm, floor_nm=args.floor_nm,
                    anchor_over_c=args.anchor_over_c, anchor_policy='explicit computational trial anchor; not a physical transition',
                    theta_nodes=theta, phi_nodes=phi, validation_gaps_nm=gaps,
                    development_boundary_limit=args.boundary_limit,
                    model_tolerance=args.model_tolerance, refinement_tolerance=args.refinement_tolerance,
                    grid_order=args.grid_order, coarse_order=args.coarse_order, fine_order=args.fine_order,
                    surface_level=args.surface_level, rim_levels=args.rim_levels, contact_levels=args.contact_levels,
                    isolated_order=6, isolated_level=1, isolated_rim_levels=1,
                    maximum_gap_nodes=args.max_gap_nodes, initial_gap_nodes=129,
                    interpolation_tolerance=1e-6, matched_model='dissipative-increment', defaults=defaults)
    tasks, groups = [], []

    def add_task(ar, pose, gap, role, case_id, resolution_label, order):
        task_id = len(tasks)
        task = dict(task_id=task_id, aspect_ratio=ar, c_nm=args.c_nm, theta_i=pose[0],
                    theta_j=pose[1], phi=pose[2], gap_nm=gap, role=role, case_id=case_id,
                    query_id=case_id,
                    resolution_label=resolution_label, output_subdir=f'tasks/task_{task_id:06d}',
                    resolution=dict(order=order, level=args.surface_level, rim_levels=args.rim_levels,
                                    contact_levels=args.contact_levels, resolution_level=order*100+args.surface_level),
                    isolated_resolution=dict(order=6, level=1, rim_levels=1, contact_levels=0, resolution_level=601))
        tasks.append(task)
        return task_id

    for ar in ars:
        name = label(ar)
        group = dict(id=name, aspect_ratio=ar, grid_task_ids=[], cases=[])
        grid = [(a,b,c) for a in theta for b in theta for c in phi]
        for k, pose in enumerate(grid):
            group['grid_task_ids'].append(add_task(ar, pose, gap_max, 'grid', f'{name}_g{k:04d}', 'grid', args.grid_order))
        poses = [(p, 'angle') for p in validation_poses]+[((theta[1], theta[-2], phi[1]), 'gap')]
        for k, (pose, kind) in enumerate(poses):
            for j, gap in enumerate(gaps):
                if kind == 'gap' and j == len(gaps)-1:
                    continue  # A training point cannot serve as a held-out comparison.
                query_id = f'{name}_v{k:02d}_h{j:02d}'
                actual_kind = kind if kind == 'gap' or j == len(gaps)-1 else 'both'
                case = dict(id=query_id, query_id=query_id, theta_i=pose[0], theta_j=pose[1], phi=pose[2],
                            gap_nm=gap, heldout_kind=actual_kind, reference_task_ids=[])
                for resolution_label, order in [('coarse',args.coarse_order), ('fine',args.fine_order)]:
                    case['reference_task_ids'].append(add_task(ar, pose, gap, 'validation', query_id, resolution_label, order))
                group['cases'].append(case)
        groups.append(group)
    config_hash = identity(settings)
    path = output/'campaign.json'
    if path.exists():
        existing = read(path)
        if (existing.get('schema') != SCHEMA or existing.get('configuration_sha256') != config_hash
                or existing.get('settings') != settings or existing.get('defaults') != defaults
                or existing.get('tasks') != tasks or existing.get('groups') != groups):
            raise ValueError('Existing campaign has different settings; use a new output directory')
        return path, existing
    if output.exists() and any(output.iterdir()):
        raise ValueError('New campaign requires an empty output directory')
    plan = dict(schema=SCHEMA, created_utc=datetime.now(timezone.utc).isoformat(), output_dir=str(output),
                configuration_sha256=config_hash, settings=settings, defaults=defaults,
                development_boundary_limit=args.boundary_limit, physical_accuracy_qualified=False,
                source_root=str(HERE.parents[1]), task_count=len(tasks), tasks=tasks, groups=groups,
                task_failure_policy='Retain failed task and continue other tasks/ARs; finalization reports unresolved coverage',
                accuracy_scope='Two-resolution empirical held-out screening; not a global error bound or bulk rheology qualification')
    save(path, plan)
    return path, plan


def task_report(root, task, campaign_path=None):
    path = root/task['output_subdir']/'task.json'
    if not path.is_file():
        return dict(task_id=task['task_id'], state='missing', path=str(path))
    try:
        record = read(path)
        bound = record.get('identity', {})
        if (bound.get('task') != task or record.get('identity_sha256') != identity(bound)
                or record.get('physical_accuracy_qualified') is not False):
            raise ValueError('Worker identity is not bound to this task')
        if campaign_path is not None and bound.get('campaign_sha256') != sha(campaign_path):
            raise ValueError('Worker belongs to a different campaign manifest')
        chosen = record.get('selected_pair_report')
        if isinstance(chosen, dict):
            chosen, expected = chosen.get('path'), chosen.get('sha256')
        else:
            expected = record.get('selected_pair_report_sha256')
        if not chosen:
            return dict(task_id=task['task_id'], state='no_pair_report', path=str(path), worker_status=record.get('status'))
        report = Path(chosen)
        if not report.is_absolute(): report = path.parent/report
        if not expected or sha(report) != expected:
            raise ValueError('Selected report SHA mismatch')
        data = read(report)
        for key in ('aspect_ratio','theta_i','theta_j','phi'):
            if not math.isclose(data[key],task[key], rel_tol=1e-12, abs_tol=1e-13):
                raise ValueError('Selected report task geometry mismatch: '+key)
        if not math.isclose(data['h_nm'],task['gap_nm'],rel_tol=1e-12,abs_tol=1e-12):
            raise ValueError('Selected report gap mismatch')
        if not math.isclose(data.get('length_unit_nm',data.get('c_nm',0)),task['c_nm'],rel_tol=1e-12):
            raise ValueError('Selected report physical size mismatch')
        for key in ('order','level'):
            if data.get(key)!=task['resolution'][key]:
                raise ValueError('Selected report resolution mismatch: '+key)
        return dict(task_id=task['task_id'], state='available', path=str(report.resolve()), sha256=expected,
                    accepted=record.get('accepted_for_development',record.get('development_accepted',False)) is True,
                    worker_status=record.get('status'))
    except (ValueError, OSError, KeyError, TypeError) as exc:
        return dict(task_id=task['task_id'], state='invalid', path=str(path), error=str(exc))


def run_logged(command, folder, name):
    folder.mkdir(parents=True,exist_ok=True)
    with (folder/(name+'.stdout.log')).open('w') as out, (folder/(name+'.stderr.log')).open('w') as err:
        process = subprocess.run([str(x) for x in command], stdout=out, stderr=err)
    return dict(command=[str(x) for x in command], returncode=process.returncode)


def verified_pipeline(path, inputs, executables, settings=None):
    try:
        p=read(path)
        if (p.get('schema')!='provisional_reference_table_pipeline_v1' or p.get('status')!='complete'
                or p.get('layout')!='full-grid' or p.get('matched_model')!='dissipative-increment'
                or p.get('workflow_completed') is not True or p.get('physical_accuracy_qualified') is not False):
            return False
        required={'reference.source17','reference.source17.manifest.json','reference.sft','generation.json','runtime.json'}
        if not required.issubset(p['outputs']):return False
        if [(x['stage'],x['returncode']) for x in p['stages']]!=[('import',0),('generate',0),('runtime',0)]:return False
        if sorted(x['sha256'] for x in p['input_reports']) != sorted(x['sha256'] for x in inputs):return False
        if p['executables'] != {str(x):sha(x) for x in executables}:return False
        for name,meta in p['outputs'].items():
            if Path(name).name!=name:return False
            file=path.parent/name
            if file.stat().st_size!=meta['bytes'] or sha(file)!=meta['sha256']:return False
        generation=read(path.parent/'generation.json');runtime=read(path.parent/'runtime.json')
        source=read(path.parent/'reference.source17.manifest.json')
        if (generation.get('status')!='provisional_numerical_transport_passed'
                or generation.get('layout')!='complete_actual_source_grid'
                or runtime.get('runtime_layout')!='complete_actual_source_grid'
                or runtime.get('pipeline_algebra_passed') is not True
                or runtime.get('poses_checked')!=len(inputs)
                or source.get('source_text_sha256')!=p['outputs']['reference.source17']['sha256']
                or any(x.get('physical_accuracy_qualified') is not False for x in (generation,runtime,source))
                or generation.get('model_revision')!='JANOSCHEK_DISSIPATIVE_INCREMENT_FULL17_V1'
                or generation.get('model_revision')!=runtime.get('model_revision')
                or generation.get('source_fnv1a64')!=runtime.get('source_fnv1a64')
                or len(generation.get('numerical_source_fingerprint',''))!=64
                or generation.get('numerical_source_fingerprint')!=runtime.get('numerical_source_fingerprint')):return False
        if settings and (not math.isclose(generation['floor_m'],settings['floor_nm']*1e-9,rel_tol=1e-12)
                or not math.isclose(generation['c_m'],settings['c_nm']*1e-9,rel_tol=1e-12)
                or not math.isclose(generation['hmax_over_c'],settings['anchor_over_c'],rel_tol=1e-12)):return False
        return True
    except (OSError,ValueError,KeyError,TypeError):return False


def finalize(args):
    campaign_path=args.campaign.resolve(strict=True); campaign=read(campaign_path)
    if campaign.get('schema')!=SCHEMA:raise ValueError('Unsupported campaign schema')
    root=Path(campaign['output_dir']);settings=campaign['settings']
    tools=[args.generator.resolve(strict=True),args.checker.resolve(strict=True),args.query_tool.resolve(strict=True)]
    status={task['task_id']:task_report(root,task,campaign_path) for task in campaign['tasks']}
    summary=dict(schema='reference_table_campaign_summary_v1',created_utc=datetime.now(timezone.utc).isoformat(),
                 campaign_sha256=sha(campaign_path),physical_accuracy_qualified=False,groups=[],
                 task_counts={state:sum(x['state']==state for x in status.values()) for state in ('available','missing','invalid','no_pair_report')})
    assessment_groups=[]
    def assessment_group(group, completed, query_output):
        return dict(id=group['id'],query_result=str(query_output),pipeline_path=str(completed),
            table_path=str(completed.parent/'reference.sft'),
            training_source=str(completed.parent/'reference.source17'),
            training_manifest=str(completed.parent/'reference.source17.manifest.json'),
            generation_report=str(completed.parent/'generation.json'),cases=[dict(id=c['id'],query_id=c['query_id'],
                heldout_kind=c['heldout_kind'],geometry=dict(aspect_ratio=group['aspect_ratio'],c_m=settings['c_nm']*1e-9,
                    theta_i=c['theta_i'],theta_j=c['theta_j'],phi=c['phi'],h_over_c=c['gap_nm']/settings['c_nm']),
                references=[status[t]['path'] if status[t]['state']=='available' else
                    str(root/campaign['tasks'][t]['output_subdir']/'unavailable_reference.json')
                    for t in c['reference_task_ids']]) for c in group['cases']])
    for group in campaign['groups']:
        destination=root/group['id'];destination.mkdir(parents=True,exist_ok=True)
        result=dict(id=group['id'],aspect_ratio=group['aspect_ratio'],table_status='waiting_for_references',
                    missing_or_rejected_grid_tasks=[status[t] for t in group['grid_task_ids'] if status[t]['state']!='available' or not status[t].get('accepted')],
                    accuracy_status='UNRESOLVED',physical_accuracy_qualified=False)
        summary['groups'].append(result)
        assessment_groups.append(assessment_group(group,destination/'not_yet_generated'/'pipeline.json',
                                                   destination/'not_yet_queried.json'))
        if result['missing_or_rejected_grid_tasks']:continue
        inputs=[status[t] for t in group['grid_task_ids']]
        attempts=destination/'table_attempts';attempts.mkdir(exist_ok=True)
        completed=next((x for x in sorted(attempts.glob('attempt_*/pipeline.json'),reverse=True)
                        if verified_pipeline(x,inputs,tools[:2],settings)),None)
        if completed is None:
            attempt=attempts/f'attempt_{len(list(attempts.glob("attempt_*")))+1:04d}'
            command=[sys.executable,HERE/'run_reference_table_pipeline.py',*[x['path'] for x in inputs],
                     '--output-dir',attempt,'--generator',tools[0],'--checker',tools[1],
                     '--matched-model',settings['matched_model'],'--layout','full-grid',
                     '--floor-nm',settings['floor_nm'],'--boundary-limit',settings['development_boundary_limit'],
                     '--gap-nodes',settings['initial_gap_nodes'],'--max-gap-nodes',settings['maximum_gap_nodes'],
                     '--interpolation-tolerance',settings['interpolation_tolerance']]
            invocation=run_logged(command,destination,'table_generation')
            result['generation_invocation']=invocation
            completed=attempt/'pipeline.json'
            if invocation['returncode'] or not verified_pipeline(completed,inputs,tools[:2],settings):
                result['table_status']='generation_failed';result['pipeline']=str(completed);continue
        result.update(table_status='complete',pipeline=str(completed),table=str(completed.parent/'reference.sft'))
        query_dir=destination/'accuracy_queries';query_dir.mkdir(exist_ok=True)
        query_input=query_dir/'queries.txt'
        lines=['QUERY_REFERENCE_TABLE17_V1','CONVENTION slurry_full17_c_eta_1_v1',
               f'UNITS {settings["c_nm"]*1e-9:.17g} {settings["floor_nm"]*1e-9:.17g}',
               f'AR {group["aspect_ratio"]:.17g}',f'QUERIES {len(group["cases"])}']
        for case in group['cases']:
            lines.append(case['query_id']+' '+' '.join(format(case[k],'.17g') for k in ('theta_i','theta_j','phi','gap_nm')))
        lines.append('END');query_input.write_text('\n'.join(lines)+'\n')
        query_output=query_dir/('queries_'+sha(query_input)[:12]+'_'+sha(completed.parent/'reference.sft')[:12]+'_'+sha(tools[2])[:12]+'.json')
        if not query_output.exists():
            invocation=run_logged([tools[2],'--table',completed.parent/'reference.sft','--queries',query_input,'--output',query_output],query_dir,'query')
            result['query_invocation']=invocation
            if invocation['returncode']:result['accuracy_status']='QUERY_FAILED';continue
        assessment_groups[-1]=assessment_group(group,completed,query_output)
    assessment_input=root/'accuracy_input.json'
    save(assessment_input,dict(schema='reference17_accuracy_campaign_v1',groups=assessment_groups))
    if assessment_groups:
        invocation=run_logged([sys.executable,HERE/'assess_campaign_accuracy.py',assessment_input,
            '--output',root/'accuracy'/'accuracy.json','--csv',root/'accuracy'/'accuracy.csv',
            '--plot',root/'accuracy'/'accuracy.png','--model-tolerance',settings['model_tolerance'],
            '--refinement-tolerance',settings['refinement_tolerance'],'--boundary-tolerance',settings['development_boundary_limit']],root,'accuracy_assessment')
        summary['assessment_invocation']=invocation
        summary['assessment_report']=str(root/'accuracy'/'accuracy.json')
        if invocation['returncode']==0:
            assessment=read(root/'accuracy'/'accuracy.json')
            for judged in assessment['groups']:
                target=next(x for x in summary['groups'] if x['id']==judged['id'])
                target['accuracy_counts']=judged['counts']
                counts=judged['counts']
                target['accuracy_status']=('FAIL' if counts['FAIL'] else
                    'UNRESOLVED' if counts['UNRESOLVED'] or not counts['PASS'] else 'PASS')
                target['accuracy_scope']='Empirical held-out screening only; no global or bulk-viscosity error bound'
    summary['next_action']='Inspect campaign_summary.json and accuracy/accuracy.csv; rerun missing tasks or create a finer campaign. A completed table alone does not establish physical accuracy.'
    save(root/'campaign_summary.json',summary)
    return summary


def parser():
    p=argparse.ArgumentParser(description=__doc__);subs=p.add_subparsers(dest='command',required=True)
    plan=subs.add_parser('plan');plan.add_argument('--output-dir',type=Path,required=True)
    plan.add_argument('--ars',default='2,5,10,20');plan.add_argument('--c-nm',type=float,default=150.)
    plan.add_argument('--floor-nm',type=float,default=1.);plan.add_argument('--anchor-over-c',type=float,default=1.)
    plan.add_argument('--theta-nodes',type=int,default=4);plan.add_argument('--phi-nodes',type=int,default=8)
    plan.add_argument('--validation-gap-count',type=int,default=5)
    plan.add_argument('--boundary-limit',type=float,default=.01)
    plan.add_argument('--model-tolerance',type=float,default=.05);plan.add_argument('--refinement-tolerance',type=float,default=.01)
    plan.add_argument('--grid-order',type=int,default=8);plan.add_argument('--coarse-order',type=int,default=6);plan.add_argument('--fine-order',type=int,default=8)
    plan.add_argument('--surface-level',type=int,default=1);plan.add_argument('--rim-levels',type=int,default=1);plan.add_argument('--contact-levels',type=int,default=1)
    plan.add_argument('--solver',choices=['dense','mmap-dense','out-of-core-dense'],default='out-of-core-dense')
    plan.add_argument('--max-memory-gb',type=float,default=28.);plan.add_argument('--quad-tol',type=float,default=1e-8)
    plan.add_argument('--max-refinements',type=int,default=2);plan.add_argument('--max-gap-nodes',type=int,default=1024)
    end=subs.add_parser('finalize');end.add_argument('--campaign',type=Path,required=True)
    for name in ('generator','checker','query-tool'):end.add_argument('--'+name,type=Path,required=True)
    status=subs.add_parser('status');status.add_argument('--campaign',type=Path,required=True)
    return p


def main():
    p=parser();args=p.parse_args()
    try:
        if args.command=='plan':
            path,plan=make_plan(args);print(json.dumps(dict(campaign=str(path),task_count=plan['task_count'],aspect_ratios=plan['settings']['aspect_ratios'])))
        elif args.command=='finalize':
            summary=finalize(args);print(json.dumps(dict(summary=str(Path(read(args.campaign)['output_dir'])/'campaign_summary.json'),task_counts=summary['task_counts'])))
        else:
            plan=read(args.campaign);root=Path(plan['output_dir'])
            rows=[task_report(root,t,args.campaign) for t in plan['tasks']]
            print(json.dumps(dict(task_count=len(rows),states={k:sum(r['state']==k for r in rows) for k in ('available','missing','invalid','no_pair_report')},accepted=sum(r.get('accepted',False) for r in rows)),indent=2))
        return 0
    except (ValueError,OSError,KeyError,TypeError) as exc:p.error(str(exc))


if __name__=='__main__':raise SystemExit(main())
