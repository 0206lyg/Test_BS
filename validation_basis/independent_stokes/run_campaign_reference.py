#!/usr/bin/env python3
"""Run one explicit generic BIE campaign task, preserving every attempt.

Two gap-independent isolated checks are shared under a hash-bound locked cache.
The pair may receive a bounded, explicitly configured residual-driven refinement;
resource/solver failures never trigger a larger automatic solve. A completed task
is not physical qualification: development acceptance uses the existing importer
with the campaign's explicit boundary ceiling, while strict accuracy stays false.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import uuid

HERE = Path(__file__).resolve().parent
SCHEMA = 'independent_reference_campaign_v1'
RESULT_SCHEMA = 'independent_reference_campaign_task_v1'
CONTROL_DEFAULTS = dict(solver='out-of-core-dense', max_memory_gb=28., quad_tol=1e-8,
    check_points_per_patch=3, alpha=1., beta=1., lu_panel_columns=512, lu_stripe_columns=256,
    mmap_factor_mode='shared', gmres_rtol=1e-9, gmres_restart=80, gmres_maxiter=20,
    preconditioner='patch', far_order_increment=0, near_radius_factor=1.,
    geometry_refine_tolerance=0., geometry_max_refinements=3)
PAIR_RESOLUTION = dict(order=6, level=1, rim_levels=1, rim_width=.05,
                       contact_levels=1, contact_radius=.25, focus=0.)
ISOLATED_RESOLUTION = dict(PAIR_RESOLUTION, contact_levels=0)
SOURCE_NAMES = ('run_campaign_reference.py', 'run_reference.py', 'geometry.py', 'backend.py',
    'matrix_free.py', 'near_apply.py', 'memory_bounded_dense.py', 'out_of_core_dense.py',
    'out_of_core_lu.py', 'ram_dense.py', 'geometry_preflight.py', 'contact_check_sites.py',
    'prepare_reference_source.py', 'compare_reference.py')


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read_json(path):
    def reject(value):
        raise ValueError('Nonfinite JSON constant: '+value)
    data = json.loads(Path(path).read_text(), parse_constant=reject)
    if not isinstance(data, dict):
        raise ValueError('Expected JSON object: '+str(path))
    return data


def save(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name+'.', suffix='.pending',
                                         mode='w', delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(record, stream, indent=2, allow_nan=False)
            stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value, label, minimum=None, maximum=None, strict_minimum=False):
    require(isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value),
            label+' must be finite numeric')
    if minimum is not None:
        require(value > minimum if strict_minimum else value >= minimum, label+' is below its allowed range')
    if maximum is not None:
        require(value <= maximum, label+' is above its allowed range')
    return value


def integer(value, label, minimum=0):
    require(type(value) is int and value >= minimum, label+' must be an integer >= '+str(minimum))
    return value


def resolution(values, defaults):
    require(isinstance(values, dict), 'resolution must be an object')
    require(set(values) <= set(defaults)|{'resolution_level'}, 'Unknown resolution control')
    result = dict(defaults, **values)
    for key in ('order','level','rim_levels','contact_levels'):
        integer(result[key],key,2 if key=='order' else 0)
    for key in ('rim_width','contact_radius'):
        number(result[key],key,0.,strict_minimum=True)
    number(result['focus'],'focus',0.,1.)
    require(result['focus'] < 1., 'focus must be below 1')
    require(not result['rim_levels'] or result['focus']==0., 'Rim refinement requires focus=0')
    result.setdefault('resolution_level',result['level']*100+result['order'])
    integer(result['resolution_level'],'resolution_level')
    return result


def select_task(campaign, task_id):
    require(campaign.get('schema')==SCHEMA,'Unsupported campaign schema')
    tasks=campaign.get('tasks')
    require(isinstance(tasks,list) and bool(tasks),'Campaign tasks must be a nonempty list')
    seen=set(); selected=None
    for task in tasks:
        require(isinstance(task,dict),'Every task must be an object')
        identity=integer(task.get('task_id'),'task_id')
        require(identity not in seen,'Duplicate task_id in campaign'); seen.add(identity)
        if identity==task_id: selected=task
    require(selected is not None,'Requested task_id is absent from campaign')
    from geometry import spheroid_pose
    physical={key:selected.get(key) for key in ('aspect_ratio','c_nm','theta_i','theta_j','phi','gap_nm')}
    for key,value in physical.items(): number(value,key)
    require(physical['gap_nm']>0.,'Pair gap_nm must be positive')
    spheroid_pose(physical['gap_nm'],**{key:value for key,value in physical.items() if key!='gap_nm'})
    require(selected.get('role') in ('grid','validation'),'Task role must be grid or validation')
    for key in ('case_id','resolution_label','query_id'):
        require(isinstance(selected.get(key),str) and selected[key],key+' must be a nonempty string')
    require(selected['resolution_label'] in ('grid','coarse','fine'),'Unknown resolution_label')
    return selected,physical


def controls(campaign, max_memory_override=None):
    supplied=campaign.get('defaults',{})
    require(isinstance(supplied,dict) and set(supplied)<=set(CONTROL_DEFAULTS)|{'max_refinements','adapt_fraction','isolated_max_refinements','isolated_require_pass'},'Unknown solver default control')
    # These are worker orchestration controls, not run_reference CLI flags.
    values=dict(CONTROL_DEFAULTS,**{key:value for key,value in supplied.items() if key in CONTROL_DEFAULTS})
    if max_memory_override is not None: values['max_memory_gb']=max_memory_override
    require(values['solver'] in ('dense','ram-dense','mmap-dense','out-of-core-dense','all-near','fmm'),'Unknown solver')
    require(values['mmap_factor_mode'] in ('shared','cow'),'Unknown mmap factor mode')
    require(values['preconditioner'] in ('body','patch','none'),'Unknown preconditioner')
    for key in ('max_memory_gb','quad_tol','alpha','beta','gmres_rtol','near_radius_factor'):
        number(values[key],key,0.,strict_minimum=True)
    require(values['quad_tol']<.01 and values['gmres_rtol']<1.,'Invalid solver tolerance')
    for key in ('check_points_per_patch','lu_panel_columns','lu_stripe_columns','gmres_restart','gmres_maxiter'):
        integer(values[key],key,1)
    integer(values['far_order_increment'],'far_order_increment')
    number(values['geometry_refine_tolerance'],'geometry_refine_tolerance',0.)
    require(values['geometry_refine_tolerance']<1.,'Invalid geometry preflight tolerance')
    integer(values['geometry_max_refinements'],'geometry_max_refinements')
    require(values['geometry_refine_tolerance']==0 or values['solver']=='ram-dense',
            'Geometry preflight requires ram-dense solver')
    return values


@contextmanager
def locked(path, blocking=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as stream:
        try:
            fcntl.flock(stream,fcntl.LOCK_EX|(0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError as exc:
            raise ValueError('This campaign task is already running: '+str(path.parent)) from exc
        try: yield
        finally: fcntl.flock(stream,fcntl.LOCK_UN)


def recorded_file(report, key, report_path, fallback):
    value=report.get(key)
    require(isinstance(value,str) and bool(value),'Report lacks '+key)
    path=Path(value)
    if not path.is_absolute(): path=report_path.parent/path
    return path if path.exists() else fallback


def validate_report(report_path, config, physical, res, isolated_body):
    """Bind a finished report to exact source/solver/geometry and checkpoint bytes."""
    from run_reference import validate_pose
    from geometry import spheroid_pose
    path=Path(report_path); data=read_json(path)
    require(data.get('status')=='complete','Driver report is not complete')
    require(data.get('independent_reference') is True and data.get('shares_production_solver') is False and
            data.get('physical_accuracy_qualified') is False and data.get('synthetic') is not True,
            'Driver report scope/physical flags are invalid')
    validate_pose(data,spheroid_pose(physical['gap_nm'],**{k:v for k,v in physical.items() if k!='gap_nm'}).metadata())
    require(data.get('isolated_body')==isolated_body,'Driver isolated body mismatch')
    require(data.get('library_sha256')==config['library']['sha256'],'Driver library identity mismatch')
    require(data.get('operator_export',{}).get('sha256')==config['helper']['sha256'],'Driver operator helper identity mismatch')
    for key in ('run_reference.py','geometry.py','backend.py'):
        require(data.get('source_hashes',{}).get(key)==config['source_hashes'][key],'Driver source identity mismatch: '+key)
    for key,value in data.get('source_hashes',{}).items():
        require(config['source_hashes'].get(key)==value,'Driver dependency identity mismatch: '+key)
    rename={'quad_tol':'quadrature_tolerance'}
    reported={'solver','alpha','beta','far_order_increment','near_radius_factor','mmap_factor_mode',
              'lu_panel_columns','lu_stripe_columns','quad_tol'}
    for key in reported:
        require(data.get(rename.get(key,key))==config['controls'][key],'Driver solver control mismatch: '+key)
    for key,default in (('geometry_refine_tolerance',0.),('geometry_max_refinements',3)):
        require(data.get(key,default)==config['controls'][key],'Driver geometry control mismatch: '+key)
    for key,value in res.items(): require(data.get(key)==value,'Driver resolution mismatch: '+key)
    checkpoint=recorded_file(data,'solution_file',path,path.with_suffix('.solution.npz'))
    require(checkpoint.is_file() and digest(checkpoint)==data.get('solution_sha256'),'Driver checkpoint missing or SHA mismatch')
    archive=data.get('source_archive',{})
    require(isinstance(archive,dict),'Driver source archive missing')
    archive_path=Path(archive.get('path',''))
    if not archive_path.is_absolute(): archive_path=path.parent/archive_path
    if not archive_path.is_file(): archive_path=path.with_suffix('.sources.zip')
    require(archive_path.is_file() and digest(archive_path)==archive.get('sha256'),'Driver source archive SHA mismatch')
    return data,dict(path=str(path.resolve()),sha256=digest(path),solution_path=str(checkpoint.resolve()),
                     solution_sha256=digest(checkpoint))


def source_configuration(reference_library, operator_export, backend_manifest, values):
    library=Path(reference_library).resolve(strict=True); helper=Path(operator_export).resolve(strict=True)
    require(library.is_file() and helper.is_file(),'Reference library/helper must be files')
    require(os.access(helper,os.X_OK),'Operator helper is not executable')
    selected=Path(backend_manifest).resolve(strict=True) if backend_manifest is not None else library.parent.parent/'patch_manifest.json'
    manifest=None
    require(backend_manifest is None or selected.is_file(),'Explicit backend manifest must be a regular file')
    if selected.is_file():
        value=read_json(selected)
        require(value.get('library_sha256')==digest(library),'Backend build manifest identifies a different library')
        manifest=dict(path=str(selected),sha256=digest(selected))
    return dict(library=dict(path=str(library),sha256=digest(library)),
                helper=dict(path=str(helper),sha256=digest(helper)),backend_manifest=manifest,
                controls=values,source_hashes={name:digest(HERE/name) for name in SOURCE_NAMES},
                python_executable=sys.executable,python_version=sys.version)


def command_for(config, physical, res, report, isolated_body=None, isolated_reports=(), adapt_from=None, adapt_fraction=.2):
    command=[config['python_executable'],'-B',str(HERE/'run_reference.py'),
             '--library',config['library']['path'],'--operator-export',config['helper']['path'],
             '--output',str(report),'--matrix-workdir',str(report.parent/'matrix_work')]
    if config['backend_manifest'] is not None:
        command+=['--backend-manifest',config['backend_manifest']['path']]
    for key,value in dict(physical,**config['controls'],**res).items():
        command+=['--'+key.replace('_','-'),str(value)]
    if isolated_body is not None: command+=['--isolated-body',str(isolated_body)]
    for path in isolated_reports: command+=['--isolated-validation',str(path)]
    if adapt_from is not None:
        command+=['--adapt-from',str(adapt_from),'--adapt-fraction',str(adapt_fraction)]
    return command


def unchanged(config):
    for item in (config['library'],config['helper'],config['backend_manifest']):
        if item is not None:
            require(digest(item['path'])==item['sha256'],'Executable/library/build manifest changed during campaign task')
    for name,value in config['source_hashes'].items():
        require(digest(HERE/name)==value,'Worker/solver source changed during campaign task: '+name)


def run_stage(state, state_path, stage_name, directory, config, physical, res, isolated_body=None,
              isolated_reports=(), adapt_from=None, adapt_fraction=.2, isolated_retry_token=None):
    directory.mkdir(parents=True,exist_ok=False)
    report=directory/'reference.json'
    command=command_for(config,physical,res,report,isolated_body,isolated_reports,adapt_from,adapt_fraction)
    attempt=dict(stage=stage_name,status='running',command=command,directory=str(directory),resolution=dict(res),
                 isolated_retry_token=isolated_retry_token,
                 started_utc=datetime.now(timezone.utc).isoformat(),report_path=str(report),
                 physical_accuracy_qualified=False)
    state.setdefault('stageattempts',[]).append(attempt);save(state_path,state)
    start=time.monotonic()
    print(json.dumps(dict(event='reference_stage_started',stage=stage_name,report=str(report),
                          aspect_ratio=physical['aspect_ratio'],gap_nm=physical['gap_nm'],
                          order=res['order'],level=res['level'])),flush=True)
    with (directory/'stdout.log').open('w') as stdout,(directory/'stderr.log').open('w') as stderr:
        completed=subprocess.run(command,stdout=stdout,stderr=stderr,check=False)
    attempt.update(returncode=completed.returncode,seconds=time.monotonic()-start,
                   stdout_sha256=digest(directory/'stdout.log'),stderr_sha256=digest(directory/'stderr.log'))
    if report.is_file():
        attempt['report_sha256']=digest(report)
        try:
            progress=read_json(report)
            attempt['driver_status']=progress.get('status')
            attempt['driver_failure']=progress.get('failure_message',progress.get('failure'))
        except (OSError,ValueError): pass
    try:
        unchanged(config)
        require(completed.returncode==0,'Driver subprocess failed with exit '+str(completed.returncode))
        data,identity=validate_report(report,config,physical,res,isolated_body)
        attempt.update(status='complete',report=identity)
        print(json.dumps(dict(event='reference_stage_complete',stage=stage_name,report=str(report),
                              seconds=attempt['seconds'])),flush=True)
        save(state_path,state)
        return data,identity
    except (OSError,ValueError) as exc:
        attempt.update(status='failed',failure=str(exc));save(state_path,state)
        raise


def reused(stage, config, physical, res, body):
    identity=stage['report'];path=Path(identity['path'])
    require(digest(path)==identity['sha256'],'Previously completed report changed')
    data,actual=validate_report(path,config,physical,res,body)
    require(actual['solution_sha256']==identity['solution_sha256'],'Previously completed checkpoint changed')
    return data,actual


def isolated_policy(campaign, task):
    """Opt-in isolated accuracy retries; worker policy is never a driver CLI flag."""
    defaults=campaign.get('defaults',{})
    count=integer(task.get('isolated_max_refinements',campaign.get('isolated_max_refinements',
                  defaults.get('isolated_max_refinements',0))),'isolated_max_refinements')
    required=task.get('isolated_require_pass',campaign.get('isolated_require_pass',
                     defaults.get('isolated_require_pass',False)))
    require(type(required) is bool,'isolated_require_pass must be a boolean')
    return dict(max_refinements=count,require_pass=required,order_increment=2,
                boundary_limit=.001,moment_block_limit=.002)


def isolated_cache(root, config, physical, res, policy=None, *, retry_failed=False, retry_token=None):
    isolated_physical=dict(physical,gap_nm=physical['c_nm']) # h/c=1; isolated validation excludes pair gap.
    policy=policy if policy is not None else isolated_policy({}, {})
    identity=dict(configuration=config,physical=isolated_physical,resolution=res,isolated_policy=policy)
    cache=root/'isolated_cache'/canonical_digest(identity)
    cache.mkdir(parents=True,exist_ok=True)
    state_path=cache/'isolated.json'
    with locked(cache/'cache.lock'):
        state=read_json(state_path) if state_path.exists() else dict(schema='campaign_isolated_cache_v1',identity=identity,stageattempts=[])
        require(state.get('identity')==identity,'Isolated cache identity mismatch')
        paths=[];evidence=[]
        for body in (0,1):
            label='isolated'+str(body)
            for refinement in range(policy['max_refinements']+1):
                increment=policy['order_increment']*refinement
                current_res=dict(res,order=res['order']+increment,
                                 resolution_level=res['resolution_level']+increment)
                previous=[x for x in state['stageattempts'] if x['stage']==label and x['status']=='complete'
                          and x.get('resolution',res)==current_res]
                if previous:
                    stage=previous[-1]
                    data,result=reused(stage,config,isolated_physical,current_res,body)
                else:
                    failed=[x for x in state['stageattempts'] if x['stage']==label and x['status']=='failed'
                            and x.get('resolution',res)==current_res]
                    if policy['require_pass'] and failed:
                        last=failed[-1]
                        if not retry_failed or last.get('isolated_retry_token')==retry_token:
                            raise ValueError('Cached isolated resource/solver failure; pair was not run: '+
                                json.dumps(dict(body=body,order=current_res['order'],
                                    directory=last['directory'],failure=last.get('driver_failure',last.get('failure')),
                                    retry='Use --retry-failed with a new run token to retry once'),sort_keys=True))
                    index=len(state['stageattempts'])
                    data,result=run_stage(state,state_path,label,cache/('attempt_%04d'%index),config,
                                          isolated_physical,current_res,body,isolated_retry_token=retry_token)
                    stage=state['stageattempts'][-1]
                # Stage metadata also binds the exact p refinement on restart.
                stage['resolution']=current_res
                if not (policy['require_pass'] or policy['max_refinements']):
                    save(state_path,state)
                    break # Preserve legacy acceptance policy when not explicitly enabled.
                from run_reference import isolated_evidence
                assessment=isolated_evidence(result['path'],data)
                stage['isolated_assessment']=assessment
                result=dict(result,resolution=current_res,isolated_validation=assessment)
                save(state_path,state)
                print(json.dumps(dict(event='isolated_validation_assessed',body=body,
                    aspect_ratio=physical['aspect_ratio'],order=current_res['order'],
                    passed=assessment['passed'],failures=assessment['failures'],
                    refinements_used=refinement,max_refinements=policy['max_refinements'])),flush=True)
                if assessment['passed']:
                    break
            paths.append(Path(result['path']));evidence.append(result)
        save(state_path,state)
    return paths,evidence


def development_assessment(path, boundary_limit, ar):
    from prepare_reference_source import load_source
    try:
        accepted=load_source(path,boundary_limit=boundary_limit,expected_ar=ar)
        return dict(accepted_for_development=True,provisional_accepted=True,
                    physical_accuracy_qualified=False,development_boundary_limit=boundary_limit,
                    measurements=accepted['measurements'],strict_qualification_failures=accepted['strict_qualification_failures'])
    except (OSError,ValueError,KeyError) as exc:
        return dict(accepted_for_development=False,provisional_accepted=False,
                    physical_accuracy_qualified=False,development_boundary_limit=boundary_limit,failure=str(exc))


def needs_boundary_refinement(data, ceiling):
    validation=data.get('validation',{})
    errors=[validation.get(key) for key in ('boundary_max_scaled_error','gap_patch_max_scaled_error')]
    return (data.get('moments',{}).get('isolated_spheroid_validation_passed') is True and
            all(isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x) and x>=0 for x in errors) and
            max(errors)>ceiling)


def run_task(campaign_path, task_id, reference_library, operator_export, *, backend_manifest=None,
             output_root=None, max_memory_gb=None, retry_failed_isolated=False, isolated_retry_token=None):
    require(type(retry_failed_isolated) is bool,'retry_failed_isolated must be boolean')
    require(isolated_retry_token is None or isinstance(isolated_retry_token,str) and bool(isolated_retry_token.strip()),
            'isolated_retry_token must be a nonempty string')
    if retry_failed_isolated and isolated_retry_token is None:
        isolated_retry_token=uuid.uuid4().hex # Direct worker invocation gets one explicit retry opportunity.
    campaign_path=Path(campaign_path).resolve(strict=True);campaign=read_json(campaign_path)
    task,physical=select_task(campaign,task_id)
    values=controls(campaign,max_memory_gb)
    pair_res=resolution(task.get('resolution',{}),PAIR_RESOLUTION)
    iso_res=resolution(task.get('isolated_resolution',campaign.get('isolated_resolution',{})),ISOLATED_RESOLUTION)
    boundary=number(campaign.get('development_boundary_limit',.01),'development_boundary_limit',0.,1.,True)
    retry_defaults=campaign.get('defaults',{})
    iso_policy=isolated_policy(campaign,task)
    refinements=integer(task.get('max_refinements',campaign.get('max_refinements',retry_defaults.get('max_refinements',2))),'max_refinements')
    fraction=number(task.get('adapt_fraction',campaign.get('adapt_fraction',retry_defaults.get('adapt_fraction',.2))),'adapt_fraction',0.,1.,True)
    root=Path(output_root if output_root is not None else campaign.get('output_dir','reference_campaign'))
    if not root.is_absolute():root=campaign_path.parent/root
    root=root.resolve();subdir=Path(task.get('output_subdir','tasks/task_%06d'%task_id))
    require(not subdir.is_absolute() and '..' not in subdir.parts,'Task output_subdir must remain inside output root')
    folder=root/subdir;folder.mkdir(parents=True,exist_ok=True)
    config=source_configuration(reference_library,operator_export,backend_manifest,values)
    identity=dict(task=task,physical=physical,pair_resolution=pair_res,isolated_resolution=iso_res,
                  development_boundary_limit=boundary,max_refinements=refinements,adapt_fraction=fraction,
                  isolated_policy=iso_policy,
                  configuration=config,campaign_sha256=digest(campaign_path))
    state_path=folder/'task.json'
    with locked(folder/'task.lock',blocking=False):
        if state_path.exists():
            state=read_json(state_path)
            require(state.get('identity_sha256')==canonical_digest(identity),
                    'Task/config/source identity changed; select a new campaign output root instead of reusing old results')
        else:
            state=dict(schema=RESULT_SCHEMA,task_id=task_id,case_id=task['case_id'],query_id=task['query_id'],
                       role=task['role'],resolution_label=task['resolution_label'],identity=identity,
                       identity_sha256=canonical_digest(identity),campaign_path=str(campaign_path),
                       status='running',workflow_completed=False,accepted_for_development=False,
                       physical_accuracy_qualified=False,stageattempts=[])
        try:
            unchanged(config)
            state.update(status='running',workflow_completed=False);save(state_path,state)
            iso_paths,iso_evidence=isolated_cache(root,config,physical,iso_res,iso_policy,
                retry_failed=retry_failed_isolated,retry_token=isolated_retry_token)
            state['isolated_reports']=iso_evidence;save(state_path,state)
            if iso_policy['require_pass']:
                failed=[item['isolated_validation'] for item in iso_evidence
                        if item['isolated_validation']['passed'] is not True]
                require(not failed,'Isolated validation exhausted configured refinements; pair was not run: '+
                        json.dumps(failed,sort_keys=True))
            completed=[x for x in state['stageattempts'] if x['stage']=='pair' and x['status']=='complete']
            selected=None;data=None
            if completed:
                data,selected=reused(completed[-1],config,physical,pair_res,None)
            while True:
                if selected is None or (needs_boundary_refinement(data,boundary) and len(completed)<=refinements):
                    previous=Path(selected['path']) if selected is not None else None
                    index=len(state['stageattempts'])
                    data,selected=run_stage(state,state_path,'pair',folder/('attempt_%04d'%index),config,
                        physical,pair_res,None,iso_paths,previous,fraction)
                    completed=[x for x in state['stageattempts'] if x['stage']=='pair' and x['status']=='complete']
                state['selected_pair_report']=selected
                assessment=development_assessment(Path(selected['path']),boundary,physical['aspect_ratio'])
                state['development_assessment']=assessment
                state['accepted_for_development']=assessment['accepted_for_development']
                save(state_path,state)
                if assessment['accepted_for_development'] or not needs_boundary_refinement(data,boundary) or len(completed)>refinements:
                    break
            state.update(status='complete' if state['accepted_for_development'] else 'completed_unaccepted',workflow_completed=True,
                         refinement_attempts=max(0,len(completed)-1),physical_accuracy_qualified=False)
            save(state_path,state)
            return state,0
        except (OSError,ValueError,RuntimeError,KeyError) as exc:
            state.update(status='unresolved',workflow_completed=False,accepted_for_development=False,
                         failure=str(exc),physical_accuracy_qualified=False)
            save(state_path,state)
            return state,2


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--task-id',type=int,required=True)
    parser.add_argument('--reference-library',type=Path,required=True)
    parser.add_argument('--operator-export',type=Path,required=True)
    parser.add_argument('--backend-manifest',type=Path)
    parser.add_argument('--output-root',type=Path)
    parser.add_argument('--max-memory-gb',type=float)
    parser.add_argument('--retry-failed-isolated',action='store_true',
                        help='Explicitly retry a cached isolated resource failure once per run token')
    parser.add_argument('--isolated-retry-token',
                        help='Shared coordinator run token; not part of physical/task identity')
    args=parser.parse_args()
    try:
        state,code=run_task(args.campaign,args.task_id,args.reference_library,args.operator_export,
                           backend_manifest=args.backend_manifest,output_root=args.output_root,max_memory_gb=args.max_memory_gb,
                           retry_failed_isolated=args.retry_failed_isolated,isolated_retry_token=args.isolated_retry_token)
        print(json.dumps({key:state.get(key) for key in ('task_id','status','workflow_completed',
            'accepted_for_development','physical_accuracy_qualified','selected_pair_report','failure')},allow_nan=False),flush=True)
        return code
    except (OSError,ValueError,RuntimeError,KeyError) as exc:
        print('Campaign reference worker: '+str(exc),file=sys.stderr)
        return 2


if __name__=='__main__':
    raise SystemExit(main())
