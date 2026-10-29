#!/usr/bin/env python3
"""Actual low-resolution backend routing fixture; never an accuracy campaign."""
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
HERE = ROOT.parents[2]
sys.path.insert(0, str(HERE))
import reference_table_campaign as planner
import run_campaign_reference as worker

LIBRARY = Path('/workspace/scratch/581ed100c5ad/reference_deps_quadparams_highorder_v1/lib/libfmm3dbie.so')
MANIFEST = LIBRARY.parent.parent/'patch_manifest.json'
HELPER = Path('/dev/shm/build-reference-pipeline/slurry_reference_operator_export')


def prepare():
    args = planner.parser().parse_args(['plan', '--output-dir', str(ROOT/'planner_full'),
        '--ars', '2,5,10,20', '--grid-order', '2', '--coarse-order', '2', '--fine-order', '2',
        '--surface-level', '0', '--rim-levels', '0', '--contact-levels', '0',
        '--solver', 'dense', '--max-memory-gb', '2', '--max-refinements', '0'])
    path, original = planner.make_plan(args)
    plan = deepcopy(original)
    tasks = []
    for ar in (2., 5., 10., 20.):
        # This is a genuine non-polar angular-grid node, not a manufactured pose.
        group = next(g for g in original['groups'] if g['aspect_ratio'] == ar)
        task = deepcopy(original['tasks'][group['grid_task_ids'][53]])
        task.update(task_id=len(tasks), output_subdir='tasks/task_%06d'%len(tasks),
                    max_refinements=0, test_only_low_resolution=True)
        task['isolated_resolution'] = deepcopy(task['resolution'])
        tasks.append(task)
    plan.update(output_dir=str(ROOT/'run'), tasks=tasks, task_count=len(tasks),
                groups=[dict(id=planner.label(t['aspect_ratio']),aspect_ratio=t['aspect_ratio'],
                             grid_task_ids=[t['task_id']],cases=[]) for t in tasks],
                test_only_low_resolution=True, physical_accuracy_qualified=False,
                complete_grid_coverage=False,
                scope='Real backend routing only: p2/l0 with no rim/contact/adaptive refinement; insufficient for accuracy.',
                planner_fixture_parent=dict(path=str(path),sha256=planner.sha(path)),
                test_fixture_selection='One actual grid task (local index53) per AR; IDs reassigned before execution.')
    plan['settings'].update(isolated_order=2,isolated_level=0,isolated_rim_levels=0)
    plan['configuration_sha256'] = planner.identity(plan['settings'])
    worker.save(ROOT/'campaign.json',plan)
    print(json.dumps(dict(campaign=str(ROOT/'campaign.json'),task_count=len(tasks),
                         test_only_low_resolution=True)))


def run(task_id):
    command=[sys.executable,'-B',str(HERE/'run_campaign_reference.py'),'--campaign',str(ROOT/'campaign.json'),
             '--task-id',str(task_id),'--reference-library',str(LIBRARY),
             '--backend-manifest',str(MANIFEST),'--operator-export',str(HELPER)]
    environment=dict(os.environ,TMPDIR='/dev/shm',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    environment['PATH']='/workspace/scratch/581ed100c5ad/reference_deps/gfortran13/bin:'+environment.get('PATH','')
    with (ROOT/('worker_%d.stdout.log'%task_id)).open('w') as out, (ROOT/('worker_%d.stderr.log'%task_id)).open('w') as err:
        result=subprocess.run(command,env=environment,stdout=out,stderr=err,timeout=180)
    print(json.dumps(dict(task_id=task_id,returncode=result.returncode,test_only_low_resolution=True)))
    return result.returncode


def summarize():
    campaign=worker.read_json(ROOT/'campaign.json'); result=[]
    for task in campaign['tasks']:
        state_path=Path(campaign['output_dir'])/task['output_subdir']/'task.json'
        state=worker.read_json(state_path)
        reports=[]
        identities=state.get('isolated_reports',[])+([state['selected_pair_report']] if state.get('selected_pair_report') else [])
        for identity in identities:
            path=Path(identity['path']); report=worker.read_json(path)
            if worker.digest(path)!=identity['sha256']: raise ValueError('Report changed: '+str(path))
            if report['schema']!='independent_spheroid_panel_bie_v1' or report['status']!='complete':
                raise ValueError('Actual generic driver did not complete')
            for key in ('aspect_ratio','c_nm','theta_i','theta_j','phi'):
                if report[key]!=task[key]: raise ValueError('Geometry mismatch: '+key)
            if report['h_nm']!=task['gap_nm']: raise ValueError('Gap mismatch')
            if report['physical_accuracy_qualified'] is not False: raise ValueError('Unexpected physical qualification')
            checkpoint=Path(identity['solution_path'])
            if worker.digest(checkpoint)!=identity['solution_sha256']: raise ValueError('Checkpoint changed')
            reports.append(dict(path=str(path),sha256=identity['sha256'],solution_sha256=identity['solution_sha256'],
                schema=report['schema'],isolated_body=report['isolated_body'],node_count=report['node_count'],
                status=report['status'],geometry=report.get('geometry'),
                raw_reciprocity=report['raw_reciprocity'],validation=report['validation'],
                isolated_exact_validation=report.get('isolated_exact_validation'),
                physical_accuracy_qualified=False))
        result.append(dict(task_id=task['task_id'],aspect_ratio=task['aspect_ratio'],
            worker_status=state['status'],workflow_completed=state['workflow_completed'],
            accepted_for_development=state['accepted_for_development'],
            development_assessment=state.get('development_assessment'),
            driver_completion_count=len(reports),reports=reports))
    summary=dict(schema='actual_campaign_backend_routing_check_v1',
        completed_utc=datetime.now(timezone.utc).isoformat(),test_only_low_resolution=True,
        physical_accuracy_qualified=False,new_physical_stokes_solves=12,
        scope='Tiny actual BIE routing runs only; development rejection is expected and not bypassed.',
        campaign_sha256=worker.digest(ROOT/'campaign.json'),script_sha256=worker.digest(__file__),
        library_sha256=worker.digest(LIBRARY),operator_export_sha256=worker.digest(HELPER),
        controls=dict(order=2,level=0,rim_levels=0,contact_levels=0,max_refinements=0,solver='dense',max_memory_gb=2),
        routing_passed=all(x['workflow_completed'] and x['driver_completion_count']==3 for x in result),tasks=result)
    worker.save(ROOT/'summary.json',summary)
    print(json.dumps({k:summary[k] for k in ('routing_passed','test_only_low_resolution','physical_accuracy_qualified')}))


if __name__=='__main__':
    if sys.argv[1:] == ['prepare']: prepare()
    elif sys.argv[1:] == ['summarize']: summarize()
    elif len(sys.argv)==3 and sys.argv[1]=='run': raise SystemExit(run(int(sys.argv[2])))
    else: raise SystemExit('Usage: run_backend_routing.py prepare | run TASK_ID | summarize')
