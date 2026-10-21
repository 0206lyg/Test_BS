"""Worker orchestration fixtures only; no BIE solves or physical qualification."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

import geometry
import run_campaign_reference as worker


class CampaignWorker(unittest.TestCase):
    def setUp(self):
        directory=tempfile.TemporaryDirectory(prefix='worker_contract_only_')
        self.addCleanup(directory.cleanup); self.root=Path(directory.name)
        self.library=self.root/'reference.so';self.library.write_bytes(b'test-only mock library')
        self.helper=self.root/'helper';self.helper.write_text('#!/bin/sh\nexit 0\n');self.helper.chmod(0o700)
        self.campaign=self.root/'campaign.json'
        task=dict(task_id=0,case_id='ar5',role='grid',resolution_label='grid',query_id='pose0',
                  aspect_ratio=5.,c_nm=100.,theta_i=.2,theta_j=.7,phi=1.,gap_nm=100.,
                  resolution=dict(order=6,level=1,rim_levels=1,contact_levels=1),output_subdir='tasks/task_000000')
        self.plan=dict(schema=worker.SCHEMA,output_dir='output',development_boundary_limit=.01,max_refinements=2,
                       defaults=dict(max_memory_gb=2.),tasks=[task])
        self.write_plan();self.commands=[];self.pair_errors=[.008];self.fail_pair=False

    def write_plan(self):self.campaign.write_text(json.dumps(self.plan))

    def fake_driver(self,command,**kwargs):
        self.commands.append(command)
        options=dict(zip(command[3::2],command[4::2]))
        path=Path(options['--output']);body=int(options['--isolated-body']) if '--isolated-body' in options else None
        if body is None and self.fail_pair:
            path.write_text(json.dumps(dict(status='linear_solve_failed',failure_message='fixture resource failure')))
            return subprocess.CompletedProcess(command,2)
        physical={key:float(options['--'+key.replace('_','-')]) for key in
                  ('aspect_ratio','c_nm','theta_i','theta_j','phi','gap_nm')}
        pose=geometry.spheroid_pose(physical['gap_nm'],**{k:v for k,v in physical.items() if k!='gap_nm'})
        report=dict(pose.metadata(),schema='independent_spheroid_panel_bie_v1',status='complete',
            independent_reference=True,shares_production_solver=False,physical_accuracy_qualified=False,
            isolated_body=body,library_sha256=worker.digest(self.library),
            operator_export={'sha256':worker.digest(self.helper)},test_fixture_only=True)
        report['source_hashes']={name:worker.digest(worker.HERE/name) for name in
            ('backend.py','geometry.py','run_reference.py','matrix_free.py','near_apply.py','memory_bounded_dense.py',
             'out_of_core_dense.py','out_of_core_lu.py')}
        for key in worker.CONTROL_DEFAULTS:
            value=options['--'+key.replace('_','-')]
            if isinstance(worker.CONTROL_DEFAULTS[key],int):value=int(value)
            elif isinstance(worker.CONTROL_DEFAULTS[key],float):value=float(value)
            report['quadrature_tolerance' if key=='quad_tol' else key]=value
        for key in list(worker.PAIR_RESOLUTION)+['resolution_level']:
            value=options['--'+key.replace('_','-')]
            report[key]=int(value) if key in ('order','level','rim_levels','contact_levels','resolution_level') else float(value)
        solution=path.with_suffix('.solution.npz');solution.write_bytes(b'test-only fake checkpoint')
        archive=path.with_suffix('.sources.zip');archive.write_bytes(b'test-only fake source archive')
        report.update(solution_file=str(solution),solution_sha256=worker.digest(solution),
                      source_archive=dict(path=str(archive),sha256=worker.digest(archive)))
        error=0.0004 if body is not None else self.pair_errors.pop(0) if len(self.pair_errors)>1 else self.pair_errors[0]
        report['validation']=dict(boundary_max_scaled_error=error,gap_patch_max_scaled_error=error)
        report['moments']=dict(isolated_spheroid_validation_passed=True)
        path.write_text(json.dumps(report))
        return subprocess.CompletedProcess(command,0)

    def assess(self,path,ceiling,ar):
        error=worker.read_json(path)['validation']['boundary_max_scaled_error']
        return dict(accepted_for_development=error<=ceiling,physical_accuracy_qualified=False,
                    test_fixture_only=True,development_boundary_limit=ceiling)

    def execute(self,task_id=0):
        with mock.patch.object(worker.subprocess,'run',side_effect=self.fake_driver), \
             mock.patch.object(worker,'development_assessment',side_effect=self.assess):
            return worker.run_task(self.campaign,task_id,self.library,self.helper)

    def test_runs_two_isolated_and_pair_then_reuses_matching_complete_stages(self):
        state,code=self.execute()
        self.assertEqual(code,0);self.assertTrue(state['workflow_completed'])
        self.assertTrue(state['accepted_for_development']);self.assertFalse(state['physical_accuracy_qualified'])
        self.assertEqual(len(self.commands),3)
        self.assertNotIn('--isolated-body',self.commands[-1])
        self.assertEqual(self.commands[-1].count('--isolated-validation'),2)
        original=deepcopy(state['selected_pair_report'])
        state,code=self.execute()
        self.assertEqual(code,0);self.assertEqual(len(self.commands),3)
        self.assertEqual(state['selected_pair_report'],original)

    def test_shared_isolated_cache_excludes_pair_gap(self):
        task=deepcopy(self.plan['tasks'][0]);task.update(task_id=1,gap_nm=25.,query_id='holdout',
            role='validation',resolution_label='coarse',output_subdir='tasks/task_000001')
        self.plan['tasks'].append(task);self.write_plan()
        first,code=self.execute(0);second,code2=self.execute(1)
        self.assertEqual((code,code2),(0,0));self.assertEqual(len(self.commands),4)
        self.assertEqual(first['isolated_reports'],second['isolated_reports'])
        self.assertEqual(self.commands[-1][self.commands[-1].index('--gap-nm')+1],'25.0')

    def test_boundary_adaptation_is_bounded_and_preserves_reports(self):
        self.pair_errors=[.03,.02,.009]
        state,code=self.execute()
        self.assertEqual(code,0);self.assertTrue(state['accepted_for_development'])
        self.assertEqual(state['refinement_attempts'],2);self.assertEqual(len(self.commands),5)
        pairs=state['stageattempts'];self.assertEqual(len(pairs),3)
        for attempt in pairs:self.assertTrue(Path(attempt['report']['path']).is_file())
        for command in self.commands[-2:]:
            self.assertIn('--adapt-from',command)
            self.assertEqual(command[command.index('--order')+1],'6')

    def test_accuracy_rejection_is_complete_but_never_qualified(self):
        self.pair_errors=[.03]
        state,code=self.execute()
        self.assertEqual(code,0);self.assertTrue(state['workflow_completed'])
        self.assertEqual(state['status'],'completed_unaccepted')
        self.assertFalse(state['accepted_for_development']);self.assertFalse(state['physical_accuracy_qualified'])
        self.assertEqual(len(state['stageattempts']),3)

    def test_resource_failure_never_triggers_larger_retry(self):
        self.fail_pair=True
        state,code=self.execute()
        self.assertEqual(code,2);self.assertFalse(state['workflow_completed'])
        self.assertEqual(state['status'],'unresolved');self.assertEqual(len(self.commands),3)
        self.assertEqual(len(state['stageattempts']),1)
        self.assertIn('fixture resource failure',state['stageattempts'][0]['driver_failure'])
        self.fail_pair=False
        state,code=self.execute()
        self.assertEqual(code,0);self.assertEqual(len(self.commands),4)
        self.assertEqual(len(state['stageattempts']),2)
        self.assertEqual(state['stageattempts'][0]['status'],'failed')

    def test_modified_checkpoint_and_config_cannot_be_reused(self):
        state,code=self.execute()
        checkpoint=Path(state['selected_pair_report']['solution_path']);checkpoint.write_bytes(b'changed')
        state,code=self.execute()
        self.assertEqual(code,2);self.assertIn('checkpoint',state['failure'])
        self.plan['development_boundary_limit']=.02;self.write_plan()
        with self.assertRaisesRegex(ValueError,'identity changed'):self.execute()

    def test_actual_planner_contract_for_all_four_aspect_ratios(self):
        import reference_table_campaign as planner
        args=planner.parser().parse_args(['plan','--output-dir',str(self.root/'planned'),
            '--ars','2,5,10,20','--phi-nodes','4','--validation-gap-count','3'])
        path,plan=planner.make_plan(args)
        self.assertEqual(set(t['aspect_ratio'] for t in plan['tasks']),{2.,5.,10.,20.})
        for task in plan['tasks']:
            selected,physical=worker.select_task(plan,task['task_id'])
            self.assertEqual(selected['case_id'],task['case_id'])
            worker.resolution(task['resolution'],worker.PAIR_RESOLUTION)
        selected_controls=worker.controls(plan)
        self.assertNotIn('max_refinements',selected_controls)
        self.assertNotIn('adapt_fraction',selected_controls)
        self.campaign=path
        state,code=self.execute(0)
        self.assertEqual(code,0)
        self.assertEqual(state['identity']['max_refinements'],args.max_refinements)
        self.assertNotIn('--max-refinements',self.commands[-1])
        self.assertNotIn('--adapt-fraction',self.commands[-1])

    def test_invalid_task_and_output_path_are_rejected(self):
        self.plan['tasks'][0]['output_subdir']='../escape';self.write_plan()
        with self.assertRaisesRegex(ValueError,'inside output root'):self.execute()
        self.plan['tasks'].append(deepcopy(self.plan['tasks'][0]));self.write_plan()
        with self.assertRaisesRegex(ValueError,'Duplicate'):self.execute()


if __name__=='__main__':unittest.main()
