"""Small worker-only tests; fixture reports do not claim physical validation."""
import json
from pathlib import Path
import unittest
from unittest import mock

import run_campaign_reference as worker
from run_reference import CONVENTION
import test_campaign_reference as fixtures


class IsolatedRetries(unittest.TestCase):
    write_plan = fixtures.CampaignWorker.write_plan
    assess = fixtures.CampaignWorker.assess
    def execute(self,task_id=0,**kwargs):
        with mock.patch.object(worker.subprocess,'run',side_effect=self.fake_driver), \
             mock.patch.object(worker,'development_assessment',side_effect=self.assess):
            return worker.run_task(self.campaign,task_id,self.library,self.helper,**kwargs)

    def setUp(self):
        fixtures.CampaignWorker.setUp(self)
        self.plan['isolated_resolution']={'order':8}
        self.plan['defaults'].update(isolated_max_refinements=2,isolated_require_pass=True)
        self.pass_orders={0:8,1:10}
        self.fail_isolated_body=None
        self.write_plan()

    def fake_driver(self,command,**kwargs):
        options=dict(zip(command[3::2],command[4::2]))
        if '--isolated-body' in options and int(options['--isolated-body'])==self.fail_isolated_body:
            import subprocess
            self.commands.append(command)
            Path(options['--output']).write_text(json.dumps(dict(status='memory_preflight_failed',
                failure_message='fixture memory ceiling')))
            return subprocess.CompletedProcess(command,2)
        completed=fixtures.CampaignWorker.fake_driver(self,command,**kwargs)
        if '--isolated-body' not in options:
            return completed
        body=int(options['--isolated-body']); order=int(options['--order'])
        path=Path(options['--output']); data=worker.read_json(path)
        passed=order>=self.pass_orders[body]
        data.update(convention_id=CONVENTION,raw_reciprocity=0.,
                    raw_full17=[[float(i==j) for j in range(17)] for i in range(17)])
        data['validation'].update(off_grid_boundary=True,boundary_evaluation_valid=True,
            boundary_max_scaled_error=.0004 if passed else .003,
            exact_surface_position_max_error_over_gap=.0001,linear_relative_residual=1e-12)
        data['isolated_exact_validation']={'passed':passed,'block_relative_errors':
            {key:.001 if passed else .003 for key in ('force','torque','stresslet')}}
        data['moments']['isolated_spheroid_validation_passed']=passed
        path.write_text(json.dumps(data))
        return completed

    def isolated_commands(self):
        return [cmd for cmd in self.commands if '--isolated-body' in cmd]

    def test_retry_preserves_failed_attempt_and_reuses_passing_refinement(self):
        state,code=self.execute()
        self.assertEqual(code,0)
        commands=self.isolated_commands()
        self.assertEqual([int(cmd[cmd.index('--order')+1]) for cmd in commands],[8,8,10])
        self.assertEqual(state['identity']['isolated_policy']['max_refinements'],2)
        self.assertEqual(state['isolated_reports'][1]['resolution']['order'],10)
        self.assertEqual(state['isolated_reports'][1]['resolution']['resolution_level'],110)
        cache=list((self.root/'output'/'isolated_cache').glob('*/isolated.json'))
        attempts=worker.read_json(cache[0])['stageattempts']
        self.assertEqual(len(attempts),3)
        self.assertFalse(attempts[1]['isolated_assessment']['passed'])
        self.assertTrue(attempts[2]['isolated_assessment']['passed'])
        state,code=self.execute()
        self.assertEqual(code,0)
        self.assertEqual(len(self.commands),4)
        self.assertNotIn('--isolated-max-refinements',self.commands[-1])
        self.assertNotIn('--isolated-require-pass',self.commands[-1])

    def test_exhausted_cache_is_unresolved_and_pair_never_starts(self):
        self.pass_orders[1]=99
        state,code=self.execute()
        self.assertEqual(code,2)
        self.assertEqual(state['status'],'unresolved')
        self.assertFalse(state['workflow_completed'])
        self.assertIn('pair was not run',state['failure'])
        self.assertEqual(len(self.commands),4)
        self.assertEqual(len(self.isolated_commands()),4)
        self.assertEqual(state['isolated_reports'][1]['resolution']['order'],12)
        self.assertEqual(state['stageattempts'],[])
        state,code=self.execute()
        self.assertEqual(code,2)
        self.assertEqual(len(self.commands),4)

    def test_refined_isolated_cache_is_shared_across_gaps(self):
        from copy import deepcopy
        task=deepcopy(self.plan['tasks'][0]); task.update(task_id=1,gap_nm=10.,
            query_id='gap10',output_subdir='tasks/task_000001')
        self.plan['tasks'].append(task); self.write_plan()
        first,code=self.execute(0); second,code2=self.execute(1)
        self.assertEqual((code,code2),(0,0))
        self.assertEqual(first['isolated_reports'],second['isolated_reports'])
        self.assertEqual(len(self.isolated_commands()),3)
        self.assertEqual(len(self.commands),5)

    def test_resource_failure_does_not_start_larger_isolated_retry(self):
        self.fail_isolated_body=0
        state,code=self.execute()
        self.assertEqual(code,2)
        self.assertEqual(state['status'],'unresolved')
        self.assertEqual(len(self.commands),1)
        self.assertEqual(self.commands[0][self.commands[0].index('--order')+1],'8')
        self.assertEqual(state['stageattempts'],[])
        cache=list((self.root/'output'/'isolated_cache').glob('*/isolated.json'))
        attempt=worker.read_json(cache[0])['stageattempts'][0]
        self.assertEqual(attempt['status'],'failed')
        self.assertIn('memory ceiling',attempt['driver_failure'])

    def test_failed_shared_isolated_stage_retries_once_per_explicit_run_token(self):
        from copy import deepcopy
        task=deepcopy(self.plan['tasks'][0]);task.update(task_id=1,gap_nm=10.,
            query_id='gap10',output_subdir='tasks/task_000001')
        self.plan['tasks'].append(task);self.write_plan()
        self.fail_isolated_body=0
        self.assertEqual(self.execute(0)[1],2)
        self.assertEqual(self.execute(1)[1],2)
        self.assertEqual(len(self.commands),1)
        state,_=self.execute(1)
        self.assertIn('Cached isolated resource/solver failure',state['failure'])
        self.assertEqual(len(self.commands),1)
        options=dict(retry_failed_isolated=True,isolated_retry_token='runA')
        self.assertEqual(self.execute(0,**options)[1],2)
        self.assertEqual(self.execute(1,**options)[1],2)
        self.assertEqual(len(self.commands),2)
        self.fail_isolated_body=None
        options['isolated_retry_token']='runB'
        self.assertEqual(self.execute(0,**options)[1],0)
        self.assertEqual(len(self.commands),6)
        self.assertEqual(self.execute(1,**options)[1],0)
        self.assertEqual(len(self.commands),7)
        cache=list((self.root/'output'/'isolated_cache').glob('*/isolated.json'))
        attempts=worker.read_json(cache[0])['stageattempts']
        self.assertEqual([a['isolated_retry_token'] for a in attempts[:3]],[None,'runA','runB'])
        self.assertEqual(attempts[0]['resolution']['order'],8)

    def test_interrupted_isolated_stage_is_retried_without_explicit_retry(self):
        self.fail_isolated_body=0
        self.execute()
        cache=list((self.root/'output'/'isolated_cache').glob('*/isolated.json'))[0]
        state=worker.read_json(cache);state['stageattempts'][0]['status']='running'
        worker.save(cache,state)
        self.fail_isolated_body=None
        result,code=self.execute()
        self.assertEqual(code,0)
        self.assertEqual(len(self.commands),5)
        self.assertTrue(result['accepted_for_development'])

    def test_policy_is_hash_bound_and_not_silently_changed_on_restart(self):
        self.execute()
        self.plan['defaults']['isolated_max_refinements']=1; self.write_plan()
        with self.assertRaisesRegex(ValueError,'identity changed'):
            self.execute()

    def test_policy_requires_integer_retry_count_and_boolean_gate(self):
        for key,value in [('isolated_max_refinements',True),('isolated_require_pass',1)]:
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    worker.isolated_policy({'defaults':{key:value}}, {})


if __name__=='__main__': unittest.main()
