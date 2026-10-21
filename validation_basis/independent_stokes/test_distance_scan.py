"""Scheduler/state tests with explicit mock workers; no physical claims."""
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest import mock

import distance_scan as scan


class DistanceScanTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix='distance_scan_contract_')
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def plan(self, ar=2, extra=()):
        args = scan.parser().parse_args(['plan', '--ar', str(ar), '--output-dir', str(self.root/f'AR{ar}'), *extra])
        with mock.patch('geometry_preflight.choose_orders', return_value=dict(
                coarse_order=6, fine_order=8, candidates=[], requires_h_refinement=False)):
            return scan.make_plan(args)

    def test_distance_grid_is_not_one_anchor_and_counts_are_frozen(self):
        for ar, expected in ((2, 88), (5, 96), (10, 96), (20, 96)):
            path, plan = self.plan(ar)
            self.assertEqual(plan['task_count'], expected)
            self.assertEqual(len(plan['points'])*2, expected)
            self.assertEqual(plan['defaults']['isolated_max_refinements'], 2)
            self.assertIs(plan['defaults']['isolated_require_pass'], True)
            self.assertEqual(scan.worker.isolated_policy(plan, plan['tasks'][0])['max_refinements'], 2)
            self.assertTrue(scan.worker.isolated_policy(plan, plan['tasks'][0])['require_pass'])
            self.assertEqual(plan['settings']['gaps_nm'][0], 1.)
            self.assertEqual(plan['settings']['gaps_nm'][-1], ar*150*100)
            self.assertEqual(plan['settings']['gap_execution_order_nm'][:2], [150., ar*150*100])
            self.assertEqual({t['orientation'] for t in plan['tasks']}, {x[0] for x in scan.ORIENTATIONS})
            for start in range(0, expected, 8):
                self.assertEqual(len({t['gap_nm'] for t in plan['tasks'][start:start+8]}), 1)
            before = path.read_bytes()
            again, reloaded = self.plan(ar)
            self.assertEqual(again, path)
            self.assertEqual(path.read_bytes(), before)

    def test_invalid_resolution_pairs_fail_before_geometry_selection(self):
        for coarse, fine in ((11,12), (6,10), (5,7)):
            with self.assertRaisesRegex(ValueError, 'coarse-order'):
                self.plan(extra=['--coarse-order',str(coarse),'--fine-order',str(fine)])

    def test_slurm_comment_reports_active_and_completed_gap(self):
        path, plan = self.plan(extra=['--gaps-nm','1,10'])
        state = dict(status='running', current_task=dict(gap_nm=1.), tasks={
            str(i):dict(status='complete',updated_utc='2026-10-19T10:00:00Z') for i in range(8)})
        progress = scan.summarize(plan,path.parent,state)
        with mock.patch.dict(scan.os.environ, {'SLURM_JOB_ID':'123'}), mock.patch.object(scan.subprocess,'run') as update:
            scan.slurm_comment(progress)
        comment = update.call_args.args[0][-1]
        self.assertIn('h=1nm',comment)
        self.assertIn('last_h=10nm',comment)
        self.assertIn('8/16',comment)

    def test_changed_config_and_manifest_tampering_are_rejected(self):
        path, plan = self.plan(extra=['--gaps-nm', '1,10'])
        with self.assertRaisesRegex(ValueError, 'different settings'):
            self.plan(extra=['--gaps-nm', '1,20'])
        plan['tasks'][0]['gap_nm'] = 99
        scan.worker.save(path, plan)
        with self.assertRaisesRegex(ValueError, 'digest mismatch'):
            scan.read_plan(path)

    def test_progress_distinguishes_failed_processed_and_finished_solves(self):
        path, plan = self.plan(extra=['--gaps-nm', '1,10'])
        state = dict(status='running', tasks={'0':dict(status='complete'),
            '1':dict(status='completed_unaccepted'), '2':dict(status='failed'),
            '3':dict(status='interrupted')})
        counts = scan.summarize(plan, path.parent, state)['counts']
        self.assertEqual(counts['planned'], 16)
        self.assertEqual(counts['processed'], 3)
        self.assertEqual(counts['pair_completed'], 2)
        self.assertEqual(counts['development_accepted'], 1)
        self.assertEqual(counts['processed_percent'], 18.75)
        self.assertEqual(counts['pair_completed_percent'], 12.5)

    def fixture_run(self, path):
        args = scan.parser().parse_args(['run', '--campaign', str(path), '--build-manifest', str(self.root/'build.json')])
        build = dict(path=str(self.root/'build.json'), sha256='fixture',
                     paths=dict(python=sys.executable, reference_library='fixture.so', operator_export='fixture_helper', backend_manifest='fixture.json'))
        return args, build

    def test_worker_failure_resume_and_explicit_retry(self):
        path, plan = self.plan(extra=['--gaps-nm', '1,10'])
        args, build = self.fixture_run(path)
        reports, calls = {}, []
        fail = {1}
        def finished(root, task, campaign):
            return reports.get(task['task_id'], dict(state='missing'))
        def child(command, out, err, control):
            index = int(command[command.index('--task-id')+1]); calls.append(index)
            if args.retry_failed:
                self.assertIn('--retry-failed-isolated', command)
                self.assertIn('--isolated-retry-token', command)
            else:
                self.assertNotIn('--isolated-retry-token', command)
            if index in fail:
                return 2, False
            reports[index] = dict(state='available', accepted=True, path=f'fixture-{index}', sha256='fixture', worker_status='complete')
            return 0, False
        with mock.patch.object(scan, 'verify_build', return_value=build), \
             mock.patch.object(scan, 'verify_finished_task', side_effect=finished), \
             mock.patch.object(scan, 'execute_child', side_effect=child), \
             mock.patch.object(scan, 'assess'), mock.patch.object(scan, 'slurm_comment'), redirect_stdout(StringIO()):
            self.assertEqual(scan.run(args), 2)
            self.assertEqual(len(calls), 16)
            self.assertEqual(scan.run(args), 2)
            self.assertEqual(len(calls), 16)
            args.retry_failed = True; fail.clear()
            self.assertEqual(scan.run(args), 0)
            self.assertEqual(calls[-1], 1)
            self.assertEqual(len(calls), 17)
        state = scan.worker.read_json(path.parent/'run_state.json')
        self.assertEqual(len(state['attempts']), 17)
        self.assertEqual(state['attempts'][1]['status'], 'failed')
        self.assertEqual(scan.summarize(plan, path.parent, state)['counts']['pair_completed'], 16)

    def test_interrupted_worker_resumes_and_identity_change_is_blocked(self):
        path, plan = self.plan(extra=['--gaps-nm', '1,10'])
        args, build = self.fixture_run(path)
        reports, calls = {}, []
        def finished(root, task, campaign):
            return reports.get(task['task_id'], dict(state='missing'))
        def child(command, out, err, control):
            index = int(command[command.index('--task-id')+1]); calls.append(index)
            if len(calls) == 1:
                control.stop_requested = True; control.reason = 'fixture signal'
                return -15, True
            reports[index] = dict(state='available', accepted=True, path='fixture', sha256='fixture', worker_status='complete')
            return 0, False
        with mock.patch.object(scan, 'verify_build', return_value=build), \
             mock.patch.object(scan, 'verify_finished_task', side_effect=finished), \
             mock.patch.object(scan, 'execute_child', side_effect=child), \
             mock.patch.object(scan, 'assess'), mock.patch.object(scan, 'slurm_comment'), redirect_stdout(StringIO()):
            self.assertEqual(scan.run(args), 75)
            self.assertEqual(scan.worker.read_json(path.parent/'run_state.json')['status'], 'paused')
            self.assertEqual(scan.run(args), 0)
            self.assertEqual(calls[:2], [0, 0])
            build['sha256'] = 'other-build'
            with self.assertRaisesRegex(ValueError, 'identity changed'):
                scan.run(args)

    def test_real_child_is_stopped_at_wall_budget(self):
        control = scan.StopControl(time.monotonic()+.15)
        start = time.monotonic()
        code, stopped = scan.execute_child([sys.executable, '-c', 'import time; time.sleep(20)'],
            self.root/'stdout', self.root/'stderr', control)
        self.assertTrue(stopped)
        self.assertNotEqual(code, 0)
        self.assertLess(time.monotonic()-start, 5)
        self.assertEqual(control.reason, 'wall_budget')

    def test_parent_stop_race_preserves_interrupted_status(self):
        control = scan.StopControl(time.monotonic()+60)
        control.immediate = True
        child = mock.Mock()
        child.poll.return_value = -15
        with mock.patch.object(scan.subprocess, 'Popen', return_value=child):
            code, interrupted = scan.execute_child(['fixture'], self.root/'stdout', self.root/'stderr', control)
        self.assertEqual(code, -15)
        self.assertTrue(interrupted)
        self.assertTrue(control.stop_requested)

    def test_build_source_and_artifact_identity_are_enforced(self):
        source = self.root/'source'; package = source/'validation_basis'/'independent_stokes'
        package.mkdir(parents=True)
        names = ['distance_scan.py', 'run_campaign_reference.py', 'run_reference.py', 'distance_regime_assessment.py',
                 'ram_dense.py', 'geometry_preflight.py', 'contact_check_sites.py']
        hashes = {}
        for name in names:
            item = package/name; item.write_text('# fixture source\n')
            hashes[str(item.relative_to(source))] = scan.worker.digest(item)
        paths, artifacts = {}, {}
        for name in ('python', 'reference_library', 'backend_manifest', 'operator_export'):
            item = self.root/name; item.write_text(name)
            paths[name] = str(item)
        scan.worker.save(paths['backend_manifest'], dict(library_sha256=scan.worker.digest(paths['reference_library']),
            patch_policy='get_quadparams_adap_high_order_p_ge_4_v1'))
        for name, path in paths.items():
            artifacts[name] = dict(path=path, sha256=scan.worker.digest(path))
        manifest = self.root/'build.json'
        scan.worker.save(manifest, dict(schema='distance_scan_build_v1', status='ready', source_root=str(source),
            paths=paths, artifacts=artifacts, source_sha256=hashes))
        with mock.patch.object(scan, 'ROOT', source):
            result = scan.verify_build(manifest)
            self.assertEqual(result['paths'], paths)
            (package/'distance_scan.py').write_text('# changed\n')
            with self.assertRaisesRegex(ValueError, 'Source changed'):
                scan.verify_build(manifest)


if __name__ == '__main__':
    unittest.main()
