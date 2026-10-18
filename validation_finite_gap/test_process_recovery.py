#!/usr/bin/env python3
"""Test actual SIGBUS child processes and the table launcher's bounded retry.

No numerical physics is mocked or claimed here. Child programs deliberately
signal themselves, with core dumps disabled. Python 3.6 standard library only.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import signal
import socket
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'lammps_slurry/tools/prepare_case.py'
spec = importlib.util.spec_from_file_location('slurry_prepare_recovery_test', str(SOURCE))
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)

CHILD = r'''
import json
import os
from pathlib import Path
import resource
import signal
import sys
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
mode, state_name, diagnostic_name = sys.argv[1:]
state_path, diagnostic_path = Path(state_name), Path(diagnostic_name)
state = json.loads(state_path.read_text()) if state_path.exists() else []
prior = json.loads(diagnostic_path.read_text()) if diagnostic_path.exists() else None
state.append({'pid': os.getpid(), 'prior_diagnostic': prior})
state_path.write_text(json.dumps(state))
if mode == 'numerical':
    sys.exit(2)
if mode == 'sigbus_always' or (mode in ('recover', 'sigbus_then_numerical') and len(state) == 1):
    os.kill(os.getpid(), signal.SIGBUS)
    sys.exit(99)
if mode == 'sigbus_then_numerical':
    sys.exit(2)
sys.exit(0)
'''


@unittest.skipUnless(hasattr(signal, 'SIGBUS'), 'SIGBUS is required')
class ProcessRecovery(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='slurry-process-recovery-')
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.child = self.directory / 'signal_child.py'
        self.child.write_text(CHILD)
        self.state = self.directory / 'state.json'
        self.diagnostic = self.directory / 'diagnostic.json'

    def execute(self, mode):
        self.command = [sys.executable, str(self.child), mode, str(self.state), str(self.diagnostic)]
        return prepare.run_fast_generator(self.command, self.diagnostic)

    def verify_records(self, codes, completed):
        state = json.loads(self.state.read_text())
        self.assertEqual(len(state), len(codes))
        self.assertEqual(len({record['pid'] for record in state}), len(codes))
        report = json.loads(self.diagnostic.read_text())
        self.assertEqual(report['hostname'], socket.gethostname())
        self.assertEqual(report['command'], self.command)
        self.assertEqual(report['completed_after_sigbus_restart'], completed)
        self.assertEqual([item['returncode'] for item in report['attempts']], codes)
        self.assertEqual([item['attempt'] for item in report['attempts']], list(range(1, len(codes) + 1)))
        for attempt in report['attempts']:
            self.assertGreaterEqual(attempt['elapsed_seconds'], 0.)
        if len(codes) == 2:
            # The failure record must be installed before the replacement child
            # starts, and remain present in the eventual success/failure record.
            prior = state[1]['prior_diagnostic']
            self.assertEqual([item['returncode'] for item in prior['attempts']], [codes[0]])
            self.assertFalse(prior['completed_after_sigbus_restart'])
            self.assertEqual(prior['attempts'][0], report['attempts'][0])
        self.assertEqual(list(self.directory.glob('core*')), [])
        return report

    def test_sigbus_once_then_success_uses_one_fresh_retry(self):
        self.execute('recover')
        self.verify_records([-signal.SIGBUS, 0], True)

    def test_numerical_exit_two_is_not_retried(self):
        with self.assertRaisesRegex(ValueError, 'exit 2'):
            self.execute('numerical')
        self.verify_records([2], False)

    def test_persistent_sigbus_stops_after_two_processes(self):
        with self.assertRaisesRegex(ValueError, 'exit -{}'.format(signal.SIGBUS)):
            self.execute('sigbus_always')
        self.verify_records([-signal.SIGBUS, -signal.SIGBUS], False)

    def test_sigbus_then_numerical_failure_preserves_both_records(self):
        with self.assertRaisesRegex(ValueError, 'exit 2'):
            self.execute('sigbus_then_numerical')
        self.verify_records([-signal.SIGBUS, 2], False)

    def test_first_success_is_not_retried(self):
        self.execute('success')
        self.assertEqual(len(json.loads(self.state.read_text())), 1)
        self.assertFalse(self.diagnostic.exists())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ProcessRecovery)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {'passed': result.wasSuccessful(), 'tests': result.testsRun,
              'failures': len(result.failures), 'errors': len(result.errors),
              'skipped': len(result.skipped),
              'scope': 'Actual OS-signaled child processes; bounded recovery and diagnostic preservation only',
              'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
              'checks': ['SIGBUS then success: exactly one fresh-process retry',
                         'numerical exit 2: no retry', 'persistent SIGBUS: maximum two processes',
                         'SIGBUS then numerical failure: both attempts retained',
                         'first success: one process', 'failure diagnostic installed before retry',
                         'core dumps disabled in signaled children']}
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == '__main__':
    main()
