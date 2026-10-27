#!/usr/bin/env python3
"""Run retained contact/globalization fixtures with an explicitly chosen plugin."""
import argparse
import csv
import json
import os
from pathlib import Path
import re
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lmp', required=True, type=Path)
    parser.add_argument('--plugin', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--case', choices=('contacts', 'n500'), default='contacts')
    parser.add_argument('--backend', choices=('mesh', 'direct'), default='mesh')
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--threads', type=int, default=1)
    args = parser.parse_args()
    if args.steps < 1 or args.threads < 1:
        parser.error('steps and threads must be positive')
    lmp, plugin = args.lmp.resolve(), args.plugin.resolve()
    if not lmp.is_file() or not plugin.is_file():
        parser.error('lmp and plugin must name existing files')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    fixture_root = Path(__file__).resolve().parent / 'fixtures'
    cases = ['n500'] if args.case == 'n500' else ['ar1', 'ar2', 'ar5']
    results = []
    for case in cases:
        source, target = fixture_root / case, output / case
        target.mkdir()
        shutil.copyfile(source / 'particles.data', target / 'particles.data')
        text = (source / 'in.bridge').read_text()
        text = re.sub(r'^plugin load .*$', 'plugin load ${slurry_plugin}', text, flags=re.M)
        text = re.sub(r'^run \d+$', 'run {}'.format(args.steps), text, flags=re.M)
        text = re.sub(r'mobility_backend (mesh|direct)', 'mobility_backend '+args.backend, text)
        (target / 'in.bridge').write_text(text)
        environment = dict(os.environ, OMP_NUM_THREADS=str(args.threads),
                           MKL_NUM_THREADS=str(args.threads), SLURRY_BRIDGE_TRACE='1')
        with (target / 'console.log').open('w') as log:
            completed = subprocess.run(
                [str(lmp), '-var', 'slurry_plugin', '"'+str(plugin)+'"', '-in', 'in.bridge'],
                cwd=target, env=environment, stdout=log, stderr=subprocess.STDOUT)
        rows = []
        if (target / 'history.csv').is_file():
            with (target / 'history.csv').open() as handle:
                rows = list(csv.DictReader(handle))
        results.append(dict(case=case, returncode=completed.returncode,
            last_step=int(rows[-1]['step']) if rows else None,
            max_relative_residual=max((float(row['solver_relative_residual']) for row in rows), default=None),
            max_gap_consistency_m=max((float(row['contact_gap_consistency_m']) for row in rows), default=None),
            continuation_rescues=(target / 'console.log').read_text().count('phase=prox_retry')))
        (output / 'summary.json').write_text(json.dumps(results, indent=2)+'\n')
        print(json.dumps(results[-1]), flush=True)
    return int(any(item['returncode'] or item['last_step'] != args.steps for item in results))


if __name__ == '__main__':
    raise SystemExit(main())
