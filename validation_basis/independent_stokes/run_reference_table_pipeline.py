#!/usr/bin/env python3
"""Build a provisional table from explicitly selected independent references.

This reuses completed Stokes reports. It performs no Stokes solves, does not
qualify their accuracy, and never fills missing orientations with one pose.
Use a new output directory for every replacement reference or model revision.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('references', type=Path, nargs='+')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--generator', type=Path, required=True)
    p.add_argument('--checker', type=Path, required=True)
    p.add_argument('--layout', choices=('sparse', 'full-grid'), default='sparse')
    p.add_argument('--matched-model', choices=('constant-residual', 'dissipative-increment'), required=True)
    p.add_argument('--floor-nm', type=float, default=1.)
    p.add_argument('--boundary-limit', type=float, default=.01)
    p.add_argument('--gap-nodes', type=int, default=129)
    p.add_argument('--max-gap-nodes', type=int, default=256)
    p.add_argument('--interpolation-tolerance', type=float, default=1e-6)
    args = p.parse_args()
    output = args.output_dir.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        p.error('Use a new or empty output directory; previous runs are preserved')
    try:
        refs = [x.resolve(strict=True) for x in args.references]
        generator = args.generator.resolve(strict=True)
        checker = args.checker.resolve(strict=True)
    except OSError as exc:
        p.error(str(exc))
    importer = Path(__file__).with_name('prepare_reference_source.py')
    output.mkdir(parents=True, exist_ok=True)
    record = dict(schema='provisional_reference_table_pipeline_v1',
                  created_utc=datetime.now(timezone.utc).isoformat(),
                  status='running', workflow_completed=False,
                  physical_accuracy_qualified=False, new_stokes_solve_performed=False,
                  matched_model=args.matched_model, layout=args.layout,
                  input_reports=[dict(path=str(x), sha256=digest(x)) for x in refs],
                  executables={str(x):digest(x) for x in (generator, checker)},
                  python_sources={x.name:digest(x) for x in (Path(__file__), importer)},
                  stages=[], outputs={})
    summary = output/'pipeline.json'

    def save():
        temporary = output/('pipeline.json.tmp.'+str(os.getpid()))
        temporary.write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
        temporary.replace(summary)

    def run(stage, command):
        start = time.monotonic()
        with (output/(stage+'.stdout.log')).open('w') as stdout, (output/(stage+'.stderr.log')).open('w') as stderr:
            proc = subprocess.run([str(x) for x in command], stdout=stdout, stderr=stderr)
        record['stages'].append(dict(stage=stage, command=[str(x) for x in command],
                                    returncode=proc.returncode, seconds=time.monotonic()-start))
        save()
        if proc.returncode:
            raise RuntimeError(f'{stage} failed with exit {proc.returncode}; inspect retained logs')

    def verify_unchanged_inputs():
        for item in record['input_reports']:
            if digest(item['path']) != item['sha256']:
                raise ValueError('Reference report changed during the pipeline: '+item['path'])
        for path, expected in record['executables'].items():
            if digest(path) != expected:
                raise ValueError('Executable changed during the pipeline: '+path)
        for path in (Path(__file__), importer):
            if digest(path) != record['python_sources'][path.name]:
                raise ValueError('Python source changed during the pipeline: '+str(path))

    def read_record(name):
        result = json.loads((output/name).read_text())
        if not isinstance(result, dict) or result.get('physical_accuracy_qualified') is not False:
            raise ValueError('Missing explicit provisional status in '+name)
        return result

    source = output/'reference.source17'
    table = output/'reference.sft'
    try:
        save()
        run('import', [sys.executable, '-B', importer, *refs, '--output', source,
                       '--floor-nm', args.floor_nm, '--boundary-limit', args.boundary_limit,
                       '--layout', args.layout,
                       '--provisional-reason', 'User authorized current approximate references for end-to-end development before final mesh refinement.'])
        verify_unchanged_inputs()
        imported = read_record('reference.source17.manifest.json')
        expected_inputs = sorted((x['path'], x['sha256']) for x in record['input_reports'])
        received_inputs = sorted((x['path'], x['source_json_sha256']) for x in imported['sources'])
        if expected_inputs != received_inputs or imported['source_text_sha256'] != digest(source):
            raise ValueError('Imported source identity does not match the explicitly selected inputs')
        run('generate', [generator, '--source', source, '--output', table,
                         '--report', output/'generation.json', '--layout', args.layout,
                         '--matched-model', args.matched_model, '--gap-nodes', args.gap_nodes,
                         '--max-gap-nodes', args.max_gap_nodes,
                         '--interpolation-tolerance', args.interpolation_tolerance])
        run('runtime', [checker, '--source', source, '--table', table,
                        '--matched-model', args.matched_model, '--layout', args.layout, '--output', output/'runtime.json'])
        verify_unchanged_inputs()
        generation = read_record('generation.json')
        runtime = read_record('runtime.json')
        expected_model = {'constant-residual':'JANOSCHEK_MATCHED_HARDFLOOR_FULL17_V4',
                          'dissipative-increment':'JANOSCHEK_DISSIPATIVE_INCREMENT_FULL17_V1'}[args.matched_model]
        if (generation.get('status') != 'provisional_numerical_transport_passed' or
                runtime.get('pipeline_algebra_passed') is not True or
                generation.get('model_revision') != expected_model or
                runtime.get('model_revision') != expected_model or
                len(generation.get('numerical_source_fingerprint', '')) != 64 or
                generation.get('numerical_source_fingerprint') != runtime.get('numerical_source_fingerprint') or
                generation.get('source_fnv1a64') != runtime.get('source_fnv1a64') or
                imported['source_text_sha256'] != digest(source)):
            raise ValueError('Generation/runtime results do not form a passed, matching provisional pipeline')
        record['input_identity_verified_before_and_after'] = True
        record.update(status='complete', workflow_completed=True)
    except (ValueError, OSError, RuntimeError, KeyError, TypeError) as exc:
        record.update(status='failed', failure=str(exc))
    finally:
        for x in output.iterdir():
            if x.is_file() and x != summary:
                record['outputs'][x.name] = dict(bytes=x.stat().st_size, sha256=digest(x))
        save()
    print(json.dumps(dict(output=str(summary), workflow_completed=record['workflow_completed'],
                          physical_accuracy_qualified=False, failure=record.get('failure'))))
    return 0 if record['workflow_completed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
