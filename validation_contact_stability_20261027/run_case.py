#!/usr/bin/env python3
"""Replay a fixed slurry fixture and summarize numerical/physical diagnostics."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import time


def summarize(directory):
    path = Path(directory)
    with (path / 'history.csv').open() as handle:
        rows = list(csv.DictReader(handle))
    output = {'rows': len(rows), 'last_step': int(rows[-1]['step']) if rows else None}
    keys = ['time_s', 'strain', 'eta_relative', 'sigma_xz_Pa', 'max_overlap_m',
            'actual_endpoint_overlap_m', 'contacts', 'rolling_yield_contacts',
            'solver_relative_residual', 'force_residual', 'mobility_residual',
            'linear_iterations', 'newton_iterations', 'geometry_iterations',
            'continuation_stages', 'continuation_retries', 'profile_total_s',
            'hydro_dissipation_W', 'power_balance_error_W', 'stress_power_W']
    for key in keys:
        if not rows or key not in rows[0]:
            continue
        values = [float(row[key]) for row in rows]
        finite = [value for value in values if math.isfinite(value)]
        output[key] = dict(min=min(finite) if finite else None,
                           max=max(finite) if finite else None,
                           last=values[-1], nonfinite=len(values)-len(finite))
    if rows:
        output['overlap_radius_fraction'] = max(
            float(row.get('max_overlap_m', '0')) for row in rows) / 4e-6
        output['endpoint_overlap_radius_fraction'] = max(
            float(row.get('actual_endpoint_overlap_m', '0')) for row in rows) / 4e-6
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture', required=True, type=Path)
    parser.add_argument('--plugin', required=True, type=Path)
    parser.add_argument('--lmp', type=Path, required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--dt', type=float, default=2.5e-6)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--dump-interval', type=int, default=100)
    parser.add_argument('--backend', choices=('mesh','direct'), default='mesh')
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(args.fixture / 'particles.data', out / 'particles.data')
    text = (args.fixture / 'in.bridge').read_text()
    text = re.sub(r'^plugin load .*$', 'plugin load ${slurry_plugin}', text, flags=re.M)
    text = re.sub(r'^run \d+$', f'run {args.steps}', text, flags=re.M)
    text = re.sub(r'^timestep .*$', f'timestep {args.dt:.17g}', text, flags=re.M)
    text = re.sub(r'mobility_backend (mesh|direct)', 'mobility_backend '+args.backend, text)
    text = re.sub(r'history history.csv \d+ ', 'history history.csv 1 ', text)
    text = re.sub(r'^thermo \d+$', 'thermo 1', text, flags=re.M)
    text = re.sub(r'dump trajectory all custom \d+', f'dump trajectory all custom {args.dump_interval}', text)
    (out / 'in.bridge').write_text(text)
    env = dict(os.environ, OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads))
    env.pop('SLURRY_BRIDGE_TRACE', None)
    metadata = dict(fixture=str(args.fixture.resolve()), plugin=str(args.plugin.resolve()),
                    plugin_sha256=hashlib.sha256(args.plugin.read_bytes()).hexdigest(),
                    initial_particles_sha256=hashlib.sha256((out/'particles.data').read_bytes()).hexdigest(),
                    steps=args.steps, dt=args.dt, threads=args.threads, backend=args.backend)
    (out / 'run_metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    start = time.monotonic()
    with (out/'console.log').open('w') as log:
        process = subprocess.run([str(args.lmp), '-var', 'slurry_plugin', '"'+str(args.plugin.resolve())+'"',
                                  '-in', 'in.bridge'], cwd=out, env=env, stdout=log, stderr=subprocess.STDOUT)
    summary = summarize(out) if (out/'history.csv').exists() else {}
    summary.update(metadata)
    summary.update(returncode=process.returncode, wall_seconds=time.monotonic()-start)
    (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary), flush=True)
    return process.returncode


if __name__ == '__main__':
    raise SystemExit(main())
