#!/usr/bin/env python3
"""Rebuild scratch-only timers and replay the two existing frozen equations."""
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import os
import shutil
import subprocess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True,
                        help='New output/build directory; must not exist')
    parser.add_argument('--compiler', default='c++')
    parser.add_argument('--build-only', action='store_true')
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    repo = here.parents[1]
    frozen = here.parent / 'frozen'
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    src = output / 'src'
    src.mkdir()
    (output / 'vendor').symlink_to(repo / 'lammps_slurry' / 'vendor', target_is_directory=True)
    for item in json.loads((here / 'instrumentation.json').read_text()):
        source = repo / 'lammps_slurry' / 'src' / item['name']
        if sha(source) != item['original_sha256']:
            raise RuntimeError('Production header differs from profiled version: ' + item['name'])
        text = source.read_text()
        for change in item['replacements']:
            if text.count(change['old']) != 1:
                raise RuntimeError('Instrumentation marker is not unique: ' + item['name'])
            text = text.replace(change['old'], change['new'], 1)
        destination = src / item['name']
        destination.write_text(text)
        if sha(destination) != item['instrumented_sha256']:
            raise RuntimeError('Instrumented header checksum mismatch')
    shutil.copy2(here / 'profile_cpu.h', src / 'profile_cpu.h')
    executable = output / 'replay_cost'
    command = [args.compiler, '-O3', '-DNDEBUG', '-std=c++17', '-DEIGEN_DONT_PARALLELIZE',
               '-DSLURRY_HAVE_EIGEN=1', '-I' + str(src),
               '-I' + str(repo / 'lammps_slurry' / 'src'), '-I' + str(frozen),
               '-isystem', str(repo / 'lammps_slurry' / 'vendor' / 'eigen3'),
               str(here / 'replay_cost.cpp'), '-o', str(executable)]
    (output / 'compile_command.json').write_text(json.dumps(command, indent=2) + '\n')
    subprocess.run(command, check=True)
    if args.build_only:
        return
    environment = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    for step in (595, 683):
        snapshot = output / ('step' + str(step) + '.snapshot')
        snapshot.write_bytes(gzip.decompress((frozen / f'ar5_phi020_step{step}_inner.snapshot.gz').read_bytes()))
        with (output / f'step{step}.jsonl').open('w') as log:
            subprocess.run([str(executable), str(snapshot),
                            str(frozen / f'ar5_phi020_step{step}_caps.txt'), f'step{step}'],
                           stdout=log, env=environment, check=True)
    print('Profiles written to', output)


if __name__ == '__main__':
    main()
