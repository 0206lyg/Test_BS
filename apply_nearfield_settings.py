#!/usr/bin/env python3
"""Select near-field hydrodynamics and stronger contact springs in an existing run script."""
import argparse
import math
from pathlib import Path
import re
import shutil


def update(text, normal=30., tangential=24., rolling=24.):
    values = (normal, tangential, rolling)
    if not all(math.isfinite(x) and x > 0 for x in values):
        raise ValueError('Contact stiffnesses must be finite and positive')
    settings = {'MOBILITY_BACKEND': 'nearfield',
                'NORMAL_STIFFNESS': format(normal, '.12g'),
                'TANGENTIAL_STIFFNESS': format(tangential, '.12g'),
                'ROLLING_STIFFNESS': format(rolling, '.12g')}
    for key, value in settings.items():
        if len(re.findall(r'^' + key + r'=', text, re.M)) != 1:
            raise ValueError('Expected exactly one assignment for ' + key)
        comment = ('# isolated spheroid drag + local lubrication; no FFT'
                   if key == 'MOBILITY_BACKEND' else '# N/m')
        text = re.sub(r'^' + key + r'=[^\n]*', key + '=' + value + '  ' + comment,
                      text, flags=re.M)
    return text, settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--file', type=Path,
                        default=Path(__file__).resolve().parent / 'run_slurry_cpu.sbatch')
    parser.add_argument('--normal-stiffness', type=float, default=30.)
    parser.add_argument('--tangential-stiffness', type=float, default=24.)
    parser.add_argument('--rolling-stiffness', type=float, default=24.)
    args = parser.parse_args()
    path = args.file.resolve()
    original = path.read_text()
    revised, settings = update(original, args.normal_stiffness,
                               args.tangential_stiffness, args.rolling_stiffness)
    if revised != original:
        backup = path.with_name(path.name + '.before_nearfield_update')
        if not backup.exists():
            shutil.copy2(path, backup)
        path.write_text(revised)
    print('Applied:', ', '.join(k + '=' + v for k, v in settings.items()))
    print('Timestep, duration, output cadence, AR, count, paths and resources preserved.')


if __name__ == '__main__':
    main()
