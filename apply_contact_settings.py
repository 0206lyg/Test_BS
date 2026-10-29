#!/usr/bin/env python3
"""Apply the calibrated contact settings while retaining run-specific choices."""
import argparse
import math
from pathlib import Path
import re
import shutil


def update(text):
    def number(key):
        matches = re.findall(r'^' + key + r'=([^#\n]+)', text, re.M)
        if len(matches) != 1:
            raise ValueError('Expected exactly one numeric assignment for ' + key)
        return float(matches[0].strip())

    old_dt = number('TIMESTEP')
    old_steps = number('STEPS')
    old_interval = number('OUTPUT_INTERVAL')
    if not all(math.isfinite(v) and v > 0 for v in (old_dt, old_steps, old_interval)):
        raise ValueError('TIMESTEP, STEPS and OUTPUT_INTERVAL must be positive numbers')
    new_dt = 2.5e-6
    settings = {
        'NORMAL_STIFFNESS': '0.3',
        'TANGENTIAL_STIFFNESS': '0.24',
        'ROLLING_STIFFNESS': '0.24',
        'TIMESTEP': '2.5e-6',
        'STEPS': str(max(1, math.ceil(old_steps * old_dt / new_dt - 1e-9))),
        'OUTPUT_INTERVAL': str(max(1, int(round(old_interval * old_dt / new_dt)))),
    }
    comments = {
        'TIMESTEP': '# s; explicit Heun predictor-corrector',
        'NORMAL_STIFFNESS': '# N/m; numerical penalty, check overlap and stiffness convergence',
        'TANGENTIAL_STIFFNESS': '# N/m',
        'ROLLING_STIFFNESS': '# N/m; rolling quasi-force spring',
    }
    for key, value in settings.items():
        # Validate all replacements before writing anything to disk.
        number(key)
        text, count = re.subn(
            r'^' + key + r'=[^\n]*',
            key + '=' + value + ('  ' + comments[key] if key in comments else ''),
            text, flags=re.M)
        if count != 1:
            raise ValueError('Ambiguous assignment for ' + key)
    return text, settings, old_steps * old_dt, old_interval * old_dt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--file', type=Path,
                        default=Path(__file__).resolve().parent / 'run_slurry_cpu.sbatch')
    args = parser.parse_args()
    path = args.file.resolve()
    original = path.read_text()
    revised, settings, duration, interval = update(original)
    backup = path.with_name(path.name + '.before_contact_update')
    if revised != original:
        if not backup.exists():
            shutil.copy2(path, backup)
        path.write_text(revised)
    print('Applied:', ', '.join(k + '=' + v for k, v in settings.items()))
    print('Run-specific AR, PHI, COUNT, friction, paths and Slurm resources retained.')
    print('Duration: {:.12g} -> {:.12g} s; output interval: {:.12g} -> {:.12g} s'.format(
        duration, int(settings['STEPS']) * float(settings['TIMESTEP']),
        interval, int(settings['OUTPUT_INTERVAL']) * float(settings['TIMESTEP'])))


if __name__ == '__main__':
    main()
