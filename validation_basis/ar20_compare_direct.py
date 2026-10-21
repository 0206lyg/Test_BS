#!/usr/bin/env python3
"""Compare two direct results for the exact recorded AR20 pose; no table runs."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.linalg import eigvalsh


def matrices(record):
    full = np.asarray(record['symmetric_full17'])
    velocity, cross, strain = full[:12, :12], full[:12, 12:], full[12:, 12:]
    return {'full17': full, 'velocity12': velocity, 'clamped_strain5': strain,
            'force_torque_free_strain5': strain-cross.T@np.linalg.solve(velocity, cross),
            'reduced11': np.asarray(record['reduced11'])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('left');parser.add_argument('right');parser.add_argument('--output', required=True)
    args = parser.parse_args();left = json.loads(Path(args.left).read_text());right = json.loads(Path(args.right).read_text())
    for key in ['aspect_ratio', 'theta_i', 'theta_j', 'phi', 'h_over_c']:
        if left[key] != right[key]:
            raise ValueError('Different geometries: '+key)
    lm, rm = matrices(left), matrices(right)
    errors = {key: float(np.max(np.abs(eigvalsh(lm[key], rm[key])-1))) for key in lm}
    lraw, rraw = np.asarray(left['raw_full17']), np.asarray(right['raw_full17'])
    changes = {}
    for name, rows in [('force', [0,1,2,6,7,8]), ('torque_over_a', [3,4,5,9,10,11]), ('stresslet_over_a', [12,13,14,15,16])]:
        delta = lraw[rows]-rraw[rows]
        changes[name] = {'relative_frobenius': float(np.linalg.norm(delta)/np.linalg.norm(rraw[rows])),
                         'max_column_change_over_largest_reference_column': float(np.linalg.norm(delta, axis=0).max()/np.linalg.norm(rraw[rows], axis=0).max())}
    result = {'schema': 'ar20_direct_refinement_v1', 'left': str(Path(args.left).resolve()),
              'right': str(Path(args.right).resolve()), 'h_over_c': left['h_over_c'],
              'relative_energy_changes': errors, 'raw_moment_response_changes': changes, 'limit': .01,
              'all_energy_changes_below_one_percent': max(errors.values()) <= .01,
              'scope': 'One refinement in the same MFS family; not independent accuracy certification'}
    Path(args.output).write_text(json.dumps(result, indent=2)+'\n');print(json.dumps(result))


if __name__ == '__main__':
    main()
