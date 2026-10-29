#!/usr/bin/env python3
"""Algebra-only full17/153-entry encoding check of a completed saved run."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from compare_reference import (energy_scaled_reciprocity, helper_export, matrix,
                               response_metrics, symmetric)


def check(reference_path, helper_path, tolerance):
    payload = reference_path.read_bytes()
    source = json.loads(payload)
    if source.get('status') != 'complete':
        raise ValueError('Wait for status=complete; never consume a live partial solve')
    raw = matrix(source['raw_full17'])
    total = symmetric(raw)
    if 'symmetric_full17' in source:
        stored = matrix(source['symmetric_full17'])
        if np.max(abs(stored-total)) > 1e-10*max(1.,np.max(abs(total))):
            raise ValueError('Stored symmetric matrix disagrees with raw symmetrization')
    gap = float(source['h_over_c'])
    minimum = float(np.linalg.eigvalsh(total)[0])
    if minimum <= 0:
        raise ValueError('Reference symmetric total is not positive definite; no clipping permitted')
    exported = helper_export(helper_path,gap,total)
    for field in ('aspect_ratio','theta_i','theta_j','phi'):
        value = float(source[field])
        if not np.isfinite(value) or abs(value-exported[field]) > 1e-13:
            raise ValueError('Saved source is not the same recorded AR20 geometry as the encoder')
    decoded = matrix(exported['decoded_total17'])
    metrics = response_metrics(decoded,total,gap)
    passed = (exported['encoded_values']==153 and
              metrics['full17_max_relative_energy_error'] <= tolerance and
              all(item['energy_input_scaled_operator_relative'] <= tolerance
                  for item in metrics['blocks'].values()))
    return dict(schema='completed_reference_full17_encoding_algebra_v1',
        physical_accuracy_qualified=False,algebra_gate_passed=bool(passed),
        scope='One completed raw17 matrix, symmetrized after recording reciprocity; actual153-value production table encoding and decoding. No spatial interpolation, physical-source qualification, or full table generation.',
        reference=str(reference_path.resolve()),reference_sha256=hashlib.sha256(payload).hexdigest(),
        source_status=source['status'],source_library_sha256=source.get('library_sha256'),
        source_discretization_sha256=source.get('discretization_sha256'),
        source_claimed_physical_qualified=source.get('physical_accuracy_qualified',False),
        helper=str(helper_path.resolve()),helper_sha256=hashlib.sha256(helper_path.read_bytes()).hexdigest(),
        convention_id=exported['convention_id'],h_over_c=gap,aspect_ratio=source['aspect_ratio'],
        independent_stored_values=exported['encoded_values'],relative_energy_and_block_tolerance=tolerance,
        raw_reciprocity_frobenius=float(np.linalg.norm(raw-raw.T)/np.linalg.norm(raw)),
        raw_reciprocity_energy_scaled=energy_scaled_reciprocity(raw,total),
        symmetric_total_minimum_eigenvalue=minimum,
        decoded_total_minimum_eigenvalue=float(np.linalg.eigvalsh(symmetric(decoded))[0]),
        roundtrip=metrics,source_boundary_validation=source.get('validation',{}),
        source_moment_validation=source.get('moments',{}),
        qualification_note='Successful storage preserves this numerical matrix only. Direct no-slip, F/T/S spatial refinement, quadrature refinement and matching overlap remain separate gates.')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--operator-helper',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--tolerance',type=float,default=1e-9)
    args=parser.parse_args()
    if args.reference.resolve()==args.output.resolve():
        parser.error('Output must not replace the original source report')
    if not np.isfinite(args.tolerance) or args.tolerance <= 0:
        parser.error('Positive finite tolerance required')
    report=check(args.reference,args.operator_helper,args.tolerance)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(args.output),algebra_gate_passed=report['algebra_gate_passed'],
                         energy_error=report['roundtrip']['full17_max_relative_energy_error'],
                         physical_accuracy_qualified=False)))
    raise SystemExit(0 if report['algebra_gate_passed'] else 2)
