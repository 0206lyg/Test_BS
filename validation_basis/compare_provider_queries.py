#!/usr/bin/env python3
"""Optional offline comparison of bound direct-source and matched queries.

NumPy/SciPy are used only by this audit. This is numerical source convergence,
not an independent physical reference and not fast-table interpolation testing.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import numpy as np
from scipy.linalg import eigvalsh

REFERENCE_LIMIT = .005
PRODUCT_LIMIT = .05


def load_query(path, mode):
    path = Path(path)
    raw = path.read_bytes()
    query = json.loads(raw)
    if not (query.get('query_completed') is True and query.get('inputs_unchanged_during_run') is True
            and query.get('evaluated_path') == mode and query.get('positive_total') is True):
        raise ValueError('Incomplete or wrong-path query: '+str(path))
    fingerprint = query.get('source_fingerprint', '')
    if len(fingerprint) != 64 or query.get('compiled_source_fingerprint') != fingerprint:
        raise ValueError('Unbound source fingerprint: '+str(path))
    if mode == 'matched_provider' and query.get('reference_only'):
        raise ValueError('Reference-only overlay cannot stand in for the actual product')
    for key in ('basis_sha256', 'executable_sha256'):
        value = query.get(key, '')
        if len(value) != 64 or any(x not in '0123456789abcdef' for x in value):
            raise ValueError('Missing input hash '+key+': '+str(path))
    matrix = np.asarray(query.get('reduced11'), float)
    if matrix.shape != (11, 11) or not np.all(np.isfinite(matrix)):
        raise ValueError('Invalid total11 matrix: '+str(path))
    if np.linalg.norm(matrix-matrix.T) > 1e-11*np.linalg.norm(matrix):
        raise ValueError('Nonreciprocal total11 matrix: '+str(path))
    np.linalg.cholesky(matrix)
    return query, matrix, dict(filename=path.name, sha256=hashlib.sha256(raw).hexdigest())


def check_geometry(query, reference):
    for key in ('aspect_ratio', 'h_over_c', 'epsilon_over_c'):
        if query.get(key) != reference.get(key):
            raise ValueError('Query geometry/regularization mismatch: '+key)
    axes, ref_axes = np.asarray(query.get('axes')), np.asarray(reference.get('axes'))
    if axes.shape != (2, 3) or not np.allclose(axes, ref_axes, rtol=0, atol=2e-14):
        raise ValueError('Query axes mismatch')


def free_strain(matrix):
    result = matrix[6:, 6:] - matrix[6:, :6] @ np.linalg.solve(matrix[:6, :6], matrix[:6, 6:])
    return .5*(result+result.T)


def compare(prediction, reference, limit):
    energy = float(np.max(np.abs(eigvalsh(prediction, reference)-1)))
    strain = float(np.max(np.abs(eigvalsh(free_strain(prediction), free_strain(reference))-1)))
    return dict(all11_error=energy, free_strain_error=strain,
                normal_entry_relative_error=float(prediction[2, 2]/reference[2, 2]-1),
                limit=limit, passed=bool(energy <= limit and strain <= limit))


def evaluate(args):
    result = dict(schema_version=1, gate='refined_heldout_bridge_consistency', passed=False,
                  independent_physical_reference=False, reference_qualified=False,
                  product_accuracy_tested=False, reference_limit=REFERENCE_LIMIT,
                  product_limit=PRODUCT_LIMIT,
                  scope='One pose. Direct-source order and quadrature convergence, followed by actual matched-provider accuracy. Not an independent physical reference, interpolation test, or bulk validation.')
    inputs = {}
    records = {}
    matrices = {}
    try:
        for slot in ('coarse', 'fine', 'quadrature'):
            records[slot], matrices[slot], inputs[slot] = load_query(getattr(args, slot), 'direct_basis')
        anchor = records['quadrature']
        for query in records.values():
            check_geometry(query, anchor)
        if not (records['fine'].get('source_degree', 0) > records['coarse'].get('source_degree', 0)
                and records['fine'].get('source_degree') == anchor.get('source_degree')
                and anchor.get('source_nphi', 0) > records['fine'].get('source_nphi', 0)):
            raise ValueError('Expected increasing source degree and same-degree quadrature refinement')
        rings = [records[slot].get('source_rings', 0) for slot in ('coarse', 'fine', 'quadrature')]
        if not (rings[0] > 0 and rings[0] == rings[1] == rings[2]):
            raise ValueError('Source-order refinement also changed the source-ring count')
        coefficient_hash = records['fine'].get('basis_coefficients_sha256', '')
        if (len(coefficient_hash) != 64 or any(x not in '0123456789abcdef' for x in coefficient_hash)
                or coefficient_hash != anchor.get('basis_coefficients_sha256')):
            raise ValueError('Quadrature refinement changed serialized coefficients beyond the runtime phi count')
        result.update(aspect_ratio=anchor['aspect_ratio'], h_over_c=anchor['h_over_c'], axes=anchor['axes'])
        result['order_refinement'] = compare(matrices['coarse'], matrices['quadrature'], REFERENCE_LIMIT)
        result['quadrature_refinement'] = compare(matrices['fine'], matrices['quadrature'], REFERENCE_LIMIT)
        result['reference_qualified'] = result['order_refinement']['passed'] and result['quadrature_refinement']['passed']
        if not result['reference_qualified']:
            result['not_run'] = ['actual_matched_provider_comparison']
        elif args.product is None:
            result['not_run'] = ['actual_matched_provider_query_and_comparison']
        else:
            records['product'], matrices['product'], inputs['product'] = load_query(args.product, 'matched_provider')
            check_geometry(records['product'], anchor)
            result['product_comparison'] = compare(matrices['product'], matrices['quadrature'], PRODUCT_LIMIT)
            result['product_accuracy_tested'] = True
            result['passed'] = result['product_comparison']['passed']
            result['not_run'] = []
    except Exception as error:
        result['error'] = str(error)
    result['inputs'] = inputs
    result['queries'] = records
    result['comparison_helper_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('coarse', 'fine', 'quadrature'):
        parser.add_argument('--'+key, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--product', type=Path)
    parser.add_argument('--replay-report', type=Path,
                        help='Recompare the bound query objects embedded in a compact report; no new physical calculation')
    args = parser.parse_args()
    if args.replay_report is not None:
        if any(getattr(args, key) is not None for key in ('coarse', 'fine', 'quadrature', 'product')):
            parser.error('--replay-report cannot be combined with individual input queries')
        original_bytes = args.replay_report.read_bytes()
        original = json.loads(original_bytes)
        records = original.get('queries', {})
        if any(key not in records for key in ('coarse', 'fine', 'quadrature')):
            parser.error('Replay report does not contain the three required reference queries')
        with tempfile.TemporaryDirectory(prefix='slurry_query_replay_') as directory:
            for key in ('coarse', 'fine', 'quadrature', 'product'):
                if key in records:
                    path = Path(directory) / (key+'.json')
                    path.write_text(json.dumps(records[key], sort_keys=True, allow_nan=False)+'\n')
                    setattr(args, key, path)
            result = evaluate(args)
        result['replay'] = dict(original_report_filename=args.replay_report.name,
                               original_report_sha256=hashlib.sha256(original_bytes).hexdigest(),
                               original_input_hashes=original.get('inputs', {}),
                               new_physical_calculation=False)
    else:
        if any(getattr(args, key) is None for key in ('coarse', 'fine', 'quadrature')):
            parser.error('Supply --coarse, --fine, --quadrature or --replay-report')
        result = evaluate(args)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+'\n')
    print(json.dumps({key:result.get(key) for key in ('passed', 'reference_qualified', 'product_accuracy_tested', 'order_refinement', 'quadrature_refinement', 'product_comparison', 'error')}, sort_keys=True))
    return 0 if result['passed'] or (result['reference_qualified'] and args.product is None) else 2


if __name__ == '__main__':
    sys.exit(main())
