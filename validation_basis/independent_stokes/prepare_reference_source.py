#!/usr/bin/env python3
"""Import explicitly selected independent full17 reports as provisional sources.

Input matrices must already use slurry_full17_c_eta_1_v1: c=eta=1,
a=aspect_ratio; inputs [Ui,a*Oi,Uj,a*Oj,a*E5] and outputs
[Fi,Ti/a,Fj,Tj/a,intrinsic_S5/a], with nonaffine rigid velocities.
This program performs no dimensional matrix conversion, rotation,
symmetrization, eigenvalue clipping, Stokes solve, or physical qualification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np

from compare_reference import operator_identity

CONVENTION = 'slurry_full17_c_eta_1_v1'
MAGIC = 'SLURRY_REFERENCE_SOURCE17_V1'
MODEL = 'JANOSCHEK_MATCHED_HARDFLOOR_FULL17_V4'
METHODS = {'physical_traction', 'reciprocal_work', 'combined_layer_farfield_moments'}
PANEL_SCHEMAS = {'independent_ar20_panel_bie_v1', 'independent_spheroid_panel_bie_v1'}
STRICT_BOUNDARY_LIMIT = .001
LINEAR_LIMIT = 1e-8
RECIPROCITY_LIMIT = .001
SURFACE_LIMIT = .001
ISOLATED_MOMENT_LIMIT = .002


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha_string(value, label):
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdefABCDEF' for c in value):
        raise ValueError(f'{label}: a SHA256 digest is required')
    return value.lower()


def number(value, label, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{label}: finite numeric value required')
    value = float(value)
    if positive and value <= 0:
        raise ValueError(f'{label}: positive value required')
    return value


def error_value(value, label, limit):
    value = number(value, label)
    if value < 0 or value > limit:
        raise ValueError(f'{label}: measured {value:.17g} exceeds allowed {limit:.17g}')
    return value


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    def invalid_constant(value):
        raise ValueError(f'{path}: nonfinite JSON constant {value}')
    data = json.loads(Path(path).read_text(), parse_constant=invalid_constant)
    require(isinstance(data, dict), f'{path}: JSON object required')
    # Reject nonfinite metadata too, including a numeric exponent that
    # overflows even though it is legal JSON syntax.
    def check(value):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f'{path}: nonfinite numeric metadata')
        if isinstance(value, dict):
            for item in value.values():
                check(item)
        elif isinstance(value, list):
            for item in value:
                check(item)
    check(data)
    return data


def matrix17(value, label):
    require(isinstance(value, list) and len(value) == 17 and
            all(isinstance(row, list) and len(row) == 17 for row in value),
            f'{label}: a17 by17 matrix is required')
    for row in value:
        for entry in row:
            number(entry, label)
    return np.asarray(value, dtype=float)


def canonical_phi(phi):
    phi = number(phi, 'phi')
    return phi if -math.pi <= phi < math.pi else (phi+math.pi) % (2*math.pi)-math.pi


def metadata(data, c_nm_if_missing=None):
    geometry = data.get('geometry_metadata', data)
    require(isinstance(geometry, dict), 'geometry_metadata must be an object')
    require(data.get('convention_id') == CONVENTION, 'Explicit normalized full17 convention is missing or different')
    ar = number(geometry.get('aspect_ratio'), 'aspect_ratio', positive=True)
    require(ar >= 1, 'The oblate reference source requires aspect_ratio>=1')
    ti, tj = (number(geometry.get(key), key) for key in ('theta_i', 'theta_j'))
    require(0 <= ti <= math.pi/2 and 0 <= tj <= math.pi/2,
            'Canonical table theta_i and theta_j must lie in[0,pi/2]; noncanonical director angles require a corresponding matrix transformation outside this importer')
    phi = canonical_phi(geometry.get('phi'))
    h = number(geometry.get('h_over_c'), 'h_over_c', positive=True)
    if 'length_unit_nm' in geometry:
        c_nm = number(geometry['length_unit_nm'], 'length_unit_nm', positive=True)
        origin = 'report.length_unit_nm'
        if c_nm_if_missing is not None:
            require(c_nm == number(c_nm_if_missing, '--c-nm', positive=True), '--c-nm cannot override existing report length units')
    else:
        require(c_nm_if_missing is not None, 'length_unit_nm is missing; supply explicit --c-nm without converting the matrix')
        c_nm = number(c_nm_if_missing, '--c-nm', positive=True)
        origin = 'explicit_cli_c_nm_for_missing_length_unit_nm'
    expected = {'c_nm': c_nm, 'a_nm': ar*c_nm, 'h_nm': h*c_nm,
                'effective_gap_nm': h*c_nm, 'normalized_c': 1.,
                'normalized_a': ar, 'normalized_viscosity': 1.}
    for key, expected_value in expected.items():
        for container in (geometry, data) if geometry is not data else (data,):
            if key in container:
                actual = number(container[key], key)
                require(math.isclose(actual, expected_value, rel_tol=1e-12, abs_tol=1e-14),
                        f'{key}: incompatible dimensions or normalization')
    for container in (geometry, data) if geometry is not data else (data,):
        if 'coordinate_units' in container:
            require(container['coordinate_units'] == 'c=1', 'coordinate_units must be c=1; dimensional matrices are not converted')
        if 'matrix_units' in container:
            require(container['matrix_units'] == CONVENTION, 'matrix_units conflicts with the normalized full17 convention')
    if geometry is not data:
        for key in ('aspect_ratio', 'theta_i', 'theta_j', 'phi', 'h_over_c', 'length_unit_nm'):
            if key in data and key in geometry:
                left, right = number(data[key], key), number(geometry[key], key)
                difference = abs(canonical_phi(left-right)) if key == 'phi' else abs(left-right)
                require(difference <= 1e-13, f'{key}: conflicting top-level and geometry metadata')
    return {'aspect_ratio': ar, 'theta_i': ti, 'theta_j': tj, 'phi': phi,
            'h_over_c': h, 'c_nm': c_nm, 'c_m': c_nm*1e-9,
            'length_unit_origin': origin, 'original_phi': geometry['phi']}


def resolve_recorded(recorded, parent_path, adjacent):
    require(isinstance(recorded, str) and bool(recorded), 'Recorded evidence path must be nonempty')
    path = Path(recorded)
    if not path.is_absolute():
        path = parent_path.parent/path
    return path if path.exists() else adjacent


def checkpoint(data, parent_path):
    supplied = 'solution_file' in data or 'solution_sha256' in data
    if not supplied:
        require(data.get('schema') not in PANEL_SCHEMAS, 'Panel BIE source must provide its solution checkpoint')
        return None
    expected = sha_string(data.get('solution_sha256'), 'solution_sha256')
    path = resolve_recorded(data.get('solution_file'), parent_path, parent_path.with_suffix('.solution.npz'))
    require(digest(path) == expected, 'Solution checkpoint SHA256 mismatch; existing wrong-hash files are never bypassed')
    return {'recorded_path': data['solution_file'], 'resolved_path': str(path.resolve()),
            'sha256': expected, 'verified': True}


def base_identity(data):
    require(data.get('status') == 'complete', 'Source process must be complete')
    require(data.get('independent_reference') is True and data.get('shares_production_solver') is False,
            'An independent reference that does not share the production solver must be declared')
    require('synthetic' not in data or data['synthetic'] is False,
            'Synthetic diagnostics or ambiguous synthetic declarations cannot become reference sources')
    # Keep the old exact-AR20 qualification comparator unchanged.  The generic
    # panel driver uses the identical numerical identity contract; only its
    # schema label differs.  Do not modify the original source metadata.
    identity_input = (dict(data, schema='independent_ar20_panel_bie_v1')
                      if data.get('schema') in PANEL_SCHEMAS else data)
    identity, failures = operator_identity(identity_input)
    require(identity is not None and not failures, 'Invalid numerical operator provenance: '+','.join(failures))
    return identity


def boundary_checks(data, limit, *, require_gap):
    validation = data.get('validation')
    require(isinstance(validation, dict), 'Source validation object is required')
    require(validation.get('off_grid_boundary') is True and validation.get('boundary_evaluation_valid') is True,
            'Independent off-grid boundary evaluation must be explicitly valid')
    count = validation.get('boundary_sample_count')
    require(type(count) is int and count > 0, 'Positive boundary_sample_count required')
    result = {
        'boundary_max_scaled_error': error_value(validation.get('boundary_max_scaled_error'), 'boundary_max_scaled_error', limit),
        'linear_relative_residual': error_value(validation.get('linear_relative_residual', data.get('linear_residual')), 'linear_relative_residual', LINEAR_LIMIT),
        'exact_surface_position_max_error_over_gap': error_value(validation.get('exact_surface_position_max_error_over_gap'), 'exact_surface_position_max_error_over_gap', SURFACE_LIMIT),
        'boundary_sample_count': count,
    }
    if require_gap:
        count = validation.get('gap_patch_sample_count')
        require(type(count) is int and count > 0, 'Positive gap_patch_sample_count required')
        result['gap_patch_sample_count'] = count
        result['gap_patch_max_scaled_error'] = error_value(validation.get('gap_patch_max_scaled_error'), 'gap_patch_max_scaled_error', limit)
    if 'linear_residual' in data:
        error_value(data['linear_residual'], 'reported linear_residual', LINEAR_LIMIT)
    return result


def reciprocal_measure(raw, *, positive):
    norm = np.linalg.norm(raw)
    require(math.isfinite(float(norm)) and norm > 0, 'Raw17 matrix must have a finite nonzero norm')
    reciprocity = float(np.linalg.norm(raw-raw.T)/norm)
    error_value(reciprocity, 'recomputed raw reciprocity', RECIPROCITY_LIMIT)
    result = {'raw_reciprocity': reciprocity}
    if positive:
        symmetric = (raw+raw.T)/2
        eigenvalues = np.linalg.eigvalsh(symmetric)
        require(eigenvalues[0] > 0, 'Symmetric part of raw17 must be positive definite')
        lower = np.linalg.cholesky(symmetric)
        normalized = np.linalg.solve(lower, raw-raw.T)
        normalized = np.linalg.solve(lower, normalized.T).T
        energy_reciprocity = float(np.linalg.norm(normalized, ord=2))
        error_value(energy_reciprocity, 'energy-scaled raw reciprocity', RECIPROCITY_LIMIT)
        result.update(minimum_symmetric_eigenvalue=float(eigenvalues[0]),
                      maximum_symmetric_eigenvalue=float(eigenvalues[-1]),
                      symmetric_condition_number=float(eigenvalues[-1]/eigenvalues[0]),
                      energy_scaled_raw_reciprocity=energy_reciprocity)
    return result


def layout_checks(sources, layout):
    require(layout in ('sparse', 'full-grid'), 'layout must be sparse or full-grid')
    geometries = [source['geometry'] for source in sources]
    if layout == 'sparse':
        for geometry in geometries:
            if min(geometry['theta_i'], geometry['theta_j']) <= 1e-13:
                require(abs(geometry['phi']) <= 1e-13,
                        'Sparse source requires phi=0 when either director is polar; do not rotate raw matrices silently')
        return {'layout': layout, 'complete_grid_verified': False}
    def unique(values):
        result = []
        for value in sorted(values):
            if not result or abs(value-result[-1]) > 1e-13:
                result.append(value)
        return result
    ti = unique(g['theta_i'] for g in geometries)
    tj = unique(g['theta_j'] for g in geometries)
    phi = unique(g['phi'] for g in geometries)
    require(len(ti) >= 4 and len(ti) == len(tj) and
            np.allclose(ti, tj, rtol=0, atol=1e-13),
            'Full-grid requires one shared theta axis with at least4 nodes')
    require(abs(ti[0]) <= 1e-13 and abs(ti[-1]-math.pi/2) <= 1e-13,
            'Full-grid theta endpoints must be0 andpi/2')
    require(len(phi) >= 4 and len(phi) % 2 == 0, 'Full-grid requires an even periodic phi count>=4')
    expected_phi = -math.pi + 2*math.pi*np.arange(len(phi))/len(phi)
    require(np.allclose(phi, expected_phi, rtol=0, atol=1e-13),
            'Full-grid phi must be the wrapped zero-centered uniform periodic grid')
    require(len(geometries) == len(ti)**2*len(phi),
            'Full-grid requires every theta_i x theta_j x phi row from actual supplied evidence')
    return {'layout': layout, 'complete_grid_verified': True,
            'theta_axis': ti, 'phi_axis': phi,
            'pole_rows': 'Explicit grid-frame azimuth rows retained; no symmetry rows synthesized.'}


def isolated_evidence(data, parent_path, geometry, identity):
    moments = data.get('moments')
    require(isinstance(moments, dict) and moments.get('method') in METHODS,
            'Physical traction, reciprocal work, or combined-layer far-field moments must be identified')
    require(moments.get('isolated_spheroid_validation_passed') is True,
            'Isolated spheroid moment validation is missing or failed')
    if moments['method'] == 'combined_layer_farfield_moments':
        require(moments.get('density_is_traction') is False, 'Combined-layer density must not be identified as physical traction')
        error_value(moments.get('max_scaled_flux_error'), 'combined-layer flux error', STRICT_BOUNDARY_LIMIT)
    evidence = moments.get('isolated_validation_evidence')
    require(isinstance(evidence, list) and len(evidence) == 2, 'SHA-backed isolated evidence for both bodies is required')
    verified, bodies = [], set()
    for item in evidence:
        require(isinstance(item, dict), 'Isolated evidence entry must be an object')
        require(item.get('passed') is True and item.get('failures') == [], 'Recorded isolated evidence did not pass')
        body = item.get('isolated_body')
        require(type(body) is int and body in (0, 1) and body not in bodies, 'Isolated evidence must contain each body exactly once')
        bodies.add(body)
        expected = sha_string(item.get('sha256'), 'isolated evidence SHA256')
        recorded = item.get('path')
        require(isinstance(recorded, str) and recorded, 'Isolated evidence path is missing')
        path = resolve_recorded(recorded, parent_path, parent_path.parent/Path(recorded).name)
        require(digest(path) == expected, 'Isolated evidence SHA256 mismatch')
        single = read_json(path)
        require(type(single.get('isolated_body')) is int and single['isolated_body'] == body,
                'Isolated evidence body mismatch')
        single_geometry = metadata(single, geometry['c_nm'])
        for key in ('aspect_ratio', 'theta_i', 'theta_j', 'phi', 'c_m'):
            require(abs(single_geometry[key]-geometry[key]) <= 1e-13, 'Isolated evidence geometry mismatch: '+key)
        require(base_identity(single) == identity, 'Isolated evidence numerical operator identity mismatch')
        measured = boundary_checks(single, STRICT_BOUNDARY_LIMIT, require_gap=False)
        exact = single.get('isolated_exact_validation', {})
        require(exact.get('passed') is True, 'Isolated exact moment comparison failed')
        measured['moment_block_relative_errors'] = {
            key: error_value(exact.get('block_relative_errors', {}).get(key), 'isolated '+key, ISOLATED_MOMENT_LIMIT)
            for key in ('force', 'torque', 'stresslet')}
        measured.update(reciprocal_measure(matrix17(single.get('raw_full17'), 'isolated raw_full17'), positive=False))
        single_moments = single.get('moments', {})
        require(single_moments.get('method') == moments['method'], 'Isolated moment extraction method differs from the pair source')
        if moments['method'] == 'combined_layer_farfield_moments':
            require(single_moments.get('density_is_traction') is False,
                    'Isolated combined-layer density is not physical traction')
            error_value(single_moments.get('max_scaled_flux_error'), 'isolated combined-layer flux', STRICT_BOUNDARY_LIMIT)
        verified.append({'recorded_path': recorded, 'resolved_path': str(path.resolve()),
                         'sha256': expected, 'isolated_body': body, 'verified': True,
                         'measurements': measured, 'original_metadata': single})
    return verified


def load_source(path, *, boundary_limit=.01, c_nm_if_missing=None, expected_ar=None):
    path = Path(path).resolve()
    data = read_json(path)
    geometry = metadata(data, c_nm_if_missing)
    if expected_ar is not None:
        require(geometry['aspect_ratio'] == number(expected_ar, '--expect-ar', positive=True), 'Source AR differs from --expect-ar')
    identity = base_identity(data)
    require(data.get('isolated_body') is None, 'An isolated-body report cannot be used as a pair source')
    raw = matrix17(data.get('raw_full17'), 'raw_full17')
    measurements = reciprocal_measure(raw, positive=True)
    if 'raw_reciprocity' in data:
        error_value(data['raw_reciprocity'], 'reported raw_reciprocity', RECIPROCITY_LIMIT)
    if 'symmetric_full17' in data:
        supplied_symmetric = matrix17(data['symmetric_full17'], 'symmetric_full17')
        require(np.allclose(supplied_symmetric, (raw+raw.T)/2, rtol=1e-12, atol=1e-12),
                'Reported symmetric_full17 is inconsistent with raw_full17')
    measurements.update(boundary_checks(data, boundary_limit, require_gap=True))
    evidence = isolated_evidence(data, path, geometry, identity)
    solution = checkpoint(data, path)
    strict_failures = ['three_compatible_passing_resolution_sources_not_established_by_this_import']
    if max(measurements['boundary_max_scaled_error'], measurements['gap_patch_max_scaled_error']) > STRICT_BOUNDARY_LIMIT:
        strict_failures.append('sampled_boundary_error_exceeds_strict_0.1_percent_gate')
    return {'path': str(path), 'source_json_sha256': digest(path), 'geometry': geometry,
            'raw': raw, 'original_metadata': data, 'operator_identity': identity,
            'solution_checkpoint': solution, 'isolated_evidence': evidence,
            'measurements': measurements, 'strict_qualification_failures': strict_failures,
            'physical_accuracy_qualified': False, 'provisional_accepted': True}


def atomic_bytes(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name+'.pending.', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def export_sources(paths, output, *, floor_nm=1., boundary_limit=.01,
                   c_nm_if_missing=None, expected_ar=None,
                   layout='sparse',
                   provisional_reason='Explicit provisional development import; strict physical qualification is not established.'):
    floor_nm = number(floor_nm, 'floor_nm', positive=True)
    boundary_limit = number(boundary_limit, 'boundary_limit', positive=True)
    require(isinstance(provisional_reason, str) and provisional_reason.strip(), 'A provisional acceptance reason is required')
    require(bool(paths), 'At least one explicit reference report path is required')
    sources = [load_source(path, boundary_limit=boundary_limit,
                           c_nm_if_missing=c_nm_if_missing, expected_ar=expected_ar) for path in paths]
    first = sources[0]['geometry']
    for i, source in enumerate(sources):
        geometry = source['geometry']
        for key in ('aspect_ratio', 'c_m', 'h_over_c'):
            require(geometry[key] == first[key], 'V1 sources require identical '+key+' across all poses; select reports explicitly')
        require(geometry['h_over_c']*geometry['c_nm'] >= floor_nm, 'The source anchor gap must be at or above the hard floor')
        for earlier in sources[:i]:
            e = earlier['geometry']
            same = abs(geometry['theta_i']-e['theta_i']) <= 1e-13 and abs(geometry['theta_j']-e['theta_j']) <= 1e-13 and abs(canonical_phi(geometry['phi']-e['phi'])) <= 1e-13
            require(not same, 'Duplicate canonical angular pose; choose one source explicitly, including when replacing a mesh')
    layout_evidence = layout_checks(sources, layout)
    f = lambda value: format(float(value), '.17g')
    lines = [MAGIC, 'MODEL '+MODEL, 'CONVENTION '+CONVENTION, 'STATUS PROVISIONAL 0',
             'UNITS '+f(first['c_m'])+' '+f(floor_nm*1e-9),
             'AR '+f(first['aspect_ratio']), 'POSES '+str(len(sources))]
    for source in sources:
        geometry = source['geometry']
        solution = source['solution_checkpoint']
        lines.append('POSE '+' '.join(f(geometry[k]) for k in ('theta_i', 'theta_j', 'phi', 'h_over_c'))+
                     ' '+source['source_json_sha256']+' '+(solution['sha256'] if solution else '-'))
        lines.append('RAW17')
        lines.extend(' '.join(f(v) for v in row) for row in source['raw'])
        lines.append('END_POSE')
    lines.append('END')
    payload = ('\n'.join(lines)+'\n').encode()
    manifest_sources = [{key: value for key, value in source.items() if key != 'raw'} for source in sources]
    manifest = {'schema': 'slurry_reference_source17_manifest_v1', 'source_format': MAGIC,
                'model': MODEL, 'convention_id': CONVENTION, 'status': 'PROVISIONAL',
                'physical_accuracy_qualified': False, 'strict_physical_qualification': False,
                'provisional_acceptance_reason': provisional_reason,
                'development_boundary_acceptance_limit': boundary_limit,
                'strict_boundary_gate_unchanged': STRICT_BOUNDARY_LIMIT,
                'source_text_sha256': hashlib.sha256(payload).hexdigest(),
                'exporter_sha256': digest(__file__), 'aspect_ratio': first['aspect_ratio'],
                'c_m': first['c_m'], 'floor_m': floor_nm*1e-9, 'anchor_h_over_c': first['h_over_c'],
                'pose_count': len(sources), 'sources': manifest_sources,
                'layout': layout_evidence,
                'warnings': ['This development acceptance ceiling is not a physical error bound.',
                             'RAW17 values are preserved; no symmetrization or eigenvalue clipping was performed.',
                             'No new Stokes calculation or three-resolution convergence qualification was performed.',
                             'Replacement requires explicitly selected input report paths; report and verified checkpoint hashes change the source identity.',
                             'Only phi modulo2pi is normalized; matrices and Cartesian axes are not rotated.']}
    output = Path(output)
    manifest_path = output.with_suffix(output.suffix+'.manifest.json')
    # Validate and serialize both artifacts before replacing either path.
    manifest_payload = (json.dumps(manifest, indent=2, allow_nan=False)+'\n').encode()
    atomic_bytes(output, payload)
    atomic_bytes(manifest_path, manifest_payload)
    return manifest, manifest_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('references', type=Path, nargs='+', help='Explicit actual independent JSON reports; never auto-select latest')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--floor-nm', type=float, default=1.)
    parser.add_argument('--boundary-limit', type=float, default=.01, help='Explicit provisional development ceiling, not a physical qualification tolerance')
    parser.add_argument('--c-nm', type=float, help='Supply c in nm only for reports missing length_unit_nm; never converts a matrix')
    parser.add_argument('--expect-ar', type=float)
    parser.add_argument('--layout', choices=('sparse', 'full-grid'), default='sparse',
                        help='Full-grid requires every supplied actual grid-frame row; no pole/symmetry copies are created')
    parser.add_argument('--provisional-reason', default='Explicit provisional development import; strict physical qualification is not established.')
    args = parser.parse_args()
    try:
        manifest, sidecar = export_sources(args.references, args.output, floor_nm=args.floor_nm,
                                           boundary_limit=args.boundary_limit, c_nm_if_missing=args.c_nm,
                                           expected_ar=args.expect_ar, layout=args.layout,
                                           provisional_reason=args.provisional_reason)
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))
    print(json.dumps({'source': str(args.output.resolve()), 'manifest': str(sidecar.resolve()),
                      'source_text_sha256': manifest['source_text_sha256'],
                      'pose_count': manifest['pose_count'], 'physical_accuracy_qualified': False}))


if __name__ == '__main__':
    main()
