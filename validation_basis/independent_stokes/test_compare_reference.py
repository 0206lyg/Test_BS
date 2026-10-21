"""Manufactured algebra/gate tests, never physical reference certificates."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

import compare_reference as compare


def tolerances():
    return argparse.Namespace(reference_budget=.002, model_budget=.01,
                              boundary_tolerance=.001, surface_tolerance=.001,
                              reciprocity_tolerance=.001, linear_tolerance=1e-8)


def synthetic_metadata():
    record = compare.canonical_geometry(.5).metadata()
    record.update(raw_full17=np.eye(17).tolist(), convention_id=compare.CONVENTION,
                  independent_reference=True, shares_production_solver=False,
                  synthetic=True, resolution_level=1,
                  validation=dict(off_grid_boundary=True, boundary_max_scaled_error=0.,
                                  exact_surface_position_max_error_over_gap=0.,
                                  linear_relative_residual=0.),
                  moments=dict(method='combined_layer_farfield_moments',
                               isolated_spheroid_validation_passed=True,
                               max_scaled_flux_error=0.))
    record.update(manufactured_operator_data())
    return record


def manufactured_operator_data(panel=False):
    """Unit-test identities only; no physical reference files are produced."""
    if panel:
        return dict(schema=compare.PANEL_SCHEMA, convention_id=compare.CONVENTION,
                    library_sha256='a'*64, alpha=1., beta=1.,
                    source_hashes={'backend.py':'b'*64, 'geometry.py':'c'*64})
    return dict(convention_id=compare.CONVENTION,
                reference_operator_identity=dict(solver_family='manufactured_unit_gate',
                    implementation_sha256='d'*64, geometry_sha256='e'*64,
                    parameters={'viscosity':1.}))


def manufactured_refs(panel=False):
    refs = []
    for level, amplitude in enumerate([1.,1.001,1.0015],1):
        g = amplitude*np.eye(17)
        refs.append(compare.Reference(Path(f'manufactured_level{level}'),
            manufactured_operator_data(panel),.5,level,g,g,
            dict(minimum_symmetric_eigenvalue=amplitude,
                 discretization_signature=f'manufactured_mesh{level}',
                 source_gates_passed=True)))
    return refs


class ComparisonTests(unittest.TestCase):
    def test_energy_metric_and_blocks_scale_exactly(self):
        rng = np.random.default_rng(23)
        a = rng.normal(size=(17, 17))
        reference = a @ a.T + np.eye(17)
        delta = .0125*reference
        self.assertAlmostEqual(compare.energy_delta(delta, reference), .0125, places=13)
        for block in compare.block_errors(delta, reference).values():
            self.assertAlmostEqual(block['energy_input_scaled_operator_relative'], .0125, places=13)

    def test_energy_metric_detects_weak_common_motion(self):
        reference = np.eye(17)
        reference[12:, 12:] *= 1e9  # A dominant squeeze/strain response hides weak modes.
        delta = np.zeros((17, 17))
        common = np.zeros(17)
        common[0] = common[6] = 1/np.sqrt(2)
        delta += .1*np.outer(common, common)
        self.assertLess(np.linalg.norm(delta)/np.linalg.norm(reference), 1e-9)
        self.assertAlmostEqual(compare.energy_delta(delta, reference), .1, places=14)
        maps, _ = compare.prescribed_maps(.5)
        self.assertAlmostEqual(compare.energy_delta(maps.T@delta@maps, maps.T@reference@maps), .1, places=14)

    def test_held_body_strain_has_nonzero_center_velocities(self):
        gap = .5
        pose = compare.canonical_geometry(gap)
        _, held = compare.prescribed_maps(gap)
        es = compare.strain_basis()
        origin = np.array([0., 0., gap/2])
        for body, center in enumerate(pose.centers):
            for mode in range(5):
                expected = -es[mode]@(center-origin)/pose.a
                np.testing.assert_allclose(held[6*body:6*body+3, mode], expected)
        self.assertGreater(np.linalg.norm(held[:12]), .1)

    def test_synthetic_and_missing_geometry_checks_cannot_qualify(self):
        record = synthetic_metadata()
        del record['validation']['exact_surface_position_max_error_over_gap']
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'synthetic.json'
            path.write_text(json.dumps(record))
            parsed = compare.read_reference(path, tolerances())
        self.assertFalse(parsed.report['source_gates_passed'])
        self.assertIn('synthetic_diagnostic_is_not_physical_reference', parsed.report['source_gate_failures'])
        self.assertIn('surface_position_error_over_gap_missing_or_above_tolerance', parsed.report['source_gate_failures'])

    def test_two_consecutive_small_refinements_are_required(self):
        # Direct dataclass construction isolates this logical gate from source
        # qualification. These matrices are manufactured, not reference files.
        refs = manufactured_refs()
        self.assertFalse(compare.qualify_group(refs[:2], tolerances())['qualified_reference'])
        self.assertTrue(compare.qualify_group(refs, tolerances())['qualified_reference'])
        refs[-1].report['discretization_signature'] = refs[-2].report['discretization_signature']
        self.assertFalse(compare.qualify_group(refs, tolerances())['qualified_reference'])
        refs[-1].report['discretization_signature'] = 'manufactured_mesh3'
        refs[-1].total = refs[-1].raw = 1.02*np.eye(17)
        self.assertFalse(compare.qualify_group(refs, tolerances())['qualified_reference'])

    def test_matching_panel_operator_accepts_changed_driver_and_storage(self):
        refs = manufactured_refs(panel=True)
        for index, reference in enumerate(refs):
            reference.data['source_hashes']['run_reference.py'] = str(index)*64
            reference.data['source_hashes']['memory_bounded_dense.py'] = str(index+1)*64
            reference.data['solver_contract'] = f'manufactured_storage_contract{index}'
        report = compare.qualify_group(refs,tolerances())
        self.assertTrue(report['operator_identity_gate']['passed'])
        self.assertTrue(report['qualified_reference'])

    def test_panel_mixed_or_missing_library_cannot_qualify(self):
        for replacement in (None,'','not_a_digest','f'*64):
            with self.subTest(replacement=replacement):
                refs = manufactured_refs(panel=True)
                if replacement is None:
                    del refs[1].data['library_sha256']
                else:
                    refs[1].data['library_sha256'] = replacement
                report = compare.qualify_group(refs,tolerances())
                self.assertFalse(report['qualified_reference'])
                self.assertFalse(report['operator_identity_gate']['passed'])
                self.assertTrue(all(row['refinement_gate_passed'] for row in report['refinements']))

    def test_panel_missing_operator_components_cannot_qualify(self):
        for field in ('backend.py','geometry.py','alpha','beta','convention_id'):
            with self.subTest(field=field):
                refs = manufactured_refs(panel=True)
                location = refs[1].data['source_hashes'] if field.endswith('.py') else refs[1].data
                del location[field]
                self.assertFalse(compare.qualify_group(refs,tolerances())['qualified_reference'])

    def test_panel_changed_operator_components_cannot_qualify(self):
        for field,value in [('backend.py','f'*64),('geometry.py','f'*64),
                            ('alpha',2.),('beta',.5),('convention_id','other')]:
            with self.subTest(field=field):
                refs = manufactured_refs(panel=True)
                location = refs[1].data['source_hashes'] if field.endswith('.py') else refs[1].data
                location[field] = value
                self.assertFalse(compare.qualify_group(refs,tolerances())['qualified_reference'])

    def test_identity_gate_uses_latest_three_sources(self):
        refs = manufactured_refs(panel=True)
        old = deepcopy(refs[0])
        old.level = 0
        old.path = Path('manufactured_old_operator')
        old.data['library_sha256'] = 'f'*64
        old.report['discretization_signature'] = 'manufactured_old_mesh'
        old.report['source_gates_passed'] = False
        report = compare.qualify_group([old]+refs,tolerances())
        self.assertFalse(report['same_library_across_inputs'])
        self.assertTrue(report['operator_identity_gate']['passed'])
        self.assertTrue(report['qualified_reference'])

    def test_external_identity_supports_no_panel_library_and_requires_match(self):
        refs = manufactured_refs()
        self.assertTrue(compare.qualify_group(refs,tolerances())['qualified_reference'])
        for field,value in [('implementation_sha256','f'*64),('geometry_sha256','f'*64),
                            ('solver_family','another_solver'),('parameters',{'viscosity':2.})]:
            with self.subTest(field=field):
                changed = deepcopy(refs)
                changed[1].data['reference_operator_identity'][field] = value
                self.assertFalse(compare.qualify_group(changed,tolerances())['qualified_reference'])
        del refs[1].data['reference_operator_identity']
        self.assertFalse(compare.qualify_group(refs,tolerances())['qualified_reference'])

    def test_panel_identity_cannot_be_overridden_by_external_declaration(self):
        refs = manufactured_refs(panel=True)
        for reference in refs:
            reference.data['reference_operator_identity'] = manufactured_operator_data()['reference_operator_identity']
            del reference.data['library_sha256']
        self.assertFalse(compare.qualify_group(refs,tolerances())['qualified_reference'])
        record = synthetic_metadata()
        record.update(manufactured_operator_data(panel=True))
        del record['source_hashes']['backend.py']
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'synthetic.json'
            path.write_text(json.dumps(record))
            parsed = compare.read_reference(path,tolerances())
        self.assertIn('panel_operator_backend_sha256_missing_or_invalid',parsed.report['source_gate_failures'])

    def test_scaled_reciprocity_detects_hidden_weak_mode_asymmetry(self):
        reference = np.eye(17)
        reference[12:, 12:] *= 1e9
        raw = reference.copy()
        raw[0, 6] = .01
        raw[6, 0] = -.01
        self.assertLess(np.linalg.norm(raw-raw.T)/np.linalg.norm(raw), 1e-9)
        self.assertAlmostEqual(compare.energy_scaled_reciprocity(raw, reference), .02, places=14)

    def test_indefinite_reference_is_not_repaired(self):
        record = synthetic_metadata()
        record['raw_full17'][0][0] = -1.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'synthetic.json'
            path.write_text(json.dumps(record))
            parsed = compare.read_reference(path, tolerances())
        self.assertEqual(parsed.report['minimum_symmetric_eigenvalue'], -1.)
        self.assertIn('symmetric_reference_not_positive_definite', parsed.report['source_gate_failures'])

    def test_panel_backend_requires_clean_boundary_evaluation(self):
        record = synthetic_metadata()
        record['schema'] = 'independent_ar20_panel_bie_v1'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'synthetic.json'
            path.write_text(json.dumps(record))
            parsed = compare.read_reference(path, tolerances())
        self.assertIn('boundary_evaluator_failed_or_clean_execution_not_verified', parsed.report['source_gate_failures'])

    def test_partial_panel_report_and_missing_density_flux_are_rejected(self):
        record = synthetic_metadata()
        record.update(schema='independent_ar20_panel_bie_v1', status='boundary_check')
        del record['moments']['max_scaled_flux_error']
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'partial.json'
            path.write_text(json.dumps(record))
            parsed = compare.read_reference(path, tolerances())
        self.assertIn('panel_reference_process_not_complete', parsed.report['source_gate_failures'])
        self.assertIn('combined_layer_flux_missing_or_nonfinite', parsed.report['source_gate_failures'])

    def test_negative_error_sentinels_are_rejected(self):
        record = synthetic_metadata()
        for key in ('boundary_max_scaled_error', 'exact_surface_position_max_error_over_gap', 'linear_relative_residual'):
            record['validation'][key] = -1.
        record['moments']['max_scaled_flux_error'] = -1.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'synthetic.json'
            path.write_text(json.dumps(record))
            parsed = compare.read_reference(path, tolerances())
        self.assertGreaterEqual(len(parsed.report['source_gate_failures']), 5)

    def test_quadrature_changes_do_not_count_as_spatial_refinement(self):
        first = dict(discretization_sha256='same_mesh', quadrature_tolerance=1e-6)
        second = dict(discretization_sha256='same_mesh', quadrature_tolerance=1e-9)
        self.assertEqual(compare.discretization_signature(first), compare.discretization_signature(second))
        self.assertIsNone(compare.discretization_signature(dict(level=1)))

    def test_helper_wrong_pose_or_nonfinite_gap_is_rejected(self):
        for key, value in [('aspect_ratio', 19), ('h_over_c', float('nan'))]:
            data = compare.canonical_geometry(.5).metadata()
            data['convention_id'] = compare.CONVENTION
            data[key] = value
            process = argparse.Namespace(returncode=0, stdout=json.dumps(data), stderr='')
            with mock.patch.object(compare.subprocess, 'run', return_value=process):
                with self.assertRaises(ValueError):
                    compare.helper_export(Path('/synthetic/helper'), .5)


if __name__ == '__main__':
    unittest.main()
