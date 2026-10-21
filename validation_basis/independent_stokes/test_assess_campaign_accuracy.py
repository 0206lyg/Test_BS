"""Synthetic unit data only; no file here is physical reference evidence."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import assess_campaign_accuracy as assess


class CampaignAccuracyTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.folder = Path(self.temporary.name)
        self.geometry = {'aspect_ratio': 5., 'theta_i': .2, 'theta_j': .3, 'phi': .4,
                         'h_over_c': .2, 'c_m': 150e-9}
        self.query = {'geometry': self.geometry, 'effective_h_over_c': .2,
                      'total': np.eye(17)*1.02, 'surrogate': None}
        training_geometry = dict(self.geometry, h_over_c=1.)
        self.training = [{'geometry': training_geometry, 'source_json_sha256': 'c'*64}]
        self.coarse = self.make_reference('coarse', np.eye(17)*1.002)
        self.fine = self.make_reference('fine', np.eye(17))
        self.case = {'id': 'gap1', 'query_id': 'q1', 'heldout_kind': 'gap',
                     'references': [self.coarse.name, self.fine.name]}

    def tearDown(self):
        self.temporary.cleanup()

    def write(self, path, data):
        path.write_text(json.dumps(data, indent=2)+'\n')
        return path

    def make_reference(self, name, raw, boundary=.0075, implementation='a'*64):
        geometry = {key: value for key, value in self.geometry.items() if key != 'c_m'}
        geometry['length_unit_nm'] = 150.
        data = {
            'schema': 'external_reference_unit_fixture_only', 'unit_test_data': True,
            'status': 'complete', 'convention_id': assess.CONVENTION,
            'independent_reference': True, 'shares_production_solver': False,
            'geometry_metadata': geometry, 'raw_full17': raw.tolist(),
            'node_count': 100 if name == 'coarse' else 200, 'order': 4 if name == 'coarse' else 6,
            'discretization_sha256': hashlib.sha256(name.encode()).hexdigest(),
            'reference_operator_identity': {'solver_family': 'unit_fixture_only',
                'implementation_sha256': implementation, 'geometry_sha256': 'b'*64},
            'validation': {'off_grid_boundary': True, 'boundary_evaluation_valid': True,
                'boundary_sample_count': 20, 'gap_patch_sample_count': 10,
                'boundary_max_scaled_error': boundary, 'gap_patch_max_scaled_error': boundary,
                'linear_relative_residual': 1e-12, 'exact_surface_position_max_error_over_gap': 1e-8},
            'moments': {'method': 'physical_traction', 'isolated_spheroid_validation_passed': True},
        }
        evidence = []
        for body in (0, 1):
            single = copy.deepcopy(data)
            single.update(isolated_body=body, isolated_exact_validation={
                'passed': True, 'block_relative_errors': {'force': 1e-6, 'torque': 1e-6, 'stresslet': 1e-6}})
            single['validation']['boundary_max_scaled_error'] = 1e-5
            single['validation']['gap_patch_max_scaled_error'] = 1e-5
            path = self.write(self.folder/f'{name}_isolated{body}.json', single)
            evidence.append({'path': path.name, 'sha256': assess.digest(path),
                             'passed': True, 'failures': [], 'isolated_body': body})
        data['moments']['isolated_validation_evidence'] = evidence
        return self.write(self.folder/f'{name}.json', data)

    def run_case(self, case=None, query=None):
        return assess.assess_case(case or self.case, query or self.query, self.training,
                                  self.folder, assess.DEFAULTS)

    def test_two_resolution_empirical_pass_does_not_claim_strict_physics(self):
        result = self.run_case()
        self.assertEqual(result['status'], 'PASS', result['reasons'])
        self.assertFalse(result['physical_accuracy_qualified'])
        self.assertFalse(result['strict_three_resolution_qualification_established'])
        self.assertFalse(result['references'][-1]['strict_0_1_percent_boundary_diagnostic'])
        self.assertAlmostEqual(result['table_vs_independent_reference']['full17_relative_energy_error'], .02)
        self.assertAlmostEqual(result['paired_refinement']['full17_relative_energy_error'], .002)

    def test_all_requested_aspect_ratios_use_the_same_normalized_contract(self):
        for ar in (2., 5., 10., 20.):
            with self.subTest(ar=ar):
                self.geometry['aspect_ratio'] = ar
                self.training[0]['geometry']['aspect_ratio'] = ar
                self.make_reference('coarse', 1.002*np.eye(17))
                self.make_reference('fine', np.eye(17))
                self.assertEqual(self.run_case()['status'], 'PASS')

    def test_resolved_large_model_error_is_empirical_failure(self):
        query = dict(self.query, total=1.2*np.eye(17))
        result = self.run_case(query=query)
        self.assertEqual(result['status'], 'FAIL')
        self.assertFalse(result['physical_accuracy_qualified'])

    def test_underresolved_reference_is_not_declared_model_failure(self):
        data = json.loads(self.fine.read_text())
        data['validation']['boundary_max_scaled_error'] = .02
        self.write(self.fine, data)
        result = self.run_case(query=dict(self.query, total=1.2*np.eye(17)))
        self.assertEqual(result['status'], 'UNRESOLVED')
        self.assertIsNotNone(result['table_vs_independent_reference'])
        self.assertFalse(result['boundary_residual_is_force_error_bound'])

    def test_missing_fine_reference_retains_explicit_coarse_comparison(self):
        self.fine.unlink()
        result = self.run_case()
        self.assertEqual(result['status'], 'UNRESOLVED')
        self.assertIsNotNone(result['table_vs_independent_reference'])
        self.assertFalse(result['comparison_uses_requested_finest_reference'])
        self.assertIsNone(result['paired_refinement'])

    def test_operator_change_is_not_spatial_convergence(self):
        self.make_reference('fine', np.eye(17), implementation='d'*64)
        result = self.run_case()
        self.assertEqual(result['status'], 'UNRESOLVED')
        self.assertIsNone(result['paired_refinement'])
        self.assertIn('uncontrolled_operator_change', result)

    def test_duplicate_resolution_cannot_pass(self):
        case = dict(self.case, references=[self.fine.name, self.fine.name])
        result = self.run_case(case=case)
        self.assertEqual(result['status'], 'UNRESOLVED')
        self.assertIn('coarse_fine_report_reused', result['reasons'])

    def test_large_refinement_change_remains_unresolved(self):
        self.make_reference('coarse', 1.1*np.eye(17))
        result = self.run_case()
        self.assertEqual(result['status'], 'UNRESOLVED')
        self.assertIn('paired_resolution_change_exceeds_screening_tolerance', result['reasons'])

    def test_geometry_mismatch_prevents_wrong_matrix_comparison(self):
        data = json.loads(self.fine.read_text())
        data['geometry_metadata']['h_over_c'] = .3
        self.write(self.fine, data)
        result = self.run_case()
        self.assertEqual(result['status'], 'UNRESOLVED')
        self.assertFalse(result['comparison_uses_requested_finest_reference'])
        self.assertIsNone(result['paired_refinement'])

    def test_unverified_isolated_stresslet_evidence_is_unresolved(self):
        data = json.loads(self.fine.read_text())
        data['moments']['isolated_validation_evidence'][0]['sha256'] = 'e'*64
        self.write(self.fine, data)
        result = self.run_case()
        self.assertEqual(result['status'], 'UNRESOLVED')
        self.assertTrue(any('isolated_moment_evidence_unverified' in value for value in result['reasons']))

    def test_training_geometry_or_report_reuse_is_not_heldout(self):
        self.training[0]['geometry']['h_over_c'] = .2
        self.assertEqual(self.run_case()['status'], 'UNRESOLVED')
        self.training[0]['geometry']['h_over_c'] = 1.
        self.training[0]['source_json_sha256'] = assess.digest(self.fine)
        result = self.run_case()
        self.assertTrue(result['heldout']['training_report_hash_reused'])
        self.assertEqual(result['status'], 'UNRESOLVED')

    def test_polar_and_equatorial_equivalence_prevents_fake_angular_holdouts(self):
        a = dict(self.geometry, theta_i=0., phi=-1.2)
        b = dict(a, phi=.7)
        self.assertTrue(assess.same_pose(a, b))
        a = dict(self.geometry, theta_i=np.pi/2)
        self.assertTrue(assess.same_pose(a, dict(a, phi=a['phi']+np.pi)))
        self.assertFalse(assess.same_pose(self.geometry, dict(self.geometry, phi=.7)))

    def test_energy_metric_detects_weak_modes_hidden_by_frobenius(self):
        reference = np.diag(np.geomspace(1., 1e8, 17))
        predicted = reference.copy();predicted[0, 0] += .2
        metrics = assess.response_metrics(predicted, reference)
        self.assertAlmostEqual(metrics['full17_relative_energy_error'], .2)
        self.assertLess(metrics['raw_response_relative_frobenius_error'], 1e-8)

    def test_raw_antisymmetric_force_response_is_not_hidden(self):
        reference = np.eye(17);reference[0, 6] = .0002;reference[6, 0] = -.0002
        metrics = assess.response_metrics(np.eye(17), reference)
        self.assertEqual(metrics['full17_relative_energy_error'], 0.)
        self.assertGreater(metrics['blocks']['force']['energy_input_scaled_operator_relative'], 0.)
        self.assertGreater(metrics['reference_energy_scaled_reciprocity'], 0.)

    def campaign_fixture(self):
        source = self.folder/'reference.source17';source.write_text('UNIT TEST FIXTURE ONLY\n')
        table = self.folder/'reference.sft';table.write_bytes(b'UNIT TEST QUERY BINDING ONLY')
        manifest = self.write(self.folder/'reference.source17.manifest.json', {
            'source_text_sha256': assess.digest(source), 'aspect_ratio': 5., 'c_m': 150e-9,
            'floor_m': 1e-9, 'sources': self.training})
        generation = self.write(self.folder/'generation.json', {
            'source_fnv1a64': assess.fnv_file(source), 'model_revision': 'unit_model',
            'status': 'provisional_numerical_transport_passed',
            'maximum_roundtrip_relative_energy_error': 1e-12,
            'maximum_gap_interpolation_relative_energy_error': 1e-7,
            'interpolation_tolerance': 1e-6, 'interior_samples_per_interval': 7})
        query = self.write(self.folder/'query.json', {
            'schema': 'reference17_query_v1', 'convention': assess.CONVENTION,
            'operator': 'symmetric_full17_total',
            'ar': 5., 'c_m': 150e-9, 'floor_m': 1e-9, 'model_revision': 'unit_model',
            'table_fnv1a64': assess.fnv_file(table), 'physical_accuracy_qualified': False,
            'rows': [{'id': 'q1', **{key: self.geometry[key] for key in ('theta_i', 'theta_j', 'phi', 'h_over_c')},
                      'gap_nm': 30., 'effective_h_over_c': .2, 'total_full17': self.query['total'].tolist()}]})
        pipeline = self.write(self.folder/'pipeline.json', {'status': 'complete', 'workflow_completed': True,
            'outputs': {path.name: {'sha256': assess.digest(path)} for path in (source, table, manifest, generation)}})
        return {'schema': assess.SCHEMA, 'groups': [{'id': 'AR5', 'query_result': query.name,
            'training_source': source.name, 'training_manifest': manifest.name, 'generation_report': generation.name,
            'table_path': table.name, 'pipeline_path': pipeline.name, 'cases': [self.case]}]}

    def test_complete_manifest_binds_table_source_and_separates_transport(self):
        report = assess.assess_campaign(self.campaign_fixture(), self.folder)
        self.assertEqual(report['groups'][0]['counts'], {'PASS': 1, 'FAIL': 0, 'UNRESOLVED': 0})
        self.assertFalse(report['physical_accuracy_qualified'])
        self.assertEqual(report['groups'][0]['surrogate_to_table_transport']['maximum_gap_interpolation_relative_energy_error'], 1e-7)
        self.assertFalse(report['groups'][0]['cases'][0]['surrogate_vs_table_at_this_query']['available'])
        self.assertAlmostEqual(next(assess.flat_rows(report))['model_energy'], .02)

    def test_tampered_table_is_unresolved_not_silent_pass(self):
        fixture = self.campaign_fixture()
        (self.folder/'reference.sft').write_bytes(b'CHANGED')
        report = assess.assess_campaign(fixture, self.folder)
        result = report['groups'][0]['cases'][0]
        self.assertEqual(result['status'], 'UNRESOLVED')
        self.assertIn('query_table_hash_mismatch', result['reasons'])

    def test_all_missing_campaign_retains_expected_geometry_and_unresolved_rows(self):
        fixture = {'schema': assess.SCHEMA, 'groups': [
            {'id': f'AR{ar}', 'query_result': f'missing_AR{ar}.json',
             'cases': [{'id': f'case{ar}', 'query_id': 'q', 'geometry': dict(self.geometry, aspect_ratio=ar)}]}
            for ar in (2., 5., 10., 20.)]}
        report = assess.assess_campaign(fixture, self.folder)
        rows = list(assess.flat_rows(report))
        self.assertEqual([row['AR'] for row in rows], [2., 5., 10., 20.])
        self.assertTrue(all(row['status'] == 'UNRESOLVED' for row in rows))
        for row in rows:
            self.assertAlmostEqual(row['gap_nm'], 30.)


if __name__ == '__main__':
    unittest.main()
