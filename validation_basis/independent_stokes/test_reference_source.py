"""Import/export checks only; manufactured cases never certify Stokes flow."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np

import prepare_reference_source as source


HERE = Path(__file__).resolve().parent
ACTUAL = HERE/'results/pair_h150_p8_anisotropic_chunked.json'


class ReferenceSourceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(dir='/dev/shm' if Path('/dev/shm').is_dir() else None)
        self.root = Path(self.directory.name)
        self.actual = json.loads(ACTUAL.read_text())

    def tearDown(self):
        self.directory.cleanup()

    def report(self, data=None, name='reference.json'):
        path = self.root/name
        path.write_text(json.dumps(self.actual if data is None else data))
        return path

    def reject(self, data, message):
        with self.assertRaisesRegex((ValueError, OSError), message):
            source.export_sources([self.report(data)], self.root/'rejected.source')
        self.assertFalse((self.root/'rejected.source').exists())

    def test_actual_n9963_export_preserves_raw_and_provisional_status(self):
        output = self.root/'actual.source'
        manifest, sidecar = source.export_sources([ACTUAL], output)
        lines = output.read_text().splitlines()
        self.assertEqual(lines[0], source.MAGIC)
        self.assertEqual(lines[3], 'STATUS PROVISIONAL 0')
        begin = lines.index('RAW17')+1
        exported = np.array([[float(v) for v in row.split()] for row in lines[begin:begin+17]])
        np.testing.assert_array_equal(exported, self.actual['raw_full17'])
        self.assertFalse(manifest['physical_accuracy_qualified'])
        self.assertFalse(manifest['strict_physical_qualification'])
        record = manifest['sources'][0]
        self.assertEqual(record['source_json_sha256'], source.digest(ACTUAL))
        self.assertEqual(record['original_metadata'], self.actual)
        self.assertEqual(record['measurements']['boundary_max_scaled_error'], .007532226462396109)
        self.assertIn('sampled_boundary_error_exceeds_strict_0.1_percent_gate', record['strict_qualification_failures'])
        self.assertEqual(manifest['source_text_sha256'], source.digest(output))
        self.assertEqual(json.loads(sidecar.read_text()), manifest)

    def test_malformed_nonfinite_boolean_and_nonpositive_matrix_rejected(self):
        cases = []
        x = deepcopy(self.actual); x['raw_full17'].pop(); cases.append((x, '17 by17'))
        x = deepcopy(self.actual); x['raw_full17'][0][0] = float('nan'); cases.append((x, 'nonfinite'))
        x = deepcopy(self.actual); x['raw_full17'][0][0] = True; cases.append((x, 'finite numeric'))
        x = deepcopy(self.actual); x['raw_full17'] = (-np.eye(17)).tolist(); cases.append((x, 'positive definite'))
        x = deepcopy(self.actual); x['raw_full17'][0][1] += 1e5; cases.append((x, 'reciprocity'))
        for data, message in cases:
            with self.subTest(message=message):
                self.reject(data, message)

    def test_wrong_ar_and_units_rejected(self):
        x = deepcopy(self.actual); x['aspect_ratio'] = 10
        self.reject(x, 'a_nm')
        x = deepcopy(self.actual); x['coordinate_units'] = 'meters'
        self.reject(x, 'coordinate_units')
        x = deepcopy(self.actual); x['length_unit_nm'] = 1
        self.reject(x, 'c_nm')
        x = deepcopy(self.actual); x['convention_id'] = 'dimensional_force_torque_stress'
        self.reject(x, 'normalized full17 convention')
        with self.assertRaisesRegex(ValueError, '--expect-ar'):
            source.export_sources([ACTUAL], self.root/'wrong-ar.source', expected_ar=10)

    def test_missing_units_require_explicit_c_nm_and_other_ar_metadata_supported(self):
        x = deepcopy(self.actual); del x['length_unit_nm']
        with self.assertRaisesRegex(ValueError, 'length_unit_nm is missing'):
            source.metadata(x)
        self.assertEqual(source.metadata(x, 150)['c_nm'], 150)
        x['aspect_ratio'] = 10; x['a_nm'] = 1500
        self.assertEqual(source.metadata(x, 150)['aspect_ratio'], 10)
        with self.assertRaisesRegex(ValueError, 'cannot override'):
            source.metadata(self.actual, 151)

    def test_generic_panel_schema_has_same_identity_and_checkpoint_requirements(self):
        x = deepcopy(self.actual)
        x['schema'] = 'independent_spheroid_panel_bie_v1'
        self.assertEqual(source.base_identity(x), source.base_identity(self.actual))
        manifest, _ = source.export_sources([self.report(x)], self.root/'generic-panel.source')
        self.assertFalse(manifest['physical_accuracy_qualified'])
        del x['solution_file']; del x['solution_sha256']
        self.reject(x, 'Panel BIE source must provide')

    def test_duplicate_pose_and_periodic_duplicate_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Duplicate canonical'):
            source.export_sources([ACTUAL, ACTUAL], self.root/'duplicate.source')
        # No manufactured physical report is created: periodic equivalence is
        # tested through the actual metadata function and the duplicate check.
        x = deepcopy(self.actual); x['phi'] += 2*math.pi
        path = self.report(x, 'periodic.json')
        with self.assertRaisesRegex(ValueError, 'Duplicate canonical'):
            source.export_sources([ACTUAL, path], self.root/'periodic.source')

    def test_strict_boundary_failure_is_not_promoted_and_ceiling_is_explicit(self):
        with self.assertRaisesRegex(ValueError, 'boundary_max_scaled_error'):
            source.export_sources([ACTUAL], self.root/'strict.source', boundary_limit=.001)
        x = deepcopy(self.actual); x['physical_accuracy_qualified'] = True
        m, _ = source.export_sources([self.report(x)], self.root/'claimed.source')
        self.assertFalse(m['physical_accuracy_qualified'])
        self.assertTrue(m['sources'][0]['original_metadata']['physical_accuracy_qualified'])
        x = deepcopy(self.actual); x['validation']['boundary_max_scaled_error'] = .02
        self.reject(x, 'boundary_max_scaled_error')

    def test_core_validation_failures_are_not_waived(self):
        for key, value in [('linear_relative_residual', 1e-4), ('boundary_evaluation_valid', False),
                           ('exact_surface_position_max_error_over_gap', .1)]:
            x = deepcopy(self.actual); x['validation'][key] = value
            with self.subTest(key=key):
                self.reject(x, 'linear_relative_residual|boundary evaluation|surface_position')
        x = deepcopy(self.actual); x['moments']['isolated_validation_evidence'][0]['sha256'] = '0'*64
        self.reject(x, 'Isolated evidence SHA256 mismatch')

    def test_portable_checkpoint_and_isolated_evidence_fallback_remains_hashed(self):
        x = deepcopy(self.actual)
        original_solution = Path(x['solution_file'])
        if not original_solution.exists():
            original_solution = ACTUAL.with_suffix('.solution.npz')
        x['solution_file'] = str(self.root/'absent-recorded.solution.npz')
        path = self.report(x)
        shutil.copyfile(original_solution, path.with_suffix('.solution.npz'))
        for entry in x['moments']['isolated_validation_evidence']:
            old = Path(entry['path'])
            if not old.exists():
                old = ACTUAL.parent/old.name
            shutil.copyfile(old, self.root/old.name)
            entry['path'] = str(self.root/'missing-folder'/old.name)
        path = self.report(x)
        manifest, _ = source.export_sources([path], self.root/'portable.source')
        self.assertEqual(manifest['sources'][0]['solution_checkpoint']['resolved_path'], str(path.with_suffix('.solution.npz')))
        for entry in manifest['sources'][0]['isolated_evidence']:
            self.assertTrue(entry['verified'])
        Path(x['solution_file']).write_bytes(b'wrong existing checkpoint')
        with self.assertRaisesRegex(ValueError, 'checkpoint SHA256 mismatch'):
            source.export_sources([path], self.root/'bad-checkpoint.source')

    def test_explicit_replacement_changes_identity_synthetic_replacement_rejected(self):
        output = self.root/'replace.source'
        first, _ = source.export_sources([ACTUAL], output)
        # Same actual physical evidence, different JSON bytes. This tests cache
        # identity only, and is not presented as a new or finer Stokes solve.
        replacement = self.root/'actual-report-reserialized.json'
        replacement.write_text(json.dumps(self.actual, separators=(',', ':'))+'\n')
        second, _ = source.export_sources([replacement], output)
        self.assertNotEqual(first['source_text_sha256'], second['source_text_sha256'])
        self.assertNotEqual(first['sources'][0]['source_json_sha256'], second['sources'][0]['source_json_sha256'])
        before = output.read_bytes()
        x = deepcopy(self.actual); x['synthetic'] = True
        x['test_fixture_marker'] = 'Manufactured higher-resolution replacement; must not become a physical reference.'
        with self.assertRaisesRegex(ValueError, 'Synthetic'):
            source.export_sources([self.report(x, 'synthetic-replacement.json')], output)
        self.assertEqual(before, output.read_bytes())

    def test_sparse_and_complete_grid_layout_only_manufactured_metadata(self):
        geometry = source.metadata(self.actual)
        polar = dict(geometry, theta_i=0., phi=.5)
        with self.assertRaisesRegex(ValueError, 'Sparse source requires'):
            source.layout_checks([{'geometry': polar}], 'sparse')
        rows = [{'geometry': dict(geometry, theta_i=ti, theta_j=tj, phi=phi)}
                for ti in np.linspace(0, math.pi/2, 4)
                for tj in np.linspace(0, math.pi/2, 4)
                for phi in np.linspace(-math.pi, math.pi, 4, endpoint=False)]
        self.assertTrue(source.layout_checks(rows, 'full-grid')['complete_grid_verified'])
        with self.assertRaisesRegex(ValueError, 'every theta_i'):
            source.layout_checks(rows[:-1], 'full-grid')
        shifted = deepcopy(rows)
        for row in shifted:
            row['geometry']['phi'] += .1
        with self.assertRaisesRegex(ValueError, 'zero-centered'):
            source.layout_checks(shifted, 'full-grid')


if __name__ == '__main__':
    unittest.main()
