"""Checkpoint/evidence/native-log contracts; no Stokes solve or qualification."""
import copy
import ctypes
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import run_reference as driver


class DriverContracts(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='reference_driver_contract_')
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.pose = driver.recorded_pose(150.)
        self.mesh = dict(srcvals=np.zeros((12,2)), weights=np.ones(2), body_ids=np.array([0,1]),
                         norders=np.array([0,0]), ixyzs=np.array([1,2,3]), iptype=np.array([11,11]),
                         centers=self.pose.centers)
        self.mesh['srcvals'][:3] = self.pose.centers.T
        self.mesh['srcvals'][11] = 1.
        self.density = np.arange(102,dtype=float).reshape(2,3,17)/100
        self.rhs = np.zeros((6,17))
        raw,_ = driver.representation_moments(self.mesh['srcvals'][:3].T,self.mesh['srcvals'][9:12].T,
            self.mesh['weights'],self.density,self.mesh['body_ids'],self.pose.centers,driver.strain_basis())
        self.current = dict(self.pose.metadata(),schema=driver.SCHEMA,convention_id=driver.CONVENTION,
            status='complete',order=2,level=0,focus=0.,alpha=1.,beta=1.,isolated_body=None,
            quadrature_tolerance=1e-7,node_count=2,library_sha256='synthetic_library',
            discretization_sha256='synthetic_mesh',rim_levels=0,rim_width=.05,contact_levels=0,contact_radius=.25,
            source_hashes={'backend.py':'synthetic_backend','geometry.py':'synthetic_geometry'},
            raw_full17=raw.tolist(),linear_residual=1e-12)
        self.path = self.directory/'density.npz'
        digest = driver.write_checkpoint(self.path,self.mesh,self.density,self.rhs)
        self.previous = dict(self.current,solution_file=str(self.path),solution_sha256=digest,
                             rhs_sha256=driver.array_sha256(self.rhs))

    def load(self, previous=None, mesh=None, rhs=None):
        return driver.load_checkpoint(self.directory/'report.json',previous or self.previous,
            self.current,mesh or self.mesh,self.rhs if rhs is None else rhs,self.pose)

    def test_complete_checkpoint_and_legacy_identity(self):
        restored, identity = self.load()
        np.testing.assert_array_equal(restored,self.density)
        self.assertEqual(identity['solution_sha256'],driver.file_sha256(self.path))
        legacy = dict(self.previous)
        legacy.pop('solution_sha256'); legacy.pop('rhs_sha256')
        restored, identity = self.load(legacy)
        self.assertIn('legacy_complete_arrays',identity['checkpoint_identity_verification'])
        np.testing.assert_array_equal(restored,self.density)

    def test_hash_mesh_rhs_source_and_moment_mismatches_rejected(self):
        for defect in ('sha','weights','body_ids','rhs','geometry_source','raw_moments'):
            previous = copy.deepcopy(self.previous)
            mesh = copy.deepcopy(self.mesh)
            rhs = self.rhs.copy()
            if defect == 'sha': previous['solution_sha256'] = 'wrong'
            elif defect == 'weights': mesh['weights'][0] *= 2
            elif defect == 'body_ids': mesh['body_ids'] = mesh['body_ids'][::-1].copy()
            elif defect == 'rhs': rhs[0,0] = 1
            elif defect == 'geometry_source': previous['source_hashes']['geometry.py'] = 'wrong'
            elif defect == 'raw_moments': previous['raw_full17'][0][0] += .1
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                self.load(previous,mesh,rhs)

    def test_out_of_core_resume_binds_completed_density_and_both_solver_sources(self):
        self.current.update(solver='out-of-core-dense',
            solver_contract='out-of-core-dense_combined_dirichlet_full17_v1',
            lu_panel_columns=1024, lu_stripe_columns=512)
        self.current['source_hashes'].update({name:'synthetic_'+name for name in
            ('matrix_free.py','near_apply.py','memory_bounded_dense.py',
             'out_of_core_dense.py','out_of_core_lu.py')})
        previous = dict(copy.deepcopy(self.current), solution_file=str(self.path),
            solution_sha256=driver.file_sha256(self.path), rhs_sha256=driver.array_sha256(self.rhs))
        restored, _ = self.load(previous)
        np.testing.assert_array_equal(restored,self.density)
        for defect in ('out_of_core_dense.py','out_of_core_lu.py',
                       'missing_out_of_core_lu.py','lu_panel_columns','lu_stripe_columns',
                       'partial_factorization'):
            changed = copy.deepcopy(previous)
            if defect == 'missing_out_of_core_lu.py':
                changed['source_hashes'].pop('out_of_core_lu.py')
            elif defect.endswith('.py'):
                changed['source_hashes'][defect] = 'different_source'
            elif defect.startswith('lu_'):
                changed[defect] += 1
            else:
                changed['status'] = 'linear_solve_failed'
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                self.load(changed)

    def test_partial_npz_write_preserves_existing_checkpoint(self):
        before = self.path.read_bytes()
        def fail(stream, **kwargs):
            stream.write(b'partial')
            raise OSError('simulated interrupted checkpoint')
        with mock.patch.object(driver.np,'savez_compressed',side_effect=fail), self.assertRaises(OSError):
            driver.write_checkpoint(self.path,self.mesh,self.density,self.rhs)
        self.assertEqual(self.path.read_bytes(),before)
        self.assertFalse(list(self.directory.glob('*.tmp')))

    def test_failed_resume_keeps_original_report_and_density(self):
        report = self.directory/'reference.json'
        previous = dict(self.previous,assembly_seconds=0.,solve_seconds=0.)
        driver.write_json(report,previous)
        original_report = report.read_bytes()
        original_density = self.path.read_bytes()
        mesh = dict(self.mesh,patches=[(0,np.zeros((4,3))),(1,np.zeros((4,3)))],contact_patch_id_uv=[])
        backend = mock.Mock()
        backend.lib = None
        backend.discretization.return_value = (np.zeros((2,9)),)
        backend.coefficients.side_effect = lambda selected:selected
        backend.boundary_values.side_effect = RuntimeError('simulated failed resumed boundary check')
        library = self.directory/'fake_library'
        library.write_bytes(b'fake library identity only')
        identity = dict(solution_file=str(self.path),solution_sha256=driver.file_sha256(self.path),
                        rhs_sha256=driver.array_sha256(self.rhs))
        argv = ['run_reference','--library',str(library),'--gap-nm','150','--order','2',
                '--resume','--output',str(report)]
        with mock.patch.object(sys,'argv',argv), mock.patch.object(driver,'Backend',return_value=backend), \
                mock.patch.object(driver,'build_mesh',return_value=mesh), \
                mock.patch.object(driver,'load_checkpoint',return_value=(self.density,identity)), \
                self.assertRaisesRegex(RuntimeError,'simulated failed resumed'):
            driver.main()
        self.assertEqual(report.read_bytes(),original_report)
        self.assertEqual(self.path.read_bytes(),original_density)
        attempts = list(self.directory.glob('reference.resume.*.json'))
        preserved = list(self.directory.glob('reference.before_resume.*.json'))
        self.assertEqual(len(attempts),1)
        self.assertEqual(len(preserved),1)
        self.assertEqual(preserved[0].read_bytes(),original_report)
        self.assertEqual(json.loads(attempts[0].read_text())['status'],'boundary_check')

    def test_isolated_evidence_rechecks_limits_and_identity(self):
        record = dict(self.current,isolated_body=0,independent_reference=True,shares_production_solver=False,
            raw_full17=np.eye(17).tolist(),raw_reciprocity=0.,
            isolated_exact_validation={'passed':True,'block_relative_errors':dict(force=0.,torque=0.,stresslet=0.)},
            validation=dict(off_grid_boundary=True,boundary_evaluation_valid=True,boundary_max_scaled_error=0.,
                            exact_surface_position_max_error_over_gap=0.,linear_relative_residual=0.))
        path = self.directory/'isolated.json'
        for defect in (None,'boundary_flag','negative_error','moment_error','geometry_source','wrong_pose','synthetic'):
            changed = copy.deepcopy(record)
            if defect == 'boundary_flag': changed['validation'].pop('boundary_evaluation_valid')
            elif defect == 'negative_error': changed['validation']['boundary_max_scaled_error'] = -1
            elif defect == 'moment_error': changed['isolated_exact_validation']['block_relative_errors']['torque'] = .01
            elif defect == 'geometry_source': changed['source_hashes']['geometry.py'] = 'wrong'
            elif defect == 'wrong_pose': changed['theta_i'] += .1
            elif defect == 'synthetic': changed['synthetic'] = True
            path.write_text(json.dumps(changed))
            evidence = driver.isolated_evidence(path,self.current)
            with self.subTest(defect=defect):
                self.assertEqual(evidence['passed'],defect is None)
                self.assertEqual(evidence['sha256'],driver.file_sha256(path))

    @unittest.skipUnless(shutil.which('gfortran'),'gfortran needed only for native-buffer fixture')
    def test_buffered_fortran_failure_is_captured_and_rejected(self):
        source = self.directory/'diagnostic.f90'
        source.write_text("subroutine diagnostic() bind(C)\nprint *, 'Failed to synthetic GGQ'\nend subroutine\n")
        library = self.directory/'diagnostic.so'
        subprocess.run(['gfortran','-shared','-fPIC',str(source),'-o',str(library)],check=True,capture_output=True)
        native = ctypes.CDLL(str(library))
        native.diagnostic.argtypes=[]; native.diagnostic.restype=None
        path = self.directory/'native.log'
        with self.assertRaisesRegex(RuntimeError,'Native quadrature failed'):
            driver.logged_fortran(path,native.diagnostic,native)
        self.assertIn('Failed to synthetic GGQ',path.read_text())


if __name__ == '__main__':
    unittest.main()
