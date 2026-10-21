"""RAM admission, actual LAPACK, independent residual and cleanup checks."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

import ram_dense as storage
import out_of_core_dense as residual_storage


class RamDenseTests(unittest.TestCase):
    def inputs(self, n=13):
        rng = np.random.default_rng(20261019)
        d = 3*n
        # Deliberately nonsymmetric matrix with pivoting, matching the fact
        # that a combined-layer Dirichlet matrix need not be reciprocal.
        matrix = rng.normal(size=(d,d)) + 5*np.eye(d)
        matrix[[0,1]] = matrix[[1,0]]
        known = rng.normal(size=(d,17))
        mesh = {'srcvals':np.zeros((12,n)), 'norders':np.array([4])}

        def assemble(backend, mesh, eps, alpha, beta, output, chunk, progress):
            for first in range(0,n,chunk):
                last = min(n,first+chunk)
                output[3*first:3*last,:] = matrix[3*first:3*last,:]
                progress(first,last,1.25)

        return matrix, known, mesh, assemble

    def test_in_place_block_solve_original_residual_and_no_matrix_files(self):
        matrix, known, mesh, assemble = self.inputs()
        rhs = matrix@known
        original_rhs = rhs.copy()
        with tempfile.TemporaryDirectory() as folder, \
             mock.patch.object(storage,'_assemble_target_chunks',side_effect=assemble) as assembly, \
             mock.patch.object(residual_storage,'_assemble_target_chunks',side_effect=assemble) as reassembly:
            solution, report = storage.solve_ram_dense(SimpleNamespace(),mesh,rhs,1e-8,folder,
                                                       assembly_target_chunk=5,max_memory_gb=2.)
            np.testing.assert_allclose(solution,known,rtol=1e-11,atol=1e-11)
            np.testing.assert_array_equal(rhs,original_rhs)
            self.assertEqual(assembly.call_count,1)
            self.assertEqual(reassembly.call_count,1)
            self.assertTrue(report['lu_aliases_original_allocation'])
            self.assertTrue(report['original_operator_residual_checked'])
            self.assertLess(report['relative_residual_max'],1e-12)
            self.assertEqual(report['residual_completed_targets'],13)
            self.assertEqual(report['matrix_files_created'],[])
            self.assertEqual([p.suffix for p in Path(folder).rglob('*') if p.is_file()],['.json'])
            self.assertEqual(json.loads(Path(report['diagnostics_path']).read_text())['status'],'complete')
            self.assertFalse(report['physical_accuracy_qualified'])

    def test_vector_rhs_and_no_rhs_overwrite(self):
        matrix, known, mesh, assemble = self.inputs(2)
        rhs = matrix@known[:,0]
        with tempfile.TemporaryDirectory() as folder, \
             mock.patch.object(storage,'_assemble_target_chunks',side_effect=assemble), \
             mock.patch.object(residual_storage,'_assemble_target_chunks',side_effect=assemble):
            solution, report = storage.solve_ram_dense(SimpleNamespace(),mesh,rhs,1e-8,folder,
                                                       max_memory_gb=2.)
        self.assertEqual(solution.shape,(6,))
        np.testing.assert_allclose(solution,known[:,0],rtol=1e-12,atol=1e-12)
        self.assertEqual(report['rhs_count'],1)

    def test_budget_and_live_cgroup_reject_before_allocation_or_workdir(self):
        matrix, known, mesh, assemble = self.inputs(2)
        with tempfile.TemporaryDirectory() as folder:
            for budget in (.01,2.):
                destination = Path(folder)/str(budget)
                with mock.patch.object(storage,'_check_cgroup_headroom',side_effect=MemoryError('headroom')) as guard, \
                     mock.patch.object(storage.np,'empty') as allocate, \
                     self.assertRaises(MemoryError):
                    storage.solve_ram_dense(SimpleNamespace(),mesh,matrix@known,1e-8,destination,
                                            max_memory_gb=budget)
                allocate.assert_not_called()
                self.assertFalse(destination.exists())
                self.assertEqual(guard.call_count, int(budget==2.))

    def test_failed_solve_retains_diagnostics_only(self):
        matrix, known, mesh, assemble = self.inputs(2)
        with tempfile.TemporaryDirectory() as folder, \
             mock.patch.object(storage,'_assemble_target_chunks',side_effect=assemble), \
             mock.patch.object(storage,'lu_solve',side_effect=RuntimeError('injected solve failure')), \
             self.assertRaises(storage.RamDenseError) as caught:
            storage.solve_ram_dense(SimpleNamespace(),mesh,matrix@known,1e-8,folder,max_memory_gb=2.)
        report = caught.exception.diagnostics
        self.assertEqual(report['status'],'failed')
        self.assertEqual(report['matrix_files_created'],[])
        self.assertFalse(report['original_operator_residual_checked'])
        self.assertIn('injected solve failure',report['failure_message'])
        self.assertIn('heap_release_at_exit',report)

    def test_wrong_solution_fails_fresh_operator_residual(self):
        matrix, known, mesh, assemble = self.inputs(2)
        real_solve = storage.lu_solve
        def corrupt(*args,**kwargs):
            return 1.01*real_solve(*args,**kwargs)
        with tempfile.TemporaryDirectory() as folder, \
             mock.patch.object(storage,'_assemble_target_chunks',side_effect=assemble), \
             mock.patch.object(residual_storage,'_assemble_target_chunks',side_effect=assemble), \
             mock.patch.object(storage,'lu_solve',side_effect=corrupt), \
             self.assertRaises(storage.RamDenseError) as caught:
            storage.solve_ram_dense(SimpleNamespace(),mesh,matrix@known,1e-8,folder,max_memory_gb=2.)
        self.assertGreater(caught.exception.diagnostics['relative_residual_max'],.009)
        self.assertTrue(caught.exception.diagnostics['original_operator_residual_checked'])


if __name__ == '__main__':
    unittest.main()
