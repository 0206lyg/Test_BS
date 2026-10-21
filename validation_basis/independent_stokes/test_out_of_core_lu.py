"""Exact linear algebra/storage checks; these do not qualify Stokes physics."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
from scipy.linalg import lu_factor, solve

import out_of_core_lu as ooc


class OutOfCoreLUTests(unittest.TestCase):
    def run_matrix(self,matrix,panel=3,stripe=2,vector=False):
        d = len(matrix)
        rng = np.random.default_rng(20261018)
        known = rng.standard_normal((d,17))
        rhs = matrix @ known
        if vector:
            rhs,known = rhs[:,0],known[:,0]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'matrix.f64'
            mm = np.memmap(path,dtype=float,mode='w+',shape=(d,d),order='F')
            mm[:] = matrix; mm.flush(); mm._mmap.close()
            states = []
            solution,record = ooc.factor_solve(path,d,rhs,panel_columns=panel,
                stripe_columns=stripe,progress=lambda stage,record:states.append(stage))
            factors = np.array(np.memmap(path,dtype=float,mode='r',shape=(d,d),order='F'))
            self.assertLess(np.linalg.norm(solution-known)/np.linalg.norm(known),1e-10)
            self.assertLess(np.linalg.norm(solution-solve(matrix,rhs))/np.linalg.norm(solution),1e-11)
            residual = np.linalg.norm(matrix@solution-rhs)/np.linalg.norm(rhs)
            self.assertLess(residual,1e-11)
            # Match complete global pivoting and standard storage, not merely x.
            native,pivots = lu_factor(matrix)
            permuted = matrix.copy()
            for i,p in enumerate(pivots):
                permuted[[i,p]] = permuted[[p,i]]
            lower = np.tril(factors,-1)+np.eye(d)
            upper = np.triu(factors)
            self.assertLess(np.linalg.norm(permuted-lower@upper)/np.linalg.norm(matrix),1e-11)
            self.assertEqual(record['pivot_sha256'],
                             hashlib.sha256(pivots.astype(np.int64).tobytes()).hexdigest())
            self.assertLessEqual(record['maximum_explicit_buffer_bytes'],record['estimated_working_bytes'])
            self.assertFalse(record['original_operator_residual_checked'])
            self.assertFalse(record['physical_accuracy_qualified'])
            self.assertEqual(record['factor_completed_columns'],d)
            self.assertEqual(record['forward_completed_columns'],d)
            self.assertEqual(record['backward_completed_columns'],d)
            self.assertIn('forward_solving',states); self.assertIn('backward_solving',states)
            return record,residual

    def test_cross_panel_previous_l_pivots_noninvolutive_permutation(self):
        matrix = np.array([[2,4,1,-1,3],[7,2,5,4,-2],[1,8,6,3,5],
                           [9,3,2,7,1],[4,6,9,2,8]],dtype=float)
        _,pivots = lu_factor(matrix)
        self.assertEqual(pivots.tolist(),[3,2,4,3,4])
        record,_ = self.run_matrix(matrix,panel=2,stripe=3)
        self.assertEqual(record['pivot_swap_count'],3)
        self.assertEqual(record['maximum_pivot_distance'],3)

    def test_random_nonsymmetric_nondivisible_and_vector(self):
        rng = np.random.default_rng(818)
        matrix = rng.standard_normal((23,23))
        self.run_matrix(matrix,panel=7,stripe=5)
        self.run_matrix(matrix,panel=8,stripe=4,vector=True)

    def test_single_final_panel_and_one_by_one(self):
        self.run_matrix(np.array([[2.,3.],[5.,7.]]),panel=99,stripe=99)
        self.run_matrix(np.array([[3.]]),panel=1,stripe=1)

    def test_singular_last_partial_panel_fails_before_solve(self):
        matrix = np.array([[2,4,1,-1,0],[7,2,5,4,0],[1,8,6,3,0],
                           [9,3,2,7,0],[4,6,9,2,0]],dtype=float)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'singular.f64'
            mm = np.memmap(path,dtype=float,mode='w+',shape=(5,5),order='F')
            mm[:] = matrix;mm.flush();mm._mmap.close()
            with self.assertRaises(ooc.OutOfCoreLUError) as caught:
                ooc.factor_solve(path,5,np.ones((5,17)),panel_columns=2,stripe_columns=3)
            record = caught.exception.diagnostics
            self.assertEqual(record['singular_pivot_index_zero_based'],4)
            self.assertEqual(record['status'],'failed')
            self.assertEqual(record['forward_completed_columns'],0)
            self.assertTrue(record['destructive_in_place'])

    def test_guard_rejects_before_overwrite_and_buffer_estimate_is_linear(self):
        self.assertLess(ooc.working_memory_bytes(38637,17,1024,512),2_000_000_000)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'guard.f64'
            original = np.eye(8,dtype=float).tobytes(order='F');path.write_bytes(original)
            with self.assertRaises(MemoryError):
                ooc.factor_solve(path,8,np.ones((8,17)),max_memory_gb=.001)
            self.assertEqual(path.read_bytes(),original)
            with mock.patch.object(ooc,'_check_cgroup_headroom',side_effect=MemoryError('guard')):
                with self.assertRaises(MemoryError):
                    ooc.factor_solve(path,8,np.ones((8,17)))
            self.assertEqual(path.read_bytes(),original)

    @unittest.skipUnless(os.environ.get('OOC_LU_BENCH_DIMENSION'),'optional moderate benchmark')
    def test_moderate_benchmark(self):
        d = int(os.environ['OOC_LU_BENCH_DIMENSION'])
        panel = int(os.environ.get('OOC_LU_BENCH_PANEL','512'))
        stripe = int(os.environ.get('OOC_LU_BENCH_STRIPE','256'))
        rng = np.random.default_rng(20261019)
        # Dense nonsymmetric, well-conditioned, with global row pivots forced.
        matrix = rng.standard_normal((d,d))/np.sqrt(d)+2*np.eye(d)
        matrix = matrix[np.roll(np.arange(d),d//3)]
        record,residual = self.run_matrix(matrix,panel=panel,stripe=stripe)
        output = os.environ.get('OOC_LU_BENCH_REPORT')
        if output:
            Path(output).write_text(json.dumps({'schema':'out_of_core_lu_synthetic_check_v1',
                'synthetic':True,'physical_accuracy_qualified':False,
                'true_original_operator_relative_residual':float(residual),
                'known_solution_and_scipy_and_standard_LU_comparison_passed':True,
                'solver':record,
                'test_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2)+'\n')


if __name__ == '__main__':
    unittest.main()
