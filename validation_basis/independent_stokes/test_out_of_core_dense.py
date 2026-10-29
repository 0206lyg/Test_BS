"""Original-operator residual and rolling-row storage checks, not physics."""
import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

import out_of_core_dense as storage
from out_of_core_lu import working_memory_bytes


class OutOfCoreStokesStorageTests(unittest.TestCase):
    def inputs(self):
        n = 13
        rng = np.random.default_rng(819)
        kernels = rng.normal(size=(6, n, n))
        matrix = np.empty((3*n, 3*n))
        for k, (i, j) in enumerate([(0,0),(0,1),(0,2),(1,1),(1,2),(2,2)]):
            matrix[i::3, j::3] = kernels[k]
            matrix[j::3, i::3] = kernels[k]
        matrix += 45*np.eye(3*n)
        expected = matrix + .5*np.eye(3*n)
        known = rng.normal(size=(3*n, 17))
        mesh = dict(srcvals=np.zeros((12,n)),norders=np.array([4]))

        def assemble(backend, mesh, eps, alpha, beta, output, chunk, progress):
            for first in range(0, n, chunk):
                last = min(first+chunk, n)
                for i, j in [(0,0),(0,1),(0,2),(1,1),(1,2),(2,2)]:
                    output[3*first+i:3*last+i:3, j::3] = matrix[3*first+i:3*last+i:3,j::3]
                    if i != j:
                        output[3*first+j:3*last+j:3, i::3] = matrix[3*first+j:3*last+j:3,i::3]
                diagonal = np.arange(3*first,3*last)
                output[diagonal,diagonal] += beta/2
                progress(first,last,1.25)
        return expected, known, mesh, assemble

    def test_all_rhs_and_original_residual_with_short_final_chunk(self):
        matrix, known, mesh, assemble = self.inputs()
        rhs = matrix@known
        original_rhs = rhs.copy()
        with tempfile.TemporaryDirectory() as folder, \
             mock.patch.object(storage,'_assemble_target_chunks',side_effect=assemble) as calls:
            solved, report = storage.solve_out_of_core_dense(SimpleNamespace(),mesh,rhs,
                1e-7,folder,panel_columns=11,stripe_columns=7,assembly_target_chunk=5)
            self.assertEqual(calls.call_count,2)
            self.assertFalse(list(Path(folder).rglob('*.f64')))
        np.testing.assert_allclose(solved,known,rtol=1e-12,atol=1e-12)
        np.testing.assert_array_equal(rhs,original_rhs)
        self.assertLess(report['relative_residual_max'],1e-12)
        self.assertTrue(report['original_operator_residual_checked'])
        self.assertEqual(report['residual_completed_targets'],13)
        self.assertFalse(report['physical_accuracy_qualified'])
        self.assertEqual(report['schema'],'independent_stokes_out_of_core_dense_v3')
        self.assertTrue(report['original_operator_hash']['digest_valid'])
        self.assertEqual(report['original_operator_sha256_before_factorization'],
                         hashlib.sha256(matrix.tobytes(order='F')).hexdigest())

    def test_reassembled_operator_mismatch_is_not_hidden_by_factor_residual(self):
        matrix, known, mesh, assemble = self.inputs()
        real_residual = storage.reassembled_residual
        def changed_solution(*args,**kwargs):
            changed = list(args)
            changed[3] = 1.01*changed[3]
            return real_residual(*changed,**kwargs)
        with tempfile.TemporaryDirectory() as folder, \
             mock.patch.object(storage,'_assemble_target_chunks',side_effect=assemble), \
             mock.patch.object(storage,'reassembled_residual',side_effect=changed_solution), \
             self.assertRaises(storage.MMapDenseError) as caught:
            storage.solve_out_of_core_dense(SimpleNamespace(),mesh,matrix@known,1e-7,
                folder,panel_columns=11,stripe_columns=7,assembly_target_chunk=5)
        report = caught.exception.diagnostics
        self.assertEqual(report['status'],'failed')
        self.assertGreater(report['relative_residual_max'],.009)
        self.assertFalse(report['physical_accuracy_qualified'])

    def test_bounded_hash_same_bytes_and_own_read_ranges(self):
        payload = bytes(range(251))*41
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'owned_operator.f64';path.write_bytes(payload)
            details = {}
            with mock.patch.object(storage.os,'posix_fadvise') as advise:
                digest = storage._hash_file_cache_bounded(path,details,chunk_bytes=4096)
            self.assertEqual(digest,hashlib.sha256(payload).hexdigest())
            self.assertEqual(path.read_bytes(),payload)
            self.assertEqual(details['bytes_read'],len(payload))
            self.assertTrue(details['digest_valid'])
            self.assertFalse(details['cache_eviction_guaranteed'])
            self.assertEqual([(call.args[1],call.args[2]) for call in advise.call_args_list],
                             [(0,4096),(4096,4096),(8192,len(payload)-8192),(0,0)])
            self.assertEqual(details['advice_calls_accepted'],4)

    def test_hash_advice_failure_does_not_invent_eviction_or_change_digest(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'owned_operator.f64';path.write_bytes(b'0123456789')
            details = {}
            with mock.patch.object(storage.os,'posix_fadvise',side_effect=OSError(22,'unsupported')):
                digest = storage._hash_file_cache_bounded(path,details,chunk_bytes=4)
            self.assertEqual(digest,hashlib.sha256(b'0123456789').hexdigest())
            self.assertEqual(details['advice_calls_accepted'],0)
            self.assertEqual(details['advice_calls_failed'],4)
            self.assertTrue(details['digest_valid'])
            self.assertFalse(details['cache_eviction_guaranteed'])

    def test_hash_read_failure_never_publishes_digest(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'owned_operator.f64';path.write_bytes(b'0123456789')
            real_open = Path.open
            class FailingReader:
                def __init__(self):
                    self.stream = real_open(path,'rb',buffering=0);self.calls = 0
                def __enter__(self):return self
                def __exit__(self,*args):self.stream.close()
                def fileno(self):return self.stream.fileno()
                def read(self,count):
                    self.calls += 1
                    if self.calls > 1:raise OSError('injected read failure')
                    return self.stream.read(count)
            details = {}
            with mock.patch.object(Path,'open',return_value=FailingReader()), \
                 self.assertRaises(OSError):
                storage._hash_file_cache_bounded(path,details,chunk_bytes=4)
            self.assertEqual(details['status'],'failed')
            self.assertEqual(details['bytes_read'],4)
            self.assertFalse(details['digest_valid'])
            self.assertNotIn('sha256',details)

    def test_maximum_stage_admission_precedes_any_matrix_write(self):
        matrix,known,mesh,assemble = self.inputs()
        rhs=matrix@known;n=13;d=39;chunk=5
        assembly=(256*n*chunk+16*chunk+8*d*3*chunk+256*1024**2+4*rhs.nbytes)
        factor=working_memory_bytes(d,17,d,d)
        self.assertGreater(factor,assembly)
        with tempfile.TemporaryDirectory() as folder:
            for budget,actual_failure in [((assembly+factor)/2/1e9,False),(1.,True)]:
                destination=Path(folder)/str(actual_failure)
                with mock.patch.object(storage,'_check_cgroup_headroom',side_effect=MemoryError('actual guard')) as guard, \
                     mock.patch.object(storage.np,'memmap') as mapping, \
                     mock.patch.object(storage,'_assemble_target_chunks') as assembler, \
                     self.assertRaises(MemoryError):
                    storage.solve_out_of_core_dense(SimpleNamespace(),mesh,rhs,1e-7,destination,
                        max_memory_gb=budget,panel_columns=d,stripe_columns=d,assembly_target_chunk=chunk)
                self.assertFalse(destination.exists());mapping.assert_not_called();assembler.assert_not_called()
                if actual_failure:guard.assert_called_once_with(factor,250000000)
                else:guard.assert_not_called()


if __name__ == '__main__':
    unittest.main()
