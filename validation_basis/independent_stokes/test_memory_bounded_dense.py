"""Synthetic storage/linear-algebra checks; no particle or Stokes solves."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import memory_bounded_dense as storage


class StorageModes(unittest.TestCase):
    def inputs(self,n):
        rng = np.random.default_rng(20261018)
        d = 3*n
        matrix = rng.standard_normal((d,d))
        matrix = np.asfortranarray((matrix+matrix.T)*.5)
        matrix.flat[::d+1] = np.abs(matrix).sum(axis=1)+1.
        known = rng.standard_normal((d,17))
        mesh = {'srcvals':np.zeros((12,n),order='F'),'norders':np.array([1],dtype=np.int64)}
        def assemble(backend,mesh,eps,alpha,beta,output,chunk,progress):
            output[:] = matrix
            progress(0,n,1.25)
        return matrix,known,matrix@known,mesh,assemble

    def test_modes_equal_original_preserved_and_private_pages_not_discarded(self):
        n = int(os.environ.get('COW_TEST_NODES','32'))
        matrix,known,rhs,mesh,assemble = self.inputs(n)
        results = {};solutions = {}
        original_drop = storage._drop_clean_pages
        def checked_drop(mapping,path,record):
            self.assertNotEqual(mapping.mode,'c')
            return original_drop(mapping,path,record)
        with tempfile.TemporaryDirectory() as folder:
            for mode in ('shared','cow'):
                with mock.patch.object(storage,'_assemble_target_chunks',side_effect=assemble), \
                     mock.patch.object(storage,'_drop_clean_pages',side_effect=checked_drop):
                    solution,record = storage.solve_mmap_dense(SimpleNamespace(path='synthetic'),
                        mesh,rhs,1e-7,folder,factor_mode=mode,keep_matrices=True,max_memory_gb=2.)
                mapped = np.memmap(record['operator_path'],dtype=np.float64,mode='r',shape=matrix.shape,order='F')
                self.assertTrue(np.array_equal(mapped,matrix));mapped._mmap.close()
                error = float(np.linalg.norm(solution-known)/np.linalg.norm(known))
                self.assertLess(error,1e-10)
                self.assertLess(record['relative_residual_max'],1e-10)
                self.assertTrue(record['lu_aliases_memmap'])
                if mode == 'cow':
                    self.assertTrue(record['original_operator_unchanged'])
                    self.assertIsNone(record['lu_path'])
                    self.assertFalse(Path(record['operator_path']).with_name('lu.f64').exists())
                else:
                    self.assertTrue(Path(record['lu_path']).exists())
                results[mode] = {'known_solution_relative_error':error,'solver':record}
                solutions[mode] = solution
            difference = float(np.linalg.norm(solutions['cow']-solutions['shared'])/np.linalg.norm(solutions['shared']))
            self.assertLess(difference,1e-12)
            if os.environ.get('COW_TEST_REPORT'):
                report = {'schema':'mmap_cow_modes_synthetic_check_v2','synthetic':True,
                    'physical_accuracy_qualified':False,'scope':'Storage/linear algebra only; assembly is replaced by an identical deterministic SPD matrix in both modes.',
                    'equivalent_node_count':n,'matrix_dimension':3*n,'rhs_count':17,
                    'cow_vs_shared_solution_relative_error':difference,'modes':results,
                    'source_hashes':{name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                     for name in ['memory_bounded_dense.py','test_memory_bounded_dense.py']},
                    'performance_scope':'Small matrix fits RAM; timings do not predict N9963 performance under memory pressure.'}
                Path(os.environ['COW_TEST_REPORT']).write_text(json.dumps(report,indent=2)+'\n')

    def test_private_failure_cleanup_and_retained_original(self):
        matrix,known,rhs,mesh,assemble = self.inputs(2)
        with tempfile.TemporaryDirectory() as folder:
            for keep in (False,True):
                with mock.patch.object(storage,'_assemble_target_chunks',side_effect=assemble), \
                     mock.patch.object(storage,'lu_solve',side_effect=RuntimeError('injected solve failure')), \
                     self.assertRaises(storage.MMapDenseError) as caught:
                    storage.solve_mmap_dense(SimpleNamespace(path='synthetic'),mesh,rhs,1e-7,
                                             folder,factor_mode='cow',keep_matrices=keep)
                record = caught.exception.diagnostics
                self.assertEqual(record['status'],'failed')
                if keep:
                    mapped = np.memmap(record['operator_path'],dtype=np.float64,mode='r',shape=matrix.shape,order='F')
                    self.assertTrue(np.array_equal(mapped,matrix));mapped._mmap.close()
                else:
                    self.assertFalse(any(record['matrix_files_retained'].values()))

    def test_dirty_inactive_cache_is_not_credited_as_clean(self):
        files = {'/proc/self/cgroup':'0::/\n','/sys/fs/cgroup/memory.max':'1000',
                 '/sys/fs/cgroup/memory.current':'900',
                 '/sys/fs/cgroup/memory.stat':'file 800\ninactive_file 400\nfile_dirty 300\nfile_writeback 0\nshmem 0\n'}
        def read(path,*args,**kwargs):
            if str(path) not in files:
                raise FileNotFoundError(str(path))
            return files[str(path)]
        with mock.patch.object(Path,'read_text',read):
            snapshot = storage._check_cgroup_headroom(100,100)
            self.assertEqual(snapshot['reclaimable_clean_inactive_file_bytes'],100)
            self.assertEqual(snapshot['effective_current_bytes'],800)
            with self.assertRaises(MemoryError):
                storage._check_cgroup_headroom(101,100)

    def test_stage_admission_keeps_strict_actual_factor_guard(self):
        matrix,known,rhs,mesh,assemble = self.inputs(2)
        n=2;d=6;reserve=250000000;slack=256*1024**2
        assembly=256*n*n+16*n+slack
        factor=8*d*d+8*d*d+8*rhs.nbytes+slack
        capacity=2_000_000_000
        # Assembly fits current headroom; LU only fits nominal capacity.
        baseline=capacity-reserve-assembly-1024
        snapshot={'path':'synthetic','limit_bytes':capacity,'current_bytes':baseline,
                  'effective_current_bytes':baseline,'minimum_hierarchy_limit_bytes':capacity}
        with tempfile.TemporaryDirectory() as folder, \
             mock.patch.object(storage,'_cgroup_memory_snapshot',side_effect=lambda:dict(snapshot)), \
             mock.patch.object(storage,'_assemble_target_chunks',side_effect=assemble) as prepared, \
             mock.patch.object(storage,'lu_factor') as factorize, \
             self.assertRaises(storage.MMapDenseError) as caught:
            storage.solve_mmap_dense(SimpleNamespace(path='synthetic'),mesh,rhs,1e-7,
                                     folder,factor_mode='cow',cgroup_reserve_gb=.25)
        self.assertEqual(prepared.call_count,1)
        factorize.assert_not_called()
        record=caught.exception.diagnostics
        self.assertEqual(record['cgroup_preflight']['required_additional_bytes'],assembly)
        self.assertEqual(record['cgroup_capacity_preflight']['required_factor_bytes'],factor)
        self.assertTrue(record['cgroup_capacity_preflight']['actual_factor_headroom_check_required'])
        self.assertEqual(record['cgroup_guard_failure']['required_additional_bytes'],factor)
        self.assertFalse(any(record['matrix_files_retained'].values()))
        # Intrinsically oversized LU is rejected even with zero baseline.
        snapshot.update(current_bytes=0,effective_current_bytes=0,limit_bytes=factor+reserve-1,
                        minimum_hierarchy_limit_bytes=factor+reserve-1)
        with mock.patch.object(storage,'_cgroup_memory_snapshot',return_value=dict(snapshot)), \
             self.assertRaises(MemoryError):
            storage._check_cgroup_capacity(factor,reserve)


if __name__ == '__main__':
    unittest.main()
