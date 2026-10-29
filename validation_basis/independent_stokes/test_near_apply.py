"""Focused ABI/order/roundoff and bounded-memory timing checks; no physics claim."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
import time
import unittest

import numpy as np

from near_apply import NearApply, build


def reference(weights, vectors):
    n = vectors.shape[0]//3
    sigma = vectors.reshape(n, 3, -1)
    result = np.zeros_like(sigma)
    kernels = weights.reshape(6, n, n)
    for k, (i, j) in enumerate([(0,0), (0,1), (0,2), (1,1), (1,2), (2,2)]):
        result[:, i] += np.einsum('ts,sr->tr', kernels[k], sigma[:, j], optimize=False)
        if i != j:
            result[:, j] += np.einsum('ts,sr->tr', kernels[k], sigma[:, i], optimize=False)
    return result.reshape(vectors.shape)


class NearApplyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory(prefix='near_apply_test_')
        cls.path = Path(cls.folder.name)/'libnear_apply.so'
        build(cls.path)
        cls.one = NearApply(cls.path, threads=1)
        cls.many = NearApply(cls.path, threads=4)
        rng = np.random.default_rng(71923)
        cls.weights = np.asfortranarray(rng.standard_normal((6, 200*200)))
        cls.vector = rng.standard_normal(600)

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def test_matvec_matches_einsum_and_is_thread_deterministic(self):
        expected = reference(self.weights, self.vector.reshape(-1,1)).ravel()
        got = self.one.matvec(self.weights, self.vector)
        np.testing.assert_allclose(got, expected, rtol=5e-14, atol=2e-13)
        np.testing.assert_array_equal(self.many.matvec(self.weights, self.vector), got)

    def test_all_seventeen_rhs_and_output_buffer(self):
        rhs = np.random.default_rng(991).standard_normal((600,17))
        out = np.empty_like(rhs)
        got = self.many.matmat(self.weights, rhs, out=out)
        self.assertIs(got, out)
        np.testing.assert_allclose(got, reference(self.weights, rhs), rtol=5e-14, atol=2e-13)
        for j in range(17):
            np.testing.assert_array_equal(got[:,j], self.many.matvec(self.weights, rhs[:,j]))

    def test_point_shape_and_inputs_unchanged(self):
        weights = self.weights.copy(order='F')
        vector = self.vector.copy().reshape(-1,3)
        got = self.many.matvec(weights, vector)
        self.assertEqual(got.shape, (200,3))
        np.testing.assert_array_equal(weights, self.weights)
        np.testing.assert_array_equal(vector.ravel(), self.vector)

    def test_input_validation_and_alias_rejection(self):
        for bad in [self.weights.copy(order='C'), self.weights.astype(np.float32),
                    self.weights[:,:-1], np.empty((6,0), order='F')]:
            with self.assertRaises(ValueError):
                self.one.matvec(bad, self.vector)
        with self.assertRaises(ValueError):
            self.one.matvec(self.weights, self.vector, out=self.vector)
        with self.assertRaises(ValueError):
            self.one.matvec(self.weights, self.vector[:-1])
        with self.assertRaises(ValueError):
            self.one.matvec(self.weights, np.full(600, np.nan))
        with self.assertRaises(ValueError):
            NearApply(self.path, threads=-1)

    def test_explicit_build_provenance(self):
        metadata = self.one.metadata()
        self.assertEqual(metadata['schema'], 'spheroid_near_apply_build_v1')
        self.assertEqual(len(metadata['library_sha256']), 64)
        self.assertIs(metadata['fast_math'], False)
        with self.assertRaises(FileExistsError):
            build(self.path)
        copy = Path(self.folder.name)/'corrupt.so'
        copy.write_bytes(self.path.read_bytes()+b'x')
        Path(str(copy)+'.json').write_text(Path(str(self.path)+'.json').read_text())
        with self.assertRaises(ValueError):
            NearApply(copy)


def timed(function, repetitions=3):
    function()
    samples = []
    for _ in range(repetitions):
        start = time.monotonic()
        function()
        samples.append(time.monotonic()-start)
    return {'minimum_seconds': min(samples), 'median_seconds': float(np.median(samples)),
            'samples_seconds': samples}


def benchmark(library, output, sizes=(2352,4000), threads=8):
    helper = NearApply(library, threads=threads)
    rng = np.random.default_rng(207198)
    report = {'schema': 'near_apply_verification_v1', 'physical_accuracy_qualified': False,
              'scope': 'Stored-kernel application algebra and wall time only',
              'provenance': helper.metadata(), 'cpu_count': os.cpu_count(), 'cases': []}
    for n in sizes:
        # Generate directly into Fortran storage; no second N*N allocation.
        weights = np.empty((6,n*n), dtype=np.float64, order='F')
        rng.standard_normal(out=weights.ravel(order='F'))
        vector = rng.standard_normal(3*n)
        expected = reference(weights, vector.reshape(-1,1)).ravel()
        got = helper.matvec(weights, vector)
        difference = got-expected
        native = timed(lambda: helper.matvec(weights, vector))
        baseline = timed(lambda: reference(weights, vector.reshape(-1,1)))
        report['cases'].append({'npts': n, 'weights_bytes': weights.nbytes,
                               'relative_l2_error': float(np.linalg.norm(difference)/np.linalg.norm(expected)),
                               'max_absolute_error': float(np.max(np.abs(difference))),
                               'native': native, 'einsum': baseline,
                               'median_speedup': baseline['median_seconds']/native['median_seconds']})
        del weights
    Path(output).write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark-library')
    parser.add_argument('--benchmark-output')
    parser.add_argument('--threads', type=int, default=8)
    args, rest = parser.parse_known_args()
    if args.benchmark_library:
        if not args.benchmark_output:
            parser.error('--benchmark-output is required for timing')
        benchmark(args.benchmark_library, args.benchmark_output, threads=args.threads)
    else:
        unittest.main(argv=[__file__, *rest])
