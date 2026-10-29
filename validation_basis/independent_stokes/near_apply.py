"""Optional fused application of all-near Stokes quadrature weights.

Build explicitly; importing or constructing this class never compiles code::

    python near_apply.py build --output /absolute/path/libnear_apply.so

``NearApply(path, threads=8).matvec(wnear, vector)`` returns the raw integral
potential; the caller still adds its original exterior jump. Weights are
float64 Fortran-contiguous (6,N*N), ordered (xx,xy,xz,yy,yz,zz), with columns
target*N+source. Vectors use point-interleaved components. No full matrix or
weights copy is made. Build provenance is required and checked on loading.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
from pathlib import Path
import subprocess
import tempfile

import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def build(output, compiler='g++'):
    """Compile only on explicit request and write a hash-bound manifest."""
    source = Path(__file__).with_suffix('.cpp').resolve()
    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or Path(str(output)+'.json').exists():
        raise FileExistsError('Use a fresh output path; loaded libraries must not be replaced')
    flags = ['-std=c++11', '-O3', '-fopenmp', '-shared', '-fPIC', '-ffp-contract=off']
    compiler_version = subprocess.check_output([compiler, '--version'], text=True)
    with tempfile.TemporaryDirectory(prefix='near_apply_build_', dir=output.parent) as folder:
        temporary = Path(folder)/output.name
        command = [compiler, *flags, str(source), '-o', str(temporary)]
        subprocess.run(command, check=True)
        manifest = {'schema': 'spheroid_near_apply_build_v1', 'abi': 1,
                    'source_path': str(source), 'source_sha256': sha256(source),
                    'library_path': str(output), 'library_sha256': sha256(temporary),
                    'compiler': compiler, 'compiler_version': compiler_version,
                    'compile_flags': flags, 'fast_math': False}
        temporary.rename(output)
        Path(str(output)+'.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest


class NearApply:
    """Immutable loaded helper; allocation is O(N*nrhs), never O(N*N)."""
    def __init__(self, library, threads=0):
        if int(threads) != threads or threads < 0:
            raise ValueError('threads must be a nonnegative integer; zero uses OpenMP default')
        self.threads = int(threads)
        library = Path(library).expanduser().resolve()
        manifest_path = Path(str(library)+'.json')
        manifest = json.loads(manifest_path.read_text())
        actual_hash = sha256(library)
        if manifest.get('schema') != 'spheroid_near_apply_build_v1' or manifest.get('abi') != 1:
            raise ValueError('Unsupported near-apply manifest')
        if manifest.get('library_sha256') != actual_hash:
            raise ValueError('Near-apply library hash does not match its build manifest')
        if manifest.get('source_sha256') != sha256(Path(__file__).with_suffix('.cpp')):
            raise ValueError('Near-apply source changed; explicitly rebuild to a fresh path')
        if manifest.get('fast_math') is not False:
            raise ValueError('Near-apply build must preserve ordinary floating-point arithmetic')
        self.provenance = {**manifest, 'loaded_library_path': str(library),
                           'manifest_path': str(manifest_path),
                           'manifest_sha256': sha256(manifest_path),
                           'threads': self.threads}
        self.path = str(library)
        self.library = ctypes.CDLL(str(library))
        self.library.spheroid_near_apply_abi.argtypes = []
        self.library.spheroid_near_apply_abi.restype = ctypes.c_int
        if self.library.spheroid_near_apply_abi() != 1:
            raise ValueError('Near-apply native ABI mismatch')
        self._apply = self.library.spheroid_near_apply
        ptr = ctypes.POINTER(ctypes.c_double)
        self._apply.argtypes = [ctypes.c_int64, ctypes.c_int64, ptr, ptr, ptr, ctypes.c_int]
        self._apply.restype = ctypes.c_int

    def metadata(self):
        return dict(self.provenance)

    def _weights(self, weights):
        if not isinstance(weights, np.ndarray) or weights.dtype != np.float64:
            raise ValueError('Weights must already be a float64 ndarray; no implicit large copies')
        if weights.ndim != 2 or weights.shape[0] != 6 or not weights.flags.f_contiguous:
            raise ValueError('Weights must be Fortran-contiguous with shape (6,N*N)')
        n = math.isqrt(weights.shape[1])
        if n < 1 or n*n != weights.shape[1]:
            raise ValueError('Weights must have exactly N*N columns')
        # Repeated O(N*N) finiteness scans would defeat the fused application.
        # The context preparing these weights is responsible for checking them.
        return n

    def _run(self, weights, vectors, out):
        n = self._weights(weights)
        vectors = np.asarray(vectors, dtype=np.float64)
        if vectors.ndim != 2 or vectors.shape[0] != 3*n or vectors.shape[1] < 1:
            raise ValueError('Vectors must have shape (3*N,nrhs)')
        if not np.isfinite(vectors).all():
            raise ValueError('Vectors must be finite')
        vectors = np.ascontiguousarray(vectors)
        if out is None:
            out = np.empty(vectors.shape, dtype=np.float64, order='C')
        elif (not isinstance(out, np.ndarray) or out.shape != vectors.shape or
              out.dtype != np.float64 or not out.flags.c_contiguous or not out.flags.writeable):
            raise ValueError('Output must be writable float64 C-contiguous with the requested shape')
        if np.shares_memory(out, weights) or np.shares_memory(out, vectors):
            raise ValueError('Output must not overlap either input')
        pointer = ctypes.POINTER(ctypes.c_double)
        status = self._apply(n, vectors.shape[1], weights.ctypes.data_as(pointer),
                             vectors.ctypes.data_as(pointer), out.ctypes.data_as(pointer),
                             self.threads)
        if status:
            raise RuntimeError(f'Near-apply native argument check failed: status {status}')
        if not np.isfinite(out).all():
            raise FloatingPointError('Non-finite near-apply output')
        return out

    def matvec(self, weights, vector, out=None):
        """Return raw potential in the input's flat (3N) or point (N,3) shape."""
        vector = np.asarray(vector, dtype=np.float64)
        n = self._weights(weights)
        if vector.shape not in [(3*n,), (n, 3)]:
            raise ValueError('Vector must be flat (3*N,) or point-interleaved (N,3)')
        if out is not None and (not isinstance(out, np.ndarray) or out.shape != vector.shape or
                                not out.flags.c_contiguous):
            raise ValueError('Output must match the input shape and be C-contiguous')
        result = self._run(weights, vector.reshape(3*n, 1),
                           None if out is None else out.reshape(3*n, 1))
        return result.reshape(vector.shape)

    def matmat(self, weights, vectors, out=None):
        """Apply to (3*N,nrhs) inputs in one weights pass per target."""
        return self._run(weights, vectors, out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    builder = sub.add_parser('build', help='Explicitly compile and record provenance')
    builder.add_argument('--output', required=True)
    builder.add_argument('--compiler', default='g++')
    args = parser.parse_args()
    if args.command == 'build':
        print(json.dumps(build(args.output, args.compiler), indent=2))


if __name__ == '__main__':
    main()
