#!/usr/bin/env python3
"""Build the distance-scan adapter against a verified existing BIE installation.

The upstream Fortran library is reused byte-for-byte. New Python sources and
the rebuilt C++ exporter receive a separate, immutable build record.
"""
from __future__ import annotations
import argparse
import ctypes
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def source_identity(root=ROOT):
    paths = set()
    for directory in ('validation_basis/independent_stokes', 'lammps_slurry/src',
                      'lammps_slurry/vendor', 'lammps_slurry/tools'):
        paths.update(p for p in (root / directory).rglob('*')
                     if p.is_file() and (p.suffix in ('.py', '.cpp', '.h', '.hpp', '.patch', '.sh')
                                        or 'Eigen' in p.parts) and '__pycache__' not in p.parts)
    paths.update(p for p in root.iterdir() if p.is_file() and p.suffix in ('.sh', '.sbatch'))
    return {str(p.relative_to(root)): sha(p) for p in sorted(paths)}


def verified_backend(old_path):
    old = json.loads(old_path.read_text())
    if old.get('schema') not in ('reference_table_build_v1', 'distance_scan_build_v1') or old.get('status') != 'ready':
        raise ValueError('Expected a ready reference build manifest')
    paths = {}
    for key in ('reference_library', 'backend_manifest'):
        p = Path(old['paths'][key]).resolve(strict=True)
        entry = old['artifacts'][key]
        if p != Path(entry['path']).resolve() or sha(p) != entry['sha256']:
            raise ValueError('Existing backend artifact changed: ' + key)
        paths[key] = str(p)
    backend = json.loads(Path(paths['backend_manifest']).read_text())
    if (backend.get('library_sha256') != sha(paths['reference_library']) or
            backend.get('patch_policy') != 'get_quadparams_adap_high_order_p_ge_4_v1' or
            backend.get('upstream_revision') != 'be6b2b3700b77e9cbf20c512ff4f0922aaee45d9' or
            backend.get('supported_panel_order_minimum') != 4):
        raise ValueError('Existing BIE backend pin/patch identity is incompatible')
    return old, paths


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    base = Path('/scratch/negishi') / os.environ.get('USER', 'unknown')
    p.add_argument('--existing-build-manifest', type=Path,
                   default=base/'slurry_reference_build/build_reference_manifest.json')
    p.add_argument('--prefix', type=Path, default=base/'slurry_distance_scan_build')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args(argv)
    old_path = a.existing_build_manifest.expanduser().resolve(strict=True)
    old, paths = verified_backend(old_path)
    os.environ.update(old.get('blas_configuration', {}).get('runtime_environment', {}))
    compiler = shutil.which(os.environ.get('CXX', 'g++'))
    if not compiler:
        raise ValueError('C++17 compiler is missing')
    prefix = a.prefix.expanduser().resolve()
    if prefix == ROOT or ROOT in prefix.parents:
        raise ValueError('--prefix must be outside this source directory')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    build = prefix / ('build_' + stamp)
    exporter = build/'slurry_reference_operator_export'
    command = [compiler, '-std=c++17', '-O3', '-DNDEBUG', '-DSLURRY_HAVE_EIGEN=1',
               '-DEIGEN_DONT_PARALLELIZE', '-I'+str(ROOT/'lammps_slurry/src'),
               '-I'+str(ROOT/'lammps_slurry/vendor/eigen3'),
               '-I'+str(ROOT/'lammps_slurry/vendor/gr_re2'),
               str(HERE/'operator_export.cpp'), '-o', str(exporter)]
    if a.dry_run:
        print(json.dumps({'command': command, 'reused_backend': paths,
                          'manifest': str(prefix/'build_distance_scan_manifest.json')}, indent=2))
        return 0
    prefix.mkdir(parents=True, exist_ok=True)
    with (prefix/'build.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        build.mkdir()
        sources = source_identity()
        packages = {name: importlib.metadata.version(name) for name in ('numpy', 'scipy', 'threadpoolctl')}
        lib = ctypes.CDLL(paths['reference_library'], mode=ctypes.RTLD_LOCAL)
        for symbol in ('get_disc_exps_', 'surf_vals_to_coefs_', 'get_qwts_',
                       'stok_comb_vel_matgen_', 'getnearquad_stok_comb_vel_eval_', 'get_basis_pols_'):
            getattr(lib, symbol)
        print('Compiling distance-scan operator exporter', flush=True)
        subprocess.run(command, check=True)
        for ar in (2, 5, 10, 20):
            exported = json.loads(subprocess.check_output([str(exporter), '--gap-over-c', '1',
                '--aspect-ratio', str(ar), '--theta-i', '0', '--theta-j', '0', '--phi', '0'], text=True))
            if exported.get('convention_id') != 'slurry_full17_c_eta_1_v1':
                raise ValueError('Operator export convention mismatch')
            for key in ('isolated17', 'leading17'):
                if len(exported[key]) != 17 or any(len(row) != 17 for row in exported[key]):
                    raise ValueError('Operator export shape mismatch: '+key)
        # A real, deliberately tiny BIE solve checks the installed ABI, new
        # storage path and moment transport. It does not establish convergence.
        smoke = build/'ram_bie_smoke.json'
        smoke_cmd = [sys.executable, '-B', str(HERE/'run_reference.py'),
            '--library', paths['reference_library'], '--backend-manifest', paths['backend_manifest'],
            '--operator-export', str(exporter), '--output', str(smoke),
            '--aspect-ratio', '2', '--c-nm', '150', '--gap-nm', '30000',
            '--theta-i', '0', '--theta-j', '0', '--phi', '0', '--order', '4', '--level', '0',
            '--rim-levels', '0', '--contact-levels', '0', '--focus', '0',
            '--solver', 'ram-dense', '--max-memory-gb', '8', '--check-points-per-patch', '1',
            '--matrix-workdir', str(build/'smoke_work')]
        print('Checking a small real BIE solve (installation check only)', flush=True)
        with (build/'smoke.log').open('w') as log:
            subprocess.run(smoke_cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
        report = json.loads(smoke.read_text())
        import numpy as np
        matrix = np.asarray(report.get('raw_full17'))
        if (report.get('status') != 'complete' or not 0 <= report.get('linear_residual', float('inf')) < 1e-8
                or matrix.shape != (17, 17) or not np.isfinite(matrix).all()):
            raise ValueError('Small BIE numerical installation check failed; see '+str(smoke))
        if source_identity() != sources:
            raise ValueError('Source changed during build; no ready manifest published')
        paths.update(python=str(Path(sys.executable).resolve()), operator_export=str(exporter))
        record = dict(schema='distance_scan_build_v1', status='ready', source_root=str(ROOT),
            built_at_utc=datetime.now(timezone.utc).isoformat(), paths=paths,
            artifacts={k: dict(path=v, sha256=sha(v), bytes=Path(v).stat().st_size) for k,v in paths.items()},
            source_sha256=sources, python_packages=packages,
            blas_configuration=old.get('blas_configuration', {}),
            reused_build=dict(path=str(old_path), sha256=sha(old_path)),
            compiler=dict(path=compiler, command=command,
                version=subprocess.check_output([compiler, '--version'], text=True).splitlines()[0]),
            installation_smoke=dict(path=str(smoke), sha256=sha(smoke), passed=True,
                linear_residual=report['linear_residual'], physical_accuracy_qualified=False),
            physical_accuracy_qualified=False)
        manifest = prefix/'build_distance_scan_manifest.json'
        temporary = manifest.with_suffix('.pending')
        temporary.write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
        temporary.replace(manifest)
        print('READY '+str(manifest), flush=True)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print('Distance build failed: '+str(error), file=sys.stderr)
        raise SystemExit(2)
