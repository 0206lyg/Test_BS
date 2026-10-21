#!/usr/bin/env python3
"""Select and numerically check an LP64 BLAS/LAPACK link before the long build."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


PROBE_SOURCE = r'''
#include <stdint.h>
#include <math.h>
extern void dgemm_(const char *, const char *, const int32_t *, const int32_t *,
                  const int32_t *, const double *, const double *, const int32_t *,
                  const double *, const int32_t *, const double *, double *, const int32_t *);
extern void dgesv_(const int32_t *, const int32_t *, double *, const int32_t *,
                  int32_t *, double *, const int32_t *, int32_t *);
int reference_blas_probe(void) {
    /* The zero upper words allow an ILP64 routine to execute the tiny problem;
       guarded output integers then detect an incompatible 64-bit write. */
    struct { int32_t value, upper; } two = {2, 0}, one = {1, 0};
    struct { int32_t value, guard; } info = {-99, 0x13579bdf};
    struct { int32_t value[2]; int32_t guard[2]; } piv = {{0, 0}, {0x13579bdf, 0x2468ace}};
    char normal = 'N';
    double alpha = 1., beta = 0.;
    double a[4] = {1., 3., 2., 4.}, identity[4] = {1., 0., 0., 1.}, c[4] = {0.};
    dgemm_(&normal, &normal, &two.value, &two.value, &two.value, &alpha,
           a, &two.value, identity, &two.value, &beta, c, &two.value);
    for (int i = 0; i < 4; ++i) if (!isfinite(c[i]) || fabs(c[i] - a[i]) > 1e-12) return 1;
    double matrix[4] = {3., 1., 1., 2.}, rhs[2] = {9., 8.};
    dgesv_(&two.value, &one.value, matrix, &two.value, piv.value, rhs, &two.value, &info.value);
    if (info.guard != 0x13579bdf || piv.guard[0] != 0x13579bdf || piv.guard[1] != 0x2468ace) return 2;
    if (info.value != 0 || piv.value[0] < 1 || piv.value[0] > 2 || piv.value[1] < 1 || piv.value[1] > 2) return 3;
    if (!isfinite(rhs[0]) || !isfinite(rhs[1]) || fabs(rhs[0] - 2.) > 1e-12 || fabs(rhs[1] - 3.) > 1e-12) return 4;
    return 0;
}
'''


def _search_dirs(compiler):
    candidates = []
    for prefix in (os.environ.get('CONDA_PREFIX'), sys.prefix, sys.base_prefix):
        if prefix:
            candidates.extend([Path(prefix) / 'lib', Path(prefix) / 'lib64'])
    for name in ('LIBRARY_PATH', 'LD_LIBRARY_PATH'):
        candidates.extend(Path(p) for p in os.environ.get(name, '').split(os.pathsep) if p)
    try:
        result = subprocess.run([str(compiler), '-print-search-dirs'], capture_output=True,
                                text=True, timeout=5, check=True)
        for line in result.stdout.splitlines():
            if line.startswith('libraries:'):
                candidates.extend(Path(p) for p in line.split(':', 1)[1].strip().lstrip('=').split(os.pathsep) if p)
    except (OSError, subprocess.SubprocessError):
        pass
    result = []
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved.is_dir() and resolved not in result:
            result.append(resolved)
    return result


def _files(directory, name):
    return sorted((p for p in directory.glob('lib' + name + '.so*') if p.is_file()),
                  key=lambda p: (p.name != 'lib' + name + '.so', len(p.name), p.name))


def _with_rpath(flags, extra_dirs=()):
    """Normalize explicit paths and make their runtime lookup independent of modules."""
    result, directories = [], list(extra_dirs)
    index = 0
    while index < len(flags):
        token = flags[index]
        if token == '-L' and index + 1 < len(flags):
            index += 1
            directory = Path(flags[index]).expanduser().resolve()
            result.extend(['-L', str(directory)])
            directories.append(directory)
        elif token.startswith('-L') and len(token) > 2:
            directory = Path(token[2:]).expanduser().resolve()
            result.append('-L' + str(directory))
            directories.append(directory)
        elif not token.startswith('-') and Path(token).expanduser().is_file():
            library = Path(token).expanduser().resolve()
            result.append(str(library))
            directories.append(library.parent)
        else:
            result.append(token)
        index += 1
    for directory in dict.fromkeys(str(Path(p).resolve()) for p in directories):
        result.extend(['-Xlinker', '-rpath', '-Xlinker', directory])
    return result


def resolve_blas(compiler, workspace):
    """Return verified link flags, provider, runtime environment and probe diagnostics.

    The compiler is a C compiler. Only active environment directories are inspected;
    no system-wide search, library renaming or dependency installation is performed.
    """
    workspace = Path(workspace).expanduser().resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    source = workspace / 'blas_lapack_probe.c'
    source.write_text(PROBE_SOURCE)
    candidates = []
    explicit = os.environ.get('REFERENCE_BLAS')
    if explicit is not None:
        flags = shlex.split(explicit)
        if not flags:
            raise RuntimeError('REFERENCE_BLAS is empty; provide BLAS/LAPACK link flags')
        candidates.append(('REFERENCE_BLAS', _with_rpath(flags)))
    else:
        candidates.append(('system BLAS/LAPACK', ['-llapack', '-lblas']))
        directories = _search_dirs(compiler)
        # Prefer standard BLAS/LAPACK names before changing provider.
        for directory in directories:
            lapack, blas = _files(directory, 'lapack'), _files(directory, 'blas')
            if lapack and blas:
                candidates.append(('BLAS/LAPACK in ' + str(directory),
                                   _with_rpath([str(lapack[0]), str(blas[0])], [directory])))
        for directory in directories:
            for name in ('openblas', 'mkl_rt'):
                libraries = _files(directory, name)
                if libraries:
                    flags = [str(libraries[0])]
                    if name == 'mkl_rt':
                        flags += ['-lpthread', '-lm', '-ldl']
                    candidates.append((name + ' in ' + str(directory), _with_rpath(flags, [directory])))
    diagnostics = []
    seen = set()
    for provider, flags in candidates:
        key = tuple(flags)
        if key in seen:
            continue
        seen.add(key)
        library = workspace / ('probe_' + str(len(diagnostics)) + '.so')
        runtime_environment = ({'MKL_INTERFACE_LAYER': 'LP64', 'MKL_THREADING_LAYER': 'SEQUENTIAL'}
                               if any('mkl_rt' in value for value in flags) else {})
        command = [str(compiler), '-O2', '-fPIC', '-shared', str(source), '-o', str(library),
                   '-Wl,-z,defs', *flags, '-lm']
        record = {'provider': provider, 'link_flags': shlex.join(flags)}
        try:
            compiled = subprocess.run(command, capture_output=True, text=True, timeout=45)
            if compiled.returncode:
                record.update(passed=False, stage='link', error=(compiled.stderr or compiled.stdout)[-1600:])
                diagnostics.append(record)
                continue
            env = dict(os.environ)
            for name in ('LD_LIBRARY_PATH', 'LD_PRELOAD', 'LIBRARY_PATH', 'LD_RUN_PATH'):
                env.pop(name, None)
            env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', **runtime_environment)
            checked = subprocess.run([sys.executable, '-c',
                'import ctypes,sys; lib=ctypes.CDLL(sys.argv[1]); f=lib.reference_blas_probe; '
                'f.restype=ctypes.c_int; f.argtypes=[]; result=f(); print(result); sys.exit(result)',
                str(library)], capture_output=True, text=True, env=env, timeout=20)
            if checked.returncode:
                record.update(passed=False, stage='runtime',
                              error=('exit=' + str(checked.returncode) + '\n' + checked.stdout + checked.stderr)[-1600:])
                diagnostics.append(record)
                continue
            record.update(passed=True, stage='DGEMM_DGESV_LP64_clean_runtime')
            diagnostics.append(record)
            result = {'provider': provider, 'link_flags': shlex.join(flags),
                      'runtime_environment': runtime_environment, 'diagnostics': diagnostics,
                      'probe_library': str(library), 'passed': True}
            (workspace / 'blas_selection.json').write_text(json.dumps(result, indent=2) + '\n')
            return result
        except (OSError, subprocess.TimeoutExpired) as error:
            record.update(passed=False, stage='probe', error=str(error))
            diagnostics.append(record)
    report = workspace / 'blas_selection.json'
    report.write_text(json.dumps({'passed': False, 'diagnostics': diagnostics}, indent=2) + '\n')
    raise RuntimeError('No usable LP64 BLAS/LAPACK library was found (DGEMM/DGESV link and runtime probe failed). '
                       'Load a BLAS/LAPACK module or set REFERENCE_BLAS to its link flags. Details: ' + str(report))
