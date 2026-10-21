#!/usr/bin/env bash
# Build pinned, external fmm3dbie sources. No upstream code is vendored here.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "$script_dir/../.." && pwd)
prefix="$(dirname -- "$repo_dir")/reference_deps"
jobs=4

usage() {
  cat <<'EOF'
Usage: build_reference_backend.sh [--prefix DIR] [--jobs N]

DIR is an external dependency directory, outside the Test_BS checkout.
Outputs: DIR/fmm3dbie/lib/libfmm3dbie.so and DIR/reference_backend_manifest.json

Environment: FC=gfortran CC=gcc CXX=g++ PYTHON=python3
             REFERENCE_BLAS='-lblas -llapack'
             REFERENCE_FFLAGS='-fPIC -O3 -std=legacy -w'

Requires git, make, GNU Fortran, C/C++, BLAS/LAPACK, OpenMP, and Python 3.
No administrator privileges or upstream Python bindings are needed.
The installation smoke checks loading and symbols, not physical accuracy.
EOF
}

while (($#)); do
  case "$1" in
    --prefix) [[ $# -ge 2 ]] || { usage >&2; exit 2; }; prefix=$2; shift 2 ;;
    --jobs) [[ $# -ge 2 ]] || { usage >&2; exit 2; }; jobs=$2; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done
[[ "$jobs" =~ ^[1-9][0-9]*$ ]] || { printf 'jobs must be positive\n' >&2; exit 2; }

fc=${FC:-gfortran}
cc=${CC:-gcc}
cxx=${CXX:-g++}
python=${PYTHON:-python3}
blas=${REFERENCE_BLAS:--lblas -llapack}
fflags=${REFERENCE_FFLAGS:--fPIC -O3 -std=legacy -w}
for tool in git make "$fc" "$cc" "$cxx" "$python"; do
  command -v "$tool" >/dev/null || { printf 'Required tool missing: %s\n' "$tool" >&2; exit 2; }
done
fc=$(command -v "$fc")
cc=$(command -v "$cc")
cxx=$(command -v "$cxx")
mkdir -p -- "$prefix"
prefix=$(cd -- "$prefix" && pwd)
case "$prefix/" in
  "$repo_dir/"*) printf 'Dependency prefix must be outside the repository\n' >&2; exit 2 ;;
esac

bie_commit=be6b2b3700b77e9cbf20c512ff4f0922aaee45d9
fmm_commit=d2b5e6e983e1ec915d15330ce74df44221a9ac42
source_dir="$prefix/fmm3dbie"
log_dir="$prefix/reference_build_logs"
mkdir -p -- "$log_dir"

if [[ ! -e "$source_dir" ]]; then
  git init -q "$source_dir"
  git -C "$source_dir" remote add origin https://github.com/fastalgorithms/fmm3dbie.git
  git -C "$source_dir" fetch --depth 1 origin "$bie_commit"
  git -C "$source_dir" checkout -q --detach "$bie_commit"
else
  current=$(git -C "$source_dir" rev-parse HEAD 2>/dev/null) || {
    printf 'Existing dependency directory is not a Git checkout: %s\n' "$source_dir" >&2; exit 2;
  }
  [[ "$current" == "$bie_commit" ]] || {
    printf 'Existing fmm3dbie commit differs from pin; choose a fresh prefix.\n' >&2; exit 2;
  }
fi
# Never silently build changed upstream sources or apply local patches.
git -C "$source_dir" diff --quiet --ignore-submodules=all
git -C "$source_dir" diff --cached --quiet --ignore-submodules=all
if git -C "$source_dir/FMM3D" rev-parse --show-toplevel 2>/dev/null | \
    "$python" -c 'import os,sys; p=sys.stdin.read().strip(); sys.exit(not p.endswith("/FMM3D"))'; then
  git -C "$source_dir/FMM3D" diff --quiet
  git -C "$source_dir/FMM3D" diff --cached --quiet
fi
git -C "$source_dir" submodule update --init --recursive --depth 1
[[ $(git -C "$source_dir/FMM3D" rev-parse HEAD) == "$fmm_commit" ]] || {
  printf 'FMM3D commit differs from pin\n' >&2; exit 2;
}
# make.inc is deliberately not used: it may contain unrecorded local overrides.
for directory in "$source_dir" "$source_dir/FMM3D"; do
  [[ ! -f "$directory/make.inc" ]] || {
    printf 'Unrecorded make.inc present at %s; choose a clean prefix.\n' "$directory" >&2; exit 2;
  }
done

"$fc" --version > "$log_dir/fortran_version.txt"
"$cc" --version > "$log_dir/c_version.txt"
"$cxx" --version > "$log_dir/cxx_version.txt"
build_log=
trap 'status=$?; if [[ $status -ne 0 && -n "$build_log" ]]; then printf "Build failed; last log lines (%s):\n" "$build_log" >&2; tail -40 "$build_log" >&2; fi' EXIT

# -B is intentional: source/object timestamps do not encode compiler or flags.
# Explicit bounded parallelism also avoids the upstream recursive unlimited -j.
printf 'Building pinned FMM3D (%s workers)...\n' "$jobs"
build_log="$log_dir/fmm3d.log"
make -C "$source_dir/FMM3D" -B -j"$jobs" lib \
  FC="$fc" CC="$cc" CXX="$cxx" FFLAGS="$fflags -fopenmp" \
  CFLAGS='-fPIC -O3 -std=gnu17 -fopenmp' \
  CXXFLAGS='-std=c++11 -DSCTL_PROFILE=-1 -fPIC -O3 -fopenmp' \
  OMP=ON > "$build_log" 2>&1

printf 'Building pinned fmm3dbie (%s workers)...\n' "$jobs"
build_log="$log_dir/fmm3dbie.log"
make -C "$source_dir" -B -j"$jobs" lib-fmm3dbie-only \
  FC="$fc" CC="$cc" CXX="$cxx" FFLAGS="$fflags -fopenmp -J .mod/" \
  FFLAGS2='-fPIC -std=legacy' LBLAS="$blas" OMP=ON \
  LFMMSTATICLIB="$source_dir/FMM3D/lib-static/libfmm3d.a" > "$build_log" 2>&1
build_log=

"$python" - "$prefix" "$fc" "$cc" "$cxx" "$fflags" "$blas" "$jobs" <<'PY'
import ctypes
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys

prefix = Path(sys.argv[1])
src = prefix / 'fmm3dbie'
shared = src / 'lib/libfmm3dbie.so'
lib = ctypes.CDLL(str(shared), mode=ctypes.RTLD_GLOBAL)
symbols = ['get_disc_exps_', 'surf_vals_to_coefs_', 'get_qwts_',
           'stok_comb_vel_matgen_', 'getnearquad_stok_comb_vel_eval_',
           'get_basis_pols_', 'stok_comb_vel_solver_']
for symbol in symbols:
    getattr(lib, symbol)
def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()
def revision(path):
    return subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
logs = prefix / 'reference_build_logs'
manifest = {
    'schema': 'independent_stokes_backend_build_v1',
    'built_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'physical_qualification': False,
    'installation_smoke': 'shared_library_loaded_and_required_symbols_found',
    'required_symbols': symbols,
    'fmm3dbie_commit': revision(src),
    'fmm3d_commit': revision(src / 'FMM3D'),
    'compiler_paths': dict(zip(['fortran', 'c', 'cxx'], sys.argv[2:5])),
    'compiler_versions': {name: (logs / (name + '_version.txt')).read_text().splitlines()[0]
                          for name in ['fortran', 'c', 'cxx']},
    'fortran_flags': sys.argv[5] + ' -fopenmp',
    'blas_link_flags': sys.argv[6],
    'jobs': int(sys.argv[7]),
    'artifacts': {str(p.relative_to(prefix)): {'sha256': sha(p), 'bytes': p.stat().st_size}
                  for p in [shared, src / 'lib-static/libfmm3dbie.a']},
}
try:
    manifest['dynamic_linkage'] = subprocess.check_output(['ldd', str(shared)], text=True)
except (FileNotFoundError, subprocess.CalledProcessError):
    manifest['dynamic_linkage'] = 'ldd unavailable; record runtime dependencies separately'
target = prefix / 'reference_backend_manifest.json'
target.write_text(json.dumps(manifest, indent=2) + '\n')
print(f'Library: {shared}')
print(f'Manifest: {target}')
print('Installation smoke passed. Physical AR20 reference remains unqualified.')
PY
