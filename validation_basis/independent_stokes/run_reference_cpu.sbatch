#!/bin/bash
#SBATCH --job-name=ar20_stokes_reference
#SBATCH --account=ardekani
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=ar20_stokes_reference_%j.out

# One recorded pose and one explicitly selected gap. This does not create tables.
set -Eeuo pipefail
trap 'printf "Independent reference failed at line %s (exit %s). Run files are retained.\n" "$LINENO" "$?" >&2' ERR
REFERENCE_ROOT="${SLURRY_ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}}"
if [[ ! -f "$REFERENCE_ROOT/validation_basis/independent_stokes/run_reference.py" ]]; then
    printf 'Set SLURRY_ROOT to the repository root containing validation_basis/independent_stokes/run_reference.py. Current root: %s\n' "$REFERENCE_ROOT" >&2
    exit 2
fi
REFERENCE_ROOT="$(cd -- "$REFERENCE_ROOT" && pwd)"
if [[ "${SLURRY_REFERENCE_LOAD_MODULES:-1}" == 1 ]] && type module &>/dev/null; then
    module purge
    module load "${SLURRY_PYTHON_MODULE:-anaconda/2024.02-py311}"
    module load "${SLURRY_GCC_MODULE:-gcc/12.2.0}"
fi
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-${OMP_NUM_THREADS:-1}}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONUNBUFFERED=1
REFERENCE_PYTHON="${SLURRY_REFERENCE_PYTHON:-python3}"
exec "$REFERENCE_PYTHON" - "$REFERENCE_ROOT" "$@" <<'PY'
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys

root = Path(sys.argv[1]).resolve()
parser = argparse.ArgumentParser(description="One recorded AR20 Stokes reference gap; explicit order:level profiles; no table generation.")
parser.add_argument("--library", type=Path, default=os.environ.get("REFERENCE_LIBRARY"))
parser.add_argument("--gap-nm", type=float, default=os.environ.get("GAP_NM"))
parser.add_argument("--profiles", default=os.environ.get("REFERENCE_PROFILES", "4:0"),
                    help="Comma-separated order:level pairs at the same gap; default 4:0 is a candidate, not a qualified resolution")
parser.add_argument("--output-dir", type=Path, default=os.environ.get("REFERENCE_OUTPUT_DIR"))
parser.add_argument("--quad-tol", type=float, default=1e-7)
parser.add_argument("--max-memory-gb", type=float, default=6.)
parser.add_argument("--check-points-per-patch", type=int, default=3)
parser.add_argument("--alpha", type=float, default=1.)
parser.add_argument("--beta", type=float, default=1.)
parser.add_argument("--isolated-body", type=int, choices=(0, 1))
parser.add_argument("--focus", type=float, default=0.)
parser.add_argument("--operator-export", type=Path)
parser.add_argument("--isolated-validation", type=Path, action="append", default=[])
parser.add_argument("--adapt-from", type=Path,
                    help="Same-pose parent report; captured into the job's input snapshot")
parser.add_argument("--mesh-from", type=Path,
                    help="Same-pose mesh proposal or report; captured into the job's input snapshot")
parser.add_argument("--near-apply-library", type=Path,
                    help="Explicit helper library; its adjacent .json build manifest is required")
parser.add_argument("--resolution-level", type=int)
parser.add_argument("--dependency-manifest", type=Path)
parser.add_argument("--backend-manifest", type=Path,
                    help="Optional selected-library patch manifest with a top-level library_sha256")
args, backend_options = parser.parse_known_args(sys.argv[2:])
if any(value.split("=", 1)[0] in ("--order", "--level", "--output") for value in backend_options):
    parser.error("Use --profiles and --output-dir; per-profile --order/--level/--output cannot be overridden")
if args.library is None or args.gap_nm is None:
    parser.error("--library and --gap-nm are required (or REFERENCE_LIBRARY and GAP_NM)")
if args.adapt_from is not None and args.mesh_from is not None:
    parser.error("--adapt-from and --mesh-from cannot be combined")
library = args.library.expanduser().resolve()
if not library.is_file():
    parser.error("Reference library is missing: " + str(library))
profiles = []
try:
    for value in args.profiles.split(","):
        order, level = map(int, value.split(":"))
        if order < 2 or level < 0 or (order, level) in profiles:
            raise ValueError()
        profiles.append((order, level))
except ValueError:
    parser.error("--profiles requires unique order:level pairs with order>=2 and level>=0, e.g. 4:0,4:1")

def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()

started = datetime.now(timezone.utc)
if args.output_dir is None:
    results = Path(os.environ.get("SLURRY_RESULTS_ROOT", "/scratch/negishi/" + os.environ.get("USER", "unknown")))
    args.output_dir = results / ("ar20_reference_" + os.environ.get("SLURM_JOB_ID", "local") + "_" + started.strftime("%Y%m%dT%H%M%SZ"))
output = args.output_dir.expanduser().resolve()
if output.exists() and (not output.is_dir() or any(output.iterdir())):
    parser.error("Use a new or empty output directory; existing run files are preserved: " + str(output))
output.mkdir(parents=True, exist_ok=True)
snapshot = output / "source_snapshot"
snapshot.mkdir()
source = root / "validation_basis/independent_stokes"
source_suffixes = (".py", ".f", ".f90", ".c", ".cpp", ".h", ".md", ".sh", ".sbatch", ".patch")
source_paths = list(source.iterdir())
if (source / "patches").is_dir():
    source_paths.extend((source / "patches").rglob("*"))
for path in source_paths:
    if path.is_file() and path.suffix in source_suffixes:
        captured_source = snapshot / path.relative_to(source)
        captured_source.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, captured_source)
runtime = output / "runtime"
runtime.mkdir()
inputs = output / "inputs"
inputs.mkdir()
input_records = []

def snapshot_input(path, name):
    original = path.expanduser().resolve()
    captured = inputs / name
    captured.parent.mkdir(parents=True, exist_ok=True)
    original_sha = digest(original)
    shutil.copy2(original, captured)
    if digest(captured) != original_sha or digest(original) != original_sha:
        raise RuntimeError("Input changed while snapshotting: " + str(original))
    input_records.append({"original": str(original), "snapshot": str(captured), "sha256": original_sha})
    return captured

if args.operator_export is not None:
    args.operator_export = snapshot_input(args.operator_export, "operator_export")
args.isolated_validation = [snapshot_input(path, "isolated_validation_{}.json".format(k))
                            for k, path in enumerate(args.isolated_validation)]
if args.adapt_from is not None:
    args.adapt_from = snapshot_input(args.adapt_from, "adaptive_parent.json")
if args.mesh_from is not None:
    args.mesh_from = snapshot_input(args.mesh_from, "mesh_parent.json")
if args.near_apply_library is not None:
    helper_original = args.near_apply_library.expanduser().resolve()
    helper_manifest_original = Path(str(helper_original) + ".json")
    if not helper_manifest_original.is_file():
        raise RuntimeError("Near-apply helper requires its adjacent build manifest: " + str(helper_manifest_original))
    args.near_apply_library = snapshot_input(helper_original, "near_apply/" + helper_original.name)
    helper_manifest = snapshot_input(helper_manifest_original, "near_apply/" + helper_original.name + ".json")
    helper_record = json.loads(helper_manifest.read_text())
    if helper_record.get("schema") != "spheroid_near_apply_build_v1" or helper_record.get("abi") != 1:
        raise RuntimeError("Unsupported near-apply build manifest")
    if helper_record.get("library_sha256") != digest(args.near_apply_library):
        raise RuntimeError("Near-apply helper hash does not match its build manifest")
    if helper_record.get("source_sha256") != digest(snapshot / "near_apply.cpp"):
        raise RuntimeError("Near-apply source changed; explicitly rebuild to a fresh path")
    if helper_record.get("fast_math") is not False:
        raise RuntimeError("Near-apply build must preserve ordinary floating-point arithmetic")
captured_library = runtime / library.name
library_sha = digest(library)
shutil.copy2(library, captured_library)
if digest(captured_library) != library_sha or digest(library) != library_sha:
    raise RuntimeError("Reference library changed while snapshotting; no solve started")
if args.backend_manifest is None:
    inferred_backend = library.parent.parent / "patch_manifest.json"
    if inferred_backend.is_file():
        args.backend_manifest = inferred_backend
if args.backend_manifest is not None:
    args.backend_manifest = snapshot_input(args.backend_manifest, "backend_manifest.json")
    backend_record = json.loads(args.backend_manifest.read_text())
    if backend_record.get("library_sha256") != library_sha:
        raise RuntimeError("Backend manifest does not identify the selected shared library")
build_manifest = args.dependency_manifest
if build_manifest is None and len(library.parents) >= 3:
    inferred = library.parents[2] / "reference_backend_manifest.json"
    if inferred.is_file():
        build_manifest = inferred
if build_manifest is not None:
    captured_manifest = snapshot_input(build_manifest, "reference_backend_manifest.json")
    build_record = json.loads(captured_manifest.read_text())
    if not any(value.get("sha256") == library_sha for value in build_record.get("artifacts", {}).values()):
        raise RuntimeError("Dependency build manifest does not identify the selected shared library")
library_dirs = [str(runtime), str(library.parent)]
for name in ("REFERENCE_LD_LIBRARY_PATH", "LD_LIBRARY_PATH"):
    library_dirs.extend(value for value in os.environ.get(name, "").split(":") if value)
os.environ["LD_LIBRARY_PATH"] = ":".join(dict.fromkeys(library_dirs))
packages = {}
for package in ("numpy", "scipy"):
    try:
        packages[package] = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        packages[package] = None
try:
    linkage = subprocess.run(["ldd", str(captured_library)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, check=False).stdout
except OSError as error:
    linkage = str(error)
if "=> not found" in linkage:
    raise RuntimeError("Shared-library dependency is missing; set REFERENCE_LD_LIBRARY_PATH before submission:\n" + linkage)
dependency_snapshots = []
for line in linkage.splitlines():
    match = re.match(r"\s*(\S+)\s+=>\s+(/.+?)\s+\(", line)
    if not match:
        continue
    soname, dependency_name = match.groups()
    dependency = Path(dependency_name).resolve()
    # System C/Fortran/BLAS runtimes remain provided by the job environment.
    # Preserve custom compiler or BLAS installations that may later be rebuilt.
    if str(dependency).startswith(("/lib/", "/lib64/", "/usr/lib/", "/usr/lib64/")):
        continue
    captured_dependency = runtime / soname
    dependency_sha = digest(dependency)
    shutil.copy2(dependency, captured_dependency)
    if digest(captured_dependency) != dependency_sha or digest(dependency) != dependency_sha:
        raise RuntimeError("Dependency changed while snapshotting: " + str(dependency))
    dependency_snapshots.append({"original": str(dependency), "snapshot": str(captured_dependency),
                                 "sha256": dependency_sha})
manifest = {
    "schema": "independent_ar20_job_v1", "physical_accuracy_qualified": False,
    "scope": "One recorded AR20 pose and one fixed gap; selected resolutions only; no lubrication table generation",
    "started_at_utc": started.isoformat(), "host": platform.node(), "python": sys.version,
    "python_executable": sys.executable, "python_packages": packages,
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "gap_nm": args.gap_nm,
    "source_sha256": {str(path.relative_to(snapshot)): digest(path)
                      for path in snapshot.rglob("*") if path.is_file()},
    "library_original": str(library), "library_snapshot": str(captured_library), "library_sha256": library_sha,
    "library_linkage": linkage, "omp_num_threads": os.environ["OMP_NUM_THREADS"],
    "input_snapshots": input_records,
    "nonstandard_library_snapshots": dependency_snapshots,
    "openblas_num_threads": os.environ["OPENBLAS_NUM_THREADS"], "runs": []
}
manifest_path = output / "launch_manifest.json"

def save_manifest():
    pending = manifest_path.with_suffix(".json.tmp")
    pending.write_text(json.dumps(manifest, indent=2) + "\n")
    pending.replace(manifest_path)

save_manifest()
print("Independent reference output:", output, flush=True)
for order, level in profiles:
    label = "pair" if args.isolated_body is None else "isolated" + str(args.isolated_body)
    report = output / (label + "_o{}_l{}.json".format(order, level))
    command = [sys.executable, str(snapshot / "run_reference.py"), "--library", str(captured_library),
               "--gap-nm", str(args.gap_nm), "--order", str(order), "--level", str(level),
               "--output", str(report), "--quad-tol", str(args.quad_tol),
               "--max-memory-gb", str(args.max_memory_gb),
               "--check-points-per-patch", str(args.check_points_per_patch),
               "--alpha", str(args.alpha), "--beta", str(args.beta), "--focus", str(args.focus)]
    if args.isolated_body is not None:
        command += ["--isolated-body", str(args.isolated_body)]
    if args.operator_export is not None:
        command += ["--operator-export", str(args.operator_export)]
    for path in args.isolated_validation:
        command += ["--isolated-validation", str(path)]
    if args.adapt_from is not None:
        command += ["--adapt-from", str(args.adapt_from)]
    if args.mesh_from is not None:
        command += ["--mesh-from", str(args.mesh_from)]
    if args.near_apply_library is not None:
        command += ["--near-apply-library", str(args.near_apply_library)]
    if args.backend_manifest is not None:
        command += ["--backend-manifest", str(args.backend_manifest)]
    if args.resolution_level is not None:
        command += ["--resolution-level", str(args.resolution_level)]
    command += backend_options
    record = {"order": order, "level": level, "command": command, "output": str(report), "status": "started"}
    manifest["runs"].append(record)
    save_manifest()
    print("RUN", json.dumps(command), flush=True)
    log_path = report.with_suffix(".log")
    with log_path.open("w") as log:
        process = subprocess.Popen(command, cwd=snapshot, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1)
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line, end="", flush=True)
        status = process.wait()
    record.update(exit_code=status, status="completed" if status == 0 else "failed", log=str(log_path))
    save_manifest()
    if status:
        print("Reference process failed; later profiles were not started. Files retained in", output, file=sys.stderr)
        sys.exit(status if status > 0 else 1)
manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
save_manifest()
print("Selected jobs completed. Physical qualification requires independent boundary and refinement checks.", flush=True)
PY
