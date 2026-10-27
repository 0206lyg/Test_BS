#!/usr/bin/env python3
"""Reject mixed source bundles and stale or relocated slurry builds."""

import argparse
import hashlib
import json
from pathlib import Path
import sys


RELEASE_ID = "20261027-calibrated-linear-contact"
MANIFEST_SCHEMA = 1
OFFLINE_TOOLS = ("slurry_generate_basis", "slurry_precompute_fast", "slurry_basis_physical_probe", "slurry_table_interpolation_probe")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError) as error:
        raise RuntimeError("Cannot read {}: {}".format(path, error)) from error


def check_source(root, expected_release=RELEASE_ID):
    root = Path(root).expanduser().resolve()
    release = read_json(root / "release.json")
    if not isinstance(release, dict) or release.get("release_id") != expected_release:
        raise RuntimeError("Source release does not match {} in {}. Extract the complete "
                           "matching ZIP directly into this directory."
                           .format(expected_release, root))
    required = (
        "lammps_slurry/CMakeLists.txt", "lammps_slurry/lammps-version.json",
        "lammps_slurry/src/fix_slurry_implicit.cpp",
        "lammps_slurry/src/fix_slurry_bridge.cpp",
        "lammps_slurry/src/slurry_compact_bridge.h",
        "lammps_slurry/src/slurry_rough_surface.h",
        "lammps_slurry/src/slurry_periodic_mobility.h",
        "lammps_slurry/src/slurry_periodic_mesh.h",
        "lammps_slurry/src/slurry_fft_plans.h",
        "lammps_slurry/src/slurry_bridge_solver.h",
        "lammps_slurry/src/slurry_bridge_workspace.h",
        "lammps_slurry/src/slurry_bridge_contact_solver.h",
        "lammps_slurry/src/slurry_bridge_linear_solver.h",
        "lammps_slurry/src/slurry_bridge_step.h",
        "lammps_slurry/src/slurry_bridge_shape.h",
        "lammps_slurry/src/slurry_cylinder_geometry.h",
        "lammps_slurry/src/slurry_cylinder_packing.h",
        "lammps_slurry/src/slurry_cylinder_film.h",
        "lammps_slurry/src/slurry_cylinder_pair.h",
        "lammps_slurry/src/slurry_bridge_images.h",
        "lammps_slurry/tools/prepare_bridge.py",
        "lammps_slurry/tools/run_bridge.py",
        "lammps_slurry/tools/summarize_bridge.py",
        "lammps_slurry/src/slurry_spheroid_hydro_math.h",
        "lammps_slurry/src/slurry_solver_workspace.h",
        "lammps_slurry/src/slurry_eigen_ilu_math.h",
        "lammps_slurry/src/slurry_spd_table.h",
        "lammps_slurry/src/slurry_spd_coordinates.h",
        "lammps_slurry/src/slurry_objective_film.h",
        "lammps_slurry/tools/prepare_case.py", "lammps_slurry/tools/build.py",
        "lammps_slurry/tools/runtime_snapshot.py",
        "lammps_slurry/tools/precompute_fast.cpp",
        "lammps_slurry/tools/slurry_generate_basis.cpp",
        "lammps_slurry/tools/basis_cylindrical_kernel.h",
        "lammps_slurry/tests/test_basis_physical.cpp",
        "validation_basis/run_physical_gate.py",
        "lammps_slurry/tests/test_table_interpolation.cpp",
        "validation_basis/run_interpolation_gate.py",
        "lammps_slurry/tools/bundle_guard.py",
        "lammps_slurry/vendor/gr_re2/ellipsoidGap.h",
        "lammps_slurry/vendor/eigen3/Eigen/Core",
    )
    for relative in required:
        if not (root / relative).is_file():
            raise RuntimeError("Incomplete source bundle: missing {}. Re-extract the "
                               "complete matching ZIP.".format(root / relative))
    for filename, field in (("prepare_case.py", "prepare_case_sha256"),
                            ("prepare_bridge.py", "prepare_bridge_sha256")):
        prepare = root / "lammps_slurry/tools" / filename
        expected_prepare = release.get(field)
        if not expected_prepare or sha256_file(prepare) != expected_prepare:
            raise RuntimeError("{} does not match release {}. The source bundle "
                               "may contain an older input generator; re-extract the complete "
                               "matching ZIP.".format(filename, expected_release))
    return release


def source_snapshot(root):
    """Fingerprint compiled source and input-generation code, not run defaults."""
    root = Path(root).expanduser().resolve()
    project = root / "lammps_slurry"
    files = {
        root / "release.json", project / "CMakeLists.txt",
        project / "lammps-version.json", project / "tools/prepare_case.py",
        project / "tools/build.py", project / "tools/bundle_guard.py",
        project / "tools/runtime_snapshot.py",
        project / "tools/prepare_bridge.py", project / "tools/run_bridge.py",
        project / "tools/summarize_bridge.py",
        project / "tools/precompute_fast.cpp",
        project / "tools/slurry_generate_basis.cpp", project / "tools/basis_cylindrical_kernel.h",
        project / "tests/test_basis_physical.cpp", root / "validation_basis/run_physical_gate.py",
        project / "tests/test_table_interpolation.cpp", root / "validation_basis/run_interpolation_gate.py",
    }
    for path in (project / "src").rglob("*"):
        if path.is_file() and path.suffix in (".cpp", ".h", ".hpp", ".c", ".cc"):
            files.add(path)
    for path in (project / "vendor").rglob("*"):
        if path.is_file():
            files.add(path)
    hashes = {str(path.relative_to(root)): sha256_file(path)
              for path in sorted(files)}
    encoded = json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {"sha256": hashlib.sha256(encoded).hexdigest(), "files": hashes}


def build_identity(root, snapshot, executable, plugin, packer):
    """Called only after a successful build, CTest and plugin load check."""
    root = Path(root).expanduser().resolve()
    release = check_source(root)
    paths = {"executable": executable, "plugin": plugin, "packer": packer}
    offline = {name: Path(packer).resolve().with_name(name) for name in OFFLINE_TOOLS}
    for path in offline.values():
        if not path.is_file():
            raise RuntimeError("Build is missing its offline generation/validation executable: {}".format(path))
    return {
        "bundle_integrity_schema": MANIFEST_SCHEMA,
        "bundle_release": release["release_id"],
        "bundle_root": str(root),
        "source_fingerprint": snapshot["sha256"],
        "source_files": snapshot["files"],
        "offline_tools": {name: {"path": str(path), "sha256": sha256_file(path)} for name, path in offline.items()},
        "binary_sha256": {name: sha256_file(Path(path).resolve())
                          for name, path in paths.items()},
        **{name: str(Path(path).expanduser().resolve()) for name, path in paths.items()},
    }


def check_build(root, expected_release, executable, plugin, packer, manifest):
    root = Path(root).expanduser().resolve()
    check_source(root, expected_release)
    record = read_json(manifest)
    if not isinstance(record, dict) or record.get("bundle_integrity_schema") != MANIFEST_SCHEMA:
        raise RuntimeError("Build manifest has no supported source/binary identity.")
    if record.get("bundle_release") != expected_release:
        raise RuntimeError("Build belongs to a different source release.")
    if record.get("bundle_root") != str(root):
        raise RuntimeError("Build belongs to another source directory: {} (current {})."
                           .format(record.get("bundle_root", "unknown"), root))
    snapshot = source_snapshot(root)
    if record.get("source_fingerprint") != snapshot["sha256"]:
        old = record.get("source_files", {})
        changed = sorted(name for name in set(old) | set(snapshot["files"])
                         if old.get(name) != snapshot["files"].get(name))
        detail = ", ".join(changed[:3]) or "unknown file"
        raise RuntimeError("Source has changed since the verified build: {}.".format(detail))
    for name in OFFLINE_TOOLS:
        path = Path(packer).resolve().with_name(name)
        expected = record.get("offline_tools", {}).get(name, {})
        if expected.get("path") != str(path) or not path.is_file() or expected.get("sha256") != sha256_file(path):
            raise RuntimeError("Offline generation/validation tool differs from the verified build: {}".format(name))
    for name, path in (("executable", executable), ("plugin", plugin), ("packer", packer)):
        actual = Path(path).expanduser().resolve()
        if not actual.is_file():
            raise RuntimeError("Missing {}: {}.".format(name, actual))
        if record.get(name) != str(actual):
            raise RuntimeError("{} path does not match the verified build: {} (expected {})."
                               .format(name, actual, record.get(name, "unknown")))
        if record.get("binary_sha256", {}).get(name) != sha256_file(actual):
            raise RuntimeError("{} content differs from the verified build: {}."
                               .format(name, actual))
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command")
    subparsers.required = True  # argparse keyword support starts at Python 3.7.
    for command in ("check-source", "check-build"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--root", type=Path, required=True)
        subparser.add_argument("--expected-release", required=True)
        if command == "check-build":
            for name in ("executable", "plugin", "packer", "manifest"):
                subparser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "check-source":
            check_source(args.root, args.expected_release)
        else:
            check_build(args.root, args.expected_release, args.executable,
                        args.plugin, args.packer, args.manifest)
    except (OSError, RuntimeError, TypeError, AttributeError) as error:
        print("Slurry version check failed: {}\nNo simulation was started. "
              "From {}, run 'sbatch build_slurry_cpu.sbatch', wait for "
              "'Build complete', then submit the run again."
              .format(error, args.root.expanduser().resolve()), file=sys.stderr)
        return 1
    print("Slurry {} verified: {} ({})".format(args.command, args.expected_release,
                                             args.root.expanduser().resolve()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
