#!/usr/bin/env python3
"""Build the pinned LAMMPS release and the separate Test_BS slurry plugin.

The default install is entirely inside this repository.  Existing LAMMPS source
trees are validated and reused without resetting, checking out, or patching them.
Only the Python standard library, Git, a C++ compiler, CMake and MPI are required.
"""

import argparse
import datetime
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile

import bundle_guard


PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent


def run(command, *, capture=False):
    command = [str(item) for item in command]
    print("+ " + " ".join(shlex.quote(item) for item in command), flush=True)
    return subprocess.run(command, check=True, universal_newlines=True,
                          stdout=subprocess.PIPE if capture else None,
                          stderr=subprocess.STDOUT if capture else None)


def git_output(source, *arguments):
    return run(["git", "-C", source, *arguments], capture=True).stdout.strip()


def require_program(name):
    if not shutil.which(name):
        raise RuntimeError("Required program not found: {}. Load its module first.".format(name))


def validate_plugin_bundle():
    for name in ("particleMath.h", "ellipsoidGap.h", "nearFieldResistance.h"):
        header = PROJECT / "vendor" / "gr_re2" / name
        if not header.is_file():
            raise RuntimeError("Incomplete slurry source bundle: missing {}. "
                               "Re-extract the complete bundle before building LAMMPS."
                               .format(header))


def prepare_source(source, version, skip_download):
    if not source.exists():
        if skip_download:
            raise RuntimeError("--skip-download was given but {} does not exist.".format(source))
        source.parent.mkdir(parents=True, exist_ok=True)
        # A failed download cannot leave a half-populated final source directory.
        with tempfile.TemporaryDirectory(prefix=source.name + ".download-",
                                         dir=str(source.parent)) as temporary:
            staging = Path(temporary) / "source"
            run(["git", "clone", "--depth", "1", "--branch", version["tag"],
                 "--single-branch", version["repository"], staging])
            if git_output(staging, "rev-parse", "HEAD") != version["commit"]:
                raise RuntimeError("Downloaded LAMMPS commit does not match lammps-version.json.")
            staging.rename(source)
    if not (source / ".git").exists():
        raise RuntimeError("{} already exists but is not a Git checkout. Use --source-dir "
                           "to choose another location; no files were overwritten.".format(source))
    commit = git_output(source, "rev-parse", "HEAD")
    if commit != version["commit"]:
        raise RuntimeError("Existing LAMMPS checkout is at {}, but this plugin requires {} ({}). "
                           "Use a separate --source-dir; existing files were preserved."
                           .format(commit, version["tag"], version["commit"]))
    changed = git_output(source, "status", "--porcelain", "--untracked-files=no")
    if changed:
        raise RuntimeError("LAMMPS source has tracked modifications. Use a pristine checkout "
                           "at {}. Existing changes were preserved.".format(version["tag"]))
    for required in ("cmake/CMakeLists.txt", "src/lammpsplugin.h", "src/ASPHERE",
                     "doc/lammps.1", "bench", "potentials"):
        if not (source / required).exists():
            raise RuntimeError("Incomplete LAMMPS checkout: missing {}.".format(source / required))
    return commit


def validate_build_directory(build, source, mpi):
    """Avoid mixing incompatible MPI/serial builds or different source trees."""
    cache = build / "CMakeCache.txt"
    if not cache.exists():
        return
    values = {}
    for line in cache.read_text().splitlines():
        if line.startswith(("#", "//")) or "=" not in line or ":" not in line:
            continue
        key_type, value = line.split("=", 1)
        values[key_type.split(":", 1)[0]] = value
    cached_source = values.get("CMAKE_HOME_DIRECTORY")
    if cached_source and Path(cached_source).resolve() != (source / "cmake").resolve():
        raise RuntimeError("{} belongs to another source tree. Choose a different --build-dir."
                           .format(build))
    cached_mpi = values.get("BUILD_MPI")
    if cached_mpi is not None and (cached_mpi.upper() in ("ON", "YES", "TRUE", "1")) != mpi:
        raise RuntimeError("{} uses a different MPI/serial setting. Choose a different "
                           "--build-dir instead of mixing builds.".format(build))


def find_mpi_launcher():
    launcher = shutil.which("mpirun") or shutil.which("mpiexec")
    if not launcher:
        raise RuntimeError("MPI build verification requires mpirun or mpiexec. "
                           "Load the same MPI module used for compilation.")
    return launcher


def check_plugin(executable, plugin, build, *, mpi):
    smoke = build / "check-slurry-plugin.in"
    log = build / "check-slurry-plugin.log"
    smoke.write_text('plugin load "{}"\nplugin list\n'.format(plugin))
    command = [executable, "-log", "none", "-in", smoke]
    if mpi:
        # Even one MPI rank must use the MPI launcher under Slurm. A direct
        # invocation can enter an unsupported PMI/PMIx startup path.
        command = [find_mpi_launcher(), "-np", "1"] + command
    try:
        result = run(command, capture=True)
    except subprocess.CalledProcessError as error:
        log.write_text(error.stdout or "")
        if error.stdout:
            print(error.stdout, file=sys.stderr)
        raise RuntimeError("LAMMPS startup/plugin verification failed. See {}.".format(log)) from error
    log.write_text(result.stdout)
    for style in ("slurry/spheroid", "slurry/drag", "slurry/implicit"):
        if style not in result.stdout:
            raise RuntimeError("Plugin check did not find style {}. See {}."
                               .format(style, build / "check-slurry-plugin.log"))
    print("Plugin load verified: pair slurry/spheroid, fix slurry/drag, fix slurry/implicit", flush=True)


def write_environment(build, executable, plugin, packer):
    environment = build / "env.sh"
    entries = {
        "SLURRY_LAMMPS_EXECUTABLE": str(executable),
        "SLURRY_LAMMPS_PLUGIN": str(plugin),
        "SLURRY_PACKER_EXECUTABLE": str(packer),
        "SLURRY_BASIS_GENERATOR_EXECUTABLE": str(packer.with_name("slurry_generate_basis")),
        "SLURRY_FAST_GENERATOR_EXECUTABLE": str(packer.with_name("slurry_precompute_fast")),
        "SLURRY_SOURCE_GATE_EXECUTABLE": str(packer.with_name("slurry_basis_physical_probe")),
        "SLURRY_INTERPOLATION_GATE_EXECUTABLE": str(packer.with_name("slurry_table_interpolation_probe")),
        "SLURRY_BUILD_MANIFEST": str(build / "build-manifest.json"),
        "LAMMPS_EXECUTABLE": str(executable),
        "LAMMPS_PLUGIN": str(plugin),
    }
    environment.write_text("# Generated by lammps_slurry/tools/build.py\n" + "".join(
        "export {}={}\n".format(name, shlex.quote(value)) for name, value in entries.items()))
    return environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=ROOT / "lammps")
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build" / "lammps")
    parser.add_argument("--install-dir", type=Path,
                        help="Default: BUILD_DIR/install")
    parser.add_argument("--plugin-build-dir", type=Path,
                        help="Default: BUILD_DIR sibling named BUILD_DIR_NAME-slurry")
    parser.add_argument("--jobs", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    parser.add_argument("--skip-download", action="store_true",
                        help="Require an existing pristine checkout; do not access the network")
    parser.add_argument("--serial", action="store_true",
                        help="Build without MPI for local development only (production uses MPI)")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be at least 1")
    bundle_guard.check_source(ROOT, bundle_guard.RELEASE_ID)
    validate_plugin_bundle()
    source_snapshot = bundle_guard.source_snapshot(ROOT)

    source = args.source_dir.expanduser().resolve()
    build = args.build_dir.expanduser().resolve()
    install = (args.install_dir or build / "install").expanduser().resolve()
    plugin_build = (args.plugin_build_dir or build.with_name(build.name + "-slurry")).expanduser().resolve()
    if build == source or plugin_build in (source, PROJECT) or plugin_build == build:
        raise RuntimeError("Source, LAMMPS build and plugin build directories must be separate.")
    for name in ("git", "cmake", "ctest"):
        require_program(name)
    if not args.serial:
        require_program("mpicxx")
        find_mpi_launcher()

    version = json.loads((PROJECT / "lammps-version.json").read_text())
    commit = prepare_source(source, version, args.skip_download)
    validate_build_directory(build, source, not args.serial)
    build.mkdir(parents=True, exist_ok=True)
    # A failed/interrupted build must not leave an earlier approval valid.
    manifest_path = build / "build-manifest.json"
    if manifest_path.exists():
        manifest_path.unlink()
    mpi = "OFF" if args.serial else "ON"
    configure = [
        "cmake", "-S", source / "cmake", "-B", build,
        "-D", "CMAKE_BUILD_TYPE=Release", "-D", "CMAKE_CXX_STANDARD=17",
        "-D", "CMAKE_INSTALL_PREFIX={}".format(install),
        "-D", "CMAKE_INSTALL_LIBDIR=lib", "-D", "BUILD_SHARED_LIBS=ON",
        "-D", "CMAKE_INSTALL_RPATH={}".format(install / "lib"),
        "-D", "BUILD_MPI={}".format(mpi), "-D", "BUILD_OMP=OFF",
        "-D", "PKG_ASPHERE=ON", "-D", "PKG_PLUGIN=ON",
        "-D", "DOWNLOAD_POTENTIALS=OFF", "-D", "ENABLE_TESTING=OFF",
    ]
    if not args.serial:
        configure.extend(["-D", "MPI_CXX_COMPILER={}".format(shutil.which("mpicxx"))])
    run(configure)
    run(["cmake", "--build", build, "--parallel", args.jobs])
    run(["cmake", "--install", build])
    run([
        "cmake", "-S", PROJECT, "-B", plugin_build,
        "-D", "CMAKE_BUILD_TYPE=Release", "-D", "BUILD_TESTING=ON",
        "-D", "LAMMPS_SOURCE_DIR={}".format(source),
        "-D", "LAMMPS_DIR={}".format(install / "lib" / "cmake" / "LAMMPS"),
        "-D", "CMAKE_INSTALL_PREFIX={}".format(install),
    ])
    run(["cmake", "--build", plugin_build, "--parallel", args.jobs])
    run(["ctest", "--test-dir", plugin_build, "--output-on-failure"])
    executable = install / "bin" / "lmp"
    plugin = plugin_build / "libslurry.so"
    packer = plugin_build / "slurry_pack"
    if not executable.is_file() or not plugin.is_file() or not packer.is_file():
        raise RuntimeError("Build finished without the expected LAMMPS executable, plugin or packer.")
    check_plugin(executable, plugin, build, mpi=not args.serial)
    identity = bundle_guard.build_identity(ROOT, source_snapshot, executable, plugin, packer)
    if bundle_guard.source_snapshot(ROOT) != source_snapshot:
        raise RuntimeError("Slurry source changed during the build. No build manifest was "
                           "approved. Run this build again after source changes are complete.")
    manifest = {
        "completed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "lammps_tag": version["tag"], "lammps_commit": commit,
        "source": str(source), "build": str(build), "install": str(install),
        "mpi": not args.serial, "executable": str(executable), "plugin": str(plugin),
        "packer": str(packer),
        "cmake_version": run(["cmake", "--version"], capture=True).stdout.splitlines()[0],
        "packages": ["ASPHERE", "PLUGIN"],
        "validation": "CTest suite and loading slurry/spheroid, slurry/drag, slurry/implicit styles",
        **identity,
    }
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary_manifest.replace(manifest_path)
    environment = write_environment(build, executable, plugin, packer)
    print("\nBuild complete.\n  LAMMPS: {}\n  Plugin: {}\n  Environment: {}"
          .format(executable, plugin, environment), flush=True)


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print("Build failed: {}".format(error), file=sys.stderr)
        sys.exit(1)
