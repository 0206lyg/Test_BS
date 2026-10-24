#!/usr/bin/env python3
"""Launch compact-bridge dynamics from a verified private runtime copy.

Run after tools/build.py. Source/build changes after the snapshot cannot replace
libraries underneath the running process. Physical flags are passed directly to
prepare_bridge.py; old table generation is never invoked.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile

import bundle_guard
import runtime_snapshot

ROOT = Path(__file__).resolve().parents[2]


def freeze_runtime(root, destination, envfile):
    env = runtime_snapshot.read_environment(envfile)
    executable = Path(env["SLURRY_LAMMPS_EXECUTABLE"]).resolve()
    plugin = Path(env["SLURRY_LAMMPS_PLUGIN"]).resolve()
    packer = Path(env["SLURRY_PACKER_EXECUTABLE"]).resolve()
    manifest = Path(env["SLURRY_BUILD_MANIFEST"]).resolve()
    release = bundle_guard.RELEASE_ID
    with runtime_snapshot.build_lock(root, False):
        approved = bundle_guard.check_build(root, release, executable, plugin, packer, manifest)
        destination.mkdir(parents=True, exist_ok=False)
        paths = {"lmp": executable, "libslurry.so": plugin, "slurry_pack": packer}
        binaries = {name: runtime_snapshot.copy_checked(path, destination/"bin"/name)
                    for name, path in paths.items()}
        libraries = runtime_snapshot.snapshot_libraries(paths.values(), destination/"lib")
        for record in list(binaries.values()) + list(libraries.values()):
            runtime_snapshot.private_link_paths(record)
        scripts = {}
        for name in ("prepare_bridge.py", "summarize_bridge.py"):
            scripts[name] = runtime_snapshot.copy_checked(root/"lammps_slurry/tools"/name, destination/"tools"/name)
        local_env = dict(os.environ)
        local_env["LD_LIBRARY_PATH"] = str(destination/"lib")+(":"+local_env["LD_LIBRARY_PATH"] if local_env.get("LD_LIBRARY_PATH") else "")
        for record in binaries.values():
            for soname, resolved in runtime_snapshot.linked_libraries(Path(record["snapshot"]), local_env).items():
                if soname not in libraries or resolved.resolve() != (destination/"lib"/soname).resolve():
                    raise RuntimeError("A runtime dependency escaped the private copy: {} -> {}".format(soname, resolved))
        runtime_snapshot.verify_runtime_records(list(binaries.values()) + list(libraries.values()) + list(scripts.values()))
        bundle_guard.check_build(root, release, executable, plugin, packer, manifest)
    identity = {"schema": 1, "model": "shape_selected_periodic_bridge", "source_root": str(root),
                "release": release, "approved_build": approved, "binaries": binaries,
                "libraries": libraries, "scripts": scripts}
    (destination/"runtime_identity.json").write_text(json.dumps(identity, indent=2, sort_keys=True)+"\n")
    return identity, local_env


def relocate_identity(value, old, new):
    if isinstance(value, dict):
        return {k: relocate_identity(v, old, new) for k, v in value.items()}
    if isinstance(value, list):
        return [relocate_identity(v, old, new) for v in value]
    if isinstance(value, str) and (value == old or value.startswith(old+os.sep)):
        return new+value[len(old):]
    return value


def lammps_quoted_path(path):
    """Variable content must supply quotes; the input reference stays unquoted.

    Substitution in `plugin load ${slurry_plugin}` then yields one filename
    token even when the runtime directory contains spaces. Literal quotes and
    line breaks cannot be represented safely by this input-generation path.
    """
    value = str(path)
    if any(character in value for character in ('"', "\n", "\r")):
        raise ValueError("LAMMPS runtime path cannot contain a double quote or line break")
    return '"'+value+'"'


THREAD_ENVIRONMENT = ("OMP_NUM_THREADS", "OMP_DYNAMIC", "OMP_PLACES", "OMP_PROC_BIND",
                      "OMP_MAX_ACTIVE_LEVELS", "MKL_NUM_THREADS", "MKL_DYNAMIC",
                      "OPENBLAS_NUM_THREADS")


def execution_environment(environ):
    """One driver, one MPI rank, with all allocated CPUs available to threads."""
    environment = dict(environ)
    # The batch script runs this driver once. Running it under MPI could
    # duplicate preparation and create nested simulation launches.
    for key in ("OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "PMIX_UNIV_SIZE"):
        if key in environment:
            raise ValueError("Run run_bridge.py once, outside MPI ranks ({} is set)".format(key))
    raw = environment.get("SLURM_CPUS_PER_TASK", environment.get("OMP_NUM_THREADS", "1"))
    try:
        threads = int(raw)
    except (TypeError, ValueError):
        raise ValueError("Thread count must be one positive integer, got {!r}".format(raw))
    if threads < 1:
        raise ValueError("Thread count must be positive")
    environment.update(OMP_NUM_THREADS=str(threads), OMP_DYNAMIC="FALSE",
                       OMP_PLACES="cores", OMP_PROC_BIND="close",
                       OMP_MAX_ACTIVE_LEVELS="1", MKL_NUM_THREADS=str(threads),
                       MKL_DYNAMIC="FALSE", OPENBLAS_NUM_THREADS="1")
    return environment, threads


def execution_command(runtime, mpi, threads, launcher=None):
    """Construct exactly one launch; failed launches are never retried."""
    runtime = Path(runtime)
    executable = [str(runtime/"bin/lmp"), "-var", "slurry_plugin",
                  lammps_quoted_path(runtime/"bin/libslurry.so"), "-in", "in.bridge"]
    if not mpi:
        return executable
    launcher = launcher or shutil.which("mpirun") or shutil.which("mpiexec")
    if not launcher:
        raise RuntimeError("The verified MPI build requires OpenMPI mpirun or mpiexec")
    return [str(launcher), "-np", "1", "--map-by", "slot:PE={}".format(threads),
            "--bind-to", "core", "--report-bindings"] + executable


def run_simulation(command, output, environment):
    """Run the single selected command once, and preserve its output on failure."""
    with (output/"console.log").open("w") as log:
        process = subprocess.Popen(command, cwd=str(output), env=environment, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, universal_newlines=True)
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        code = process.wait()
    if code:
        raise RuntimeError("LAMMPS exited {}; see {}".format(code, output/"console.log"))


def summarize_run(runtime, output, parameters, environment):
    """Report the selected averaging window without inventing missing data."""
    spec = importlib.util.spec_from_file_location("private_bridge_summary", runtime/"tools/summarize_bridge.py")
    summary = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(summary)
    # Validate the complete history before distinguishing a short averaging
    # window from corrupt/nonfinite rheology data. Invalid data remains fatal.
    rows = summary.read_intervals(output/"history.csv", allow_empty=True)
    start = parameters["average_start_strain"]
    end = rows[-1]["interval_end_strain"] if rows else 0.
    if not rows or end <= start or not summary.select(rows, start, end):
        report = {"status": "average_unavailable", "particle_shape": parameters.get("particle_shape", "spheroid"),
                  "requested_start_strain": start, "recorded_end_strain": end,
                  "reason": "No complete positive-time, positive-strain intervals in the requested averaging window.",
                  "stationarity_certified": False}
        (output/"viscosity_summary.json").write_text(json.dumps(report, indent=2)+"\n")
        (output/"viscosity_summary.txt").write_text(report["reason"]+"\n")
        print("Averaging window unavailable; see viscosity_summary.txt", flush=True)
    else:
        report = summary.summarize(output/"history.csv", start)
        report["particle_shape"] = parameters.get("particle_shape", "spheroid")
        report["model"] = parameters["model"]
        (output/"viscosity_summary.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False)+"\n")
        rendered = summary.render(report)
        (output/"viscosity_summary.txt").write_text(rendered)
        print(rendered, end="", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, add_help=False)
    p.add_argument("--help", "-h", action="store_true")
    p.add_argument("--output", type=Path, required=False)
    p.add_argument("--build-env", type=Path, default=ROOT/"build/lammps/env.sh")
    p.add_argument("--prepare-only", action="store_true")
    args, physical = p.parse_known_args()
    if args.help:
        p.print_help()
        subprocess.run([sys.executable, str(ROOT/"lammps_slurry/tools/prepare_bridge.py"), "--help"], check=True)
        return 0
    if args.output is None:
        p.error("--output NEW_RUN_DIRECTORY is required")
    if any(flag == "--packer" or flag.startswith("--packer=") for flag in physical):
        p.error("--packer is bound to the verified build, not a runtime override")
    environment, threads = execution_environment(os.environ)
    output = args.output.expanduser().resolve()
    if output.exists():
        p.error("Run directory already exists; choose a new --output")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Snapshot before any packing computation. A temporary sibling keeps the
    # requested run path absent until prepare_bridge creates it atomically.
    with tempfile.TemporaryDirectory(prefix=".bridge-runtime-", dir=str(output.parent)) as temporary:
        staging = Path(temporary)/"runtime"
        identity, snapshot_environment = freeze_runtime(ROOT, staging, args.build_env.expanduser().resolve())
        snapshot_environment.update({key: environment[key] for key in THREAD_ENVIRONMENT})
        command = [sys.executable, str(staging/"tools/prepare_bridge.py"), "--output", str(output),
                   "--packer", str(staging/"bin/slurry_pack")]+physical
        subprocess.run(command, env=snapshot_environment, check=True)
        runtime = output/"runtime"
        staging.rename(runtime)
        identity = relocate_identity(identity, str(staging), str(runtime))
        (runtime/"runtime_identity.json").write_text(json.dumps(identity, indent=2, sort_keys=True)+"\n")
        parameters_path = output/"parameters.json"
        parameters = json.loads(parameters_path.read_text())
        parameters["packer"] = str(runtime/"bin/slurry_pack")
        parameters_path.write_text(json.dumps(parameters, indent=2, sort_keys=True)+"\n")
        identity["model"] = parameters["model"]
        identity["particle_shape"] = parameters.get("particle_shape", "spheroid")
        (runtime/"runtime_identity.json").write_text(json.dumps(identity, indent=2, sort_keys=True)+"\n")
    environment["LD_LIBRARY_PATH"] = str(runtime/"lib")+(":"+environment["LD_LIBRARY_PATH"] if environment.get("LD_LIBRARY_PATH") else "")
    # An immutable input record binds the exact helper/runtime used for this run.
    record = {"arguments": physical, "runtime_identity": "runtime/runtime_identity.json",
              "input_sha256": bundle_guard.sha256_file(output/"in.bridge"),
              "particles_sha256": bundle_guard.sha256_file(output/"particles.data"),
              "mpi_ranks": 1, "threads": threads,
              "thread_environment": {key: environment[key] for key in THREAD_ENVIRONMENT}}
    (output/"launch.json").write_text(json.dumps(record, indent=2, sort_keys=True)+"\n")
    if args.prepare_only:
        print("Prepared verified runtime and input: {}".format(output))
        return 0
    executable = execution_command(runtime, identity["approved_build"].get("mpi", False), threads)
    record["command"] = executable
    (output/"launch.json").write_text(json.dumps(record, indent=2, sort_keys=True)+"\n")
    print("Run directory: {}".format(output), flush=True)
    print("One process; {} threads; command: {}".format(threads, shlex.join(executable)), flush=True)
    run_simulation(executable, output, environment)
    summarize_run(runtime, output, parameters, environment)
    print("Run completed: {}".format(output))
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError, KeyError, subprocess.SubprocessError) as error:
        print("run_bridge.py: {}".format(error), file=sys.stderr)
        sys.exit(2)
