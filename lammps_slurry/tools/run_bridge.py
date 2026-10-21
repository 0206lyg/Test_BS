#!/usr/bin/env python3
"""Launch compact-bridge dynamics from a verified private runtime copy.

Run after tools/build.py. Source/build changes after the snapshot cannot replace
libraries underneath the running process. Physical flags are passed directly to
prepare_bridge.py; old table generation is never invoked.
"""
import argparse
import json
import os
from pathlib import Path
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
    identity = {"schema": 1, "model": "compact_bridge_v4_periodic", "source_root": str(root),
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
    output = args.output.expanduser().resolve()
    if output.exists():
        p.error("Run directory already exists; choose a new --output")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Snapshot before any packing computation. A temporary sibling keeps the
    # requested run path absent until prepare_bridge creates it atomically.
    with tempfile.TemporaryDirectory(prefix=".bridge-runtime-", dir=str(output.parent)) as temporary:
        staging = Path(temporary)/"runtime"
        identity, environment = freeze_runtime(ROOT, staging, args.build_env.expanduser().resolve())
        command = [sys.executable, str(staging/"tools/prepare_bridge.py"), "--output", str(output),
                   "--packer", str(staging/"bin/slurry_pack")]+physical
        subprocess.run(command, env=environment, check=True)
        runtime = output/"runtime"
        staging.rename(runtime)
        identity = relocate_identity(identity, str(staging), str(runtime))
        (runtime/"runtime_identity.json").write_text(json.dumps(identity, indent=2, sort_keys=True)+"\n")
        parameters_path = output/"parameters.json"
        parameters = json.loads(parameters_path.read_text())
        parameters["packer"] = str(runtime/"bin/slurry_pack")
        parameters_path.write_text(json.dumps(parameters, indent=2, sort_keys=True)+"\n")
    environment = dict(os.environ)
    environment["LD_LIBRARY_PATH"] = str(runtime/"lib")+(":"+environment["LD_LIBRARY_PATH"] if environment.get("LD_LIBRARY_PATH") else "")
    # An immutable input record binds the exact helper/runtime used for this run.
    record = {"arguments": physical, "runtime_identity": "runtime/runtime_identity.json",
              "input_sha256": bundle_guard.sha256_file(output/"in.bridge"),
              "particles_sha256": bundle_guard.sha256_file(output/"particles.data")}
    (output/"launch.json").write_text(json.dumps(record, indent=2, sort_keys=True)+"\n")
    if args.prepare_only:
        print("Prepared verified runtime and input: {}".format(output))
        return 0
    executable = [str(runtime/"bin/lmp"), "-var", "slurry_plugin", lammps_quoted_path(runtime/"bin/libslurry.so"), "-in", "in.bridge"]
    if identity["approved_build"].get("mpi", False):
        launcher = shutil.which("mpirun") or shutil.which("mpiexec")
        if not launcher:
            raise RuntimeError("The verified MPI build requires mpirun or mpiexec")
        executable = [launcher, "-np", "1"]+executable
    with (output/"console.log").open("w") as log:
        process = subprocess.Popen(executable, cwd=str(output), env=environment, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, universal_newlines=True)
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        code = process.wait()
    if code:
        raise RuntimeError("LAMMPS exited {}; see {}".format(code, output/"console.log"))
    print("Run completed: {}".format(output))
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError, KeyError, subprocess.SubprocessError) as error:
        print("run_bridge.py: {}".format(error), file=sys.stderr)
        sys.exit(2)
