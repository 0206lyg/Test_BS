#!/usr/bin/env python3
"""Run from a verified private copy while source/build directories are updated.

Models and table checkpoints stay in the original tree. Only code, executables
and their shared libraries are copied. No numerical acceptance gate is changed.
"""
import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import socket
import stat
import struct
import subprocess
import sys
import time

import bundle_guard

OVERRIDES = {
    "FINITE_GAP_FAST_GENERATOR": "slurry_precompute_fast",
    "FINITE_GAP_FAST_BASIS_GENERATOR": "slurry_generate_basis",
    "FINITE_GAP_FAST_GATE_EXECUTABLE": "slurry_basis_physical_probe",
    "FINITE_GAP_FAST_INTERPOLATION_EXECUTABLE": "slurry_table_interpolation_probe",
}
# Keep the process's matching glibc and dynamic loader. Other linked libraries,
# including libgomp, libstdc++, MPI and liblammps, are private runtime copies.
GLIBC = re.compile(r"^(?:ld-linux.*|lib(?:c|m|pthread|dl|rt|resolv|util|anl|nss_[^.]+)\.so(?:\..*)?)$")


@contextmanager
def build_lock(root, exclusive):
    path = root / "build" / ".slurry-runtime.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        kind = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        try:
            fcntl.flock(stream.fileno(), kind | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Waiting for slurry {} lock: {}".format(
                "build" if exclusive else "runtime snapshot", path), flush=True)
            fcntl.flock(stream.fileno(), kind)
        yield


def read_environment(path):
    result = {}
    for line in path.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        tokens = shlex.split(line)
        if len(tokens) != 2 or tokens[0] != "export" or "=" not in tokens[1]:
            raise RuntimeError("Unsupported generated build environment line in {}".format(path))
        key, value = tokens[1].split("=", 1)
        result[key] = value
    return result


def copy_checked(source, destination, expected=None):
    source = source.resolve()
    before = bundle_guard.sha256_file(source)
    if expected is not None and before != expected:
        raise RuntimeError("Source changed before runtime snapshot: {}".format(source))
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(source), str(destination))  # New inode; never a hard link.
    if (bundle_guard.sha256_file(source) != before or
            bundle_guard.sha256_file(destination) != before):
        raise RuntimeError("File changed during runtime snapshot: {}".format(source))
    return {"original": str(source), "sha256": before,
            "snapshot": str(destination)}


def private_link_paths(record):
    """Let LD_LIBRARY_PATH select private copies, including old GNU DT_RPATH.

    Only a private copy's dynamic tag changes (RPATH=15 -> RUNPATH=29). Path
    strings, instructions and DT_NEEDED are untouched. Preserve both hashes
    so the approved original and this explicit loader-only edit remain auditable.
    Apply to executable AND DSO copies; a transitive DSO may have its own RPATH.
    """
    original, path = Path(record["original"]), Path(record["snapshot"])
    if os.path.samefile(str(original), str(path)):
        raise RuntimeError("Refusing to edit original runtime inode: {}".format(path))
    expected = record.get("snapshot_sha256", record["sha256"])
    if (bundle_guard.sha256_file(original) != record["sha256"] or
            bundle_guard.sha256_file(path) != expected):
        raise RuntimeError("Runtime file changed before link-path normalization: {}".format(path))
    data = bytearray(path.read_bytes())
    edits = []
    if data[:4] == b"\x7fELF":
        def invalid():
            raise RuntimeError("Unsupported or malformed runtime ELF: {}".format(path))

        if len(data) < 16 or data[4] not in (1, 2) or data[5] not in (1, 2) or data[6] != 1:
            invalid()
        wide = data[4] == 2
        endian = "<" if data[5] == 1 else ">"

        def unpack(fmt, offset):
            size = struct.calcsize(endian+fmt)
            if offset < 0 or offset+size > len(data):
                invalid()
            return struct.unpack_from(endian+fmt, data, offset)[0]

        phoff = unpack("Q" if wide else "I", 32 if wide else 28)
        phsize = unpack("H", 54 if wide else 42)
        phnum = unpack("H", 56 if wide else 44)
        if phnum == 0xffff or (phnum and phsize < (56 if wide else 32)) or phoff+phsize*phnum > len(data):
            invalid()
        loads, dynamic = [], []
        for index in range(phnum):
            header = phoff+index*phsize
            kind = unpack("I", header)
            offset = unpack("Q" if wide else "I", header+(8 if wide else 4))
            vaddr = unpack("Q" if wide else "I", header+(16 if wide else 8))
            size = unpack("Q" if wide else "I", header+(32 if wide else 16))
            if kind in (1, 2) and offset+size > len(data):
                invalid()
            if kind == 1:
                loads.append((vaddr, offset, size))
            if kind == 2:
                dynamic.append((offset, size))
        if len(dynamic) > 1:
            invalid()
        entries = []
        stride = 16 if wide else 8
        for start, size in dynamic:
            terminated = False
            for offset in range(start, start+size-stride+1, stride):
                tag = unpack("q" if wide else "i", offset)
                value = unpack("Q" if wide else "I", offset+stride//2)
                if tag == 0:
                    terminated = True
                    break
                entries.append((tag, value, offset))
            if not terminated:
                invalid()
        # Slash-containing DT_NEEDED bypasses search paths altogether. Do not
        # silently miss such a dependency (ldd omits the usual 'name =>' prefix).
        needed = [value for tag, value, _ in entries if tag == 1]
        if needed:
            tables = [value for tag, value, _ in entries if tag == 5]
            sizes = [value for tag, value, _ in entries if tag == 10]
            if len(tables) != 1 or len(sizes) != 1:
                invalid()
            locations = [offset+tables[0]-vaddr for vaddr, offset, size in loads
                         if vaddr <= tables[0] and tables[0]+sizes[0] <= vaddr+size]
            if len(locations) != 1:
                invalid()
            strings = data[locations[0]:locations[0]+sizes[0]]
            for value in needed:
                end = strings.find(b"\0", value)
                if value >= len(strings) or end < 0:
                    invalid()
                name = bytes(strings[value:end])
                if b"/" in name:
                    raise RuntimeError("Runtime DT_NEEDED contains a path; cannot isolate {}: {}".format(
                        path, name.decode("utf-8", "replace")))
        rpaths = [offset for tag, _, offset in entries if tag == 15]
        runpaths = [offset for tag, _, offset in entries if tag == 29]
        if len(rpaths) > 1 or len(runpaths) > 1:
            invalid()
        if rpaths and not runpaths:
            offset = rpaths[0]
            struct.pack_into(endian+("q" if wide else "i"), data, offset, 29)
            edits.append({"offset": offset, "from_tag": 15, "to_tag": 29})
    if edits:
        mode = stat.S_IMODE(path.stat().st_mode)
        try:
            path.chmod(mode | stat.S_IWUSR)
            path.write_bytes(data)
        finally:
            path.chmod(mode)
    record["snapshot_sha256"] = bundle_guard.sha256_file(path)
    record.setdefault("link_path_edits", []).extend(edits)
    if bundle_guard.sha256_file(original) != record["sha256"]:
        raise RuntimeError("Original runtime file changed during link-path normalization: {}".format(original))
    return record


def verify_runtime_records(records):
    for record in records:
        if (bundle_guard.sha256_file(Path(record["original"])) != record["sha256"] or
                bundle_guard.sha256_file(Path(record["snapshot"])) != record.get("snapshot_sha256", record["sha256"])):
            raise RuntimeError("Runtime file changed while copying: {}".format(record["original"]))


def linked_libraries(path, environment=None):
    process = subprocess.run(["ldd", str(path)], stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, universal_newlines=True,
                             env=environment)
    output = process.stdout
    if process.returncode:
        if "not a dynamic executable" in output or "statically linked" in output:
            return {}
        raise RuntimeError("Cannot inspect runtime libraries for {}: {}".format(path, output.strip()))
    result = {}
    for line in output.splitlines():
        if "=> not found" in line:
            raise RuntimeError("Missing runtime library for {}: {}".format(path, line.strip()))
        match = re.match(r"\s*(\S+)\s+=>\s+(/.+?)\s+\(0x[0-9a-fA-F]+\)", line)
        if match and not GLIBC.match(match.group(1)):
            result[match.group(1)] = Path(match.group(2))
    return result


def snapshot_libraries(executables, destination):
    records = {}
    for executable in executables:
        for soname, original in linked_libraries(executable).items():
            if soname in records:
                if records[soname]["sha256"] != bundle_guard.sha256_file(original):
                    raise RuntimeError("Conflicting runtime library {} from {}".format(soname, original))
                continue
            records[soname] = copy_checked(original, destination / soname)
    return records


def create_runtime(root, destination, expected_release, overrides):
    """Caller holds the shared build lock through all checks and copies."""
    environment = read_environment(root / "build" / "lammps" / "env.sh")
    executable = Path(environment["SLURRY_LAMMPS_EXECUTABLE"]).resolve()
    plugin = Path(environment["SLURRY_LAMMPS_PLUGIN"]).resolve()
    packer = Path(environment.get("SLURRY_PACKER_EXECUTABLE",
                                  str(root / "build/lammps-slurry/slurry_pack"))).resolve()
    manifest = Path(environment.get("SLURRY_BUILD_MANIFEST",
                                    str(root / "build/lammps/build-manifest.json")))
    approved = bundle_guard.check_build(root, expected_release, executable, plugin, packer, manifest)
    destination.mkdir(parents=True, exist_ok=False)
    source = bundle_guard.source_snapshot(root)
    files = dict(source["files"])
    # Runtime helpers and fixed gate fixtures need not all be compiled inputs.
    for relative in ("lammps_slurry/tools", "validation_basis", "lammps_slurry/tests/data"):
        for path in (root / relative).rglob("*"):
            if path.is_file() and (relative.endswith("/data") or path.suffix == ".py"):
                files[str(path.relative_to(root))] = bundle_guard.sha256_file(path)
    for relative, digest in sorted(files.items()):
        copy_checked(root / relative, destination / relative, digest)
    models = destination / "lammps_slurry/models"
    models.parent.mkdir(parents=True, exist_ok=True)
    models.symlink_to(root / "lammps_slurry/models", target_is_directory=True)
    binaries = {}
    paths = {"lmp": executable, "libslurry.so": plugin, "slurry_pack": packer}
    for name in bundle_guard.OFFLINE_TOOLS:
        paths[name] = packer.with_name(name)
    for key, name in OVERRIDES.items():
        if overrides.get(key):
            paths[name] = Path(overrides[key]).expanduser().resolve()
    for name, original in paths.items():
        if not original.is_file() or (name != "libslurry.so" and not os.access(str(original), os.X_OK)):
            raise RuntimeError("Missing runtime executable/library: {}".format(original))
        binaries[name] = copy_checked(original, destination / "build/lammps-slurry" / name)
    libraries = snapshot_libraries(paths.values(), destination / "runtime-lib")
    for record in list(binaries.values()) + list(libraries.values()):
        private_link_paths(record)
    runtime_environment = dict(os.environ)
    runtime_environment["LD_LIBRARY_PATH"] = str(destination / "runtime-lib") + (
        ":" + os.environ["LD_LIBRARY_PATH"] if os.environ.get("LD_LIBRARY_PATH") else "")
    # Verify actual resolution after normalizing old RPATH tags on private copies.
    for record in binaries.values():
        for soname, resolved in linked_libraries(Path(record["snapshot"]), runtime_environment).items():
            if soname not in libraries or resolved.resolve() != (destination / "runtime-lib" / soname).resolve():
                raise RuntimeError("Runtime dependency escaped private snapshot: {} -> {}".format(soname, resolved))
    for relative, digest in files.items():
        if (bundle_guard.sha256_file(root / relative) != digest or
                bundle_guard.sha256_file(destination / relative) != digest):
            raise RuntimeError("Source changed while runtime was copied: {}".format(relative))
    verify_runtime_records(list(binaries.values()) + list(libraries.values()))
    if bundle_guard.source_snapshot(root)["sha256"] != source["sha256"]:
        raise RuntimeError("Source changed while runtime was copied; no computation started")
    bundle_guard.check_build(root, expected_release, executable, plugin, packer, manifest)
    record = {"schema_version": 1, "source_root": str(root),
              "runtime_root": str(destination), "hostname": socket.gethostname(),
              "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
              "created_unix": time.time(), "release": expected_release,
              "source_fingerprint": source["sha256"], "source_files": files,
              "binaries": binaries, "libraries": libraries,
              "approved_build": approved, "model_directory": str(root / "lammps_slurry/models")}
    (destination / "runtime_identity.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    exports = {"SLURRY_RUNTIME_BASE": str(destination),
               "SLURRY_LAMMPS_EXECUTABLE": binaries["lmp"]["snapshot"],
               "SLURRY_LAMMPS_PLUGIN": binaries["libslurry.so"]["snapshot"],
               "SLURRY_PACKER_EXECUTABLE": binaries["slurry_pack"]["snapshot"],
               "LD_LIBRARY_PATH": runtime_environment["LD_LIBRARY_PATH"]}
    exports.update({key: binaries[name]["snapshot"] if overrides.get(key) else ""
                    for key, name in OVERRIDES.items()})
    (destination / "runtime.env").write_text("".join(
        "export {}={}\n".format(key, shlex.quote(value)) for key, value in sorted(exports.items())))
    print("Runtime snapshot verified: {} (source {}; host {})".format(
        destination, source["sha256"], socket.gethostname()), flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command")
    commands.required = True
    build = commands.add_parser("build")
    build.add_argument("--root", required=True, type=Path)
    build.add_argument("arguments", nargs=argparse.REMAINDER)
    snapshot = commands.add_parser("snapshot")
    snapshot.add_argument("--root", required=True, type=Path)
    snapshot.add_argument("--destination", required=True, type=Path)
    snapshot.add_argument("--expected-release", required=True)
    snapshot.add_argument("--override", action="append", nargs=2, default=[], metavar=("NAME", "PATH"))
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    with build_lock(root, args.command == "build"):
        if args.command == "build":
            rest = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
            return subprocess.call([sys.executable, str(root / "lammps_slurry/tools/build.py")] + rest)
        overrides = dict(args.override)
        if set(overrides) - set(OVERRIDES):
            raise RuntimeError("Unknown runtime executable override")
        create_runtime(root, args.destination.expanduser().resolve(), args.expected_release, overrides)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        print("Slurry runtime snapshot failed: {}. No computation started.".format(error), file=sys.stderr)
        sys.exit(1)
