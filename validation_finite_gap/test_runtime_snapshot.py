#!/usr/bin/env python3
"""Runtime lifecycle regression tests; no hydrodynamic accuracy claims."""
import ast
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lammps_slurry/tools"))
import runtime_snapshot as runtime


@unittest.skipUnless(shutil.which("gcc") and shutil.which("ldd"), "ELF compiler/tools unavailable")
class RuntimeSnapshotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="slurry-runtime-test-")
        cls.base = Path(cls.temporary.name)
        source = cls.base / "probe.c"
        # The child blocks before first accessing a second file-backed data page.
        # Truncating that mapped library produces SIGBUS on Linux. A copied inode
        # remains valid, independently of the exact cause of the user's crashes.
        source.write_text("volatile unsigned char payload[16384] = {3};\n"
                          "int value(void) { return payload[12000] + 7; }\n")
        subprocess.check_call(["gcc", "-shared", "-fPIC", str(source),
                               "-Wl,-soname,libsnapshot_probe.so", "-o", str(cls.base / "libsnapshot_probe.so")])
        program = cls.base / "probe_main.c"
        program.write_text('#include <stdio.h>\nextern int value(void);\n'
                           'int main(void) { puts("READY"); fflush(stdout); getchar(); printf("%d\\n", value()); return 0; }\n')
        subprocess.check_call(["gcc", str(program), "-L" + str(cls.base),
                               "-lsnapshot_probe", "-Wl,-rpath," + str(cls.base),
                               "-o", str(cls.base / "probe")])
        cls.library_bytes = (cls.base / "libsnapshot_probe.so").read_bytes()

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def setUp(self):
        (self.base / "libsnapshot_probe.so").write_bytes(self.library_bytes)

    def process(self, binary, environment=None):
        child = subprocess.Popen([str(binary)], stdin=subprocess.PIPE,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 universal_newlines=True, env=environment)
        self.assertEqual(child.stdout.readline().strip(), "READY")
        return child

    def test_mapped_library_truncation_and_private_copy(self):
        folder = self.base / "private"
        folder.mkdir(exist_ok=True)
        runtime.copy_checked(self.base / "probe", folder / "probe")
        libraries = runtime.snapshot_libraries([self.base / "probe"], folder / "lib")
        self.assertIn("libsnapshot_probe.so", libraries)
        environment = dict(os.environ, LD_LIBRARY_PATH=str(folder / "lib"))
        original = self.process(self.base / "probe")
        private = self.process(folder / "probe", environment)
        (self.base / "libsnapshot_probe.so").write_bytes(b"")
        original.communicate("\n", timeout=10)
        stdout, stderr = private.communicate("\n", timeout=10)
        self.assertEqual(original.returncode, -signal.SIGBUS)
        self.assertEqual(private.returncode, 0, stderr)
        self.assertEqual(stdout.strip(), "7")

    def test_build_lock_excludes_snapshot_and_releases(self):
        lock_root = self.base / "lock_root"
        with runtime.build_lock(lock_root, True):
            # Test in a separate interpreter: flock is process/open-file based.
            child = subprocess.Popen([sys.executable, "-c",
                "import fcntl,sys; f=open(sys.argv[1], 'a+'); "
                "fcntl.flock(f,fcntl.LOCK_SH|fcntl.LOCK_NB)",
                str(lock_root / "build/.slurry-runtime.lock")],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            child.communicate(timeout=10)
            self.assertNotEqual(child.returncode, 0)
        with runtime.build_lock(lock_root, False):
            pass

    def test_snapshot_rejects_modified_source_digest(self):
        source = self.base / "digest.txt"
        source.write_text("before")
        digest = runtime.bundle_guard.sha256_file(source)
        source.write_text("after")
        with self.assertRaisesRegex(RuntimeError, "changed before"):
            runtime.copy_checked(source, self.base / "rejected.txt", digest)
        self.assertFalse((self.base / "rejected.txt").exists())

    def test_full_snapshot_preserves_identity_and_shared_models(self):
        source_root = self.base / "fixture_source"
        snapshot = runtime.bundle_guard.source_snapshot(ROOT)
        files = set(snapshot["files"])
        files.add("lammps_slurry/tools/runtime_snapshot.py")
        for relative in files:
            target = source_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(ROOT / relative), str(target))
        for relative in ("lammps_slurry/tools", "validation_basis", "lammps_slurry/tests/data"):
            for source in (ROOT / relative).rglob("*"):
                if source.is_file() and (relative.endswith("/data") or source.suffix == ".py"):
                    target = source_root / source.relative_to(ROOT)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(str(source), str(target))
        # This is an explicitly synthetic build-identity fixture for file/runtime
        # behavior; all roles use a trivial compiled executable, not a physics solver.
        release_path = source_root / "release.json"
        release = json.loads(release_path.read_text())
        release["release_id"] = runtime.bundle_guard.RELEASE_ID
        release["prepare_case_sha256"] = runtime.bundle_guard.sha256_file(source_root / "lammps_slurry/tools/prepare_case.py")
        release_path.write_text(json.dumps(release))
        tools = source_root / "build/lammps-slurry"
        tools.mkdir(parents=True)
        for name in ("lmp", "libslurry.so", "slurry_pack") + runtime.bundle_guard.OFFLINE_TOOLS:
            shutil.copy2(str(self.base / "probe"), str(tools / name))
        environment = source_root / "build/lammps/env.sh"
        environment.parent.mkdir()
        environment.write_text("export SLURRY_LAMMPS_EXECUTABLE='{}'\n"
                               "export SLURRY_LAMMPS_PLUGIN='{}'\n"
                               "export SLURRY_PACKER_EXECUTABLE='{}'\n".format(
                                   tools / "lmp", tools / "libslurry.so", tools / "slurry_pack"))
        identity = runtime.bundle_guard.build_identity(
            source_root, runtime.bundle_guard.source_snapshot(source_root),
            tools / "lmp", tools / "libslurry.so", tools / "slurry_pack")
        (environment.parent / "build-manifest.json").write_text(json.dumps(identity))
        models = source_root / "lammps_slurry/models"
        models.mkdir()
        (models / "keep.ssb").write_text("shared model fixture")
        destination = self.base / "whole_runtime"
        with runtime.build_lock(source_root, False):
            record = runtime.create_runtime(source_root, destination, runtime.bundle_guard.RELEASE_ID, {})
        self.assertEqual(record["source_fingerprint"], identity["source_fingerprint"])
        self.assertEqual((destination / "lammps_slurry/models").resolve(), models)
        self.assertNotEqual((tools / "slurry_pack").stat().st_ino,
                            (destination / "build/lammps-slurry/slurry_pack").stat().st_ino)
        original_prepare = source_root / "lammps_slurry/tools/prepare_case.py"
        original_prepare.write_text("changed by another deployment\n")
        self.assertEqual(runtime.bundle_guard.sha256_file(destination / "lammps_slurry/tools/prepare_case.py"),
                         identity["source_files"]["lammps_slurry/tools/prepare_case.py"])
        self.assertTrue((destination / "runtime.env").is_file())
        exports = runtime.read_environment(destination / "runtime.env")
        self.assertEqual(exports["FINITE_GAP_FAST_GENERATOR"], "")

        # Exercise the actual Bash launcher end-to-end with explicitly synthetic
        # prepare/MPI stubs. This catches override routing, snapshot paths and
        # accidental finite-gap flags on an ordinary Mari run.
        original_prepare.write_text(
            "import json,sys\nfrom pathlib import Path\n"
            "a=sys.argv[1:]; p=Path(a[a.index('--output')+1])\n"
            "if next(a[i+1] for i in range(len(a)-2,-1,-1) if a[i]=='--steps') == '13': sys.exit(7)\n"
            "p.mkdir(parents=True)\n"
            "(p/'launcher_arguments.json').write_text(json.dumps(a))\n")
        (source_root / "lammps_slurry/tools/summarize_rheology.py").write_text("pass\n")
        release["prepare_case_sha256"] = runtime.bundle_guard.sha256_file(original_prepare)
        release_path.write_text(json.dumps(release))
        identity = runtime.bundle_guard.build_identity(
            source_root, runtime.bundle_guard.source_snapshot(source_root),
            tools / "lmp", tools / "libslurry.so", tools / "slurry_pack")
        (environment.parent / "build-manifest.json").write_text(json.dumps(identity))
        launcher = source_root / "lammps_slurry/run_slurry_cpu.sbatch"
        launcher.write_text((ROOT / "lammps_slurry/run_slurry_cpu.sbatch").read_text())
        stub_bin = self.base / "stub_bin"
        stub_bin.mkdir()
        for name in ("module", "mpirun"):
            stub = stub_bin / name
            stub.write_text("#!/bin/sh\nexit 0\n")
            stub.chmod(0o755)
        shell_environment = dict(os.environ, SLURRY_ROOT=str(source_root),
                                 SLURRY_RUNTIME_ROOT=str(self.base),
                                 PATH=str(stub_bin) + os.pathsep + os.environ["PATH"])
        result = self.base / "launcher_output"
        command = ["bash", str(launcher), "--lubrication-model", "mari", "--ar", "1",
                   "--output", str(result)]
        finished = subprocess.run(command, env=shell_environment, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, universal_newlines=True)
        self.assertEqual(finished.returncode, 0, finished.stdout)
        arguments = json.loads((result / "launcher_arguments.json").read_text())
        self.assertNotIn("--finite-gap-fast-generator", arguments)
        used_packer = arguments[arguments.index("--packer") + 1]
        self.assertNotEqual(used_packer, str(tools / "slurry_pack"))
        self.assertIn("slurry_runtime_", used_packer)
        self.assertTrue((result / "runtime_identity.json").is_file())
        self.assertFalse(Path(used_packer).exists(), "Temporary runtime should be cleaned up")
        custom_output = self.base / "custom_output"
        finished = subprocess.run(["bash", str(launcher), "--output", str(custom_output),
            "--lubrication-model", "finite_gap_fast", "--finite-gap-fast-generator", str(tools / "slurry_pack")],
            env=shell_environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
        self.assertEqual(finished.returncode, 0, finished.stdout)
        arguments = json.loads((custom_output / "launcher_arguments.json").read_text())
        self.assertEqual(arguments.count("--finite-gap-fast-generator"), 1)
        copied_override = arguments[arguments.index("--finite-gap-fast-generator") + 1]
        self.assertIn("slurry_runtime_", copied_override)
        self.assertNotEqual(copied_override, str(tools / "slurry_pack"))
        failed_output = self.base / "failed_output"
        finished = subprocess.run(["bash", str(launcher), "--output", str(failed_output), "--steps", "13"],
            env=shell_environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
        self.assertEqual(finished.returncode, 7, finished.stdout)
        self.assertFalse(failed_output.exists())
        self.assertTrue(Path(str(failed_output) + ".runtime.json").is_file())
        self.assertIn("Slurry job failed: exit=7", finished.stdout)

        # A relative output/runtime root must survive the launcher's cd and EXIT
        # cleanup. Assert files/inodes on disk rather than checking script text.
        relative_environment = dict(shell_environment, SLURRY_RUNTIME_ROOT="relative_runtime")
        finished = subprocess.run(["bash", str(launcher), "--output", "relative_output"],
            cwd=str(self.base), env=relative_environment, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, universal_newlines=True)
        self.assertEqual(finished.returncode, 0, finished.stdout)
        relative_output = self.base / "relative_output"
        self.assertTrue((relative_output / "runtime_identity.json").is_file())
        self.assertFalse(Path(str(relative_output) + ".runtime.json").exists())
        self.assertEqual(list((self.base / "relative_runtime").iterdir()), [])

        # Rejected reuse of an existing output must preserve that run's original
        # audit identity, instead of replacing it in an unconditional EXIT trap.
        original_identity = (relative_output / "runtime_identity.json").read_bytes()
        finished = subprocess.run(["bash", str(launcher), "--output", str(relative_output)],
            env=shell_environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            universal_newlines=True)
        self.assertNotEqual(finished.returncode, 0, finished.stdout)
        self.assertEqual((relative_output / "runtime_identity.json").read_bytes(), original_identity)
        self.assertTrue(Path(str(relative_output) + ".runtime.json").is_file())


if __name__ == "__main__":
    unittest.main()
