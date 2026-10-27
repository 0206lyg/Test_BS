#!/usr/bin/env python3
"""Release/preflight regressions; no test compiles or approves a physical model.

Build-manifest fixtures contain inert byte files in temporary directories. They
exercise identity rejection only and are never exported as verified builds.
The batch test runs the real source guard and intercepts only the build handoff.
"""
from contextlib import contextmanager
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "lammps_slurry/tools"
spec = importlib.util.spec_from_file_location("bundle_guard_test_target", TOOLS / "bundle_guard.py")
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


class BundleGuardTest(unittest.TestCase):
    @contextmanager
    def source_fixture(self):
        """Copy only the source guard's inputs; no historical data or binaries."""
        with tempfile.TemporaryDirectory(prefix="slurry-guard-test-") as temporary:
            root = Path(temporary) / "source with spaces"
            snapshot = guard.source_snapshot(ROOT)
            for relative in snapshot["files"]:
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(ROOT / relative), str(target))
            shutil.copy2(str(ROOT / "build_slurry_cpu.sbatch"), str(root / "build_slurry_cpu.sbatch"))
            yield root

    def command(self, root, operation="check-source", extra=()):
        return [sys.executable, str(root / "lammps_slurry/tools/bundle_guard.py"), operation,
                "--root", str(root), "--expected-release", guard.RELEASE_ID] + list(extra)

    def run_guard(self, root, operation="check-source", extra=()):
        return subprocess.run(self.command(root, operation, extra), capture_output=True,
                              universal_newlines=True)

    def assert_rejected(self, process, message):
        self.assertNotEqual(process.returncode, 0, process.stdout + process.stderr)
        self.assertIn(message, process.stderr)

    def test_actual_release_identity_and_source_preflight(self):
        release = json.loads((ROOT / "release.json").read_text())
        launcher = (ROOT / "build_slurry_cpu.sbatch").read_text()
        matches = re.findall(r"^readonly SLURRY_LAUNCHER_RELEASE=(\S+)\s*$", launcher, re.MULTILINE)
        self.assertEqual(matches, [guard.RELEASE_ID])
        self.assertEqual(release["release_id"], guard.RELEASE_ID)
        self.assertEqual(release["prepare_bridge_sha256"], guard.sha256_file(TOOLS / "prepare_bridge.py"))
        self.assertEqual(release["prepare_case_sha256"], guard.sha256_file(TOOLS / "prepare_case.py"))
        process = self.run_guard(ROOT)
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        self.assertIn("verified: " + guard.RELEASE_ID, process.stdout)

    def test_changed_generator_is_rejected(self):
        with self.source_fixture() as root:
            generator = root / "lammps_slurry/tools/prepare_bridge.py"
            generator.write_text(generator.read_text() + "\n# deliberately altered test copy\n")
            self.assert_rejected(self.run_guard(root), "prepare_bridge.py does not match release")

    def test_stale_release_is_rejected(self):
        with self.source_fixture() as root:
            path = root / "release.json"
            release = json.loads(path.read_text())
            release["release_id"] = "20261025-spheroid-strong-rolling"
            path.write_text(json.dumps(release))
            self.assert_rejected(self.run_guard(root), "Source release does not match")

    def inert_build_manifest(self, root):
        """Construct only a temporary identity fixture, never a production approval."""
        binary_dir = root / "inert-test-binaries"
        binary_dir.mkdir()
        paths = {"executable": binary_dir / "lmp", "plugin": binary_dir / "libslurry.so",
                 "packer": binary_dir / "slurry_pack"}
        for name, path in paths.items():
            path.write_bytes(("NOT EXECUTABLE: identity test fixture " + name).encode("ascii"))
        offline = {}
        for name in guard.OFFLINE_TOOLS:
            path = binary_dir / name
            path.write_bytes(("NOT EXECUTABLE: identity test fixture " + name).encode("ascii"))
            offline[name] = {"path": str(path), "sha256": guard.sha256_file(path)}
        snapshot = guard.source_snapshot(root)
        record = {
            "test_fixture_only": True,
            "bundle_integrity_schema": guard.MANIFEST_SCHEMA,
            "bundle_release": guard.RELEASE_ID,
            "bundle_root": str(root),
            "source_fingerprint": snapshot["sha256"],
            "source_files": snapshot["files"],
            "offline_tools": offline,
            "binary_sha256": {name: guard.sha256_file(path) for name, path in paths.items()},
        }
        record.update({name: str(path) for name, path in paths.items()})
        manifest = root / "test-only-build-identity.json"
        manifest.write_text(json.dumps(record))
        extra = []
        for name, path in list(paths.items()) + [("manifest", manifest)]:
            extra += ["--" + name, str(path)]
        return paths, manifest, record, extra

    def test_build_identity_rejects_stale_release_source_and_binary(self):
        with self.source_fixture() as root:
            paths, manifest, record, extra = self.inert_build_manifest(root)
            process = self.run_guard(root, "check-build", extra)
            self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
            record["bundle_release"] = "20261025-spheroid-strong-rolling"
            manifest.write_text(json.dumps(record))
            self.assert_rejected(self.run_guard(root, "check-build", extra),
                                 "Build belongs to a different source release")
            record["bundle_release"] = guard.RELEASE_ID
            manifest.write_text(json.dumps(record))
            header = root / "lammps_slurry/src/slurry_compact_bridge.h"
            original = header.read_bytes()
            header.write_bytes(original + b"\n// deliberately altered test source\n")
            self.assert_rejected(self.run_guard(root, "check-build", extra),
                                 "Source has changed since the verified build")
            header.write_bytes(original)
            paths["plugin"].write_bytes(b"deliberately different inert binary fixture")
            self.assert_rejected(self.run_guard(root, "check-build", extra),
                                 "plugin content differs from the verified build")

    def test_build_identity_rejects_relocated_and_changed_offline_tools(self):
        with self.source_fixture() as root:
            paths, manifest, record, extra = self.inert_build_manifest(root)
            record["bundle_root"] = str(root / "another source")
            manifest.write_text(json.dumps(record))
            self.assert_rejected(self.run_guard(root, "check-build", extra),
                                 "Build belongs to another source directory")
            record["bundle_root"] = str(root)
            manifest.write_text(json.dumps(record))
            paths["packer"].with_name(guard.OFFLINE_TOOLS[0]).write_bytes(b"changed inert tool")
            self.assert_rejected(self.run_guard(root, "check-build", extra),
                                 "Offline generation/validation tool differs")

    def batch_preflight(self, root):
        stubs = root / "test-command-stubs"
        stubs.mkdir()
        (stubs / "module").write_text("#!/bin/sh\nexit 0\n")
        (stubs / "module").chmod(0o755)
        # Forward bundle_guard.py to the real interpreter. Only the eventual
        # runtime_snapshot.py build handoff is intercepted; no guard is mocked.
        shim = stubs / "python3"
        shim.write_text(
            "#!" + sys.executable + "\n"
            "import json,os,pathlib,sys\n"
            "args=sys.argv[1:]\n"
            "name=pathlib.Path(args[0]).name if args else ''\n"
            "with open(os.environ['SLURRY_TEST_HANDOFF_LOG'],'a') as stream:\n"
            " stream.write(json.dumps(args)+'\\n')\n"
            "if name=='bundle_guard.py':\n"
            " os.execv(sys.executable,[sys.executable]+args)\n"
            "if name=='runtime_snapshot.py' and args[1:2]==['build']:\n"
            " sys.exit(0)\n"
            "raise SystemExit('Unexpected Python command in build preflight: '+repr(args))\n")
        shim.chmod(0o755)
        capture = root / "test-handoff.jsonl"
        environment = {key: value for key, value in os.environ.items()
                       if not key.startswith(("SLURM_", "BASH_FUNC_")) and key != "BASH_ENV"}
        environment.update(PATH=str(stubs) + os.pathsep + os.environ["PATH"],
                           SLURRY_ROOT=str(root), SLURM_CPUS_PER_TASK="3",
                           SLURRY_TEST_HANDOFF_LOG=str(capture))
        process = subprocess.run(["bash", str(root / "build_slurry_cpu.sbatch")],
                                 cwd=str(root), env=environment, capture_output=True,
                                 universal_newlines=True)
        calls = [json.loads(line) for line in capture.read_text().splitlines()] if capture.exists() else []
        return process, calls

    def test_batch_preflight_reaches_build_handoff_after_real_guard(self):
        with self.source_fixture() as root:
            process, calls = self.batch_preflight(root)
            self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
            self.assertEqual(calls, [
                [str(root / "lammps_slurry/tools/bundle_guard.py"), "check-source",
                 "--root", str(root), "--expected-release", guard.RELEASE_ID],
                [str(root / "lammps_slurry/tools/runtime_snapshot.py"), "build",
                 "--root", str(root), "--", "--jobs", "3"],
            ])
            self.assertIn("verified: " + guard.RELEASE_ID, process.stdout)
            self.assertFalse((root / "build").exists(), "Preflight must not invoke compilation")

    def test_batch_preflight_stops_before_build_for_mixed_source(self):
        with self.source_fixture() as root:
            generator = root / "lammps_slurry/tools/prepare_bridge.py"
            generator.write_text(generator.read_text() + "\n# deliberately altered test copy\n")
            process, calls = self.batch_preflight(root)
            self.assert_rejected(process, "prepare_bridge.py does not match release")
            self.assertEqual(len(calls), 1)
            self.assertEqual(Path(calls[0][0]).name, "bundle_guard.py")
            self.assertFalse((root / "build").exists())


if __name__ == "__main__":
    unittest.main()
