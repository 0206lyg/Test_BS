#!/usr/bin/env python3
"""Real GNU ELF regression for private-runtime DT_RPATH isolation.

Builds tiny executables and DSOs, including a dlopen plugin and a transitive
library. No LAMMPS or hydrodynamic simulation is needed. Existing old DT_RPATH
is deliberately retained in source binaries, as on the affected cluster.
"""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

TOOLS = Path(__file__).resolve().parents[1]/"tools"
sys.path.insert(0, str(TOOLS))
import bundle_guard
import runtime_snapshot


LEAF = r'''
#define _GNU_SOURCE
#include <dlfcn.h>
const char *bridge_link_leaf_path(void) {
  Dl_info info;
  return dladdr((void *)&bridge_link_leaf_path, &info) ? info.dli_fname : "unknown";
}
'''
ENGINE = r'''
#define _GNU_SOURCE
#include <dlfcn.h>
extern const char *bridge_link_leaf_path(void);
const char *bridge_link_engine_path(void) {
  Dl_info info;
  return dladdr((void *)&bridge_link_engine_path, &info) ? info.dli_fname : "unknown";
}
const char *bridge_link_engine_leaf_path(void) { return bridge_link_leaf_path(); }
'''
PLUGIN = r'''
extern const char *bridge_link_engine_path(void);
extern const char *bridge_link_engine_leaf_path(void);
const char *bridge_link_plugin_engine_path(void) { return bridge_link_engine_path(); }
const char *bridge_link_plugin_leaf_path(void) { return bridge_link_engine_leaf_path(); }
'''
LOADER = r'''
#include <dlfcn.h>
#include <stdio.h>
#ifdef DIRECT_ENGINE
extern const char *bridge_link_engine_path(void);
extern const char *bridge_link_engine_leaf_path(void);
#endif
int main(int argc, char **argv) {
#ifdef DIRECT_ENGINE
  printf("direct_engine=%s\n", bridge_link_engine_path());
  printf("direct_leaf=%s\n", bridge_link_engine_leaf_path());
#endif
  if (argc != 2) return 2;
  void *plugin = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
  if (!plugin) { fprintf(stderr, "%s\n", dlerror()); return 3; }
  const char *(*engine)(void) = (const char *(*)(void))dlsym(plugin, "bridge_link_plugin_engine_path");
  const char *(*leaf)(void) = (const char *(*)(void))dlsym(plugin, "bridge_link_plugin_leaf_path");
  if (!engine || !leaf) return 4;
  printf("plugin_engine=%s\n", engine());
  printf("plugin_leaf=%s\n", leaf());
  dlclose(plugin);
  return 0;
}
'''


@unittest.skipUnless(sys.platform.startswith("linux") and shutil.which("gcc") and shutil.which("readelf") and shutil.which("ldd"),
                     "requires Linux GNU ELF compiler and binutils")
class RuntimeLinkPathsTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="slurry link path ")
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.original, self.private = self.base/"original install", self.base/"private runtime"
        self.original.mkdir()
        self.private.mkdir()
        for name, text in (("leaf.c", LEAF), ("engine.c", ENGINE), ("plugin.c", PLUGIN), ("loader.c", LOADER)):
            (self.base/name).write_text(text)
        gcc = shutil.which("gcc")
        old = ["-Wl,--disable-new-dtags", "-Wl,-rpath,"+str(self.original)]
        new_tags = ["-Wl,--enable-new-dtags"]
        new = new_tags+["-Wl,-rpath,"+str(self.original)]
        lib = ["-L"+str(self.original)]
        common = [gcc, "-O0"]
        commands = [
            # Cluster compilers may inject an RPATH without a test requesting
            # one. Byte-preservation fixtures explicitly use new dtags so any
            # injected path is RUNPATH; on ordinary compilers they have no path.
            common+["-shared", "-fPIC", str(self.base/"leaf.c"), "-ldl", "-Wl,-soname,libbridge_link_leaf.so"]+new_tags+["-o", str(self.original/"libbridge_link_leaf.so")],
            common+["-shared", "-fPIC", str(self.base/"engine.c")]+lib+["-lbridge_link_leaf", "-ldl", "-Wl,-soname,libbridge_link_engine.so"]+old+["-o", str(self.original/"libbridge_link_engine.so")],
            common+["-shared", "-fPIC", str(self.base/"plugin.c")]+lib+["-lbridge_link_engine", "-Wl,-soname,libbridge_link_plugin.so"]+old+["-o", str(self.original/"libbridge_link_plugin.so")],
            common+["-DDIRECT_ENGINE", str(self.base/"loader.c")]+lib+["-lbridge_link_engine", "-ldl"]+old+["-o", str(self.original/"direct_old")],
            common+["-DDIRECT_ENGINE", str(self.base/"loader.c")]+lib+["-lbridge_link_engine", "-ldl"]+new+["-o", str(self.original/"direct_runpath")],
            common+[str(self.base/"loader.c"), "-ldl"]+new_tags+["-o", str(self.original/"plugin_loader")],
        ]
        for command in commands:
            subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.records = {name: runtime_snapshot.copy_checked(self.original/name, self.private/name)
                        for name in ("libbridge_link_leaf.so", "libbridge_link_engine.so", "libbridge_link_plugin.so", "direct_old", "direct_runpath", "plugin_loader")}
        self.environment = dict(os.environ)
        self.environment["LD_LIBRARY_PATH"] = str(self.private)
        # Tests use the actual configured loader search, without inherited
        # preloading/audit instrumentation changing symbol resolution.
        self.environment.pop("LD_PRELOAD", None)
        self.environment.pop("LD_AUDIT", None)
        self.assertIn("(RPATH)", self.tags(self.original/"direct_old"))
        self.assertNotIn("(RUNPATH)", self.tags(self.original/"direct_old"))

    def tags(self, path):
        return subprocess.run(["readelf", "-d", str(path)], check=True, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, universal_newlines=True).stdout

    def loaded(self, executable="direct_old"):
        process = subprocess.run([str(self.private/executable), str(self.private/"libbridge_link_plugin.so")],
                                 env=self.environment, check=True, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, universal_newlines=True)
        return {key: Path(value).resolve() for key, value in (line.split("=", 1) for line in process.stdout.splitlines())}

    def patch(self, names=None):
        for name in (names or self.records):
            record = self.records[name]
            returned = runtime_snapshot.private_link_paths(record)
            self.assertIs(returned, record)
            self.assertEqual(record["sha256"], bundle_guard.sha256_file(record["original"]))
            self.assertEqual(record["snapshot_sha256"], bundle_guard.sha256_file(record["snapshot"]))
            self.assertIsInstance(record["link_path_edits"], list)

    def assert_private(self, loaded):
        self.assertTrue(loaded)
        for name, path in loaded.items():
            self.assertEqual(path.parent, self.private.resolve(), (name, path))

    def test_old_rpath_escapes_private_directory(self):
        """Reproduce the user's escape before applying any tag conversion."""
        for executable in ("direct_old", "plugin_loader"):
            loaded = self.loaded(executable)
            self.assertTrue(loaded)
            for path in loaded.values():
                self.assertEqual(path.parent, self.original.resolve())
        resolved = runtime_snapshot.linked_libraries(self.private/"direct_old", self.environment)
        self.assertEqual(resolved["libbridge_link_engine.so"].parent, self.original)
        self.assertEqual(resolved["libbridge_link_leaf.so"].parent, self.original)

    def test_executable_and_dlopen_plugin_load_private_copies(self):
        original_hashes = {name: bundle_guard.sha256_file(self.original/name) for name in self.records}
        self.patch()
        for name in ("direct_old", "libbridge_link_engine.so", "libbridge_link_plugin.so"):
            self.assertNotIn("(RPATH)", self.tags(self.private/name))
            self.assertIn("(RUNPATH)", self.tags(self.private/name))
            self.assertIn("(RPATH)", self.tags(self.original/name))
            self.assertNotEqual(self.records[name]["snapshot_sha256"], self.records[name]["sha256"])
        for executable in ("direct_old", "plugin_loader", "direct_runpath"):
            self.assert_private(self.loaded(executable))
            for path in runtime_snapshot.linked_libraries(self.private/executable, self.environment).values():
                self.assertEqual(path.parent, self.private)
        self.assertEqual(original_hashes, {name: bundle_guard.sha256_file(self.original/name) for name in self.records})

    def test_transitive_dso_also_needs_conversion(self):
        self.patch(["direct_old"])
        loaded = self.loaded()
        self.assertEqual(loaded["direct_engine"].parent, self.private)
        self.assertEqual(loaded["direct_leaf"].parent, self.original)
        self.patch(["libbridge_link_engine.so", "libbridge_link_plugin.so", "libbridge_link_leaf.so"])
        self.assert_private(self.loaded())

    def test_private_runtime_survives_original_library_truncation(self):
        self.patch()
        private_hashes = {name: bundle_guard.sha256_file(self.private/name) for name in self.records}
        for name in ("libbridge_link_leaf.so", "libbridge_link_engine.so", "libbridge_link_plugin.so"):
            (self.original/name).write_bytes(b"")
        self.assert_private(self.loaded())
        self.assert_private(self.loaded("plugin_loader"))
        self.assertEqual(private_hashes, {name: bundle_guard.sha256_file(self.private/name) for name in self.records})

    def test_existing_runpath_and_no_rpath_files_keep_bytes(self):
        for name in ("direct_runpath", "plugin_loader", "libbridge_link_leaf.so"):
            self.assertNotIn("(RPATH)", self.tags(self.original/name), name)
        self.assertIn("(RUNPATH)", self.tags(self.original/"direct_runpath"))
        self.patch(["direct_runpath", "plugin_loader", "libbridge_link_leaf.so"])
        for name in ("direct_runpath", "plugin_loader", "libbridge_link_leaf.so"):
            record = self.records[name]
            self.assertEqual(record["sha256"], record["snapshot_sha256"], name)
            self.assertEqual(record["link_path_edits"], [], name)


    def test_absolute_dt_needed_is_rejected_without_writing(self):
        # A DSO without SONAME, linked by absolute filename, produces an actual
        # slash-containing DT_NEEDED. Search-path editing cannot isolate it.
        leaf = self.original/"absolute_leaf.so"
        engine = self.original/"absolute_engine.so"
        gcc = shutil.which("gcc")
        subprocess.run([gcc, "-shared", "-fPIC", str(self.base/"leaf.c"), "-ldl", "-o", str(leaf)],
                       check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        subprocess.run([gcc, "-shared", "-fPIC", str(self.base/"engine.c"), str(leaf), "-ldl",
                        "-Wl,--disable-new-dtags", "-Wl,-rpath,"+str(self.original), "-o", str(engine)],
                       check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.assertIn(str(leaf), self.tags(engine))
        record = runtime_snapshot.copy_checked(engine, self.private/engine.name)
        before = bundle_guard.sha256_file(engine)
        with self.assertRaisesRegex(RuntimeError, "DT_NEEDED contains a path"):
            runtime_snapshot.private_link_paths(record)
        self.assertEqual(bundle_guard.sha256_file(engine), before)
        self.assertEqual(bundle_guard.sha256_file(record["snapshot"]), before)

    def test_malformed_elf_is_rejected_without_writing(self):
        valid = (self.original/"direct_old").read_bytes()
        for index, data in enumerate((b"\x7fELF", valid[:64])):
            with self.subTest(case=index):
                source = self.original/("truncated_"+str(index))
                source.write_bytes(data)
                record = runtime_snapshot.copy_checked(source, self.private/source.name)
                before = record["sha256"]
                with self.assertRaisesRegex(RuntimeError, "malformed runtime ELF"):
                    runtime_snapshot.private_link_paths(record)
                self.assertEqual(bundle_guard.sha256_file(source), before)
                self.assertEqual(bundle_guard.sha256_file(record["snapshot"]), before)

    def test_read_only_private_copy_mode_is_preserved(self):
        path = self.private/"direct_old"
        path.chmod(0o555)
        self.patch(["direct_old"])
        self.assertEqual(path.stat().st_mode & 0o777, 0o555)
        self.assertIn("(RUNPATH)", self.tags(path))


    def test_refuses_to_patch_original_inode(self):
        original = self.original/"direct_old"
        before = bundle_guard.sha256_file(original)
        alias = self.private/"unsafe_hardlink"
        os.link(str(original), str(alias))
        record = {"original": str(original), "snapshot": str(alias), "sha256": before}
        with self.assertRaises((ValueError, RuntimeError)):
            runtime_snapshot.private_link_paths(record)
        self.assertEqual(bundle_guard.sha256_file(original), before)


if __name__ == "__main__":
    unittest.main()
