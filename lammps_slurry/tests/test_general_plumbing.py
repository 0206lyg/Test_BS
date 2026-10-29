#!/usr/bin/env python3
"""General runtime input routing; synthetic packer is not a physics test."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
PREPARE = ROOT / "lammps_slurry/tools/prepare_case.py"
spec = importlib.util.spec_from_file_location("general_prepare", str(PREPARE))
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


def arguments(**overrides):
    values = dict(lubrication_model="general", ar=2.,
        finite_gap_fast_table=None, finite_gap_fast_source=None, finite_gap_fast_generator=None,
        finite_gap_basis=None, finite_gap_table=None, finite_gap_range=.5, finite_gap_film_grid=161,
        lubrication_cutoff_nm=None, lubrication_epsilon_relative=.001,
        lubrication_cutoff_relative=.5, lubrication_epsilon_nm=.15,
        roughness_nm=3., contact_gap_nm=3., general_tolerance=.005,
        general_max_level=6, general_cutoff=1., general_normal_log="leading",
        general_closure="relative", general_near_completion="energy")
    values.update(overrides)
    return SimpleNamespace(**values)


class GeneralPlumbing(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="slurry-general-routing-")
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)

    def test_no_basis_or_table_generation_and_physical_lengths(self):
        with mock.patch.object(prepare, "resolve_fast_table", side_effect=AssertionError("table path called")):
            for ar in (1., 2., 5., 10., 20., 3.7):
                model, epsilon, cutoff = prepare.resolve_lubrication(arguments(ar=ar), ar*150e-9)
                self.assertEqual(model, "general")
                self.assertAlmostEqual(epsilon/1e-9, .15)
                self.assertAlmostEqual(cutoff/(ar*150e-9), .5)

    def test_external_model_inputs_and_invalid_controls_are_rejected(self):
        for changed in (dict(finite_gap_basis=Path("missing.ssb")),
                        dict(finite_gap_table=Path("missing.sfg")),
                        dict(finite_gap_fast_source=Path("missing.ssb")),
                        dict(general_tolerance=1.), dict(general_max_level=0),
                        dict(general_max_level=7), dict(general_cutoff=0.),
                        dict(general_normal_log="unknown"), dict(general_closure="unknown"),
                        dict(general_near_completion="unknown")):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                prepare.resolve_lubrication(arguments(**changed), 300e-9)

    def test_legacy_auto_unchanged_and_general_overrides_not_silently_ignored(self):
        self.assertEqual(prepare.resolve_lubrication(arguments(lubrication_model="auto"), 300e-9)[0], "spheroid")
        self.assertEqual(prepare.resolve_lubrication(arguments(lubrication_model="auto", ar=1), 150e-9)[0], "mari")
        with self.assertRaisesRegex(ValueError, "general options require"):
            prepare.resolve_lubrication(arguments(lubrication_model="auto", general_cutoff=2.), 300e-9)

    def test_generated_fix_uses_general_controls_and_metadata(self):
        # At very low phi the initial seed is already the requested final size.
        # This stub only bypasses the real geometric packer's executable.
        packer = self.base / "copy_seed.py"
        packer.write_text("#!" + sys.executable + "\nimport shutil,sys\na=sys.argv\n"
                          "shutil.copyfile(a[a.index('--input')+1],a[a.index('--output')+1])\n")
        packer.chmod(0o755)
        for ar, closure in ((2, "relative"), (20, "pair_additive")):
            output = self.base / ("AR" + str(ar))
            command = [sys.executable, str(PREPARE), "--ar", str(ar), "--phi", ".0001",
                "--initial-phi", ".0001", "--count", "2", "--steps", "0", "--prep-steps", "0",
                "--hydro-interaction-range", ".75", "--general-tolerance", ".01",
                "--general-max-level", "4", "--general-cutoff", "1.5",
                "--general-normal-log", "derived", "--general-closure", closure,
                "--general-near-completion", "raw",
                "--packer", str(packer), "--output", str(output)]
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    universal_newlines=True)
            self.assertEqual(result.returncode, 0, result.stdout)
            text = (output / "in.slurry").read_text()
            self.assertIn("lubrication general finite_gap_range 0.75 general_tolerance 0.01 general_max_level 4", text)
            self.assertIn("general_cutoff 1.5 general_normal_log derived general_closure " + closure, text)
            self.assertIn("general_near_completion raw", text)
            self.assertNotIn("finite_gap_basis", text)
            self.assertNotIn("finite_gap_fast_table", text)
            self.assertFalse(list(output.glob("*.sft")) + list(output.glob("*.ssb")))
            metadata = (output / "parameters.txt").read_text()
            self.assertIn("general_orientation_table = False", metadata)
            self.assertIn("lubrication_gap_rule = max(h,epsilon)", metadata)
            self.assertIn("general_closure = " + closure, metadata)

    def test_spooled_launcher_uses_submit_root_and_preserves_user_overrides(self):
        root = self.base / "source"
        target = root / "lammps_slurry/run_slurry_cpu.sbatch"
        target.parent.mkdir(parents=True)
        target.write_text("#!/bin/bash\nprintf '%s\\n' \"$@\"\n")
        spool = self.base / "slurm_script"
        spool.write_text((ROOT / "lammps_slurry/run_general_cpu.sbatch").read_text())
        environment = dict(os.environ, SLURM_SUBMIT_DIR=str(root))
        environment.pop("SLURRY_ROOT", None)
        result = subprocess.run(["bash", str(spool), "--ar", "20", "--steps", "100"],
            env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
        self.assertEqual(result.returncode, 0, result.stdout)
        values = result.stdout.splitlines()
        self.assertEqual(values[:2], ["--lubrication-model", "general"])
        self.assertEqual(values[-4:], ["--ar", "20", "--steps", "100"])


if __name__ == "__main__":
    unittest.main()
