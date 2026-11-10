#!/usr/bin/env python3
"""Exercise launcher output paths with real Bash, without Slurm or dynamics."""
import csv
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
LAUNCHERS = ("run_baseline.sbatch", "run_pureGr.sbatch", "run_cmc.sbatch")
SCRATCH = Path("/scratch/negishi/lee6456")


def configure(text, settings):
    """Change only named settings in a temporary copy of an editable script."""
    for name, value in settings.items():
        replacement = name + "=" + shlex.quote(str(value))
        text, count = re.subn(r"(?m)^" + re.escape(name) + r"=[^\n]*$",
                              lambda match: replacement, text)
        if count != 1:
            raise AssertionError("Expected one setting: " + name)
    return text


class LauncherPathTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="slurry-launcher-paths-")
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name) / "repository with spaces"
        tools = self.root / "lammps_slurry/tools"
        tools.mkdir(parents=True)
        shutil.copy2(ROOT / "lammps_slurry/tools/cmc_medium.py", tools / "cmc_medium.py")
        (tools / "run_bridge.py").write_text(
            "import json,os,sys\n"
            "with open(os.environ['LAUNCHER_TEST_CAPTURE'], 'w') as f:\n"
            " json.dump({'argv':sys.argv[1:], 'threads':os.environ['OMP_NUM_THREADS'],"
            "'newton':os.environ['SLURRY_IMEX_MAX_ITERATIONS']},f)\n")
        for name in LAUNCHERS:
            shutil.copy2(ROOT / name, self.root / name)
        self.capture = self.root / "capture.json"
        self.environment = {key: value for key, value in os.environ.items()
                            if not key.startswith(("SLURM_", "BASH_FUNC_"))
                            and key not in ("BASH_ENV", "SLURRY_ROOT", "SLURRY_RESULTS_ROOT")}
        self.environment.update(SLURRY_ROOT=str(self.root), SLURM_JOB_ID="44897439",
                                SLURM_CPUS_PER_TASK="2", LAUNCHER_TEST_CAPTURE=str(self.capture))

    def launch(self, name, **settings):
        script = self.root / name
        script.write_text(configure((ROOT / name).read_text(), settings))
        self.capture.unlink(missing_ok=True)
        process = subprocess.run(["bash", str(script)], env=self.environment,
                                 capture_output=True, text=True, timeout=20)
        record = json.loads(self.capture.read_text()) if self.capture.exists() else None
        return process, record

    def flags(self, record):
        arguments = record["argv"]
        self.assertEqual(len(arguments) % 2, 0)
        flags = dict(zip(arguments[::2], arguments[1::2]))
        self.assertEqual(len(flags) * 2, len(arguments), "Repeated options")
        self.assertTrue(all(flag.startswith("--") for flag in flags))
        return flags

    def test_exact_requested_folder_names_and_full_force_values(self):
        work = "0.00212434877286055"
        cases = (
            ("run_baseline.sbatch", {"AR": "2", "PHI": "0.32"},
             "bl_ar2_phi0.32_n1000_sr0.1_44897439"),
            ("run_pureGr.sbatch", {"COHESION_WORK": work},
             "pG_cw0.0021_n1000_sr0.1_44897439"),
            ("run_cmc.sbatch", {"CMC_TOTAL": "16.0", "COHESION_WORK": work,
                                "BACKGROUND_HAMAKER": "9.9e-20"},
             "cmc_c16_cw0.0021_h9.9_n1000_sr0.1_44897439"),
        )
        for name, settings, folder in cases:
            with self.subTest(launcher=name):
                process, record = self.launch(name, COUNT=1000, SHEAR_RATE="0.1", **settings)
                self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
                flags = self.flags(record)
                self.assertEqual(flags["--output"], str(SCRATCH / folder))
                self.assertIn(str(SCRATCH / folder), process.stdout)
                self.assertEqual(flags["--shear-rate"], "0.1")
                self.assertEqual(flags["--count"], "1000")
                self.assertEqual(flags["--build-env"], str(self.root / "build/lammps/env.sh"))
                self.assertEqual(record["threads"], "2")
                if name == "run_baseline.sbatch":
                    self.assertNotIn("--cohesion-work", flags)
                    self.assertNotIn("--background-hamaker", flags)
                else:
                    self.assertEqual(flags["--cohesion-work"], work)
                if name == "run_cmc.sbatch":
                    self.assertEqual(flags["--background-hamaker"], "9.9e-20")
                    self.assertAlmostEqual(float(flags["--viscosity"]), .007233566768075233)
                    self.assertIn("FREE_CMC=", process.stdout)

    def test_cw_rounds_only_the_tag_and_h_uses_fixed_physical_units(self):
        process, record = self.launch("run_cmc.sbatch", CMC_TOTAL="16", SHEAR_RATE="1e-1",
                                      COHESION_WORK="0.00219", BACKGROUND_HAMAKER="9.9e-22")
        self.assertEqual(process.returncode, 0, process.stderr)
        flags = self.flags(record)
        self.assertEqual(Path(flags["--output"]).name,
                         "cmc_c16_cw0.0022_h0.099_n1000_sr0.1_44897439")
        self.assertEqual(flags["--cohesion-work"], "0.00219")
        self.assertEqual(flags["--background-hamaker"], "9.9e-22")
        self.assertEqual(flags["--shear-rate"], "1e-1")

    def test_zero_force_tags_and_inputs(self):
        for name, expected in (("run_pureGr.sbatch", "pG_cw0.0000_n1000_sr0.1_44897439"),
                               ("run_cmc.sbatch", "cmc_c0_cw0.0000_h0_n1000_sr0.1_44897439")):
            settings = {"COHESION_WORK": 0, "BACKGROUND_HAMAKER": 0, "SHEAR_RATE": .1}
            if name == "run_cmc.sbatch":
                settings["CMC_TOTAL"] = 0
            with self.subTest(launcher=name):
                process, record = self.launch(name, **settings)
                self.assertEqual(process.returncode, 0, process.stderr)
                flags = self.flags(record)
                self.assertEqual(Path(flags["--output"]).name, expected)
                self.assertEqual(flags["--cohesion-work"], "0")
                self.assertEqual(flags["--background-hamaker"], "0")

    def test_all_defaults_and_slurm_logs_use_scratch(self):
        for name, prefix in zip(LAUNCHERS, ("bl", "pG", "cmc")):
            with self.subTest(launcher=name):
                text = (ROOT / name).read_text()
                outputs = re.findall(r"(?m)^#SBATCH --output=(.+)$", text)
                self.assertEqual(outputs, [str(SCRATCH / (prefix + "_%j.out"))])
                process, record = self.launch(name)
                self.assertEqual(process.returncode, 0, process.stderr)
                self.assertEqual(Path(self.flags(record)["--output"]).parent, SCRATCH)

    def test_relative_and_explicit_absolute_output_paths(self):
        for name in LAUNCHERS:
            with self.subTest(launcher=name):
                process, record = self.launch(name, OUTPUT_DIR="trial with spaces")
                self.assertEqual(process.returncode, 0, process.stderr)
                self.assertEqual(self.flags(record)["--output"], str(SCRATCH / "trial with spaces"))
                explicit = self.root.parent / "explicit output" / "case"
                process, record = self.launch(name, OUTPUT_DIR=str(explicit))
                self.assertEqual(process.returncode, 0, process.stderr)
                self.assertEqual(self.flags(record)["--output"], str(explicit))
                custom_parent = self.root.parent / "different scratch root"
                process, record = self.launch(name, RESULTS_ROOT=str(custom_parent), OUTPUT_DIR="case")
                self.assertEqual(process.returncode, 0, process.stderr)
                self.assertEqual(self.flags(record)["--output"], str(custom_parent / "case"))

    def test_sweeps_dry_run_use_new_templates_and_scratch_staging(self):
        cases = (("sweep_slurry.sh", "run_baseline.sbatch", 2),
                 ("sweep_practical_slurry.sh", "run_pureGr.sbatch", 5))
        for sweep, launcher, count in cases:
            with self.subTest(sweep=sweep):
                text = (ROOT / sweep).read_text()
                self.assertIn('SWEEP_ROOT="/scratch/negishi/lee6456/sweeps"', text)
                staging = self.root.parent / (sweep + " scratch with spaces")
                script = self.root / sweep
                script.write_text(configure(text, {"SWEEP_ROOT": str(staging)}))
                process = subprocess.run(["bash", str(script), "--dry-run"], env=self.environment,
                                         capture_output=True, text=True, timeout=20)
                self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
                tables = list(staging.rglob("jobs.tsv"))
                self.assertEqual(len(tables), 1)
                with tables[0].open() as stream:
                    rows = list(csv.DictReader(stream, delimiter="\t"))
                self.assertEqual(len(rows), count)
                for row in rows:
                    path = Path(row["script"])
                    self.assertEqual(row["job_id"], "DRY_RUN")
                    self.assertEqual(path.name, launcher)
                    self.assertTrue(path.is_relative_to(staging))
                    self.assertTrue(path.is_file())
                    subprocess.run(["bash", "-n", str(path)], check=True, capture_output=True)
                self.assertFalse((self.root / "sweeps").exists())


if __name__ == "__main__":
    unittest.main()
