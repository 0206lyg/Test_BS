#!/usr/bin/env python3
"""Packing/image and averaging regressions for the new runtime tools."""
import csv
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import sys
import unittest
from unittest import mock

TOOLS = Path(__file__).resolve().parents[1]/"tools"
def load(name):
    spec = importlib.util.spec_from_file_location(name, TOOLS/(name+".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
prepare = load("prepare_bridge")
summary = load("summarize_bridge")
sys.path.insert(0, str(TOOLS))
runner = load("run_bridge")
TEST_PACKER = Path(os.environ.get("SLURRY_TEST_PACKER", str(prepare.DEFAULT_PACKER)))


class BridgeToolsTest(unittest.TestCase):
    def test_one_mpi_rank_receives_all_allocated_cores(self):
        env, threads = runner.execution_environment({"SLURM_CPUS_PER_TASK": "16", "OMP_NUM_THREADS": "1"})
        self.assertEqual(threads, 16)
        self.assertEqual(env["OMP_NUM_THREADS"], "16")
        self.assertEqual(env["MKL_NUM_THREADS"], "16")
        self.assertEqual(env["OMP_PLACES"], "cores")
        self.assertEqual(env["OMP_PROC_BIND"], "close")
        command = runner.execution_command(Path("/tmp/with spaces/runtime"), True, threads, launcher="/usr/bin/mpirun")
        self.assertEqual(command[:9], ["/usr/bin/mpirun", "-np", "1", "--map-by", "slot:PE=16",
                                       "--bind-to", "core", "--report-bindings", "/tmp/with spaces/runtime/bin/lmp"])
        self.assertEqual(command.count("-np"), 1)
        local = runner.execution_command(Path("/tmp/runtime"), False, 8)
        self.assertEqual(local[0], "/tmp/runtime/bin/lmp")
        self.assertNotIn("-np", local)

    def test_rejects_nested_mpi_driver_and_invalid_thread_counts(self):
        for key in ("OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "PMIX_UNIV_SIZE"):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "outside MPI"):
                runner.execution_environment({key: "8"})
        for value in ("0", "-2", "8,2", "many"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                runner.execution_environment({"OMP_NUM_THREADS": value})

    def test_failed_simulation_is_not_relaunched(self):
        process = mock.Mock(stdout=["simulation failed\n"])
        process.wait.return_value = 7
        with tempfile.TemporaryDirectory() as temporary:
            with mock.patch.object(runner.subprocess, "Popen", return_value=process) as launch:
                with self.assertRaisesRegex(RuntimeError, "LAMMPS exited 7"):
                    runner.run_simulation(["lmp", "-in", "in.bridge"], Path(temporary), {})
            self.assertEqual(launch.call_count, 1)
            self.assertEqual(process.wait.call_count, 1)
            self.assertEqual((Path(temporary)/"console.log").read_text(), "simulation failed\n")

    def test_single_batch_forwards_every_physical_setting_once(self):
        source = TOOLS.parents[1]/"run_slurry_cpu.sbatch"
        text = source.read_text()
        self.assertIn("#SBATCH --ntasks=1\n", text)
        self.assertIn("#SBATCH --hint=nomultithread\n", text)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)/"source with spaces"
            tools = root/"lammps_slurry/tools"
            tools.mkdir(parents=True)
            (tools/"run_bridge.py").write_text("# command is captured before this file executes\n")
            script = root/"run_slurry_cpu.sbatch"
            stub = root/"bin/python3"
            stub.parent.mkdir()
            stub.write_text("#!"+sys.executable+"\nimport json,os,sys\n"
                            "with open(os.environ['CAPTURE'], 'a') as f:\n"
                            " f.write(json.dumps({'argv':sys.argv[1:], 'threads':os.environ['OMP_NUM_THREADS']})+'\\n')\n")
            stub.chmod(0o755)
            capture = root/"capture.jsonl"
            environment = {key: value for key, value in os.environ.items()
                           if not key.startswith(("SLURM_", "BASH_FUNC_")) and key not in ("BASH_ENV", "SLURRY_ROOT")}
            environment.update(PATH=str(stub.parent)+os.pathsep+os.environ["PATH"],
                               SLURRY_ROOT=str(root), SLURM_JOB_ID="12345", SLURM_CPUS_PER_TASK="16",
                               CAPTURE=str(capture))
            for ar, count in ((2, 500), (5, 123)):
                script.write_text(text.replace("AR=2\n", "AR={}\n".format(ar))
                                  .replace("COUNT=500\n", "COUNT={}\n".format(count)))
                subprocess.run(["bash", str(script)], env=environment, check=True, capture_output=True, text=True)
                records = [json.loads(line) for line in capture.read_text().splitlines()]
                self.assertEqual(len(records), 1 if ar == 2 else 2)
                record = records[-1]
                self.assertEqual(record["threads"], "16")
                arguments = record["argv"][1:]
                index = arguments.index("--build-env")
                del arguments[index:index+2]
                supplied = {arg for arg in arguments if arg.startswith("--")}
                expected = {option for action in prepare.parser()._actions for option in action.option_strings
                            if option not in ("-h", "--help", "--packer")}
                self.assertEqual(supplied, expected)
                args = prepare.parser().parse_args(arguments)
                self.assertEqual(args.ar, ar)
                self.assertEqual(args.count, count)
                self.assertEqual(args.phi, .24)
                self.assertEqual(args.steps, 100000)
                self.assertEqual(args.output_interval, 100)
                self.assertEqual(args.normal_stiffness, 100.)
                self.assertEqual(args.mobility_backend, "mesh")
                self.assertEqual(args.friction, .1)
                self.assertIn("12345", str(args.output))

    def test_defaults_and_explicit_physical_overrides(self):
        defaults = prepare.parser().parse_args(["--output", "unused"])
        self.assertEqual(defaults.normal_stiffness, 100.)
        self.assertEqual(defaults.contact_gap_nm, 3.)
        self.assertEqual(defaults.tangential_stiffness, 80.)
        self.assertEqual(defaults.mobility_backend, "mesh")
        explicit = prepare.parser().parse_args(["--output", "unused", "--normal-stiffness", "250",
                                                "--contact-gap-nm", "2", "--mobility-backend", "direct"])
        self.assertEqual(explicit.normal_stiffness, 250.)
        self.assertEqual(explicit.contact_gap_nm, 2.)
        self.assertEqual(explicit.mobility_backend, "direct")

    def test_small_box_all_images_including_self(self):
        q = (1., 0., 0., 0.)
        particles = [((.75, .75, .75), q, prepare.shape_matrix(q, .05))]
        with self.assertRaisesRegex(ValueError, "periodic-image overlap"):
            prepare.verify_packing(particles, 1., 1.5, 1e-6)

    def test_accepted_n108_ar20_box_smaller_than_six_a(self):
        a, c, count, phi = 20., 1., 108, .24
        box = (count*4*math.pi*a*a*c/(3*phi))**(1/3)
        particles, packing = prepare.aligned_particles(count, a, c, box, .02)
        worst, images = prepare.verify_packing(particles, a, box, 1e-6)
        self.assertLess(box, 6*a)
        self.assertEqual(len(particles), count)
        self.assertEqual(worst, 0.)
        self.assertGreater(images, 0)
        self.assertEqual(packing["lattice"], (2, 2, 27))

    def test_generated_xz_input_uses_one_nm_actual_gap_floor(self):
        self.assertTrue(TEST_PACKER.is_file(), "Set SLURRY_TEST_PACKER to the compiled geometry packer")
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)/"case"
            args = prepare.parser().parse_args(["--output", str(output), "--initialization", "aligned",
                                                "--packer", str(TEST_PACKER), "--steps", "0", "--shear-rate", "-100"])
            record = prepare.prepare(args)
            text = (output/"in.bridge").read_text()
            self.assertIn("fix shear all deform 0 xz erate -100 remap v", text)
            self.assertIn("plugin load ${slurry_plugin}\n", text)
            self.assertNotIn('plugin load "${slurry_plugin}"', text)
            self.assertIn("slurry/bridge", text)
            self.assertNotIn("finite_gap", text)
            self.assertEqual(record["gap_floor_nm"], 1.)
            self.assertEqual(record["spectral"], 2.)
            self.assertEqual(record["normal_stiffness"], 100.)
            self.assertEqual(record["mobility_backend"], "mesh")
            self.assertEqual(record["mesh_numerics"]["window_support"], 10)
            self.assertIn("not a total", record["spectral_tolerance_scope"])
            self.assertTrue(record["packing_clearance"]["certified"])
            self.assertGreater(record["packing_clearance"]["checked_pair_images"], 0)
            self.assertGreaterEqual(record["packing_clearance"]["minimum_surface_gap_m"], 3e-9)
            self.assertEqual(record["packing_clearance"]["max_clearance_deficit_m"], 0.)
            force_line = next(line for line in text.splitlines() if line.startswith("fix hydro ")).split()
            self.assertEqual(float(force_line[7]), 100.)
            self.assertIn("mobility_backend mesh", text)
            with self.assertRaisesRegex(ValueError, "already exists"):
                prepare.prepare(args)

    def test_uncertified_clearance_does_not_write_dynamics(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)/"case"
            args = prepare.parser().parse_args(["--output", str(output), "--initialization", "aligned"])
            with mock.patch.object(prepare, "certify_packing", side_effect=ValueError("clearance rejected")):
                with self.assertRaisesRegex(ValueError, "clearance rejected"):
                    prepare.prepare(args)
            self.assertFalse((output/"in.bridge").exists())
            self.assertFalse((output/"parameters.json").exists())

    @unittest.skipUnless(TEST_PACKER.is_file(), "compiled packer supplied by CTest")
    def test_random_preparation_certifies_requested_clearance_and_preserves_overrides(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)/"case"
            args = prepare.parser().parse_args([
                "--output", str(output), "--packer", str(TEST_PACKER), "--count", "16",
                "--ar", "2", "--phi", ".1", "--steps", "0", "--normal-stiffness", "250",
                "--contact-gap-nm", "2", "--mobility-backend", "direct"])
            record = prepare.prepare(args)
            self.assertEqual(record["normal_stiffness"], 250.)
            self.assertEqual(record["mobility_backend"], "direct")
            self.assertGreaterEqual(record["packing_clearance"]["minimum_surface_gap_m"], 2e-9)
            self.assertEqual(record["packing_clearance"]["max_clearance_deficit_m"], 0.)
            self.assertAlmostEqual(record["packing_clearance"]["nominal_volume_fraction"], .1)
            force_line = next(line for line in (output/"in.bridge").read_text().splitlines()
                              if line.startswith("fix hydro ")).split()
            self.assertEqual(float(force_line[7]), 250.)
            self.assertEqual(float(force_line[9]), 2e-9)

    def test_clearance_certificate_must_match_requested_gap(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            args = prepare.parser().parse_args(["--output", str(output), "--packer", sys.executable])
            q = (1., 0., 0., 0.)
            particles = [((0., 0., 0.), q, prepare.shape_matrix(q, .5))]
            def certificate(command, **kwargs):
                self.assertEqual(command[command.index("--certify-only")+1], "1")
                self.assertAlmostEqual(float(command[command.index("--contact-gap-m")+1]), 3e-9)
                (output/"packing_clearance.json").write_text(json.dumps({
                    "certified": True, "target_gap_m": 1e-9, "minimum_surface_gap_m": 4e-9,
                    "max_clearance_deficit_m": 0., "nominal_volume_fraction": args.phi,
                    "checked_pair_images": 1}))
                return prepare.subprocess.CompletedProcess(command, 0, stdout="")
            with mock.patch.object(prepare.subprocess, "run", side_effect=certificate):
                with self.assertRaisesRegex(ValueError, "requested contact gap"):
                    prepare.certify_packing(args, particles, 1., .5, 10., output)

    def test_clearance_certificate_rejects_nonfinite_or_contradictory_values(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            args = prepare.parser().parse_args(["--output", str(output), "--packer", sys.executable])
            q = (1., 0., 0., 0.)
            particles = [((0., 0., 0.), q, prepare.shape_matrix(q, .5))]
            valid = {"certified": True, "target_gap_m": 3e-9, "minimum_surface_gap_m": 4e-9,
                     "max_clearance_deficit_m": 0., "nominal_volume_fraction": args.phi,
                     "checked_pair_images": 1}
            for field, value in (("minimum_surface_gap_m", float("nan")),
                                 ("minimum_surface_gap_m", float("inf")),
                                 ("minimum_surface_gap_m", 2.9e-9),
                                 ("max_clearance_deficit_m", 1e-12),
                                 ("nominal_volume_fraction", .5), ("checked_pair_images", 0)):
                with self.subTest(field=field, value=value):
                    def certificate(command, **kwargs):
                        report = dict(valid)
                        report[field] = value
                        (output/"packing_clearance.json").write_text(json.dumps(report))
                        return prepare.subprocess.CompletedProcess(command, 0, stdout="")
                    with mock.patch.object(prepare.subprocess, "run", side_effect=certificate):
                        with self.assertRaises(ValueError):
                            prepare.certify_packing(args, particles, 1., .5, 10., output)

    def test_plugin_path_spaces_are_quoted_by_variable_value(self):
        self.assertEqual(runner.lammps_quoted_path(Path("/tmp/with spaces/libslurry.so")),
                         '"/tmp/with spaces/libslurry.so"')
        with self.assertRaises(ValueError):
            runner.lammps_quoted_path('/tmp/a"b/libslurry.so')
        with self.assertRaises(ValueError):
            runner.lammps_quoted_path('/tmp/a\nb/libslurry.so')

    def history(self, path):
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=summary.REQUIRED)
            writer.writeheader()
            for start, end, viscosity in ((0., 1., 2.), (1., 4., 4.)):
                row = {"interval_steps": 1, "interval_start_time_s": start, "interval_end_time_s": end,
                       "interval_start_strain": 2*start, "interval_end_strain": 2*end}
                values = {"eta_Pa_s": viscosity, "eta_relative": viscosity, "sigma_xz_Pa": 2*viscosity,
                          "sigma_solvent_Pa": 2., "sigma_hydro_Pa": 2*viscosity-2., "sigma_contact_Pa": 0.}
                row.update({"interval_mean_"+k: v for k, v in values.items()})
                writer.writerow(row)

    def test_viscosity_is_time_weighted_and_empty_window_errors(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/"history.csv"
            self.history(path)
            result = summary.summarize(path, start=0., blocks=5)
            self.assertAlmostEqual(result["mean"]["eta_Pa_s"], 3.5)
            self.assertAlmostEqual(result["mean"]["sigma_xz_Pa"], 7.)
            self.assertFalse(result["stationarity_certified"])
            self.assertEqual(result["block_stability_check"], "insufficient_block_coverage")
            with self.assertRaisesRegex(ValueError, "No complete output intervals"):
                summary.summarize(path, start=.1, end=1.)
            with self.assertRaises(ValueError):
                summary.summarize(path, start=100.)

if __name__ == "__main__":
    unittest.main()
