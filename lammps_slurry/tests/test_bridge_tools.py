#!/usr/bin/env python3
"""Packing/image and averaging regressions for the new runtime tools."""
import csv
import importlib.util
import contextlib
import io
import json
import math
import os
from pathlib import Path
import re
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
        settings_text, separator, runtime_text = text.partition("# End of run settings.")
        self.assertTrue(separator, "Cannot find the batch settings section")
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
            for ar, count, shape in ((2, 500, "spheroid"), (5, 123, "spheroid")):
                # Users edit these assignments for each run. Configure only the
                # temporary test copy by name, independent of their saved values.
                # Literal replacements such as AR=2 silently miss AR=20.
                settings = {
                    "PARTICLE_SHAPE": shape, "AR": ar, "COUNT": count,
                    "PHI": .24, "STEPS": 100000, "OUTPUT_INTERVAL": 100,
                    "NORMAL_STIFFNESS": 100, "MOBILITY_BACKEND": "mesh",
                    "FRICTION": 10, "ROLLING_FRICTION": 10, "ROLLING_STIFFNESS": 80,
                    "NOMINAL_DIAMETER_UM": 8, "LUBRICATION_EPSILON": .001,
                    "OUTPUT_DIR": '""', "BUILD_ENV": '""',
                    "PREPARE_ONLY": 0, "TRACE_SOLVER": 0,
                }
                configured = settings_text
                for name, value in settings.items():
                    configured, replacements = re.subn(
                        r"(?m)^[ \t]*"+re.escape(name)+r"[ \t]*=[^\n]*$",
                        lambda match: "{}={}".format(name, value), configured)
                    self.assertEqual(replacements, 1, "Missing or repeated batch setting: "+name)
                configured += separator+runtime_text
                script.write_text(configured)
                subprocess.run(["bash", str(script)], env=environment, check=True, capture_output=True, text=True)
                records = [json.loads(line) for line in capture.read_text().splitlines()]
                self.assertEqual(len(records), 1 if ar == 2 else 2)
                record = records[-1]
                self.assertEqual(record["threads"], "16")
                arguments = record["argv"][1:]
                index = arguments.index("--build-env")
                del arguments[index:index+2]
                supplied = {arg for arg in arguments if arg.startswith("--")}
                self.assertEqual(len(supplied), sum(arg.startswith("--") for arg in arguments))
                expected = {option for action in prepare.parser()._actions for option in action.option_strings
                            if option not in ("-h", "--help", "--packer")}
                self.assertEqual(supplied, expected)
                args = prepare.parser().parse_args(arguments)
                self.assertEqual(args.ar, ar)
                self.assertEqual(args.shape, shape)
                self.assertIn(shape+"_ar", str(args.output))
                self.assertEqual(args.count, count)
                self.assertEqual(args.phi, .24)
                self.assertEqual(args.steps, 100000)
                self.assertEqual(args.output_interval, 100)
                self.assertEqual(args.normal_stiffness, 100.)
                self.assertEqual(args.mobility_backend, "mesh")
                self.assertEqual(args.friction, 10.)
                self.assertEqual(args.rolling_friction, 10.)
                self.assertEqual(args.rolling_stiffness, 80.)
                self.assertEqual(args.nominal_diameter_um, 8.)
                self.assertEqual(args.lubrication_epsilon, .001)
                self.assertIn("12345", str(args.output))

    def test_defaults_and_explicit_physical_overrides(self):
        defaults = prepare.parser().parse_args(["--output", "unused"])
        prepare.validate_args(defaults)
        self.assertEqual(defaults.ar, 2.)
        self.assertEqual(defaults.count, 500)
        self.assertEqual(defaults.nominal_diameter_um, 8.)
        self.assertEqual(defaults.lubrication_epsilon, .001)
        self.assertEqual(defaults.normal_stiffness, 100.)
        self.assertEqual(defaults.tangential_stiffness, 80.)
        self.assertEqual(defaults.rolling_stiffness, 80.)
        self.assertEqual(defaults.friction, 10.)
        self.assertEqual(defaults.rolling_friction, 10.)
        self.assertEqual(defaults.mobility_backend, "mesh")
        self.assertEqual(defaults.shape, "spheroid")
        explicit = prepare.parser().parse_args([
            "--output", "unused", "--normal-stiffness", "250", "--friction", "0",
            "--rolling-friction", "0", "--rolling-stiffness", "40", "--mobility-backend", "direct"])
        prepare.validate_args(explicit)
        self.assertEqual(explicit.normal_stiffness, 250.)
        self.assertEqual(explicit.rolling_stiffness, 40.)
        self.assertEqual(explicit.rolling_friction, 0.)
        self.assertEqual(explicit.mobility_backend, "direct")

    def test_fixed_nominal_diameter_and_regularization_across_ar(self):
        with tempfile.TemporaryDirectory() as temporary:
            volumes, boxes = [], []
            for ar in (1., 2., 5.):
                output = Path(temporary)/str(ar)
                args = prepare.parser().parse_args([
                    "--output", str(output), "--initialization", "aligned", "--ar", str(ar),
                    "--count", "1", "--phi", ".01", "--steps", "0"])
                with mock.patch.object(prepare, "certify_packing", return_value={"certified": True}):
                    record = prepare.prepare(args)
                persisted = json.loads((output/"parameters.json").read_text())
                self.assertEqual(json.loads(json.dumps(record)), persisted)
                self.assertAlmostEqual(record["a_m"]/record["c_m"], ar, places=13)
                self.assertAlmostEqual(record["particle_volume_m3"]/(math.pi*(8e-6)**3/6), 1.)
                self.assertAlmostEqual(record["reference_radius_m"]/4e-6, 1.)
                self.assertEqual(record["computational_length_m"], record["reference_radius_m"])
                self.assertAlmostEqual(record["lubrication_regularization_m"]/4e-9, 1.)
                self.assertEqual(record["full_axes_m"], [2*record["a_m"], 2*record["a_m"], 2*record["c_m"]])
                self.assertEqual(record["initial_surface_gap_target_m"], 0.)
                self.assertEqual(record["rolling_contact"]["rolling_stiffness_N_per_m"], 80.)
                self.assertAlmostEqual(record["rolling_contact"]["equivalent_angular_stiffness_Nm_per_rad"]/(80*(4e-6)**2), 1.)
                self.assertEqual(record["rolling_contact"]["reference_radius_m"], record["reference_radius_m"])
                self.assertIn("not measured", record["friction_parameter_basis"])
                self.assertIn("max(actual_gap,0)", record["hydro_gap_rule"])
                for retired in ("thickness_nm", "contact_gap_nm", "gap_floor_nm", "rough_hydrodynamics",
                                "rotation_contact", "rolling_length_mode", "twisting_friction"):
                    self.assertNotIn(retired, record)
                force = next(line for line in (output/"in.bridge").read_text().splitlines()
                             if line.startswith("fix hydro ")).split()
                self.assertEqual(force[3], "slurry/bridge")
                self.assertAlmostEqual(float(force[6])/8e-6, 1.)
                self.assertEqual(float(force[7]), .001)
                self.assertEqual(float(force[8]), 100.)
                self.assertEqual(force[9:15], ["contact", "10", "80", "rolling", "10", "80"])
                self.assertNotIn("rough_hydro", force)
                self.assertNotIn("rotation_contact", force)
                axes_row = (output/"particles.data").read_text().split("Ellipsoids\n\n", 1)[1].splitlines()[0]
                axes = list(map(float, axes_row.split()[1:4]))
                self.assertEqual(axes, record["full_axes_m"])
                volumes.append(record["particle_volume_m3"])
                boxes.append(record["box_length_m"])
            for value in volumes:
                self.assertAlmostEqual(value/volumes[0], 1.)
            for value in boxes:
                self.assertAlmostEqual(value/boxes[0], 1.)

    def test_retired_surface_flags_and_flake_are_rejected(self):
        retired = ("--thickness-nm", "--gap-floor-nm", "--contact-gap-nm", "--rough-hydro",
                   "--rough-height-nm", "--rough-width-nm", "--rough-coverage", "--twisting-friction",
                   "--twisting-stiffness", "--twisting-length-nm", "--rolling-length-nm",
                   "--rolling-length-mode")
        for flag in retired:
            with self.subTest(flag=flag), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                prepare.parser().parse_args(["--output", "unused", flag, "1"])
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            prepare.parser().parse_args(["--output", "unused", "--shape", "flake"])

    def test_invalid_contact_parameters_fail_before_creating_run_directory(self):
        cases = (["--ar", ".9"], ["--ar", "5.1"], ["--nominal-diameter-um", "0"],
                 ["--nominal-diameter-um", "inf"], ["--lubrication-epsilon", "0"],
                 ["--lubrication-epsilon", "nan"], ["--rolling-friction", "-.1"],
                 ["--friction", "inf"], ["--rolling-stiffness", "0"], ["--rolling-stiffness", "nan"])
        with tempfile.TemporaryDirectory() as temporary:
            for number, flags in enumerate(cases):
                output = Path(temporary)/str(number)
                args = prepare.parser().parse_args(["--output", str(output)]+flags)
                with self.subTest(flags=flags), self.assertRaises(ValueError):
                    prepare.prepare(args)
                self.assertFalse(output.exists())

    def test_runner_rejects_retired_options_before_snapshot(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)/"case"
            with mock.patch.object(sys, "argv", ["run_bridge.py", "--output", str(output),
                                                 "--contact-gap-nm", "3"]), \
                 mock.patch.object(runner, "freeze_runtime") as freeze, \
                 contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                runner.main()
            freeze.assert_not_called()
            self.assertFalse(output.exists())

    def test_cylinder_volume_mass_and_storage_density(self):
        a, c, density = 3e-6, 150e-9, 2400.
        cylinder = prepare.particle_volume(a, c, "flake")
        spheroid = prepare.particle_volume(a, c, "spheroid")
        self.assertAlmostEqual(cylinder/spheroid, 1.5)
        q = (1., 0., 0., 0.)
        particles = [((1e-5, 1e-5, 1e-5), q, prepare.shape_matrix(q, c/a))]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"particles.data"
            prepare.write_data(path, particles, a, c, 2e-5, density, 100., "flake")
            text = path.read_text()
            self.assertIn("storage proxies", text)
            row = text.split("Atoms # ellipsoid\n\n", 1)[1].splitlines()[0].split()
            storage_density = float(row[3])
            self.assertEqual(storage_density, 3600.)
            self.assertAlmostEqual(storage_density*spheroid/(density*cylinder), 1.)

    def test_spheroid_rejects_wrong_shape_clearance_certificate(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            args = prepare.parser().parse_args(["--output", str(output), "--packer", sys.executable])
            q = (1., 0., 0., 0.)
            particles = [((0., 0., 0.), q, prepare.shape_matrix(q, .5))]
            def certificate(command, **kwargs):
                self.assertEqual(command[command.index("--shape")+1], "spheroid")
                (output/"packing_clearance.json").write_text(json.dumps({
                    "certified": True, "particle_shape": "flake", "target_gap_m": 0.,
                    "minimum_surface_gap_m": 4e-9, "max_clearance_deficit_m": 0.,
                    "nominal_volume_fraction": args.phi, "checked_pair_images": 1}))
                return prepare.subprocess.CompletedProcess(command, 0, stdout="")
            with mock.patch.object(prepare.subprocess, "run", side_effect=certificate):
                with self.assertRaisesRegex(ValueError, "particle shape"):
                    prepare.certify_packing(args, particles, 1., .5, 10., output)

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

    @unittest.skipUnless(TEST_PACKER.is_file(), "compiled packer supplied by CTest")
    def test_generated_xz_input_uses_mari_normalized_gap(self):
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
            self.assertEqual(record["lubrication_epsilon"], .001)
            self.assertAlmostEqual(record["lubrication_regularization_m"]/4e-9, 1.)
            self.assertEqual(record["spectral"], 2.)
            self.assertEqual(record["normal_stiffness"], 100.)
            self.assertEqual(record["mobility_backend"], "mesh")
            self.assertEqual(record["mesh_numerics"]["window_support"], 10)
            self.assertIn("not a total", record["spectral_tolerance_scope"])
            self.assertTrue(record["packing_clearance"]["certified"])
            self.assertGreater(record["packing_clearance"]["checked_pair_images"], 0)
            self.assertGreaterEqual(record["packing_clearance"]["minimum_surface_gap_m"], 0.)
            self.assertEqual(record["packing_clearance"]["max_clearance_deficit_m"], 0.)
            force_line = next(line for line in text.splitlines() if line.startswith("fix hydro ")).split()
            self.assertEqual(float(force_line[8]), 100.)
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
                "--lubrication-epsilon", ".002", "--mobility-backend", "direct"])
            record = prepare.prepare(args)
            self.assertEqual(record["normal_stiffness"], 250.)
            self.assertEqual(record["mobility_backend"], "direct")
            self.assertGreaterEqual(record["packing_clearance"]["minimum_surface_gap_m"], 0.)
            self.assertEqual(record["packing_clearance"]["max_clearance_deficit_m"], 0.)
            self.assertAlmostEqual(record["packing_clearance"]["nominal_volume_fraction"], .1)
            force_line = next(line for line in (output/"in.bridge").read_text().splitlines()
                              if line.startswith("fix hydro ")).split()
            self.assertEqual(float(force_line[8]), 250.)
            self.assertEqual(float(force_line[7]), .002)

    def test_clearance_certificate_must_match_requested_gap(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            args = prepare.parser().parse_args(["--output", str(output), "--packer", sys.executable])
            q = (1., 0., 0., 0.)
            particles = [((0., 0., 0.), q, prepare.shape_matrix(q, .5))]
            def certificate(command, **kwargs):
                self.assertEqual(command[command.index("--certify-only")+1], "1")
                self.assertEqual(float(command[command.index("--contact-gap-m")+1]), 0.)
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
            valid = {"certified": True, "target_gap_m": 0., "minimum_surface_gap_m": 4e-9,
                     "max_clearance_deficit_m": 0., "nominal_volume_fraction": args.phi,
                     "checked_pair_images": 1}
            for field, value in (("minimum_surface_gap_m", float("nan")),
                                 ("minimum_surface_gap_m", float("inf")),
                                 ("minimum_surface_gap_m", -1e-12),
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

    def private_summary_runtime(self, output):
        runtime = output/"runtime"
        (runtime/"tools").mkdir(parents=True)
        (runtime/"tools/summarize_bridge.py").write_text((TOOLS/"summarize_bridge.py").read_text())
        return runtime

    def test_run_summary_reports_valid_short_or_empty_history_without_average(self):
        for empty in (False, True):
            with self.subTest(empty=empty), tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary)
                runtime = self.private_summary_runtime(output)
                self.history(output/"history.csv")
                if empty:
                    header = (output/"history.csv").read_text().splitlines()[0]
                    (output/"history.csv").write_text(header+"\n")
                runner.summarize_run(runtime, output,
                                     {"average_start_strain": 20., "particle_shape": "flake",
                                      "model": "cylinder_reynolds_rim_periodic"}, {})
                report = json.loads((output/"viscosity_summary.json").read_text())
                self.assertEqual(report["status"], "average_unavailable")
                self.assertEqual(report["particle_shape"], "flake")
                self.assertEqual(report["requested_start_strain"], 20.)
                self.assertEqual(report["recorded_end_strain"], 0. if empty else 8.)
                self.assertFalse(report["stationarity_certified"])
                self.assertNotIn("mean", report)
                self.assertIn("No complete", (output/"viscosity_summary.txt").read_text())

    def test_run_summary_writes_time_weighted_window_and_shape(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            runtime = self.private_summary_runtime(output)
            self.history(output/"history.csv")
            runner.summarize_run(runtime, output,
                                 {"average_start_strain": 0., "particle_shape": "flake",
                                  "model": "cylinder_reynolds_rim_periodic"}, {})
            report = json.loads((output/"viscosity_summary.json").read_text())
            self.assertAlmostEqual(report["mean"]["eta_relative"], 3.5)
            self.assertAlmostEqual(report["mean"]["sigma_xz_Pa"], 7.)
            self.assertEqual(report["used_window"], [0., 8.])
            self.assertEqual(report["used_intervals"], 2)
            self.assertEqual(report["particle_shape"], "flake")
            self.assertEqual(report["model"], "cylinder_reynolds_rim_periodic")
            self.assertFalse(report["stationarity_certified"])
            self.assertIn("relative viscosity = 3.5", (output/"viscosity_summary.txt").read_text())

    def test_run_summary_rejects_corrupt_history_before_testing_window(self):
        for corrupt in ("missing_header", "not-a-number", "nan", "inf"):
            with self.subTest(corrupt=corrupt), tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary)
                runtime = self.private_summary_runtime(output)
                history = output/"history.csv"
                self.history(history)
                if corrupt == "missing_header":
                    history.write_text("strain,eta_relative\n0.1,3.5\n")
                else:
                    with history.open(newline="") as stream:
                        rows = list(csv.DictReader(stream))
                    rows[0]["interval_mean_eta_relative"] = corrupt
                    with history.open("w", newline="") as stream:
                        writer = csv.DictWriter(stream, fieldnames=summary.REQUIRED)
                        writer.writeheader()
                        writer.writerows(rows)
                with self.assertRaises(ValueError):
                    runner.summarize_run(runtime, output,
                                         {"average_start_strain": 100., "particle_shape": "flake",
                                          "model": "cylinder_reynolds_rim_periodic"}, {})
                self.assertFalse((output/"viscosity_summary.json").exists())
                self.assertFalse((output/"viscosity_summary.txt").exists())

if __name__ == "__main__":
    unittest.main()
