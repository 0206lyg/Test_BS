#!/usr/bin/env python3
"""Packing/image and averaging regressions for the new runtime tools."""
import csv
import importlib.util
import json
import math
import os
from pathlib import Path
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
