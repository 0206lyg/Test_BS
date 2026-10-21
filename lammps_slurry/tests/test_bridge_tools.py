#!/usr/bin/env python3
"""Packing/image and averaging regressions for the new runtime tools."""
import csv
import importlib.util
import math
from pathlib import Path
import tempfile
import sys
import unittest

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


class BridgeToolsTest(unittest.TestCase):
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
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)/"case"
            args = prepare.parser().parse_args(["--output", str(output), "--initialization", "aligned", "--steps", "0", "--shear-rate", "-100"])
            record = prepare.prepare(args)
            text = (output/"in.bridge").read_text()
            self.assertIn("fix shear all deform 0 xz erate -100 remap v", text)
            self.assertIn("plugin load ${slurry_plugin}\n", text)
            self.assertNotIn('plugin load "${slurry_plugin}"', text)
            self.assertIn("slurry/bridge", text)
            self.assertNotIn("finite_gap", text)
            self.assertEqual(record["gap_floor_nm"], 1.)
            self.assertEqual(record["spectral"], 2.)
            with self.assertRaisesRegex(ValueError, "already exists"):
                prepare.prepare(args)

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
