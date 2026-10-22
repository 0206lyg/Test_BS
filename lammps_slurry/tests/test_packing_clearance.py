#!/usr/bin/env python3
"""Actual-gap packing regressions, including periodic and self-image contacts."""
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
PACKER = Path(os.environ.get("SLURRY_TEST_PACKER", str(ROOT / "build/lammps-slurry/slurry_pack")))


def write_configuration(path, box, a, c, positions):
    with path.open("w") as out:
        out.write("{} {:.17g} {:.17g} {:.17g} 1\n".format(len(positions), box, a, c))
        for p in positions:
            out.write(" ".join("{:.17g}".format(x) for x in (*p, 1., 0., 0., 0.)) + "\n")


def orientation_tensor(path):
    particles = [list(map(float, row.split())) for row in path.read_text().splitlines()[1:]]
    tensor = [[0.] * 3 for _ in range(3)]
    for p in particles:
        w, x, y, z = p[3:]
        n = (2*(x*z+w*y), 2*(y*z-w*x), 1-2*(x*x+y*y))
        for i in range(3):
            for j in range(3):
                tensor[i][j] += n[i]*n[j]/len(particles)
    return tensor


@unittest.skipUnless(PACKER.is_file(), "Set SLURRY_TEST_PACKER to the compiled geometry packer")
class PackingClearanceTest(unittest.TestCase):
    def invoke_certificate(self, source, gap, report):
        result = subprocess.run([str(PACKER), "--input", str(source), "--contact-gap-m", str(gap),
                                 "--certify-only", "1", "--report", str(report)],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        self.assertIn(result.returncode, (0, 2), result.stdout+result.stderr)
        return result, json.loads(report.read_text())

    def test_periodic_contact_layer_is_not_nominal_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            p, report = Path(tmp)/"seed", Path(tmp)/"report.json"
            a, box, gap = 150e-9, 2e-6, 2e-9
            write_configuration(p, box, a, a, [(0.1*a, box/2, box/2),
                                               (box-1.9*a-gap, box/2, box/2)])
            result, data = self.invoke_certificate(p, 3e-9, report)
            self.assertEqual(result.returncode, 2)
            self.assertFalse(data["certified"])
            self.assertAlmostEqual(data["minimum_surface_gap_m"]/1e-9, 2., places=10)
            self.assertAlmostEqual(data["max_clearance_deficit_m"]/1e-9, 1., places=10)

    def test_self_image_certificate(self):
        with tempfile.TemporaryDirectory() as tmp:
            p, report = Path(tmp)/"seed", Path(tmp)/"report.json"
            a, box = 150e-9, 302e-9
            write_configuration(p, box, a, a, [(box/2,)*3])
            result, data = self.invoke_certificate(p, 3e-9, report)
            self.assertEqual(result.returncode, 2)
            self.assertAlmostEqual(data["minimum_surface_gap_m"]/1e-9, 2., places=10)
            result, data = self.invoke_certificate(p, 1e-9, report)
            self.assertEqual(result.returncode, 0)
            self.assertTrue(data["certified"])

    def test_clearance_relaxation_preserves_geometry_and_prepared_orientation(self):
        # These realistic N=500 fixtures already meet the old PW criterion but
        # contain compressed 3 nm contact layers. The new stage must fix them
        # without changing the particle axes, box, density, or random texture.
        for ar in (2, 20):
            source = ROOT/"validation_solver500/fixtures"/("ar"+str(ar))/"packing.final"
            with self.subTest(ar=ar), tempfile.TemporaryDirectory() as tmp:
                final, report, history = (Path(tmp)/name for name in ("final", "report.json", "history.csv"))
                result = subprocess.run([str(PACKER), "--input", str(source), "--output", str(final),
                                         "--history", str(history), "--report", str(report)],
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                        universal_newlines=True, timeout=60)
                self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
                data = json.loads(report.read_text())
                self.assertTrue(data["certified"])
                self.assertGreaterEqual(data["minimum_surface_gap_m"], 3e-9)
                self.assertEqual(data["max_clearance_deficit_m"], 0.)
                self.assertAlmostEqual(data["nominal_volume_fraction"], .24, places=12)
                self.assertEqual(list(map(float, final.read_text().splitlines()[0].split())),
                                 list(map(float, source.read_text().splitlines()[0].split())))
                before, after = orientation_tensor(source), orientation_tensor(final)
                change = math.sqrt(sum((after[i][j]-before[i][j])**2 for i in range(3) for j in range(3)))
                self.assertLess(change, .02)
                result, data = self.invoke_certificate(final, 3e-9, report)
                self.assertEqual(result.returncode, 0, result.stdout+result.stderr)


if __name__ == "__main__":
    unittest.main()
