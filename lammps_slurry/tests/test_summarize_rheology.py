#!/usr/bin/env python3
"""Regression for reporting signed general pair stress in interval averages."""
import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "summarize_rheology", ROOT / "tools/summarize_rheology.py")
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


class RheologySummary(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="rheology-summary-")
        self.addCleanup(self.temporary.cleanup)
        self.history = Path(self.temporary.name) / "history.csv"

    def write_history(self, include_pair=True):
        rows = []
        # Shear rate is 2/s. Unequal interval widths must weight all reported
        # components identically. A pair correction can have either sign.
        for start, end, contact, pair in ((0., 1., 11., -4.), (1., 4., 5., 12.)):
            components = {"solvent": 1., "single_free": 2.,
                          "contact_normal": contact}
            if include_pair:
                components["finite_gap"] = pair
            total = sum(components.values())
            row = {"interval_steps": 10, "interval_start_strain": start,
                   "interval_end_strain": end,
                   "interval_mean_sigma_xy_Pa": total,
                   "interval_mean_eta_Pa_s": total / 2.,
                   "interval_mean_eta_relative": total / 2. / .01}
            for name, stress in components.items():
                row["interval_mean_sigma_" + name + "_Pa"] = stress
                row["interval_mean_eta_" + name + "_Pa_s"] = stress / 2.
            # These endpoint values deliberately disagree with interval means.
            # The summary must not substitute the last instantaneous sample.
            row["sigma_finite_gap_Pa"] = 1000.
            row["eta_finite_gap_Pa_s"] = 500.
            rows.append(row)
        with self.history.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def components(report):
        section = report.split("component,mean_sigma_Pa,mean_eta_Pa_s\n", 1)[1]
        section = section.split("\n\n", 1)[0]
        return {row[0]: tuple(map(float, row[1:]))
                for row in csv.reader(section.splitlines())}

    def test_general_pair_is_included_with_signed_interval_weighting(self):
        self.write_history()
        report = summary.summarize(self.history, 0., None, 1)
        components = self.components(report)
        self.assertEqual(components["finite_gap"], (8., 4.))
        self.assertAlmostEqual(sum(value[0] for value in components.values()), 17.5)
        self.assertAlmostEqual(sum(value[1] for value in components.values()), 8.75)
        self.assertIn("sigma_xy_Pa,17.5,", report)
        self.assertIn("eta_Pa_s,8.75,", report)

    def test_selected_window_uses_only_its_general_pair_intervals(self):
        self.write_history()
        report = summary.summarize(self.history, 1., 4., 1)
        self.assertEqual(self.components(report)["finite_gap"], (12., 6.))

    def test_legacy_history_without_pair_columns_remains_readable(self):
        self.write_history(include_pair=False)
        report = summary.summarize(self.history, 0., None, 1)
        self.assertNotIn("finite_gap", self.components(report))
        self.assertIn("eta_Pa_s,4.75,", report)


if __name__ == "__main__":
    unittest.main()
