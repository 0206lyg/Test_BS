#!/usr/bin/env python3
"""Report acceptance checks on explicit synthetic history records, not physics."""
import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("general_smoke", str(Path(__file__).with_name("run_lammps_smoke.py")))
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


class SmokeAnalysis(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="general-smoke-analysis-")
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name)
        (self.path / "parameters.txt").write_text("lubrication_model_resolved = general\nsteps = 1\nshear_rate = 100\n")
        (self.path / "final.data").write_text("synthetic final output\n")
        (self.path / "log.lammps").write_text("Loop time of 0.5 on 1 procs for 1 steps with 2 atoms\n")
        self.row = dict.fromkeys(smoke.CORE_FIELDS, 0.)
        # Negative net stress and eta are consistent with elastic release.
        self.row.update(step=1., stress_power_W=-2., hydro_dissipation_W=1.,
                        elastic_energy_rate_W=-4., sliding_power_W=.5,
                        contact_discrete_dissipation_W=.5, power_balance_error_W=0.,
                        force_residual_N=1e-9, force_tolerance_N=1e-8,
                        sigma_xy_Pa=-2., eta_Pa_s=-.02, finite_gap_strain_bound_min=1.,
                        profile_total_s=.5, interval_steps=1., average_duration_s=1.)

    def save(self, rows):
        with (self.path / "history.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def test_negative_transient_eta_is_not_a_failure(self):
        self.save([self.row])
        result = smoke.analyze_case(self.path)
        self.assertTrue(result["passed"], result["checks"])
        self.assertEqual(result["power_balance"]["maximum_normalized_recomputed_error"], 0.)
        self.assertEqual(result["instantaneous_eta_Pa_s"]["minimum"], -.02)

    def test_force_and_recomputed_power_fail_even_if_reported_error_is_zero(self):
        self.row["force_residual_N"] = 1e-7
        self.row["hydro_dissipation_W"] = 2.
        self.save([self.row])
        result = smoke.analyze_case(self.path)
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["force_within_solver_tolerance"])
        self.assertFalse(result["checks"]["normalized_power_balance"])

    def test_startup_means_may_be_undefined_but_core_nan_may_not(self):
        startup = dict(self.row, step=0., interval_steps=0., average_duration_s=0.,
                       interval_mean_eta_Pa_s=float("nan"), mean_eta_Pa_s=float("nan"))
        final = dict(self.row, interval_mean_eta_Pa_s=-.02, mean_eta_Pa_s=-.02)
        self.save([startup, final])
        self.assertTrue(smoke.analyze_case(self.path)["passed"])
        final["force_residual_N"] = float("nan")
        self.save([startup, final])
        self.assertFalse(smoke.analyze_case(self.path)["checks"]["history_finite"])

    def test_final_file_and_requested_step_are_required(self):
        self.row["step"] = 0.
        self.save([self.row])
        (self.path / "final.data").unlink()
        result = smoke.analyze_case(self.path)
        self.assertFalse(result["checks"]["requested_steps_completed"])
        self.assertFalse(result["checks"]["final_data_written"])


if __name__ == "__main__":
    unittest.main()
