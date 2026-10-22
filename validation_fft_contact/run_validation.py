#!/usr/bin/env python3
"""Run the actual verified bridge launcher and check the FFT/contact release.

This is a short execution/consistency check, not a steady-state rheology run.
Results, complete logs, initial-gap certificates and trajectories stay in OUTPUT.
"""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def check_case(path, expected_count, ar):
    parameters = json.loads((path / "parameters.json").read_text())
    assert parameters["normal_stiffness"] == 100., parameters
    assert parameters["contact_gap_nm"] == 3., parameters
    assert parameters["mobility_backend"] == "mesh", parameters
    assert parameters["count"] == expected_count, parameters
    certificate = parameters["packing_clearance"]
    assert certificate["certified"] is True, certificate
    assert math.isfinite(certificate["minimum_surface_gap_m"]), certificate
    assert certificate["minimum_surface_gap_m"] >= 3e-9, certificate
    with (path / "history.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) >= 2, "No integrated timestep was recorded"
    required = ("actual_endpoint_min_gap_m", "actual_endpoint_overlap_m",
                "contact_gap_consistency_m", "geometry_iterations",
                "normal_elastic_energy_J", "tangential_elastic_energy_J")
    for key in required:
        assert key in rows[0], "Missing diagnostic: " + key
    for row in rows:
        for key in required:
            assert math.isfinite(float(row[key])), (key, row[key])
        iterations = float(row["geometry_iterations"])
        assert iterations.is_integer() and 1 <= iterations <= 24, row
        assert float(row["actual_endpoint_min_gap_m"]) >= -1e-15, row
        assert float(row["actual_endpoint_overlap_m"]) <= 1e-15, row
        assert float(row["contact_gap_consistency_m"]) <= 1e-14, row
        assert float(row["normal_elastic_energy_J"]) >= 0., row
        assert float(row["tangential_elastic_energy_J"]) >= 0., row
        assert float(row["min_gap_m"]) >= -1e-15, row
        assert float(row["solver_relative_residual"]) <= 1.01e-6, row
        assert float(row["hydro_dissipation_W"]) >= 0., row
        power_scale = max(abs(float(row["stress_power_W"])),
                          abs(float(row["hydro_dissipation_W"])), 1e-30)
        assert abs(float(row["power_balance_error_W"])) / power_scale < 1e-4, row
    commit_errors = []
    for previous, current in zip(rows, rows[1:]):
        assert int(current["step"]) == int(previous["step"]) + 1
        # Below onset the independent endpoint search rules out a hidden
        # contact, so this minimum must be the next committed minimum too.
        if float(previous["actual_endpoint_min_gap_m"]) < 3e-9:
            error = abs(float(current["min_gap_m"]) -
                        float(previous["actual_endpoint_min_gap_m"]))
            assert error <= 1e-14, (previous, current, error)
            commit_errors.append(error)
    result = {"AR": ar, "N": expected_count, "rows": len(rows),
              "packing": certificate,
              "minimum_endpoint_gap_m": min(float(r["actual_endpoint_min_gap_m"]) for r in rows),
              "maximum_contact_gap_consistency_m": max(float(r["contact_gap_consistency_m"]) for r in rows),
              "maximum_geometry_iterations": max(int(float(r["geometry_iterations"])) for r in rows),
              "endpoint_commit_comparisons": len(commit_errors),
              "maximum_endpoint_commit_error_m": max(commit_errors, default=0.),
              "final_record": rows[-1]}
    for name in rows[0]:
        if name.startswith("profile_") and name.endswith("_s"):
            values = [float(r[name]) for r in rows[1:]]
            if all(math.isfinite(x) for x in values):
                result["median_" + name] = statistics.median(values)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--build-env", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True, help="new validation directory")
    p.add_argument("--count", type=int, default=500)
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--ars", type=float, nargs="+", default=[2., 20., 30.])
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    environment = dict(os.environ, OMP_NUM_THREADS=str(args.threads),
                       OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    report = {"purpose": "short execution and contact consistency; not steady-state viscosity",
              "threads": args.threads, "steps": args.steps,
              "initialization": "random for AR<30; aligned for AR>=30 to isolate runtime validation from high-AR random packing convergence",
              "cases": []}
    for ar in args.ars:
        case = output / ("ar" + format(ar, "g"))
        command = [sys.executable, str(ROOT / "lammps_slurry/tools/run_bridge.py"),
                   "--build-env", str(args.build_env.resolve()), "--output", str(case),
                   "--ar", str(ar), "--count", str(args.count), "--phi", ".24",
                   "--initialization", "aligned" if ar >= 30 else "random",
                   "--steps", str(args.steps), "--output-interval", "1"]
        started = time.monotonic()
        with (output / (case.name + ".log")).open("w") as log:
            subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
        result = check_case(case, args.count, ar)
        result["wall_seconds_including_packing_and_runtime_copy"] = time.monotonic() - started
        report["cases"].append(result)
        (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
        print("PASS AR={} N={} minimum endpoint gap={:.8g} nm".format(
            ar, args.count, result["minimum_endpoint_gap_m"] * 1e9), flush=True)


if __name__ == "__main__":
    main()
