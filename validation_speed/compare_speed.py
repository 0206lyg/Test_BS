#!/usr/bin/env python3
"""Sequential, reproducible baseline/candidate runs of certified N=500 fixtures.

This benchmark measures short-step numerical equivalence and cost, not steady
state viscosity or physical accuracy of the underlying lubrication model.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
PHYSICAL = ("sigma_hydro_Pa", "sigma_contact_Pa", "sigma_xz_Pa", "eta_relative",
            "hydro_dissipation_W", "contact_power_W", "power_balance_error_W",
            "force_residual", "mobility_residual", "solver_relative_residual",
            "actual_endpoint_min_gap_m", "actual_endpoint_overlap_m")
PROFILE = ("profile_geometry_s", "profile_middle_s", "profile_solve_s", "profile_total_s")
COUNTERS = ("linear_iterations", "newton_iterations", "mobility_applications",
            "geometry_iterations", "descent_steps", "continuation_retries")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite(value):
    value = float(value)
    return value if math.isfinite(value) else None


def read_history(path):
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("Empty history: {}".format(path))
    required = ("step",) + PHYSICAL + PROFILE + COUNTERS
    missing = set(required) - set(rows[0])
    if missing:
        raise ValueError("Missing history fields: {}".format(sorted(missing)))
    output = []
    for row in rows:
        record = {key: finite(value) for key, value in row.items()}
        if any(record[key] is None for key in required):
            raise ValueError("Nonfinite required diagnostic in {} at step {}".format(path, row["step"]))
        record["step"] = int(record["step"])
        output.append(record)
    if len({row["step"] for row in output}) != len(output):
        raise ValueError("Duplicate history steps")
    return output


def read_last_frame(path):
    """Read a custom LAMMPS dump, retaining only its final complete frame."""
    final = None
    with path.open() as stream:
        while True:
            line = stream.readline()
            if not line:
                break
            if line.strip() != "ITEM: TIMESTEP":
                raise ValueError("Malformed trajectory timestep header")
            step = int(stream.readline())
            if stream.readline().strip() != "ITEM: NUMBER OF ATOMS":
                raise ValueError("Malformed trajectory particle count header")
            count = int(stream.readline())
            bounds_header = stream.readline().split()
            if bounds_header[:3] != ["ITEM:", "BOX", "BOUNDS"]:
                raise ValueError("Malformed trajectory cell header")
            bounds = [tuple(map(float, stream.readline().split())) for _ in range(3)]
            triclinic = bounds_header[3:6] == ["xy", "xz", "yz"]
            xy, xz, yz = (bounds[0][2], bounds[1][2], bounds[2][2]) if triclinic else (0., 0., 0.)
            # Restricted-triclinic dump bounds include the tilt bounding box.
            xlo = bounds[0][0] - min(0., xy, xz, xy + xz)
            xhi = bounds[0][1] - max(0., xy, xz, xy + xz)
            ylo = bounds[1][0] - min(0., yz)
            yhi = bounds[1][1] - max(0., yz)
            cell = (xhi - xlo, yhi - ylo, bounds[2][1] - bounds[2][0], xy, xz, yz)
            header = stream.readline().split()
            if header[:2] != ["ITEM:", "ATOMS"]:
                raise ValueError("Malformed trajectory field header")
            columns = header[2:]
            particles = {}
            for _ in range(count):
                values = list(map(float, stream.readline().split()))
                if len(values) != len(columns) or not all(map(math.isfinite, values)):
                    raise ValueError("Incomplete or nonfinite trajectory frame")
                record = dict(zip(columns, values))
                particle = int(record["id"])
                if particle in particles:
                    raise ValueError("Duplicate particle ID in trajectory")
                particles[particle] = record
            final = {"step": step, "cell": cell, "particles": particles}
    if final is None:
        raise ValueError("Empty trajectory")
    return final


def vector_metrics(baseline, candidate, difference=None):
    delta = difference if difference is not None else [b - a for a, b in zip(baseline, candidate)]
    base_norm = math.sqrt(math.fsum(value * value for value in baseline))
    error_norm = math.sqrt(math.fsum(value * value for value in delta))
    return {"baseline_l2": base_norm,
            "candidate_l2": math.sqrt(math.fsum(value * value for value in candidate)),
            "absolute_l2_difference": error_norm,
            "relative_l2_difference": error_norm / base_norm if base_norm else (0. if error_norm == 0 else None),
            "rms_component_difference": error_norm / math.sqrt(len(delta)),
            "maximum_absolute_component_difference": max(map(abs, delta))}


def periodic_displacement(delta, cell):
    lx, ly, lz, xy, xz, yz = cell
    sz = delta[2] / lz
    sy = (delta[1] - yz * sz) / ly
    sx = (delta[0] - xy * sy - xz * sz) / lx
    near = (round(sx), round(sy), round(sz))
    # Check adjacent images too; component-wise fractional rounding alone is
    # insufficient to minimize Euclidean distance in a tilted periodic cell.
    candidates = []
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            for k in (-1, 0, 1):
                nx, ny, nz = near[0] + i, near[1] + j, near[2] + k
                candidates.append((delta[0] - lx * nx - xy * ny - xz * nz,
                                   delta[1] - ly * ny - yz * nz, delta[2] - lz * nz))
    return min(candidates, key=lambda value: sum(component * component for component in value))


def compare_trajectories(baseline_path, candidate_path):
    """ID-matched final-frame comparison; callable without any simulation."""
    try:
        baseline, candidate = read_last_frame(Path(baseline_path)), read_last_frame(Path(candidate_path))
        if baseline["step"] != candidate["step"]:
            raise ValueError("Final trajectory steps differ: {} vs {}".format(baseline["step"], candidate["step"]))
        if set(baseline["particles"]) != set(candidate["particles"]):
            raise ValueError("Final particle ID sets differ")
        ids = sorted(baseline["particles"])
        if not ids:
            raise ValueError("No trajectory particles")
        bp, cp = baseline["particles"], candidate["particles"]
        groups = {"position_m": ("x", "y", "z"), "velocity_m_s": ("vx", "vy", "vz"),
                  "angular_momentum_kg_m2_s": tuple("c_shape[{}]".format(i) for i in range(8, 11)),
                  "hydrodynamic_force_N": tuple("f_hydro[{}]".format(i) for i in range(1, 4)),
                  "hydrodynamic_torque_N_m": tuple("f_hydro[{}]".format(i) for i in range(4, 7)),
                  "hydrodynamic_stresslet_N_m": tuple("f_hydro[{}]".format(i) for i in range(7, 12)),
                  "contact_force_N": tuple("f_hydro[{}]".format(i) for i in range(12, 15)),
                  "contact_torque_N_m": tuple("f_hydro[{}]".format(i) for i in range(15, 18))}
        result = {"available": True, "step": baseline["step"], "particle_count": len(ids), "groups": {}}
        for name, columns in groups.items():
            a = [bp[particle][column] for particle in ids for column in columns]
            b = [cp[particle][column] for particle in ids for column in columns]
            result["groups"][name] = vector_metrics(a, b)
        cell_error = max(abs(b - a) for a, b in zip(baseline["cell"], candidate["cell"]))
        result["maximum_cell_component_difference_m"] = cell_error
        if cell_error <= 1e-12 * max(baseline["cell"][:3]):
            columns = groups["position_m"]
            a = [bp[particle][column] for particle in ids for column in columns]
            b = [cp[particle][column] for particle in ids for column in columns]
            delta = [value for particle in ids for value in periodic_displacement(
                [cp[particle][column] - bp[particle][column] for column in columns], baseline["cell"])]
            result["groups"]["position_periodic_m"] = vector_metrics(a, b, delta)
        quaternion_a, quaternion_b, angles = [], [], []
        for particle in ids:
            a = [bp[particle]["c_shape[{}]".format(i)] for i in range(4, 8)]
            b = [cp[particle]["c_shape[{}]".format(i)] for i in range(4, 8)]
            anorm, bnorm = math.sqrt(sum(v * v for v in a)), math.sqrt(sum(v * v for v in b))
            if not anorm or not bnorm:
                raise ValueError("Zero orientation quaternion")
            a, b = [v / anorm for v in a], [v / bnorm for v in b]
            if sum(x * y for x, y in zip(a, b)) < 0:
                b = [-v for v in b]
            quaternion_a.extend(a)
            quaternion_b.extend(b)
            chord = math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
            angles.append(4 * math.asin(min(1., chord / 2)))
        result["groups"]["orientation_quaternion_sign_aligned"] = vector_metrics(quaternion_a, quaternion_b)
        result["orientation_rms_angle_rad"] = math.sqrt(sum(value * value for value in angles) / len(angles))
        result["orientation_maximum_angle_rad"] = max(angles)
        return result
    except (OSError, ValueError, KeyError, IndexError, ZeroDivisionError) as error:
        return {"available": False, "error": str(error)}


def verify_fixture(directory):
    manifest = json.loads((directory / "fixture.json").read_text())
    for name, expected in manifest["sha256"].items():
        if sha256(directory / name) != expected:
            raise ValueError("Fixture checksum mismatch: {}".format(directory / name))
    certificate = json.loads((directory / "packing_clearance.json").read_text())
    if certificate.get("certified") is not True or certificate["minimum_surface_gap_m"] < 3e-9:
        raise ValueError("Invalid contact-clearance certificate")
    return manifest


def environment(threads):
    result = dict(os.environ)
    if any(key in result for key in ("OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "PMIX_UNIV_SIZE")):
        raise ValueError("Launch this driver once outside MPI ranks")
    result.update(OMP_NUM_THREADS=str(threads), OMP_DYNAMIC="FALSE", OMP_PLACES="cores",
                  OMP_PROC_BIND="close", OMP_MAX_ACTIVE_LEVELS="1", OMP_DISPLAY_ENV="VERBOSE",
                  OMP_DISPLAY_AFFINITY="TRUE", MKL_NUM_THREADS=str(threads), MKL_DYNAMIC="FALSE",
                  OPENBLAS_NUM_THREADS="1", SLURRY_BRIDGE_TRACE="0")
    return result


def one_run(args, fixture, label, plugin, threads, directory):
    directory.mkdir()
    for name in ("particles.data", "in.bridge", "fixture.json", "packing_clearance.json"):
        shutil.copy2(fixture / name, directory / name)
    env = environment(threads)
    # Exactly one process launch, optionally through a single-rank OpenMPI job.
    # There is no automatic launch retry or fallback.
    plugin_text = str(plugin)
    if any(character in plugin_text for character in ('"', "\n", "\r")):
        raise ValueError("Plugin path cannot contain quotes or newlines")
    command = [str(args.lmp), "-var", "slurry_plugin", '"' + plugin_text + '"',
               "-var", "validation_steps", str(args.steps), "-in", "in.bridge"]
    if args.mpi_launcher:
        command = [str(args.mpi_launcher), "-np", "1", "--map-by", "slot:PE={}".format(threads),
                   "--bind-to", "core", "--report-bindings"] + command
    record = {"label": label, "directory": str(directory), "command": command,
              "threads": threads, "steps_requested": args.steps,
              "lmp_sha256": sha256(args.lmp), "plugin_sha256": sha256(plugin),
              "thread_environment": {key: value for key, value in env.items()
                                     if key.startswith(("OMP_", "MKL_", "OPENBLAS_"))},
              "driver_allowed_cpus": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None}
    (directory / "launch.json").write_text(json.dumps(record, indent=2) + "\n")
    print("Running {} AR={} threads={} steps={}".format(label, fixture.name[2:], threads, args.steps), flush=True)
    start = time.perf_counter()
    try:
        with (directory / "console.log").open("w") as stream:
            process = subprocess.run(command, cwd=directory, env=env, stdout=stream,
                                     stderr=subprocess.STDOUT, timeout=args.timeout or None)
        record["returncode"] = process.returncode
    except (OSError, subprocess.TimeoutExpired) as error:
        record.update(returncode=None, execution_error=str(error))
    record["wall_seconds_including_setup"] = time.perf_counter() - start
    try:
        rows = read_history(directory / "history.csv")
        record["history_rows"] = len(rows)
        record["final"] = rows[-1]
        advancing = [row for row in rows if row["step"] > 0]
        record["advancing_step_profile_sum_s"] = {key: sum(row[key] for row in advancing) for key in PROFILE}
        record["advancing_step_counter_sum"] = {key: sum(row[key] for row in advancing) for key in COUNTERS}
        record["max_solver_relative_residual"] = max(row["solver_relative_residual"] for row in rows)
        record["completed"] = record["returncode"] == 0 and rows[-1]["step"] == args.steps
    except (OSError, ValueError, TypeError) as error:
        rows = []
        record.update(completed=False, history_error=str(error))
    (directory / "result.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    return record, rows


def compare(baseline, candidate, trajectories=False):
    base, br = baseline
    cand, cr = candidate
    result = {"both_completed": base["completed"] and cand["completed"]}
    if not result["both_completed"]:
        return result
    result["wall_speedup_baseline_over_candidate"] = base["wall_seconds_including_setup"] / cand["wall_seconds_including_setup"]
    bp, cp = base["advancing_step_profile_sum_s"], cand["advancing_step_profile_sum_s"]
    result["profile_speedup_baseline_over_candidate"] = {
        key: bp[key] / cp[key] if cp[key] else None for key in PROFILE}
    result["final_difference"] = {}
    for key in PHYSICAL:
        a, b = base["final"][key], cand["final"][key]
        result["final_difference"][key] = {"baseline": a, "candidate": b, "candidate_minus_baseline": b - a,
            "absolute_difference": abs(b - a), "relative_to_baseline": abs(b - a) / abs(a) if a else None}
    bmap, cmap = {r["step"]: r for r in br}, {r["step"]: r for r in cr}
    common = sorted(set(bmap) & set(cmap))
    result["common_steps"] = common
    result["maximum_history_absolute_difference"] = {
        key: max(abs(cmap[step][key] - bmap[step][key]) for step in common) for key in PHYSICAL}
    if trajectories:
        result["final_trajectory"] = compare_trajectories(
            Path(base["directory"]) / "trajectory.lammpstrj", Path(cand["directory"]) / "trajectory.lammpstrj")
    return result


def reanalyze(output):
    """Read completed runs and write reanalysis.json without executing anything.

    summary.json is left untouched, including when an existing comparison
    driver is still appending its next run. Incomplete runs are reported.
    """
    output = Path(output).expanduser().resolve()
    summary = json.loads((output / "summary.json").read_text())
    pairs = {}
    for stored in summary["runs"]:
        directory = output / Path(stored["directory"]).name
        record = json.loads((directory / "result.json").read_text())
        record["directory"] = str(directory)
        fixture = json.loads((directory / "fixture.json").read_text())
        rows = read_history(directory / "history.csv") if record["completed"] else []
        pairs.setdefault((fixture["ar"], record["threads"]), {})[record["label"]] = record, rows
    result = {"schema": 1, "purpose": summary["purpose"], "comparisons": []}
    for (ar, threads), pair in sorted(pairs.items()):
        if "baseline" in pair and "candidate" in pair:
            comparison = compare(pair["baseline"], pair["candidate"], trajectories=True)
        else:
            comparison = {"both_completed": False, "available_labels": sorted(pair)}
        comparison.update(ar=ar, threads=threads)
        result["comparisons"].append(comparison)
    (output / "reanalysis.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lmp", type=Path)
    parser.add_argument("--baseline-plugin", type=Path)
    parser.add_argument("--candidate-plugin", type=Path)
    parser.add_argument("--threads", type=int, nargs="+", default=[1])
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--ars", type=int, nargs="+", choices=(2, 5, 10, 20), default=[2, 5, 10, 20])
    parser.add_argument("--output", type=Path, help="new directory; existing paths are refused")
    parser.add_argument("--compare-trajectories", action="store_true", help="also compare final per-particle trajectory fields")
    parser.add_argument("--reanalyze", type=Path, help="only analyze an existing comparison directory; never launch simulations")
    parser.add_argument("--mpi-launcher", type=Path, help="OpenMPI mpirun path; omit for a serial build or MPI singleton")
    parser.add_argument("--timeout", type=float, default=0., help="per-run seconds; zero means unlimited")
    args = parser.parse_args()
    if args.reanalyze:
        reanalyze(args.reanalyze)
        print("Results: {}".format(args.reanalyze.resolve() / "reanalysis.json"))
        return 0
    if any(getattr(args, key) is None for key in ("lmp", "baseline_plugin", "candidate_plugin", "output")):
        parser.error("A new run requires --lmp, --baseline-plugin, --candidate-plugin and --output")
    if args.steps < 1 or min(args.threads) < 1 or args.timeout < 0:
        parser.error("steps and threads must be positive; timeout must be nonnegative")
    if len(set(args.threads)) != len(args.threads) or len(set(args.ars)) != len(args.ars):
        parser.error("Duplicate thread counts or AR values")
    for key in ("lmp", "baseline_plugin", "candidate_plugin", "mpi_launcher"):
        path = getattr(args, key)
        if path is not None:
            path = path.expanduser().resolve(strict=True)
            if not path.is_file():
                parser.error("Not a file: {}".format(path))
            setattr(args, key, path)
    environment(args.threads[0])
    fixtures = {ar: verify_fixture(HERE / "fixtures" / ("ar" + str(ar))) for ar in args.ars}
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    summary = {"schema": 1, "purpose": "short-step numerical and cost comparison; not steady-state rheology",
               "order": "AR, thread count, baseline then candidate; strictly sequential; no retries",
               "fixtures": fixtures, "runs": [], "comparisons": []}
    for ar in args.ars:
        for threads in args.threads:
            fixture = HERE / "fixtures" / ("ar" + str(ar))
            pair = []
            for label, plugin in (("baseline", args.baseline_plugin), ("candidate", args.candidate_plugin)):
                directory = output / "ar{}_t{}_{}".format(ar, threads, label)
                pair.append(one_run(args, fixture, label, plugin, threads, directory))
                summary["runs"].append(pair[-1][0])
                (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
            result = compare(*pair, trajectories=args.compare_trajectories)
            result.update(ar=ar, threads=threads)
            summary["comparisons"].append(result)
            (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    with (output / "comparison.csv").open("w", newline="") as stream:
        fields = ["ar", "threads", "both_completed", "wall_speedup_baseline_over_candidate"]
        fields += [label + "_" + key for key in ("sigma_xz_Pa", "eta_relative", "solver_relative_residual")
                   for label in ("baseline", "candidate", "absolute_difference")]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for result in summary["comparisons"]:
            row = {key: result.get(key) for key in fields[:4]}
            for key in ("sigma_xz_Pa", "eta_relative", "solver_relative_residual"):
                for label in ("baseline", "candidate", "absolute_difference"):
                    row[label + "_" + key] = result.get("final_difference", {}).get(key, {}).get(label)
            writer.writerow(row)
    print("Results: {}".format(output / "summary.json"))
    return 0 if all(run["completed"] for run in summary["runs"]) else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError) as error:
        sys.exit("compare_speed.py: {}".format(error))
