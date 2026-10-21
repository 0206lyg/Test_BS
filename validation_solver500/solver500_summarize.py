#!/usr/bin/env python3
"""Summarize completed diagnostic solver500 runs; makes no steady-state claim."""
import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import re


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite(value, label):
    value = float(value)
    require(math.isfinite(value), "Nonfinite "+label)
    return value


def norm(values):
    return math.sqrt(sum(v*v for v in values))


def box_from_bounds(lines, header):
    bounds = [[finite(v, "trajectory box") for v in line.split()] for line in lines]
    if "xy xz yz" in header:
        require(all(len(v) == 3 for v in bounds), "Invalid triclinic bounds")
        xy, xz, yz = bounds[0][2], bounds[1][2], bounds[2][2]
        xlo = bounds[0][0]-min(0., xy, xz, xy+xz)
        xhi = bounds[0][1]-max(0., xy, xz, xy+xz)
        ylo, yhi = bounds[1][0]-min(0., yz), bounds[1][1]-max(0., yz)
        zlo, zhi = bounds[2][:2]
    else:
        require(all(len(v) == 2 for v in bounds), "Invalid orthogonal bounds")
        (xlo, xhi), (ylo, yhi), (zlo, zhi) = bounds
        xy = xz = yz = 0.
    lx, ly, lz = xhi-xlo, yhi-ylo, zhi-zlo
    require(min(lx, ly, lz) > 0., "Invalid trajectory cell")
    # Minimum-image enumeration below is intended for these short, reduced-cell runs.
    require(abs(xy) <= .5*lx and abs(xz) <= .5*lx and abs(yz) <= .5*ly,
            "Unreduced cell requires a different minimum-image enumeration")
    return lx, ly, lz, xy, xz, yz


def read_trajectory(path):
    frames = []
    with path.open() as stream:
        while True:
            line = stream.readline()
            if not line:
                break
            require(line.strip() == "ITEM: TIMESTEP", "Invalid trajectory frame")
            step = int(stream.readline())
            require(stream.readline().strip() == "ITEM: NUMBER OF ATOMS", "Missing atom count")
            count = int(stream.readline())
            header = stream.readline().strip()
            require(header.startswith("ITEM: BOX BOUNDS"), "Missing box bounds")
            box = box_from_bounds([stream.readline() for _ in range(3)], header)
            header = stream.readline().strip().split()
            require(header[:2] == ["ITEM:", "ATOMS"], "Missing trajectory atom fields")
            names = header[2:]
            required = ["id", "x", "y", "z"]+["c_shape[{}]".format(i) for i in range(4, 8)]
            require(all(name in names for name in required), "Trajectory lacks position/quaternion fields")
            atoms = {}
            for _ in range(count):
                tokens = stream.readline().split()
                require(len(tokens) == len(names), "Truncated trajectory atom record")
                values = [finite(v, "trajectory atom value") for v in tokens]
                row = dict(zip(names, values))
                ident = int(row["id"])
                require(ident not in atoms and ident == row["id"], "Invalid/duplicate trajectory ID")
                atoms[ident] = {"x": [row[k] for k in ("x", "y", "z")],
                                "q": [row["c_shape[{}]".format(i)] for i in range(4, 8)]}
            frames.append({"step": step, "count": count, "box": box, "atoms": atoms})
    require(bool(frames), "Empty trajectory")
    require(all(b["step"] > a["step"] for a, b in zip(frames, frames[1:])),
            "Trajectory timesteps are not strictly increasing")
    return frames


def minimum_image(delta, cell):
    lx, ly, lz, xy, xz, yz = cell
    sz = delta[2]/lz
    sy = (delta[1]-yz*sz)/ly
    sx = (delta[0]-xy*sy-xz*sz)/lx
    center = [round(s) for s in (sx, sy, sz)]
    best = None
    for offset in itertools.product((-1, 0, 1), repeat=3):
        nx, ny, nz = [center[i]+offset[i] for i in range(3)]
        candidate = [delta[0]-lx*nx-xy*ny-xz*nz, delta[1]-ly*ny-yz*nz, delta[2]-lz*nz]
        if best is None or norm(candidate) < norm(best):
            best = candidate
    return best


def quat_angle(q0, q1):
    n0, n1 = norm(q0), norm(q1)
    require(min(n0, n1) > 0., "Zero quaternion")
    q0, q1 = [v/n0 for v in q0], [v/n1 for v in q1]
    if sum(a*b for a, b in zip(q0, q1)) < 0.:
        q1 = [-v for v in q1]
    # Stable near zero and invariant under quaternion sign reversal.
    return 4*math.atan2(norm([a-b for a, b in zip(q0, q1)]),
                        norm([a+b for a, b in zip(q0, q1)]))


def motion_summary(frames, expected_count, expected_steps):
    first, last = frames[0], frames[-1]
    require(first["step"] == 0 and last["step"] == expected_steps,
            "Trajectory endpoints do not cover requested steps")
    ids = set(first["atoms"])
    require(all(frame["count"] == expected_count and set(frame["atoms"]) == ids for frame in frames),
            "Trajectory particle IDs/count changed")
    distances, raw_distances, angles = [], [], []
    for ident in sorted(ids):
        a, b = first["atoms"][ident], last["atoms"][ident]
        delta = [y-x for x, y in zip(a["x"], b["x"])]
        raw_distances.append(norm(delta))
        distances.append(norm(minimum_image(delta, last["box"])))
        angles.append(quat_angle(a["q"], b["q"]))
    quat_error = max(abs(norm(atom["q"])-1.) for frame in frames for atom in frame["atoms"].values())
    return {"all_trajectory_numeric_fields_finite": True, "frames": len(frames),
            "first_step": first["step"], "last_step": last["step"], "particle_count": expected_count,
            "maximum_quaternion_norm_error": quat_error,
            "maximum_position_displacement_m": max(distances),
            "rms_position_displacement_m": norm(distances)/math.sqrt(expected_count),
            "maximum_raw_wrapped_displacement_m": max(raw_distances),
            "particles_position_changed_above_1e_18_m": sum(v > 1e-18 for v in distances),
            "maximum_orientation_change_rad": max(angles),
            "rms_orientation_change_rad": norm(angles)/math.sqrt(expected_count),
            "particles_orientation_changed_above_1e_12_rad": sum(v > 1e-12 for v in angles),
            "positions_changed": max(distances) > 1e-18,
            "orientations_changed": max(angles) > 1e-12,
            "position_difference_convention": "minimum image in final reduced triclinic cell",
            "orientation_difference_convention": "quaternion sign-invariant rotation angle"}


def summarize_case(run, expected_steps):
    launch = json.loads((run/"diagnostic_launch.json").read_text())
    require(launch.get("status") == "completed" and launch.get("completed_requested_steps") is True
            and launch.get("returncode") == 0, "Diagnostic launch is not completed")
    require(launch.get("inputs_unchanged") is True and launch.get("before_hashes") == launch.get("after_hashes"),
            "Diagnostic binary/input hashes changed during run")
    params = json.loads((run/"parameters.json").read_text())
    require(params["steps"] == expected_steps, "Unexpected requested step count")
    for name in ("particles.data", "in.bridge"):
        matching = [value for path, value in launch["before_hashes"].items() if Path(path).name == name]
        require(len(matching) == 1 and sha(run/name) == matching[0], "Current input differs from launch: "+name)
    with (run/"history.csv").open() as f:
        rows = list(csv.DictReader(f))
    by_step = {}
    required = ["time_s", "strain", "profile_total_s", "profile_solve_s", "profile_geometry_s",
                "profile_middle_s", "newton_iterations", "linear_iterations", "solver_relative_residual",
                "hydro_dissipation_W", "contact_power_W", "stress_power_W", "power_balance_error_W",
                "force_residual", "mobility_residual", "eta_Pa_s", "near_pairs", "contacts", "reciprocal_modes"]
    for row in rows:
        step = int(row["step"])
        require(step not in by_step, "Duplicate history timestep")
        parsed = {key: finite(row[key], "history "+key) for key in required}
        parsed["step"] = step
        parsed["descent_steps"] = int(row["descent_steps"]) if "descent_steps" in row else None
        scale = max(abs(parsed[k]) for k in ("hydro_dissipation_W", "contact_power_W", "stress_power_W"))
        parsed["relative_power_balance_error"] = abs(parsed["power_balance_error_W"])/max(scale, 1e-300)
        by_step[step] = parsed
    require(set(by_step) == set(range(expected_steps+1)), "Missing or extra history steps")
    final_text = (run/"final.data").read_text()
    match = re.search(r"^LAMMPS data file .*timestep\s*=\s*(\d+)", final_text)
    require(match is not None and int(match.group(1)) == expected_steps, "final.data did not advance requested steps")
    count_match = re.search(r"(?m)^(\d+) atoms\s*$", final_text)
    require(count_match is not None and int(count_match.group(1)) == params["count"], "final.data particle count mismatch")
    frames = read_trajectory(run/"trajectory.lammpstrj")
    motion = motion_summary(frames, params["count"], expected_steps)
    values = [by_step[step] for step in sorted(by_step)]
    binary_hashes = {p: value for p, value in launch["before_hashes"].items()
                     if Path(p).name not in ("particles.data", "in.bridge")}
    return {"ar": params["ar"], "count": params["count"], "phi": params["phi"], "run": str(run),
            "status": "completed_and_parsed", "requested_steps": expected_steps,
            "wall_seconds": launch.get("wall_seconds"), "threads": launch.get("OMP_NUM_THREADS"),
            "startup_step0": by_step[0], "timesteps": [by_step[step] for step in range(1, expected_steps+1)],
            "max_solver_relative_residual": max(row["solver_relative_residual"] for row in values),
            "max_relative_power_balance_error": max(row["relative_power_balance_error"] for row in values),
            "final_data_timestep": int(match.group(1)), "final_data_step_advance_confirmed": True,
            "trajectory": motion, "launch_binary_hashes": binary_hashes,
            "launch_before_after_hashes_equal": True,
            "particle_sha256": sha(run/"particles.data"),
            "artifact_sha256": {name: sha(run/name) for name in
                                ("history.csv", "trajectory.lammpstrj", "final.data", "diagnostic_launch.json")},
            "steady_state_claim": False,
            "scope": "Short moving-particle diagnostic; no steady viscosity or Stokes-accuracy certification"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("paths", type=Path, nargs="+", help="case directories or parents containing diagnostic_launch.json")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=2)
    p.add_argument("--require-ar", type=float, action="append", default=[])
    args = p.parse_args()
    cases = set()
    for path in args.paths:
        path = path.resolve()
        if (path/"diagnostic_launch.json").is_file():
            cases.add(path)
        else:
            cases.update(q.parent for q in path.rglob("diagnostic_launch.json"))
    completed, excluded = [], []
    for run in sorted(cases):
        try:
            launch = json.loads((run/"diagnostic_launch.json").read_text())
            if not (launch.get("status") == "completed" and launch.get("completed_requested_steps") is True
                    and launch.get("returncode") == 0):
                excluded.append({"run": str(run), "reason": "Run not completed; history/trajectory not parsed",
                                 "recorded_status": launch.get("status"), "returncode": launch.get("returncode")})
                continue
            completed.append(summarize_case(run, args.steps))
        except (ValueError, KeyError, OSError) as error:
            excluded.append({"run": str(run), "reason": str(error)})
    missing = sorted(set(args.require_ar)-{case["ar"] for case in completed})
    result = {"schema_version": 1, "completed": sorted(completed, key=lambda c: (c["ar"], c["run"])),
              "excluded": excluded, "missing_required_ar": missing, "steady_state_claim": False,
              "relative_power_balance_definition": "abs(error)/max(abs(hydro_dissipation),abs(contact_power),abs(stress_power),1e-300)",
              "profile_time_definition": "Each history row's own solve: step0 initialization separated from later timesteps"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+"\n")
    print(json.dumps({"output": str(args.output.resolve()), "completed_cases": len(completed),
                      "excluded_cases": len(excluded), "missing_required_ar": missing}), flush=True)
    if not completed or missing or excluded:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
