#!/usr/bin/env python3
"""Actual moving LAMMPS check of periodic images, xz flip and stress signs.

A single two-body contacting trajectory is represented in equivalent lattice
bases and under a uniform translation. This is an implementation regression,
not a Stokes accuracy, steady-state viscosity or configuration-sampling test.
"""
import argparse
import csv
import hashlib
import json
import math
import shutil
from pathlib import Path
import subprocess
import time

import numpy as np

C = 150e-9
L = 20.0
SHEAR = 100.0
DT = 1e-6
STEPS = 4


def require(ok, message):
    if not ok:
        raise AssertionError(message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_positions(H, points):
    # Integer rounding only removes coordinates outside [0,1); no physical move.
    fractional = np.linalg.solve(H, np.asarray(points).T).T
    fractional -= np.floor(fractional)
    return (H @ fractional.T).T


def make_case(directory, plugin, variant, remap="v"):
    directory.mkdir(parents=True, exist_ok=True)
    tilt = .4999 * L
    H = np.array([[L, 0., tilt], [0., L, 0.], [0., 0., L]])
    points = np.array([[12., 10., 19.7], [2.002, 10., 1.7133333333333334]])
    if variant == "rebased":
        H[0, 2] -= L
    elif variant == "translated":
        # Put the repelled upper body immediately below zhi, ensuring actual
        # particle wrapping while following the same physical relative motion.
        points += np.array([0., 0., L - points[1, 2] - 1e-7])
    elif variant != "original":
        raise ValueError(variant)
    points = canonical_positions(H, points)
    data = ["Two-body compact bridge periodic implementation regression; SI", "",
            "2 atoms", "2 ellipsoids", "1 atom types", "",
            f"0 {L*C:.17g} xlo xhi", f"0 {L*C:.17g} ylo yhi", f"0 {L*C:.17g} zlo zhi",
            f"0 {H[0,2]*C:.17g} 0 xy xz yz", "", "Atoms # ellipsoid", ""]
    for i, point in enumerate(points, 1):
        data.append(f"{i} 1 1 2400 " + " ".join(f"{x*C:.17g}" for x in point))
    data.extend(["", "Ellipsoids", ""])
    for i in (1, 2):
        data.append(f"{i} {4*C:.17g} {4*C:.17g} {2*C:.17g} 1 0 0 0")
    (directory / "particles.data").write_text("\n".join(data) + "\n")
    script = f'''plugin load "{plugin}"
units si
atom_style ellipsoid
boundary p p p
read_data particles.data
comm_modify vel yes
pair_style zero {6*C:.17g}
pair_coeff * *
neighbor {0.2*C:.17g} bin
neigh_modify every 1 delay 0 check yes
fix shear all deform 0 xz erate {SHEAR:.17g} remap {remap} units box flip yes
fix hydro all slurry/bridge 0.01 {SHEAR:.17g} 1e-9 1 contact 3e-9 0.1 80 spectral 2 spectral_tolerance 1e-8 tolerance 1e-8 iterations 2000 history history.csv 1 0
timestep {DT:.17g}
thermo 1
thermo_style custom step time xz f_hydro[1] f_hydro[4] f_hydro[7] f_hydro[14]
dump state all custom 1 trajectory.lammpstrj id x y z ix iy iz vx vy vz f_hydro[*]
dump_modify state sort id format float %.17g
run {STEPS}
'''
    (directory / "in.bridge").write_text(script)
    return H, points


def read_history(path):
    with path.open() as stream:
        return [{key: float(value) for key, value in row.items()}
                for row in csv.DictReader(stream)]


def read_dump(path):
    frames = []
    lines = iter(path.read_text().splitlines())
    for line in lines:
        require(line == "ITEM: TIMESTEP", "Unexpected dump header")
        step = int(next(lines))
        require(next(lines) == "ITEM: NUMBER OF ATOMS", "Missing dump atom count")
        count = int(next(lines))
        header = next(lines)
        require(header.startswith("ITEM: BOX BOUNDS"), "Missing box header")
        bounds = [list(map(float, next(lines).split())) for _ in range(3)]
        columns = next(lines).split()[2:]
        values = np.array([list(map(float, next(lines).split())) for _ in range(count)])
        frames.append({"step": step, "bounds": bounds, "columns": columns, "values": values})
    return frames


def run_case(executable, plugin, output, variant, remap="v", launcher=None):
    H, initial = make_case(output, plugin, variant, remap)
    started = time.perf_counter()
    command = [str(executable), "-in", "in.bridge", "-log", "log.lammps"]
    if launcher:
        command = [launcher, "-np", "1"] + command
    proc = subprocess.run(command,
                          cwd=output, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, env=None)
    elapsed = time.perf_counter() - started
    (output / "stdout.txt").write_text(proc.stdout)
    if remap != "v":
        require(proc.returncode != 0 and "remap v" in proc.stdout,
                "slurry/bridge must reject fix deform remap x")
        return {"rejected": True, "returncode": proc.returncode, "elapsed_s": elapsed}
    require(proc.returncode == 0, f"LAMMPS {variant} failed; see {output / 'stdout.txt'}")
    history = read_history(output / "history.csv")
    frames = read_dump(output / "trajectory.lammpstrj")
    require(len(history) == STEPS+1 and len(frames) == STEPS+1, "Missing timestep records")
    require(all(row["contacts"] >= 1 for row in history), "Contact must remain active through flip")
    power_errors = []
    force_errors = []
    torque_errors = []
    for row, frame in zip(history, frames):
        require(int(row["step"]) == frame["step"], "Mismatched stress/dump timestep")
        power_scale = max(abs(row["stress_power_W"]), abs(row["hydro_dissipation_W"]),
                          abs(row["contact_power_W"]), 1e-100)
        power_errors.append(abs(row["power_balance_error_W"])/power_scale)
        columns = frame["columns"]
        # Per-atom columns: hydro F,T; on-fluid STF5; contact F,T.
        moments = frame["values"][:, [columns.index(f"f_hydro[{k}]") for k in range(1, 18)]]
        hydro_force, contact_force = moments[:, :3], moments[:, 11:14]
        hydro_torque, contact_torque = moments[:, 3:6], moments[:, 14:17]
        force_errors.append(float(np.linalg.norm(hydro_force+contact_force) /
                            max(np.linalg.norm(hydro_force), np.linalg.norm(contact_force), 1e-100)))
        torque_errors.append(float(np.linalg.norm(hydro_torque+contact_torque) /
                             max(np.linalg.norm(hydro_torque), np.linalg.norm(contact_torque), 1e-100)))
    require(max(power_errors) < 1e-5, "Stress-work/dissipation mismatch")
    require(max(force_errors) < 1e-5, "Hydro/contact force balance mismatch")
    require(max(torque_errors) < 1e-5, "Hydro/contact torque balance mismatch")
    tilt = [frame["bounds"][1][2] / C for frame in frames]
    iz = [frame["values"][:, frame["columns"].index("iz")].tolist() for frame in frames]
    return {"elapsed_s": elapsed, "initial_H_over_c": H.tolist(),
            "initial_positions_over_c": initial.tolist(), "history": history,
            "frames": frames, "tilt_xz_over_c": tilt, "z_image_flags": iz,
            "maximum_relative_power_balance_error": max(power_errors),
            "maximum_relative_force_balance_error": max(force_errors),
            "maximum_relative_torque_balance_error": max(torque_errors)}


def compare(reference, candidate):
    fields = ["sigma_hydro_Pa", "sigma_contact_Pa", "sigma_xz_Pa", "hydro_dissipation_W",
              "contact_power_W", "stress_power_W", "eta_Pa_s"]
    field_errors = {}
    for name in fields:
        a = np.array([row[name] for row in reference["history"]])
        b = np.array([row[name] for row in candidate["history"]])
        field_errors[name] = float(np.max(np.abs(a-b)) / max(np.max(np.abs(a)), 1e-100))
    response_errors = {"force": 0., "torque": 0., "stresslet": 0., "contact_force": 0., "contact_torque": 0.}
    for first, second in zip(reference["frames"], candidate["frames"]):
        cols = first["columns"]
        indices = [cols.index(f"f_hydro[{k}]") for k in range(1, 18)]
        a, b = first["values"][:, indices], second["values"][:, indices]
        for name, sl in (("force", slice(0,3)), ("torque", slice(3,6)), ("stresslet", slice(6,11)),
                         ("contact_force", slice(11,14)), ("contact_torque", slice(14,17))):
            response_errors[name] = max(response_errors[name], float(np.linalg.norm(a[:,sl]-b[:,sl]) /
                                          max(np.linalg.norm(a[:,sl]), 1e-100)))
    require(max(field_errors.values()) < 1e-5, "Lattice/translation equivalent stress histories disagree")
    require(max(response_errors.values()) < 1e-5, "Lattice/translation equivalent F/T/S histories disagree")
    return {"maximum_relative_history_differences": field_errors,
            "maximum_relative_per_particle_response_differences": response_errors}


def strict_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: strict_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [strict_json(item) for item in value]
    return value


def public_case(case):
    return strict_json({key: value for key, value in case.items() if key != "frames"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--plugin", type=Path, required=True)
    parser.add_argument("--mpi", action="store_true",
                        help="launch an MPI LAMMPS build through mpirun/mpiexec -np 1 (not a multi-rank test)")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "periodic_check")
    args = parser.parse_args()
    executable, plugin, output = args.executable.resolve(), args.plugin.resolve(), args.output.resolve()
    launcher = (shutil.which("mpirun") or shutil.which("mpiexec")) if args.mpi else None
    if args.mpi and not launcher:
        parser.error("--mpi requires mpirun or mpiexec from the matching MPI installation")
    output.mkdir(parents=True, exist_ok=True)
    plugin_digest = sha256(plugin)
    original = run_case(executable, plugin, output / "original", "original", launcher=launcher)
    rebased = run_case(executable, plugin, output / "rebased", "rebased", launcher=launcher)
    translated = run_case(executable, plugin, output / "translated", "translated", launcher=launcher)
    flip_steps = [k for k in range(1, len(original["tilt_xz_over_c"]))
                  if abs(original["tilt_xz_over_c"][k]-original["tilt_xz_over_c"][k-1]) > L/2]
    require(flip_steps, "Trajectory must actually undergo a triclinic box flip")
    require(any(row != translated["z_image_flags"][0] for row in translated["z_image_flags"][1:]),
            "Translated trajectory must actually wrap a particle across z boundary")
    report = {"purpose": "moving periodic integration regression; not steady-state viscosity",
              "executable": str(executable), "executable_sha256": sha256(executable),
              "plugin": str(plugin), "mpi": args.mpi, "mpi_launcher": launcher, "plugin_sha256": plugin_digest, "runner_sha256": sha256(__file__),
              "steps": STEPS, "dt_s": DT, "shear_rate_s_inv": SHEAR, "flip_steps": flip_steps,
              "original": public_case(original), "rebased": public_case(rebased),
              "translated": public_case(translated),
              "rebased_comparison": compare(original, rebased),
              "translated_comparison": compare(original, translated),
              "remap_x": run_case(executable, plugin, output / "remap_x", "original", "x", launcher=launcher),
              "passed": True}
    require(sha256(plugin) == plugin_digest, "Plugin changed during regression run")
    (output / "result.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"passed": True, "result": str(output / "result.json"), "flip_steps": flip_steps,
                      "rebased_comparison": report["rebased_comparison"],
                      "translated_comparison": report["translated_comparison"]}, indent=2))


if __name__ == "__main__":
    main()
