#!/usr/bin/env python3
"""Run the default AR=2, phi=0.24 case and check serial/MPI agreement.

Uses only Python's standard library. With --mpiexec, runs the same packed
configuration through the launcher with one and two MPI ranks. The short inertial DEM run
tests executable integration and communication, not steady-state viscosity.
"""

import argparse
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_COUNT = 108
EXPECTED_PHI = 0.24
EXPECTED_SEMIAXES = (300e-9, 300e-9, 150e-9)
EXPECTED_DIAMETERS = tuple(2*axis for axis in EXPECTED_SEMIAXES)
EXPECTED_DENSITY = 2400.0


def checked_run(command, work, timeout):
    process = subprocess.Popen(command, cwd=work, text=True, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, start_new_session=True)
    try:
        output, _ = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        # Kill MPI workers as well as their launcher. Killing only the launcher
        # can leave worker-held stdout pipes open and prevent timeout cleanup.
        os.killpg(process.pid, signal.SIGKILL)
        output, _ = process.communicate()
        raise RuntimeError(f"Command timed out after {timeout} s: {command!r}\n{output[-16000:]}") from None
    if process.returncode:
        raise RuntimeError(f"Command failed ({process.returncode}): {command!r}\n{output[-16000:]}")
    return output


def check_parameters(path):
    """Read the generated provenance record, not a simulation input file."""
    parameters = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        key, separator, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if not separator or not key or key in parameters:
            raise AssertionError(f"{path}: malformed or duplicate parameter: {line!r}")
        parameters[key] = value
    expected = {"count": EXPECTED_COUNT, "ar": 2.0, "phi": EXPECTED_PHI,
                "volume_fraction_actual": EXPECTED_PHI, "density": EXPECTED_DENSITY,
                "thickness_nm": 300.0, "roughness_nm": 3.0,
                "packing_initial_phi_actual": 0.05,
                **dict(zip(("a_m", "b_m", "c_m"), EXPECTED_SEMIAXES))}
    for key, expected_value in expected.items():
        value = float(parameters[key])
        if not math.isclose(value, expected_value, rel_tol=2e-14, abs_tol=0):
            raise AssertionError(f"{path}: {key}={value:.17g}, expected {expected_value:.17g}")
    overlap = float(parameters["packing_maximum_PW_overlap"])
    tolerance = float(parameters["overlap_tolerance"])
    if not math.isfinite(overlap) or not 0 <= overlap <= tolerance*1.05 + 1e-12:
        raise AssertionError(f"{path}: packing overlap {overlap} exceeds tolerance {tolerance}")


def check_diameters(diameters, path):
    if any(not math.isclose(actual, expected, rel_tol=2e-14, abs_tol=0)
           for actual, expected in zip(diameters, EXPECTED_DIAMETERS)):
        raise AssertionError(f"{path}: particle diameters {diameters}, expected {EXPECTED_DIAMETERS}")


def final_frame(path):
    lines = path.read_text().splitlines()
    final = max(i for i, line in enumerate(lines) if line == "ITEM: TIMESTEP")
    frame = lines[final:]
    count_line = frame.index("ITEM: NUMBER OF ATOMS")
    count = int(frame[count_line + 1])
    if count != EXPECTED_COUNT:
        raise AssertionError(f"{path}: expected {EXPECTED_COUNT} particles, found {count}")
    box_line = next(i for i, line in enumerate(frame) if line.startswith("ITEM: BOX BOUNDS"))
    bounds = [[float(v) for v in row.split()] for row in frame[box_line + 1:box_line + 4]]
    if not all(math.isfinite(v) for row in bounds for v in row):
        raise AssertionError(f"{path}: non-finite box bounds")
    if "xy xz yz" not in frame[box_line]:
        raise AssertionError(f"{path}: expected the triclinic Lees--Edwards box")
    (xlo, xhi, xy), (ylo, yhi, xz), (zlo, zhi, yz) = bounds
    lengths = (xhi-xlo-max(0, xy, xz, xy+xz)+min(0, xy, xz, xy+xz),
               yhi-ylo-max(0, yz)+min(0, yz), zhi-zlo)
    if any(length <= 0 for length in lengths):
        raise AssertionError(f"{path}: invalid cell lengths {lengths}")
    atom_line = next(i for i, line in enumerate(frame) if line.startswith("ITEM: ATOMS "))
    columns = frame[atom_line].split()[2:]
    particles = {}
    volume = 0.0
    for line in frame[atom_line + 1:atom_line + 1 + count]:
        values = [float(v) for v in line.split()]
        if len(values) != len(columns) or not all(math.isfinite(v) for v in values):
            raise AssertionError(f"{path}: malformed or non-finite atom row")
        row = dict(zip(columns, values))
        number = int(row["id"])
        if number in particles:
            raise AssertionError(f"{path}: duplicate particle ID {number}")
        particles[number] = row
        diameters = tuple(row[f"c_shape[{i}]"] for i in range(1, 4))
        if any(d <= 0 for d in diameters):
            raise AssertionError(f"{path}: non-positive particle shape")
        check_diameters(diameters, path)
        volume += math.pi*math.prod(diameters)/6
        quat_norm = sum(row[f"c_shape[{i}]"]**2 for i in range(4, 8))
        if not math.isclose(quat_norm, 1, rel_tol=0, abs_tol=2e-10):
            raise AssertionError(f"{path}: non-unit quaternion for particle {number}")
    if set(particles) != set(range(1, EXPECTED_COUNT + 1)):
        raise AssertionError(f"{path}: missing or unexpected particle IDs")
    phi = volume/math.prod(lengths)
    if not math.isclose(phi, EXPECTED_PHI, rel_tol=0, abs_tol=2e-12):
        raise AssertionError(f"{path}: volume fraction is {phi:.17g}")
    return int(frame[1]), particles, lengths


def check_final_data(path):
    """Check particle geometry, density and phi in an actual LAMMPS data file."""
    lines = path.read_text().splitlines()
    count = next(int(line.split()[0]) for line in lines if line.strip().endswith(" atoms"))
    if count != EXPECTED_COUNT:
        raise AssertionError(f"{path}: final data has {count} particles")
    dimensions = []
    for axis in "xyz":
        bound = next(line.split() for line in lines if line.split()[-2:] == [f"{axis}lo", f"{axis}hi"])
        dimensions.append(float(bound[1])-float(bound[0]))
    volume = 0.0
    section = None
    counts = {"Atoms": 0, "Velocities": 0, "Ellipsoids": 0}
    for line in lines:
        values = line.split()
        if not values:
            continue
        if values[0] in counts:
            section = values[0]
            continue
        if section is None:
            continue
        if not values[0].isdigit():
            section = None
            continue
        numbers = [float(value) for value in values]
        if not all(math.isfinite(value) for value in numbers):
            raise AssertionError(f"{path}: non-finite {section} row")
        counts[section] += 1
        if section == "Atoms":
            # atom_style ellipsoid: id type ellipsoid_flag density x y z [...].
            if len(numbers) < 7 or numbers[2] != 1 or not math.isclose(
                    numbers[3], EXPECTED_DENSITY, rel_tol=2e-14, abs_tol=0):
                raise AssertionError(f"{path}: incorrect ellipsoid density or atom row")
        elif section == "Ellipsoids":
            if len(numbers) != 8 or min(numbers[1:4]) <= 0:
                raise AssertionError(f"{path}: malformed ellipsoid")
            check_diameters(numbers[1:4], path)
            volume += math.pi*math.prod(numbers[1:4])/6
    if any(count != EXPECTED_COUNT for count in counts.values()):
        raise AssertionError(f"{path}: incomplete final data sections {counts}")
    phi = volume/math.prod(dimensions)
    if not math.isclose(phi, EXPECTED_PHI, rel_tol=0, abs_tol=2e-12):
        raise AssertionError(f"{path}: final-data volume fraction is {phi:.17g}")


def run_case(options, work, ranks):
    command = [options.lmp, "-var", "slurry_plugin", f'"{options.plugin}"',
               "-in", "in.slurry", "-log", "log.lammps"]
    if options.mpiexec:
        command = [options.mpiexec, "-n", str(ranks)] + command
    checked_run(command, work, options.timeout)
    check_final_data(work/"final.data")
    frame = final_frame(work/"final.dump")
    if frame[0] != options.steps:
        raise AssertionError(f"Run ended at step {frame[0]}, expected {options.steps}")
    # Check every saved frame for finite output as well as the forced final dump.
    trajectory = (work/"trajectory.lammpstrj").read_text().lower()
    for line in trajectory.splitlines():
        if line.startswith("item:"):
            continue
        if not all(math.isfinite(float(word)) for word in line.split()):
            raise AssertionError("Trajectory contains non-finite state values")
    print(f"PASS: default AR=2, phi=0.24 dynamics; {EXPECTED_COUNT} finite particles, ranks={ranks}")
    return frame[1], frame[2]


def compare_states(serial, parallel, lengths):
    maximum_position_error = 0.0
    maximum_velocity_error = 0.0
    maximum_quaternion_error = 0.0
    for number in sorted(serial):
        first, second = serial[number], parallel[number]
        for axis, length in zip("xyz", lengths):
            difference = abs(first[axis]-second[axis])
            maximum_position_error = max(maximum_position_error, difference)
            if difference > 2e-11*length:
                raise AssertionError(f"MPI position mismatch for particle {number}, {axis}: {difference:.9g} m")
        for axis in ("vx", "vy", "vz"):
            difference = abs(first[axis]-second[axis])
            maximum_velocity_error = max(maximum_velocity_error, difference)
            if not math.isclose(first[axis], second[axis], rel_tol=2e-8, abs_tol=1e-12):
                raise AssertionError(f"MPI velocity mismatch for particle {number}, {axis}: {difference:.9g} m/s")
        first_quat = [first[f"c_shape[{i}]"] for i in range(4, 8)]
        second_quat = [second[f"c_shape[{i}]"] for i in range(4, 8)]
        sign = 1 if sum(a*b for a, b in zip(first_quat, second_quat)) >= 0 else -1
        difference = max(abs(a-sign*b) for a, b in zip(first_quat, second_quat))
        maximum_quaternion_error = max(maximum_quaternion_error, difference)
        if difference > 2e-10:
            raise AssertionError(f"MPI quaternion mismatch for particle {number}: {difference:.9g}")
    print("PASS: one-/two-rank final states agree; "
          f"max |dx|={maximum_position_error:.3g} m, "
          f"|dv|={maximum_velocity_error:.3g} m/s, |dq|={maximum_quaternion_error:.3g}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lmp", required=True)
    parser.add_argument("--plugin", required=True)
    parser.add_argument("--packer", default=str(ROOT.parent/"build/lammps-slurry/slurry_pack"),
                        help="compiled geometric-growth packer")
    parser.add_argument("--mpiexec", help="MPI launcher; omit to check only the single-process case")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--timeout", type=int, default=180, help="seconds per LAMMPS invocation")
    options = parser.parse_args()
    if options.steps < 1 or options.timeout < 1:
        parser.error("steps and timeout must be positive")
    options.lmp = shutil.which(options.lmp) or str(Path(options.lmp).resolve())
    options.plugin = str(Path(options.plugin).resolve())
    options.packer = str(Path(options.packer).resolve())
    if any(character in options.plugin for character in ('"', '\n', '\r')):
        parser.error("plugin filename cannot contain double quotes or newlines")
    if options.mpiexec:
        options.mpiexec = shutil.which(options.mpiexec) or str(Path(options.mpiexec).resolve())
    with tempfile.TemporaryDirectory(prefix="slurry_case_check_") as temporary:
        work = Path(temporary)
        initial = work/"initial"
        checked_run([sys.executable, str(ROOT/"tools"/"prepare_case.py"),
                     "--packer", options.packer, "--output", str(initial),
                     "--steps", str(options.steps)], work, 120)
        check_parameters(initial/"parameters.txt")
        check_final_data(initial/"particles.data")
        text = (initial/"in.slurry").read_text()
        text += "write_dump all custom final.dump id type x y z vx vy vz fx fy fz c_shape[*] modify sort id format float %.17g\n"
        (initial/"in.slurry").write_text(text)
        serial = work/"serial"
        shutil.copytree(initial, serial)
        single_state, lengths = run_case(options, serial, 1)
        if options.mpiexec:
            parallel = work/"parallel"
            shutil.copytree(initial, parallel)
            parallel_state, parallel_lengths = run_case(options, parallel, 2)
            if any(not math.isclose(a, b, rel_tol=2e-14) for a, b in zip(lengths, parallel_lengths)):
                raise AssertionError("Serial and MPI cell dimensions differ")
            compare_states(single_state, parallel_state, lengths)
        else:
            print("SKIP: two-rank comparison requires --mpiexec")


if __name__ == "__main__":
    main()
