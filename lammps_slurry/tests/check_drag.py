#!/usr/bin/env python3
"""Check isolated-particle drag through the actual LAMMPS plugin interface.

python3 check_drag.py --lmp /path/lmp --plugin /path/slurry_plugin.so
python3 check_drag.py --lmp /path/lmp --plugin /path/slurry_plugin.so \
    --mpiexec mpiexec --ranks 2

Uses only the Python standard library. These force/torque checks do not validate
the complete suspension stress, which is outside the first-stage model.
"""

import argparse
import math
from pathlib import Path
import shutil
import subprocess
import tempfile


def axisymmetric_apply(equatorial, polar, axis, vector):
    projection = sum(a * v for a, v in zip(axis, vector))
    return tuple(equatorial * v + (polar - equatorial) * a * projection
                 for a, v in zip(axis, vector))


def analytic_oblate_resistance(a, c):
    """Closed-form oblate integrals, independent of the plugin quadrature."""
    d = math.sqrt(a * a - c * c)
    angle = math.atan(d / c)
    chi = 2 * angle / d
    alpha_a = (angle / d - c / (a * a)) / (d * d)
    alpha_c = 2 * (1 / c - angle / d) / (d * d)
    rt_a = 16 * math.pi / (chi + a * a * alpha_a)
    rt_c = 16 * math.pi / (chi + c * c * alpha_c)
    rr_a = 16 * math.pi / 3 * (a * a + c * c) / (
        a * a * alpha_a + c * c * alpha_c)
    rr_c = 16 * math.pi / (3 * alpha_a)
    return rt_a, rt_c, rr_a, rr_c


def format_vector(values):
    return " ".join(f"{value:.17g}" for value in values)


def run_case(options, work, name, a, c, angle, shear, slip, rotation_slip):
    eta = 0.01
    axis = (math.sin(math.radians(angle)), 0, math.cos(math.radians(angle)))
    ratio = (c * c - a * a) / (c * c + a * a)
    omega_flow = (-ratio * shear * axis[2] * axis[0] / 2,
                  0,
                  -shear / 2 + ratio * shear * axis[0] * axis[0] / 2)
    omega = tuple(w + dw for w, dw in zip(omega_flow, rotation_slip))
    mass = 2200 * 4 * math.pi * a * a * c / 3
    inertia_a = mass * (a * a + c * c) / 5
    inertia_c = mass * 2 * a * a / 5
    angmom = axisymmetric_apply(inertia_a, inertia_c, axis, omega)
    y = 5e-6
    velocity = (shear * y + slip[0], slip[1], slip[2])

    if a == c:
        rt_a = rt_c = 6 * math.pi * a
        rr_a = rr_c = 8 * math.pi * a**3
    else:
        rt_a, rt_c, rr_a, rr_c = analytic_oblate_resistance(a, c)
    expected_force = tuple(-eta * f for f in axisymmetric_apply(rt_a, rt_c, axis, slip))
    expected_torque = tuple(-eta * t for t in axisymmetric_apply(
        rr_a, rr_c, axis, rotation_slip))

    input_path = work / f"{name}.in"
    dump_path = work / f"{name}.dump"
    input_path.write_text(f"""plugin load "{options.plugin}"
units si
atom_style ellipsoid
boundary p p p
region box block 0 1e-5 0 1e-5 0 1e-5 units box
create_box 1 box
create_atoms 1 single 5e-6 {y:.17g} 5e-6 units box
set atom 1 shape {2*a:.17g} {2*a:.17g} {2*c:.17g}
set atom 1 mass {mass:.17g}
set atom 1 quat 0 1 0 {angle:.17g}
set atom 1 angmom {format_vector(angmom)}
velocity all set {format_vector(velocity)} units box
pair_style zero 2e-6
pair_coeff * *
neighbor 1e-6 bin
fix integrate all nve/asphere
fix solvent all slurry/drag {eta:.17g} {shear:.17g}
timestep 1e-12
run 0
write_dump all custom "{dump_path}" id fx fy fz tqx tqy tqz modify sort id format float %.17g
""", encoding="ascii")
    command = [options.lmp, "-log", "none", "-in", str(input_path)]
    if options.mpiexec:
        command = [options.mpiexec, "-n", str(options.ranks)] + command
    result = subprocess.run(command, cwd=work, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, timeout=60, check=False)
    if result.returncode:
        raise RuntimeError(f"{name}: LAMMPS failed\n{result.stdout}")
    rows = dump_path.read_text(encoding="ascii").splitlines()
    marker = next(i for i, row in enumerate(rows) if row.startswith("ITEM: ATOMS"))
    measured = [float(value) for value in rows[marker + 1].split()[1:]]
    expected = expected_force + expected_torque
    for component, (actual, target) in enumerate(zip(measured, expected)):
        absolute_tolerance = 1e-25 if component < 3 else 1e-30
        if not math.isclose(actual, target, rel_tol=2e-9, abs_tol=absolute_tolerance):
            raise AssertionError(
                f"{name}, component {component}: got {actual:.17g}, expected {target:.17g}")
    print(f"PASS: {name} (LAMMPS ranks={options.ranks})")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lmp", required=True, help="LAMMPS executable")
    parser.add_argument("--plugin", required=True, help="Built slurry plugin shared library")
    parser.add_argument("--mpiexec", help="MPI launcher, for example mpiexec")
    parser.add_argument("--ranks", type=int, default=1)
    options = parser.parse_args()
    options.lmp = shutil.which(options.lmp) or str(Path(options.lmp).resolve())
    options.plugin = str(Path(options.plugin).resolve())
    if options.ranks < 1 or (options.ranks != 1 and not options.mpiexec):
        parser.error("ranks must be positive; more than one rank requires --mpiexec")
    with tempfile.TemporaryDirectory(prefix="slurry_drag_check_") as temporary:
        work = Path(temporary)
        run_case(options, work, "sphere_stokes", 1.65e-6, 1.65e-6, 45, 0,
                 (1e-4, -2e-4, 0.5e-4), (7, -2, 3))
        run_case(options, work, "rotated_oblate_drag", 1.65e-6, 0.825e-6, 45, 50,
                 (1e-4, -2e-4, 0.3e-4), (2, -1, 4))
        run_case(options, work, "affine_jeffery_zero_force", 1.65e-6, 0.825e-6, 45, 50,
                 (0, 0, 0), (0, 0, 0))


if __name__ == "__main__":
    main()
