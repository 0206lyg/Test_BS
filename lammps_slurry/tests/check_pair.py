#!/usr/bin/env python3
"""Check the loaded LAMMPS pair plugin against analytical two-particle limits.

Only the Python standard library is needed.  No integration or drag fix is used:
run 0 isolates forces and torques from the actual slurry/spheroid pair style.
With --mpiexec, repeat each case on two MPI ranks and compare with one rank.
The shear-image case places particles across the periodic y boundary and tests
LAMMPS's ghost-velocity remapping against the analytical affine-velocity slip.
"""

import argparse
import math
import os
from pathlib import Path
import shlex
import subprocess
import tempfile


ETA = 0.01
HCUT = 50e-9
HMIN = 5e-9
KN = 80.0
DENSITY = 2200.0
BOX = 12e-6


def add(a, b):
    return tuple(x+y for x, y in zip(a, b))


def scale(a, value):
    return tuple(value*x for x in a)


def dot(a, b):
    return sum(x*y for x, y in zip(a, b))


def cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2],
            a[0]*b[1]-a[1]*b[0])


def angular_momentum(axes, omega):
    mass = 4*math.pi*math.prod(axes)*DENSITY/3
    return tuple(mass/5*(axes[(k+1)%3]**2+axes[(k+2)%3]**2)*omega[k]
                 for k in range(3))


def data_file(path, case):
    axes = case["axes"]
    text = ["Two-particle plugin verification (SI)", "", "2 atoms", "2 ellipsoids",
            "1 atom types", ""]
    for direction in "xyz":
        text.append(f"{-BOX/2:.17g} {BOX/2:.17g} {direction}lo {direction}hi")
    text += ["0 0 0 xy xz yz", "", "Atoms # ellipsoid", ""]
    for index, position in enumerate(case["positions"], 1):
        text.append(f"{index} 1 1 {DENSITY:.17g} " +
                    " ".join(f"{x:.17g}" for x in position))
    text += ["", "Ellipsoids", ""]
    for index in (1, 2):
        text.append(f"{index} " + " ".join(f"{2*x:.17g}" for x in axes) + " 1 0 0 0")
    text += ["", "Velocities", ""]
    for index, (velocity, omega) in enumerate(zip(case["velocities"], case["omegas"]), 1):
        values = velocity + angular_momentum(axes, omega)
        text.append(f"{index} " + " ".join(f"{x:.17g}" for x in values))
    path.write_text("\n".join(text) + "\n")


def make_case(name, gap, normal=(1., 0., 0.), velocity=(0., 0., 0.),
              omega=(0., 0., 0.), axes=(1e-6, 1e-6, 1e-6), shear=0.):
    normal_axis = next(k for k in range(3) if normal[k])
    distance = 2*axes[normal_axis] + gap
    positions = (scale(normal, -distance/2), scale(normal, distance/2))
    velocities = ((0., 0., 0.), velocity)
    omegas = (omega, (0., 0., 0.))
    if shear:
        # Atom 2's lower periodic image is separated from atom 1 by -distance*y.
        positions = ((0., -BOX/2+distance/2, 0.), (0., BOX/2-distance/2, 0.))
        velocities = tuple((shear*p[1], 0., 0.) for p in positions)
        normal = (0., -1., 0.)
    return dict(name=name, axes=axes, gap=gap, normal=normal, distance=distance,
                positions=positions, velocities=velocities, omegas=omegas, shear=shear)


def analytical(case):
    """Principal-axis local curvature formula, independently reduced to scalars."""
    normal = case["normal"]
    axis = next(k for k in range(3) if normal[k])
    axes = case["axes"]
    k1, k2 = (2*axes[axis]/axes[k]**2 for k in range(3) if k != axis)
    if case["gap"] < HCUT:
        gap = max(case["gap"], HMIN)
        rn = 12*math.pi*ETA/((k1+k2)*math.sqrt(k1*k2))*(1/gap-1/HCUT)
        rt = 2*math.pi*ETA/math.sqrt(k1*k2)*math.log(HCUT/gap)
    else:
        rn = rt = 0.0
    lever_i = scale(normal, case["distance"]/2)
    lever_j = scale(lever_i, -1)
    vi, vj = case["velocities"]
    if case["shear"]:
        vj = add(vj, (-case["shear"]*BOX, 0., 0.))
    slip = add(add(vj, cross(case["omegas"][1], lever_j)),
               scale(add(vi, cross(case["omegas"][0], lever_i)), -1))
    vn = dot(slip, normal)
    force_i = add(scale(normal, rn*vn-KN*max(-case["gap"], 0)),
                  scale(add(slip, scale(normal, -vn)), rt))
    force_j = scale(force_i, -1)
    return {1: force_i + cross(lever_i, force_i),
            2: force_j + cross(lever_j, force_j)}


def parse_dump(path):
    lines = path.read_text().splitlines()
    header_index = next(i for i, line in enumerate(lines) if line.startswith("ITEM: ATOMS"))
    names = lines[header_index].split()[2:]
    result = {}
    for line in lines[header_index+1:]:
        values = dict(zip(names, line.split()))
        result[int(values["id"])] = tuple(float(values[key]) for key in
                                         ("fx", "fy", "fz", "tqx", "tqy", "tqz"))
    if set(result) != {1, 2}:
        raise AssertionError(f"Expected atoms 1 and 2 in {path}")
    return result


def compare(observed, expected, label):
    for atom in (1, 2):
        for offset, kind in ((0, "force"), (3, "torque")):
            actual = observed[atom][offset:offset+3]
            target = expected[atom][offset:offset+3]
            floor = 1e-22 if kind == "force" else 1e-28
            tolerance = floor + 2e-8*max(abs(value) for value in target)
            if any(not math.isfinite(value) for value in actual):
                raise AssertionError(f"{label}: nonfinite {kind} on atom {atom}: {actual}")
            if any(abs(x-y) > tolerance for x, y in zip(actual, target)):
                raise AssertionError(f"{label}: atom {atom} {kind} {actual}, expected {target}")


def run_case(root, case, lmp, plugin, ranks, mpiexec, restart=False):
    directory = root/f"{case['name']}-{ranks}rank{'-restart' if restart else ''}"
    directory.mkdir()
    data_file(directory/"particles.data", case)
    grid = [1, 1, 1]
    grid[next(k for k in range(3) if case["normal"][k])] = ranks
    processor_grid = " ".join(str(value) for value in grid)
    text = [f'plugin load "{plugin}"', "units si", "atom_style ellipsoid",
            "boundary p p p", f"processors {processor_grid}", "newton on"]
    if restart:
        source = root/f"{case['name']}-{ranks}rank"/"pair.restart"
        text.append(f'read_restart "{source}"')
    else:
        text += ["read_data particles.data",
                 f"pair_style slurry/spheroid {ETA:.17g} {HCUT:.17g} {HMIN:.17g} {KN:.17g}",
                 "pair_coeff * *"]
    text += ["comm_modify vel yes", "neighbor 1e-7 bin", "neigh_modify every 1 delay 0 check yes"]
    if case["shear"]:
        text.append(f"fix shear all deform 1 xy erate {case['shear']:.17g} remap v units box")
    text += ["thermo_style custom step pe", "run 0",
             "write_dump all custom forces.dump id fx fy fz tqx tqy tqz "
             "modify sort id format float %.17g", "write_restart pair.restart"]
    (directory/"in.check").write_text("\n".join(text)+"\n")
    command = ([*mpiexec, "-n", str(ranks)] if mpiexec else []) + [
        str(lmp), "-in", "in.check", "-log", "log.lammps"]
    environment = os.environ.copy()
    environment.setdefault("OMP_NUM_THREADS", "1")
    result = subprocess.run(command, cwd=directory, env=environment, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
    (directory/"stdout.txt").write_text(result.stdout)
    if result.returncode:
        raise RuntimeError(f"{case['name']} ({ranks} ranks) failed:\n{result.stdout[-6000:]}")
    return parse_dump(directory/"forces.dump")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lmp", required=True, type=Path)
    parser.add_argument("--plugin", required=True, type=Path)
    parser.add_argument("--mpiexec", help="MPI launcher, optionally with flags; e.g. 'mpiexec --oversubscribe'")
    parser.add_argument("--output", type=Path, help="retain inputs/logs in a new directory")
    args = parser.parse_args()
    args.lmp, args.plugin = args.lmp.resolve(), args.plugin.resolve()
    for path in (args.lmp, args.plugin):
        if not path.is_file():
            parser.error(f"File does not exist: {path}")
    mpiexec = shlex.split(args.mpiexec) if args.mpiexec else []
    temporary = None
    if args.output:
        root = args.output.resolve()
        root.mkdir(parents=True, exist_ok=False)
    else:
        temporary = tempfile.TemporaryDirectory(prefix="slurry-pair-check-")
        root = Path(temporary.name)
    cases = [
        make_case("sphere-normal", 10e-9, velocity=(-1e-5, 0., 0.)),
        make_case("sphere-tangent", 10e-9, velocity=(0., 1e-5, 0.)),
        make_case("sphere-spin", 10e-9, omega=(0., 0., 10.)),
        make_case("one-nm-gap", 1e-9, velocity=(-1e-5, 0., 0.)),
        make_case("one-nm-static", 1e-9),
        make_case("one-nm-overlap", -1e-9),
        make_case("outside-cutoff", 60e-9, velocity=(-1e-5, 0., 0.)),
        make_case("oblate-pole", 10e-9, normal=(0., 0., 1.), velocity=(1e-5, 0., -1e-5),
                  axes=(1.65e-6, 1.65e-6, .825e-6)),
        make_case("shear-periodic-image", 10e-9, normal=(0., 1., 0.), shear=100.),
    ]
    try:
        for case in cases:
            expected = analytical(case)
            serial = run_case(root, case, args.lmp, args.plugin, 1, mpiexec)
            compare(serial, expected, case["name"])
            if mpiexec:
                parallel = run_case(root, case, args.lmp, args.plugin, 2, mpiexec)
                compare(parallel, expected, case["name"]+" MPI analytical")
                compare(parallel, serial, case["name"]+" MPI consistency")
            print(f"PASS: {case['name']}")
        restart = run_case(root, cases[0], args.lmp, args.plugin, 1, mpiexec, restart=True)
        compare(restart, analytical(cases[0]), "restart")
        print("PASS: restart settings/coefficients and forces")
        print("PASS: pair forces/torques; MPI checks " + ("enabled" if mpiexec else "not requested"))
    finally:
        if temporary:
            temporary.cleanup()


if __name__ == "__main__":
    main()
