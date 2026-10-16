#!/usr/bin/env python3
"""Validate the overdamped slurry/implicit integrator through LAMMPS.

Uses Python's standard library only. Every run, including one rank, is launched
through MPI. The checks distinguish numerical stability from time accuracy:

* an isolated oblate spheroid follows affine translation and the exact Jeffery
  director solution, with a half-timestep convergence check;
* a 1 nm face-to-face overlap relaxes at dt=1 microsecond without overshoot;
* a close sphere pair in the box interior is equivalent to its translated copy
  spanning the Lees--Edwards y boundary (including image velocities);
* repulsion pushes one particle through yhi, checking box-update and periodic
  migration timing against an equivalent interior trajectory;
* the same inputs give matching results with one and two MPI ranks.

These tests do not validate a total suspension stress or viscosity model.

Example from the repository root:
  python3 lammps_slurry/tests/validate_implicit.py \
    --lmp /path/to/lmp --plugin build/lammps-slurry/libslurry.so \
    --mpiexec mpirun --output build/validation/implicit
"""

import argparse
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess


ETA = 0.01
HCUT = 7.5e-9
HMIN = 3e-9
KN = 1.0
DENSITY = 2400.0
BOX = 10e-6
SHEAR = 100.0
DT = 1e-6
OBLATE = (300e-9, 300e-9, 150e-9)
SPHERE = (150e-9, 150e-9, 150e-9)
P0 = (0.6, 0.3, math.sqrt(0.55))


def formatted(values):
    return " ".join(f"{x:.17g}" for x in values)


def quaternion_for_director(p):
    """Shortest rotation taking the local z axis to p; LAMMPS order w,x,y,z."""
    w = math.sqrt((1.0 + p[2]) / 2.0)
    return (w, -p[1] / (2.0*w), p[0] / (2.0*w), 0.0)


def director(q):
    w, x, y, z = q
    return (2*(x*z+w*y), 2*(y*z-w*x), 1-2*(x*x+y*y))


def exact_jeffery(time):
    """Matrix-exponential solution of q'=(Omega+lambda E)q, p=q/|q|."""
    a, _, c = OBLATE
    lam = (c*c-a*a)/(c*c+a*a)
    upper = SHEAR*(1+lam)/2
    lower = SHEAR*(lam-1)/2
    frequency = math.sqrt(-upper*lower)
    cosine, sine = math.cos(frequency*time), math.sin(frequency*time)
    raw = (cosine*P0[0] + upper/frequency*sine*P0[1],
           cosine*P0[1] + lower/frequency*sine*P0[0], P0[2])
    length = math.sqrt(sum(x*x for x in raw))
    return tuple(x/length for x in raw)


def particle(position, axes=OBLATE, quaternion=(1.0, 0.0, 0.0, 0.0)):
    return dict(position=position, axes=axes, quaternion=quaternion)


def cases():
    isolated = [particle((1e-6, 0.8e-6, -0.3e-6), quaternion=quaternion_for_director(P0))]
    face_distance = 2*OBLATE[2] - 1e-9
    overlap = [particle((0.0, 0.0, -face_distance/2)),
               particle((0.0, 0.0, face_distance/2))]
    sphere_distance = 2*SPHERE[0] + 3e-9
    interior = [particle((0.0, -sphere_distance/2, 0.0), SPHERE),
                particle((0.0, sphere_distance/2, 0.0), SPHERE)]
    boundary = [particle((0.0, BOX/2-sphere_distance/2, 0.0), SPHERE),
                particle((0.0, -BOX/2+sphere_distance/2, 0.0), SPHERE)]
    crossing_distance = 2*SPHERE[0] - 1e-9
    translation = BOX/2-crossing_distance/2-0.1e-9
    crossing_interior = [particle((0.0, -crossing_distance/2, 0.0), SPHERE),
                         particle((0.0, crossing_distance/2, 0.0), SPHERE)]
    crossing_boundary = [particle((0.0, -crossing_distance/2+translation, 0.0), SPHERE),
                         particle((0.0, crossing_distance/2+translation, 0.0), SPHERE)]
    return [
        dict(name="isolated-dt1us", particles=isolated, shear=SHEAR, dt=DT,
             steps=10000, interval=1000),
        dict(name="isolated-dt0p5us", particles=isolated, shear=SHEAR, dt=DT/2,
             steps=20000, interval=2000),
        dict(name="face-overlap-dt1us", particles=overlap, shear=0.0, dt=DT,
             steps=100, interval=1),
        dict(name="le-interior-dt1us", particles=interior, shear=SHEAR, dt=DT,
             steps=500, interval=50),
        dict(name="le-boundary-dt1us", particles=boundary, shear=SHEAR, dt=DT,
             steps=500, interval=50),
        dict(name="le-crossing-interior-dt1us", particles=crossing_interior,
             shear=SHEAR, dt=DT, steps=100, interval=1),
        dict(name="le-crossing-boundary-dt1us", particles=crossing_boundary,
             shear=SHEAR, dt=DT, steps=100, interval=1, translation=translation),
    ]


def write_data(path, case):
    count = len(case["particles"])
    lines = ["Independent implicit-integrator verification; SI", "", f"{count} atoms",
             f"{count} ellipsoids", "1 atom types", ""]
    for axis in "xyz":
        lines.append(f"{-BOX/2:.17g} {BOX/2:.17g} {axis}lo {axis}hi")
    lines += ["0 0 0 xy xz yz", "", "Atoms # ellipsoid", ""]
    for number, item in enumerate(case["particles"], 1):
        lines.append(f"{number} 1 1 {DENSITY:.17g} {formatted(item['position'])}")
    lines += ["", "Ellipsoids", ""]
    for number, item in enumerate(case["particles"], 1):
        values = tuple(2*x for x in item["axes"]) + item["quaternion"]
        lines.append(f"{number} {formatted(values)}")
    lines += ["", "Velocities", ""]
    for number, item in enumerate(case["particles"], 1):
        # The overdamped solve sets velocities; initial affine values are benign.
        velocity = case["shear"]*(item["position"][1]+BOX/2)
        lines.append(f"{number} {velocity:.17g} 0 0 0 0 0")
    path.write_text("\n".join(lines)+"\n", encoding="ascii")


def parse_dump(path):
    lines = iter(path.read_text(encoding="ascii").splitlines())
    result = {}
    for line in lines:
        if line != "ITEM: TIMESTEP":
            raise ValueError(f"Unexpected dump line: {line}")
        step = int(next(lines))
        if next(lines) != "ITEM: NUMBER OF ATOMS":
            raise ValueError("Missing atom count")
        count = int(next(lines))
        if not next(lines).startswith("ITEM: BOX BOUNDS"):
            raise ValueError("Missing box bounds")
        bounds = [tuple(map(float, next(lines).split())) for _ in range(3)]
        names = next(lines).split()[2:]
        atoms = {}
        for _ in range(count):
            row = dict(zip(names, map(float, next(lines).split())))
            atoms[int(row.pop("id"))] = row
        result[step] = dict(atoms=atoms, xy=bounds[0][2])
    return result


def parse_diagnostics(path):
    result = []
    columns = None
    for line in path.read_text(encoding="ascii", errors="replace").splitlines():
        words = line.split()
        if words and words[0] == "Step" and "f_integrate[4]" in words:
            columns = words
        elif columns and len(words) == len(columns):
            try:
                values = list(map(float, words))
            except ValueError:
                continue
            result.append(dict(zip(columns, values)))
    if not result:
        raise AssertionError("No implicit-solver diagnostics were written")
    return result


def run_case(root, case, args, ranks):
    directory = root/f"{case['name']}-{ranks}rank"
    directory.mkdir()
    write_data(directory/"particles.data", case)
    cutoff = 2*max(max(p["axes"]) for p in case["particles"]) + HCUT
    text = [f'plugin load "{args.plugin}"', "units si", "atom_style ellipsoid",
            "boundary p p p", "atom_modify map array", f"processors 1 {ranks} 1",
            "newton on", "read_data particles.data", "comm_modify vel yes",
            f"pair_style zero {cutoff:.17g}", "pair_coeff * *",
            "neighbor 3e-8 bin", "neigh_modify every 1 delay 0 check yes"]
    if case["shear"]:
        text.append(f"fix shear all deform 0 xy erate {case['shear']:.17g} remap v units box")
    text += [f"fix integrate all slurry/implicit {ETA:.17g} {case['shear']:.17g} "
             f"{HCUT:.17g} {HMIN:.17g} {KN:.17g}",
             f"timestep {case['dt']:.17g}",
             "compute orientation all property/atom quatw quati quatj quatk",
             f"dump trajectory all custom {case['interval']} trajectory.dump "
             "id xu yu zu vx vy vz ix iy iz c_orientation[*]",
             "dump_modify trajectory sort id format float %.17g",
             f"thermo {case['interval']}",
             "thermo_style custom step time atoms f_integrate[1] f_integrate[2] "
             "f_integrate[3] f_integrate[4] f_integrate[5] f_integrate[6]",
             "thermo_modify flush yes lost error format float %.17g",
             f"run {case['steps']}"]
    (directory/"in.check").write_text("\n".join(text)+"\n", encoding="ascii")
    command = [*args.mpiexec, "-n", str(ranks), str(args.lmp), "-log", "log.lammps",
               "-in", "in.check"]
    environment = os.environ.copy()
    environment.setdefault("OMP_NUM_THREADS", "1")
    with (directory/"stdout.txt").open("w") as stream:
        process = subprocess.run(command, cwd=directory, env=environment,
                                 stdout=stream, stderr=subprocess.STDOUT,
                                 timeout=args.timeout, check=False)
    if process.returncode:
        tail = (directory/"stdout.txt").read_text(errors="replace")[-6000:]
        raise RuntimeError(f"{case['name']} ({ranks} ranks) failed:\n{tail}")
    frames = parse_dump(directory/"trajectory.dump")
    diagnostics = parse_diagnostics(directory/"log.lammps")
    expected_steps = set(range(0, case["steps"]+1, case["interval"]))
    if set(frames) != expected_steps:
        raise AssertionError(f"{case['name']}: missing trajectory snapshots")
    for frame in frames.values():
        if set(frame["atoms"]) != set(range(1, len(case["particles"])+1)):
            raise AssertionError(f"{case['name']}: lost or repeated atom IDs")
        for atom in frame["atoms"].values():
            if not all(math.isfinite(x) for x in atom.values()):
                raise AssertionError(f"{case['name']}: nonfinite trajectory")
            q = tuple(atom[f"c_orientation[{i}]"] for i in range(1, 5))
            if abs(sum(x*x for x in q)-1) > 2e-10:
                raise AssertionError(f"{case['name']}: nonunit quaternion {q}")
    max_residual = max(row["f_integrate[4]"] for row in diagnostics)
    if not math.isfinite(max_residual) or max_residual > 1.01e-10:
        raise AssertionError(f"{case['name']}: solver relative residual {max_residual}")
    return dict(frames=frames, diagnostics=diagnostics, max_residual=max_residual)


def vector_error(actual, expected):
    return math.sqrt(sum((x-y)**2 for x, y in zip(actual, expected)))


def check_isolated(case, result):
    errors = []
    x0, y0, z0 = case["particles"][0]["position"]
    for step, frame in result["frames"].items():
        time = step*case["dt"]
        atom = frame["atoms"][1]
        actual = tuple(atom[x] for x in ("xu", "yu", "zu"))
        # The plugin's background profile is u_x=gamma*(y-ylo).
        affine_velocity = SHEAR*(y0+BOX/2)
        expected = (x0+affine_velocity*time, y0, z0)
        if vector_error(actual, expected) > 2e-15:
            raise AssertionError(f"{case['name']}: affine translation mismatch {actual}, {expected}")
        if vector_error(tuple(atom[x] for x in ("vx", "vy", "vz")),
                        (affine_velocity, 0.0, 0.0)) > 1e-12:
            raise AssertionError(f"{case['name']}: affine velocity mismatch")
        q = tuple(atom[f"c_orientation[{i}]"] for i in range(1, 5))
        errors.append(vector_error(director(q), exact_jeffery(time)))
    if max(errors) > 2e-4:
        raise AssertionError(f"{case['name']}: Jeffery director error {max(errors)}")
    return errors[-1]


def check_overlap(result):
    overlaps = []
    for step, frame in result["frames"].items():
        first, second = frame["atoms"][1], frame["atoms"][2]
        gap = second["zu"]-first["zu"]-2*OBLATE[2]
        overlaps.append(max(0.0, -gap))
        if gap > 1e-10:
            raise AssertionError(f"Overlapping pair overshot contact at step {step}: gap={gap}")
        for atom in (first, second):
            if max(abs(atom[k]) for k in ("xu", "yu")) > 1e-16:
                raise AssertionError("Symmetric face pair moved laterally")
        if abs(first["zu"]+second["zu"]) > 1e-16:
            raise AssertionError("Symmetric pair centre drifted")
    if abs(overlaps[0]-1e-9) > 1e-16:
        raise AssertionError("Overlap fixture is not initially 1 nm")
    if any(right > left+1e-15 for left, right in zip(overlaps, overlaps[1:])):
        raise AssertionError("Overlap increased during zero-shear relaxation")
    if overlaps[-1] > 1e-12:
        raise AssertionError(f"Overlap did not relax: remaining {overlaps[-1]:.17g} m")
    if max(row["f_integrate[1]"] for row in result["diagnostics"]) < 1:
        raise AssertionError("Overlap fixture did not exercise lubrication")
    return overlaps[-1]


def assert_close(actual, target, field, label):
    if field in ("xu", "yu", "zu"):
        tolerance = 2e-15
    elif field in ("vx", "vy", "vz"):
        tolerance = 2e-12 + 1e-8*abs(target)
    else:
        tolerance = 5e-9
    if abs(actual-target) > tolerance:
        raise AssertionError(f"{label} {field}: {actual:.17g}, expected {target:.17g}")


def check_le_equivalence(interior, boundary, translation=BOX/2,
                         initial_images=(0, 1), require_crossing=False):
    maximum_position_error = 0.0
    for step, frame in interior["frames"].items():
        time = step*DT
        translated = boundary["frames"][step]
        for number, reference in frame["atoms"].items():
            actual = dict(translated["atoms"][number])
            # xu/yu already include dynamic image crossings. Initial images
            # omitted by constructing the data file must be restored separately.
            initial_image = initial_images[number-1]
            actual["xu"] += initial_image*translated["xy"] - SHEAR*translation*time
            actual["yu"] += initial_image*BOX - translation
            actual["vx"] += (initial_image+actual["iy"]-reference["iy"])*SHEAR*BOX \
                            - SHEAR*translation
            for field, target in reference.items():
                if field in ("ix", "iy", "iz"):
                    continue
                assert_close(actual[field], target, field, f"LE step {step} atom {number}")
            maximum_position_error = max(maximum_position_error,
                vector_error(tuple(actual[k] for k in ("xu", "yu", "zu")),
                             tuple(reference[k] for k in ("xu", "yu", "zu"))))
    if max(row["f_integrate[1]"] for row in boundary["diagnostics"]) < 1:
        raise AssertionError("LE fixture did not exercise an active lubrication pair")
    if require_crossing:
        upper_images = [frame["atoms"][2]["iy"] for frame in boundary["frames"].values()]
        if upper_images[0] != 0 or max(upper_images) < 1:
            raise AssertionError("LE crossing fixture did not cross yhi")
    return maximum_position_error


def check_mpi(serial, parallel, name):
    for step, frame in serial["frames"].items():
        for number, atom in frame["atoms"].items():
            observed = parallel["frames"][step]["atoms"][number]
            for field, target in atom.items():
                assert_close(observed[field], target, field,
                             f"MPI {name} step {step} atom {number}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lmp", type=Path, required=True)
    parser.add_argument("--plugin", type=Path, required=True)
    parser.add_argument("--mpiexec", required=True, help="MPI launcher, optionally including flags")
    parser.add_argument("--ranks", default="1,2", help="comma-separated rank counts; must include 1")
    parser.add_argument("--output", type=Path, required=True, help="new directory for retained checks")
    parser.add_argument("--timeout", type=float, default=120.0, help="seconds per LAMMPS invocation")
    args = parser.parse_args()
    args.lmp = Path(shutil.which(str(args.lmp)) or args.lmp).resolve()
    args.plugin = args.plugin.resolve()
    args.mpiexec = shlex.split(args.mpiexec)
    ranks = sorted(set(map(int, args.ranks.split(","))))
    if not ranks or ranks[0] != 1:
        parser.error("--ranks must contain 1 and only positive integers")
    if not args.mpiexec:
        parser.error("--mpiexec must identify a launcher")
    for path in (args.lmp, args.plugin):
        if not path.is_file():
            parser.error(f"File not found: {path}")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    all_cases = cases()
    results = {}
    for case in all_cases:
        results[case["name"]] = {}
        for count in ranks:
            result = run_case(root, case, args, count)
            results[case["name"]][count] = result
            if count != 1:
                check_mpi(results[case["name"]][1], result, case["name"])
            print(f"PASS: {case['name']} ({count} MPI rank(s)); "
                  f"max solver residual={result['max_residual']:.3g}", flush=True)

    report = {"dt_s": DT, "ranks": ranks, "checks": {}}
    for count in ranks:
        coarse = check_isolated(all_cases[0], results[all_cases[0]["name"]][count])
        fine = check_isolated(all_cases[1], results[all_cases[1]["name"]][count])
        if fine > 0.75*coarse + 1e-8:
            raise AssertionError(f"Jeffery half-step did not improve: {coarse} -> {fine}")
        overlap = check_overlap(results["face-overlap-dt1us"][count])
        le_error = check_le_equivalence(results["le-interior-dt1us"][count],
                                        results["le-boundary-dt1us"][count])
        crossing_error = check_le_equivalence(results["le-crossing-interior-dt1us"][count],
            results["le-crossing-boundary-dt1us"][count],
            translation=all_cases[-1]["translation"], initial_images=(0, 0),
            require_crossing=True)
        report["checks"][str(count)] = dict(jeffery_error_dt1us=coarse,
            jeffery_error_dt0p5us=fine, final_overlap_m=overlap,
            le_maximum_position_error_m=le_error,
            le_crossing_maximum_position_error_m=crossing_error)
        print(f"PASS: physics ({count} MPI rank(s)): Jeffery error "
              f"{coarse:.6g} -> {fine:.6g}; final overlap={overlap:.6g} m; "
              f"LE error={le_error:.6g} m; crossing error={crossing_error:.6g} m", flush=True)
    (root/"summary.json").write_text(json.dumps(report, indent=2)+"\n")
    print(f"PASS: implicit integration at dt=1e-6 s; report: {root/'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
