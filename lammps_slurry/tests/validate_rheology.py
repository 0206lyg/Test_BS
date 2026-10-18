#!/usr/bin/env python3
"""Independent LAMMPS checks for slurry rheology, soft contact and history.

Requires the companion validate_implicit.py, Python's standard library, and
the built plugin. Every invocation uses an MPI launcher, including one rank.
This suite checks isolated-sphere Einstein stress, analytic soft-contact
relaxation, frictional contact crossing the Lees--Edwards boundary, repeated
run/setup history, and agreement between MPI decompositions. The C++ core
tests separately exercise prescribed stick/slip/release and history rotation.

  python3 lammps_slurry/tests/validate_rheology.py --lmp /path/lmp \
    --plugin build/lammps-slurry/libslurry.so --mpiexec mpirun \
    --output build/validation/rheology
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess

import validate_implicit as base
from check_drag import analytic_oblate_resistance


CONTACT_GAP = 3e-9
MU = 0.1
KT = 80.0
STRESS_COMPONENTS = (
    "sigma_solvent_Pa", "sigma_single_free_Pa", "sigma_single_rotation_Pa",
    "sigma_lub_normal_Pa", "sigma_lub_tangent_Pa", "sigma_contact_normal_Pa",
    "sigma_contact_tangent_Pa",
)
REQUIRED = (*STRESS_COMPONENTS, "sigma_xy_Pa", "eta_Pa_s", "eta_relative",
            "contacts", "sticking", "sliding", "lubrication_pairs",
            "max_contact_compression_m", "max_overlap_m", "force_residual",
            "coulomb_violation_N")


def cases():
    result = [dict(name="sphere-einstein", particles=[base.particle(
        (0.0, 0.8e-6, 0.0), base.SPHERE)], shear=base.SHEAR,
        dt=base.DT, steps=10, runs=[10])]
    for name, gap in (("outside-contact", 4e-9), ("at-contact", 3e-9),
                      ("soft-contact", 2e-9), ("geometric-overlap", -1e-9)):
        distance = 2*base.SPHERE[0]+gap
        result.append(dict(name=name, particles=[
            base.particle((0.0, 0.0, -distance/2), base.SPHERE),
            base.particle((0.0, 0.0, distance/2), base.SPHERE)],
            shear=0.0, dt=base.DT, steps=20, runs=[20], initial_gap=gap))
    distance = 2*base.SPHERE[0]+2e-9
    translation = base.BOX/2-distance/2-0.1e-9
    interior = [base.particle((0.0, -distance/2, 0.0), base.SPHERE),
                base.particle((0.0, distance/2, 0.0), base.SPHERE)]
    boundary = [base.particle((0.0, -distance/2+translation, 0.0), base.SPHERE),
                base.particle((0.0, distance/2+translation, 0.0), base.SPHERE)]
    result += [
        dict(name="friction-interior", particles=interior, shear=base.SHEAR,
             dt=base.DT, steps=100, runs=[100]),
        dict(name="friction-crossing", particles=boundary, shear=base.SHEAR,
             dt=base.DT, steps=100, runs=[100], translation=translation),
        dict(name="friction-segmented", particles=interior, shear=base.SHEAR,
             dt=base.DT, steps=100, runs=[2, 0, 1, 0, 7, 0, 90]),
        dict(name="oblate-stress", particles=[base.particle(
             (0.0, 0.8e-6, 0.0), quaternion=base.quaternion_for_director(base.P0))],
             shear=base.SHEAR, dt=base.DT, steps=20, runs=[20]),
        dict(name="oblate-friction", particles=[base.particle(
             tuple(sign*(2*base.OBLATE[2]+2e-9)*x/2 for x in base.P0),
             quaternion=base.quaternion_for_director(base.P0)) for sign in (-1, 1)],
             shear=base.SHEAR, dt=base.DT, steps=100, runs=[100]),
    ]
    return result


def parse_history(path):
    with path.open(newline="") as stream:
        reader = csv.DictReader(line for line in stream if not line.startswith("#"))
        if reader.fieldnames is None:
            raise AssertionError(f"Missing CSV header in {path}")
        missing = set(REQUIRED)-set(reader.fieldnames)
        if missing:
            raise AssertionError(f"Missing rheology columns: {sorted(missing)}")
        rows = [{key: float(value) for key, value in row.items()} for row in reader]
    if not rows:
        raise AssertionError(f"Empty rheology output: {path}")
    steps = [int(row["step"]) for row in rows]
    if len(set(steps)) != len(steps):
        raise AssertionError("Repeated physical timestep in rheology CSV")
    return {int(row["step"]): row for row in rows}


def run_case(root, case, args, ranks):
    directory = root/f"{case['name']}-{ranks}rank"
    directory.mkdir()
    base.write_data(directory/"particles.data", case)
    cutoff = 2*max(max(p["axes"]) for p in case["particles"])+base.HCUT
    lines = ['plugin load ${slurry_plugin}', "units si", "atom_style ellipsoid",
             "boundary p p p", "atom_modify map array", f"processors 1 {ranks} 1",
             "newton on", "read_data particles.data", "comm_modify vel yes",
             f"pair_style zero {cutoff:.17g}", "pair_coeff * *",
             "neighbor 3e-8 bin", "neigh_modify every 1 delay 0 check yes"]
    if case["shear"]:
        lines.append(f"fix shear all deform 0 xy erate {case['shear']:.17g} remap v units box")
    lines += [f"fix integrate all slurry/implicit {base.ETA:.17g} {case['shear']:.17g} "
              f"{base.HCUT:.17g} {base.HMIN:.17g} {base.KN:.17g} "
              f"contact_gap {CONTACT_GAP:.17g} friction {MU:.17g} {KT:.17g} "
              "history history.csv 1 0",
              f"timestep {case['dt']:.17g}",
              "compute orientation all property/atom quatw quati quatj quatk",
              "compute momentum all property/atom angmomx angmomy angmomz",
              "dump trajectory all custom 1 trajectory.dump "
              "id xu yu zu vx vy vz ix iy iz c_orientation[*] c_momentum[*]",
              "dump_modify trajectory sort id format float %.17g",
              "thermo 1",
              "thermo_style custom step time atoms f_integrate[4] f_integrate[9] "
              "f_integrate[11] f_integrate[12] f_integrate[13] f_integrate[16]",
              "thermo_modify lost error flush yes format float %.17g"]
    lines += [f"run {steps}" for steps in case["runs"]]
    (directory/"in.check").write_text("\n".join(lines)+"\n")
    reproduce = (f"# mpirun -n {ranks} /path/lmp -var slurry_plugin /path/libslurry.so "
                 "-in in.reproduce\n")
    (directory/"in.reproduce").write_text(reproduce+"\n".join(lines)+"\n")
    metadata = dict(case=case["name"], dt_s=case["dt"], steps=case["steps"],
        run_segments=case["runs"], shear_rate_s_inv=case["shear"], mpi_ranks=ranks,
        contact_gap_m=CONTACT_GAP, friction_mu=MU, tangent_stiffness_N_m=KT,
        plugin_sha256=hashlib.sha256(args.plugin.read_bytes()).hexdigest(),
        input_data_sha256=hashlib.sha256((directory/"particles.data").read_bytes()).hexdigest(),
        started_utc=datetime.now(timezone.utc).isoformat())
    (directory/"check_parameters.json").write_text(json.dumps(metadata, indent=2)+"\n")
    command = [*args.mpiexec, "-n", str(ranks), str(args.lmp),
               "-var", "slurry_plugin", str(args.plugin),
               "-in", "in.check", "-log", "log.lammps"]
    environment = os.environ.copy()
    environment.setdefault("OMP_NUM_THREADS", "1")
    with (directory/"stdout.txt").open("w") as output:
        process = subprocess.run(command, cwd=directory, env=environment,
            stdout=output, stderr=subprocess.STDOUT, timeout=args.timeout, check=False)
    if process.returncode:
        tail = (directory/"stdout.txt").read_text(errors="replace")[-6000:]
        raise RuntimeError(f"{case['name']} ({ranks} MPI ranks) failed:\n{tail}")
    frames = base.parse_dump(directory/"trajectory.dump")
    history = parse_history(directory/"history.csv")
    if set(frames) != set(range(case["steps"]+1)):
        raise AssertionError(f"{case['name']}: incomplete trajectory")
    if case["steps"] not in history:
        raise AssertionError(f"{case['name']}: missing final rheology record")
    for frame in frames.values():
        if len(frame["atoms"]) != len(case["particles"]):
            raise AssertionError(f"{case['name']}: particle count changed")
        for atom in frame["atoms"].values():
            if not all(math.isfinite(value) for value in atom.values()):
                raise AssertionError(f"{case['name']}: nonfinite particle state")
            q = [atom[f"c_orientation[{i}]"] for i in range(1, 5)]
            if abs(sum(x*x for x in q)-1)>2e-10:
                raise AssertionError(f"{case['name']}: nonunit quaternion")
    for step, row in history.items():
        for key in (*STRESS_COMPONENTS, "sigma_xy_Pa", "force_residual",
                    "max_contact_compression_m", "max_overlap_m", "coulomb_violation_N"):
            if not math.isfinite(row[key]):
                raise AssertionError(f"{case['name']}, step {step}: nonfinite {key}")
        assert_number(row["sigma_xy_Pa"], sum(row[k] for k in STRESS_COMPONENTS),
                      f"{case['name']} stress sum", relative=1e-9, absolute=1e-12)
        check_force_residual(row, case["name"])
        if row["coulomb_violation_N"]>1e-18:
            raise AssertionError(f"{case['name']}: Coulomb bound violated")
        if case["shear"]:
            assert_number(row["eta_Pa_s"], row["sigma_xy_Pa"]/case["shear"],
                          f"{case['name']} viscosity", relative=1e-10, absolute=1e-14)
            assert_number(row["eta_relative"], row["eta_Pa_s"]/base.ETA,
                          f"{case['name']} relative viscosity", relative=1e-10, absolute=1e-12)
    return dict(frames=frames, history=history, energy=check_energy(history, case["name"]))


def assert_number(actual, expected, label, relative=1e-8, absolute=1e-14):
    if not math.isfinite(actual) or abs(actual-expected)>absolute+relative*abs(expected):
        raise AssertionError(f"{label}: got {actual:.17g}, expected {expected:.17g}")


def check_force_residual(row, label):
    """Check the actual solver criterion, including zero-force relaxation tails.

    Every runtime fixture writes every step, so this checks all accepted states.
    A purely relative ratio is not meaningful when the force scale vanishes;
    the solver has always used absolute + relative force tolerances. Older CSVs
    do not expose that absolute criterion, so retain their original check.
    """
    if "force_residual_N" in row and "force_tolerance_N" in row:
        residual, tolerance = row["force_residual_N"], row["force_tolerance_N"]
        if (not math.isfinite(residual) or not math.isfinite(tolerance)
                or residual < 0 or tolerance <= 0 or residual > (1+1e-12)*tolerance):
            raise AssertionError(f"{label}: force residual {residual} N exceeds {tolerance} N")
    else:
        for key, value in row.items():
            if "residual" in key and (not math.isfinite(value) or value > 1.01e-10):
                raise AssertionError(f"{label}: {key}={value}")


def check_energy(history, label):
    # Normalize by one run-wide scale: late relaxation tails may be many
    # orders below the initial power and should not set the denominator.
    powers = ("stress_power_W", "hydro_dissipation_W", "elastic_energy_rate_W",
              "sliding_power_W")
    dissipation = ("hydro_dissipation_W", "sliding_power_W",
                   "contact_discrete_dissipation_W")
    for row in history.values():
        for key in (*powers, *dissipation, "power_balance_error_W"):
            if key not in row or not math.isfinite(row[key]):
                raise AssertionError(f"{label}: missing/nonfinite energy field {key}")
    scale = max(abs(row[key]) for row in history.values() for key in powers)
    maximum_error = max(abs(row["power_balance_error_W"]) for row in history.values())
    relative = maximum_error/scale if scale else (0.0 if maximum_error==0 else math.inf)
    if relative>=1e-9:
        raise AssertionError(f"{label}: run-scaled power balance error {relative}")
    minima = {key:min(row[key] for row in history.values()) for key in dissipation}
    for key,value in minima.items():
        if value < -1e-10*scale:
            raise AssertionError(f"{label}: negative dissipative power {key}={value}")
    return dict(power_scale_W=scale, maximum_power_balance_error_W=maximum_error,
                maximum_scaled_power_error=relative, minimum_dissipative_powers_W=minima)


def compare_histories(first, second, label):
    if set(first)!=set(second):
        raise AssertionError(f"{label}: different CSV timesteps")
    for step, reference in first.items():
        other = second[step]
        for key, expected in reference.items():
            # Timings and solver bookkeeping are deliberately not physical
            # invariants under a change in MPI layout or preconditioner.
            if ("iterations" in key or "residual" in key or key=="kink_reselections"
                    or "profile_" in key or key.startswith("sparse_")
                    or key == "force_tolerance_N"):
                continue
            if not math.isfinite(expected) and not math.isfinite(other[key]):
                continue
            absolute = 1e-15 if key.endswith("_m") else 2e-12
            if key.endswith("_N"):
                absolute = 1e-18
            assert_number(other[key], expected, f"{label} step {step} {key}",
                          relative=2e-7, absolute=absolute)


def check_einstein(result):
    radius = base.SPHERE[0]
    phi = 4*math.pi*radius**3/(3*base.BOX**3)
    solvent = base.ETA*base.SHEAR
    particle = solvent*2.5*phi
    for row in result["history"].values():
        assert_number(row["sigma_solvent_Pa"], solvent, "Einstein solvent", absolute=1e-12)
        assert_number(row["sigma_single_free_Pa"], particle, "Einstein single stress",
                      relative=2e-8, absolute=1e-13)
        for key in STRESS_COMPONENTS[2:]:
            assert_number(row[key], 0.0, f"Einstein {key}", absolute=1e-13)
        assert_number(row["eta_relative"], 1+2.5*phi, "Einstein viscosity",
                      relative=1e-10, absolute=1e-12)
    return dict(phi=phi, expected_eta_relative=1+2.5*phi)


def check_oblate(result):
    # AR=2 torque-free stresslet eigenvalues / (eta*a^3), independently
    # reduced from Einarsson et al. Phys. Fluids 27, 063301 (2015), App. A.
    k0, k1, k2 = 10.011469147671463, 8.642608183478222, 13.682907462098928
    maximum_error = 0.0
    for step, row in result["history"].items():
        atom = result["frames"][step]["atoms"][1]
        p = base.director(tuple(atom[f"c_orientation[{i}]"] for i in range(1, 5)))
        ep = (base.SHEAR*p[1]/2, base.SHEAR*p[0]/2, 0.0)
        scalar = sum(x*y for x,y in zip(p, ep))
        q = tuple(x-scalar*y for x,y in zip(ep, p))
        e0 = 1.5*scalar*p[0]*p[1]
        e1 = p[0]*q[1]+q[0]*p[1]
        e2 = base.SHEAR/2-e0-e1
        expected = base.ETA*base.OBLATE[0]**3*(k0*e0+k1*e1+k2*e2)/base.BOX**3
        maximum_error = max(maximum_error, abs(row["sigma_single_free_Pa"]-expected))
        assert_number(row["sigma_single_free_Pa"], expected, "oblate isolated stress",
                      relative=3e-8, absolute=1e-13)
        for key in STRESS_COMPONENTS[2:]:
            assert_number(row[key], 0.0, f"oblate isolated {key}", absolute=1e-12)
    return maximum_error


def check_rotation_stress(result):
    a, _, c = base.OBLATE
    mass = 4*math.pi*a*a*c*base.DENSITY/3
    inertia_a, inertia_c = mass*(a*a+c*c)/5, 2*mass*a*a/5
    _, _, drag_a, drag_c = analytic_oblate_resistance(a, c)
    lam = (c*c-a*a)/(c*c+a*a)
    maximum_error, maximum_stress = 0.0, 0.0
    for step, row in result["history"].items():
        expected = 0.0
        for atom in result["frames"][step]["atoms"].values():
            p = base.director(tuple(atom[f"c_orientation[{i}]"] for i in range(1, 5)))
            momentum = tuple(atom[f"c_momentum[{i}]"] for i in range(1, 4))
            projected = sum(x*y for x,y in zip(momentum, p))
            omega = tuple(l/inertia_a+(1/inertia_c-1/inertia_a)*projected*n
                          for l,n in zip(momentum, p))
            jeffery = (-lam*base.SHEAR*p[0]*p[2]/2,
                       lam*base.SHEAR*p[1]*p[2]/2,
                       -base.SHEAR/2+lam*base.SHEAR*(p[0]**2-p[1]**2)/2)
            slip = tuple(x-y for x,y in zip(omega, jeffery))
            projected = sum(x*y for x,y in zip(slip, p))
            torque = tuple(-base.ETA*(drag_a*w+(drag_c-drag_a)*projected*n)
                           for w,n in zip(slip, p))
            txp = (torque[1]*p[2]-torque[2]*p[1],
                   torque[2]*p[0]-torque[0]*p[2],
                   torque[0]*p[1]-torque[1]*p[0])
            expected += lam*(txp[0]*p[1]+p[0]*txp[1])/(2*base.BOX**3)
        actual = row["sigma_single_rotation_Pa"]
        maximum_error = max(maximum_error, abs(actual-expected))
        maximum_stress = max(maximum_stress, abs(actual))
        assert_number(actual, expected, f"oblate rotation stress step {step}",
                      relative=2e-7, absolute=2e-12)
    if maximum_stress<1e-10:
        raise AssertionError("Oblate friction fixture did not exercise rotation stress")
    return dict(maximum_rotation_stress_Pa=maximum_stress,
                maximum_rotation_stress_error_Pa=maximum_error)


def check_normal_relaxation(case, result):
    radius = base.SPHERE[0]
    zeta = 6*math.pi*base.ETA*radius
    # Independent sphere reduction of the principal-curvature lubrication law.
    resistance = 1.5*math.pi*base.ETA*radius**2*(1/base.HMIN-1/base.HCUT)
    effective_drag = zeta+2*resistance
    compression0 = max(CONTACT_GAP-case["initial_gap"], 0.0)
    ratio = effective_drag/(effective_drag+2*base.KN*case["dt"])
    max_error = 0.0
    for step, frame in result["frames"].items():
        first, second = frame["atoms"][1], frame["atoms"][2]
        gap = second["zu"]-first["zu"]-2*radius
        expected_gap = (CONTACT_GAP-compression0*ratio**step
                        if compression0 else case["initial_gap"])
        error = abs(gap-expected_gap)
        max_error = max(error, max_error)
        if error>2e-15:
            raise AssertionError(f"{case['name']} step {step}: gap {gap}, expected {expected_gap}")
        expected_speed = base.KN*compression0*ratio**step/(effective_drag+2*base.KN*case["dt"])
        assert_number(first["vz"], -expected_speed, f"{case['name']} normal velocity",
                      relative=2e-6, absolute=1e-11)
        assert_number(second["vz"], expected_speed, f"{case['name']} normal velocity",
                      relative=2e-6, absolute=1e-11)
        assert_number(first["zu"]+second["zu"], 0.0,
                      f"{case['name']} centre of mass", absolute=1e-16)
    return max_error


def check_crossing(interior, boundary, translation):
    maximum_error = 0.0
    crossed = False
    for step, frame in interior["frames"].items():
        other = boundary["frames"][step]
        crossed |= other["atoms"][2]["iy"]>=1
        for number, reference in frame["atoms"].items():
            actual = dict(other["atoms"][number])
            actual["xu"] -= base.SHEAR*translation*step*base.DT
            actual["yu"] -= translation
            actual["vx"] += (actual["iy"]-reference["iy"])*base.SHEAR*base.BOX \
                            - base.SHEAR*translation
            for key, expected in reference.items():
                if key not in ("ix", "iy", "iz"):
                    base.assert_close(actual[key], expected, key,
                                      f"friction LE step {step} atom {number}")
            maximum_error = max(maximum_error, base.vector_error(
                tuple(actual[k] for k in ("xu", "yu", "zu")),
                tuple(reference[k] for k in ("xu", "yu", "zu"))))
    if not crossed:
        raise AssertionError("Frictional LE fixture did not cross yhi")
    # A Galilean translation changes absolute speeds, but no stresses/contact
    # properties. Compare the remaining CSV observables including block means.
    first = {s: {k:v for k,v in r.items() if "speed" not in k}
             for s,r in interior["history"].items()}
    second = {s: {k:v for k,v in r.items() if "speed" not in k}
              for s,r in boundary["history"].items()}
    compare_histories(first, second, "friction LE history")
    if max(r["contacts"] for r in interior["history"].values())<1:
        raise AssertionError("Frictional fixture never activated contact")
    tangential_stress = max(abs(r["sigma_contact_tangent_Pa"])
                            for r in interior["history"].values())
    if tangential_stress<1e-10:
        raise AssertionError("Frictional fixture did not exercise tangential force")
    return dict(maximum_position_error_m=maximum_error,
                maximum_tangential_stress_Pa=tangential_stress,
                maximum_sticking=max(r["sticking"] for r in interior["history"].values()),
                maximum_sliding=max(r["sliding"] for r in interior["history"].values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lmp", type=Path, required=True)
    parser.add_argument("--plugin", type=Path, required=True)
    parser.add_argument("--mpiexec", required=True)
    parser.add_argument("--ranks", default="1,2")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()
    args.lmp = Path(shutil.which(str(args.lmp)) or args.lmp).resolve()
    args.plugin = args.plugin.resolve()
    args.mpiexec = shlex.split(args.mpiexec)
    ranks = sorted(set(map(int, args.ranks.split(","))))
    if not ranks or ranks[0]!=1:
        parser.error("Rank list must include 1 and only positive integers")
    if not args.mpiexec:
        parser.error("An MPI launcher is required")
    for path in (args.lmp, args.plugin):
        if not path.is_file():
            parser.error(f"File not found: {path}")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    inputs = cases()
    results = {}
    for case in inputs:
        results[case["name"]] = {}
        for rank in ranks:
            result = run_case(root, case, args, rank)
            results[case["name"]][rank] = result
            if rank!=1:
                base.check_mpi(results[case["name"]][1], result, case["name"])
                compare_histories(results[case["name"]][1]["history"], result["history"],
                                  f"MPI {case['name']}")
            print(f"PASS: {case['name']} ({rank} MPI ranks)", flush=True)
    report = {"dt_s": base.DT, "contact_gap_m":CONTACT_GAP, "mu":MU, "kt_N_m":KT,
              "ranks":ranks, "checks":{}, "energy":{
                  name:{str(rank):result["energy"] for rank,result in by_rank.items()}
                  for name,by_rank in results.items()}}
    for rank in ranks:
        summary = {"einstein": check_einstein(results["sphere-einstein"][rank])}
        summary["oblate_maximum_stress_error_Pa"] = check_oblate(results["oblate-stress"][rank])
        summary["oblate_contact_rotation"] = check_rotation_stress(results["oblate-friction"][rank])
        for case in inputs[1:5]:
            summary[case["name"]+"_maximum_gap_error_m"] = check_normal_relaxation(
                case, results[case["name"]][rank])
        summary["crossing"] = check_crossing(results["friction-interior"][rank],
            results["friction-crossing"][rank], inputs[6]["translation"])
        base.check_mpi(results["friction-interior"][rank],
                       results["friction-segmented"][rank], "run/setup segmentation")
        compare_histories(results["friction-interior"][rank]["history"],
                          results["friction-segmented"][rank]["history"],
                          "run/setup history segmentation")
        report["checks"][str(rank)] = summary
        print(f"PASS: rheology, contact, LE and run/setup history ({rank} MPI ranks)", flush=True)
    (root/"summary.json").write_text(json.dumps(report, indent=2)+"\n")
    print(f"PASS: report {root/'summary.json'}", flush=True)


if __name__=="__main__":
    main()
