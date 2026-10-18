#!/usr/bin/env python3
"""Small end-to-end checks of the spherical Mari lubrication backend.

Uses Python's standard library and the existing validation helpers. This is a
numerical/regression suite, not a dense-suspension reproduction of a published
viscosity curve. The independent C++ pair tests check prescribed squeeze,
shear, and counterrotation pumping coefficients.

  python3 lammps_slurry/tests/validate_mari.py --lmp /path/lmp \
    --plugin build/lammps-slurry/libslurry.so --mpiexec mpirun \
    --output build/validation/mari
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess

import validate_implicit as base
import validate_rheology as rheology


RADIUS = base.SPHERE[0]
HCUT = 0.5*RADIUS
EPSILON = 0.001*RADIUS
STRESS_COMPONENTS = (*rheology.STRESS_COMPONENTS[:5], "sigma_lub_pump_Pa",
                     *rheology.STRESS_COMPONENTS[5:])


def sphere_pair(gap, normal=(0.6, -0.8, 0.0), radius_ratio=1.0):
    distance = RADIUS*(1.0+radius_ratio)+gap
    return [base.particle(tuple(-distance*x/2 for x in normal), base.SPHERE),
            base.particle(tuple(distance*x/2 for x in normal),
                          (RADIUS*radius_ratio,)*3)]


def cases():
    result = [dict(name="sphere-einstein", particles=[base.particle(
        (0.0, 0.8e-6, 0.0), base.SPHERE)], shear=base.SHEAR,
        dt=base.DT, steps=5, runs=[5], contact_gap=0.0)]
    for name, gap in (("contact-gap0p3nm", 0.3e-9),
                      ("contact-gap2nm", 2e-9)):
        result.append(dict(name=name, particles=sphere_pair(gap, (0.0, 0.0, 1.0)),
            shear=0.0, dt=base.DT, steps=20, runs=[20],
            initial_gap=gap, contact_gap=rheology.CONTACT_GAP))
    for name, ratio, gap in (("equal-oblique", 1.0, 6e-9),
                             ("unequal-oblique", 1.4, 6e-9),
                             ("equal-below3nm", 1.0, 1.5e-9)):
        result.append(dict(name=name, particles=sphere_pair(gap, radius_ratio=ratio),
            shear=base.SHEAR, dt=base.DT, steps=200, runs=[200], contact_gap=0.0))
    # Three timesteps distinguish actual convergence from a small two-run
    # difference. No active elastic/contact force introduces a rate scale here.
    for name, dt, steps, shear in (("equal-half-dt", base.DT/2, 400, base.SHEAR),
            ("equal-quarter-dt", base.DT/4, 800, base.SHEAR),
            ("equal-shear10", base.DT*10, 200, base.SHEAR/10)):
        result.append(dict(name=name, particles=sphere_pair(6e-9), shear=shear,
            dt=dt, steps=steps, runs=[steps], contact_gap=0.0))
    distance = 2*RADIUS+2e-9
    translation = base.BOX/2-distance/2-0.1e-9
    interior = sphere_pair(2e-9, (0.0, 1.0, 0.0))
    boundary = [dict(p, position=(p["position"][0], p["position"][1]+translation,
                                  p["position"][2])) for p in interior]
    for name, particles, runs in (("friction-interior", interior, [100]),
            ("friction-crossing", boundary, [100]),
            ("friction-segmented", interior, [2, 0, 1, 0, 7, 0, 90])):
        result.append(dict(name=name, particles=particles, shear=base.SHEAR,
            dt=base.DT, steps=100, runs=runs, contact_gap=rheology.CONTACT_GAP,
            translation=translation if name=="friction-crossing" else 0.0))
    return result


def run_case(root, case, args, ranks):
    directory = root/f"{case['name']}-{ranks}rank"
    directory.mkdir()
    base.write_data(directory/"particles.data", case)
    cutoff = 2*max(max(p["axes"]) for p in case["particles"])+HCUT
    lines = ['plugin load ${slurry_plugin}', "units si", "atom_style ellipsoid",
             "boundary p p p", "atom_modify map array", f"processors 1 {ranks} 1",
             "newton on", "read_data particles.data", "comm_modify vel yes",
             f"pair_style zero {cutoff:.17g}", "pair_coeff * *",
             "neighbor 3e-8 bin", "neigh_modify every 1 delay 0 check yes"]
    if case["shear"]:
        lines.append(f"fix shear all deform 0 xy erate {case['shear']:.17g} remap v units box")
    lines += [f"fix integrate all slurry/implicit {base.ETA:.17g} {case['shear']:.17g} "
              f"{HCUT:.17g} {EPSILON:.17g} {base.KN:.17g} lubrication mari "
              f"contact_gap {case['contact_gap']:.17g} "
              f"friction {rheology.MU:.17g} {rheology.KT:.17g} history history.csv 1 0",
              f"timestep {case['dt']:.17g}",
              "compute orientation all property/atom quatw quati quatj quatk",
              "compute momentum all property/atom angmomx angmomy angmomz",
              "dump trajectory all custom 1 trajectory.dump "
              "id xu yu zu vx vy vz ix iy iz c_orientation[*] c_momentum[*]",
              "dump_modify trajectory sort id format float %.17g", "thermo 1",
              "thermo_style custom step time atoms f_integrate[4]",
              "thermo_modify lost error flush yes format float %.17g"]
    lines += [f"run {steps}" for steps in case["runs"]]
    (directory/"in.check").write_text("\n".join(lines)+"\n", encoding="ascii")
    (directory/"parameters.json").write_text(json.dumps(dict(case,
        eta_Pa_s=base.ETA, hcut_m=HCUT, epsilon_m=EPSILON, kn_N_m=base.KN,
        mu=rheology.MU, kt_N_m=rheology.KT, ranks=ranks,
        plugin_sha256=hashlib.sha256(args.plugin.read_bytes()).hexdigest()), indent=2)+"\n")
    command = [*args.mpiexec, "-n", str(ranks), str(args.lmp), "-var", "slurry_plugin",
               str(args.plugin), "-in", "in.check", "-log", "log.lammps"]
    environment = os.environ.copy()
    environment.setdefault("OMP_NUM_THREADS", "1")
    with (directory/"stdout.txt").open("w") as output:
        process = subprocess.run(command, cwd=directory, env=environment, stdout=output,
            stderr=subprocess.STDOUT, timeout=args.timeout, check=False)
    if process.returncode:
        tail = (directory/"stdout.txt").read_text(errors="replace")[-6000:]
        raise RuntimeError(f"{case['name']} ({ranks} ranks) failed:\n{tail}")
    frames = base.parse_dump(directory/"trajectory.dump")
    history = rheology.parse_history(directory/"history.csv")
    if set(frames)!=set(range(case["steps"]+1)) or case["steps"] not in history:
        raise AssertionError(f"{case['name']}: incomplete trajectory/history")
    for step, frame in frames.items():
        if set(frame["atoms"])!=set(range(1, len(case["particles"])+1)):
            raise AssertionError(f"{case['name']}: missing/duplicate particle ID")
        for atom in frame["atoms"].values():
            if not all(math.isfinite(x) for x in atom.values()):
                raise AssertionError(f"{case['name']}: nonfinite particle state at {step}")
            q = [atom[f"c_orientation[{i}]"] for i in range(1,5)]
            if abs(sum(x*x for x in q)-1)>2e-10:
                raise AssertionError(f"{case['name']}: nonunit quaternion")
    for step, row in history.items():
        for key in (*STRESS_COMPONENTS, "sigma_xy_Pa", "force_residual"):
            if key not in row or not math.isfinite(row[key]):
                raise AssertionError(f"{case['name']}: missing/nonfinite {key} at {step}")
        rheology.assert_number(row["sigma_xy_Pa"], sum(row[k] for k in STRESS_COMPONENTS),
            f"{case['name']} stress sum", relative=1e-9, absolute=1e-12)
        rheology.check_force_residual(row, case["name"])
        if row["coulomb_violation_N"]>1e-18:
            raise AssertionError(f"{case['name']}: Coulomb bound violation")
        if case["shear"]:
            rheology.assert_number(row["eta_Pa_s"], row["sigma_xy_Pa"]/case["shear"],
                f"{case['name']} viscosity", relative=1e-10, absolute=1e-14)
            rheology.assert_number(row["eta_relative"], row["eta_Pa_s"]/base.ETA,
                f"{case['name']} relative viscosity", relative=1e-10, absolute=1e-12)
        if case["contact_gap"]==0 and row["contacts"]!=0:
            raise AssertionError(f"{case['name']}: unexpected contact in hydro fixture")
    energy = rheology.check_energy(history, case["name"])
    mode_minima = {}
    for key in ("lub_squeeze_dissipation_W", "lub_shear_dissipation_W",
                "lub_pump_dissipation_W"):
        for row in history.values():
            if key not in row or not math.isfinite(row[key]):
                raise AssertionError(f"{case['name']}: missing/nonfinite {key}")
        mode_minima[key] = min(row[key] for row in history.values())
        if mode_minima[key] < -1e-10*energy["power_scale_W"]:
            raise AssertionError(f"{case['name']}: negative modal dissipation {key}")
    energy["minimum_lubrication_mode_powers_W"] = mode_minima
    return dict(frames=frames, history=history, energy=energy)


def check_normal_relaxation(case, result):
    """Independent analytic force balance at each frozen pair geometry.

    Rn=6*pi*eta*(a*b/(a+b))**2/(max(h,0)+epsilon). Equal spheres
    have zeta=6*pi*eta*a. The implicit spring force uses end-step
    compression, while the resistance is evaluated at the current gap.
    """
    zeta = 6*math.pi*base.ETA*RADIUS
    maximum_speed_error = 0.0
    maximum_step_error = 0.0
    inferred_initial = None
    for step, frame in result["frames"].items():
        first, second = frame["atoms"][1], frame["atoms"][2]
        gap = second["zu"]-first["zu"]-2*RADIUS
        resistance = 1.5*math.pi*base.ETA*RADIUS**2/(max(gap,0.0)+EPSILON)
        compression = max(case["contact_gap"]-gap, 0.0)
        expected = base.KN*compression/(zeta+2*resistance+2*base.KN*case["dt"])
        maximum_speed_error = max(maximum_speed_error, abs(second["vz"]-expected))
        rheology.assert_number(second["vz"], expected, f"{case['name']} normal velocity",
            relative=3e-6, absolute=2e-11)
        rheology.assert_number(first["vz"], -expected, f"{case['name']} opposite velocity",
            relative=3e-6, absolute=2e-11)
        if step==0:
            inferred_initial = (base.KN*compression/second["vz"]-zeta
                                -2*base.KN*case["dt"])/2
        if step<case["steps"]:
            next_frame = result["frames"][step+1]
            next_gap = next_frame["atoms"][2]["zu"]-next_frame["atoms"][1]["zu"]-2*RADIUS
            error = abs(next_gap-gap-2*case["dt"]*expected)
            maximum_step_error = max(maximum_step_error, error)
            if error>2e-15:
                raise AssertionError(f"{case['name']}: implicit relaxation step error {error}")
    return dict(maximum_speed_error_m_s=maximum_speed_error,
                maximum_gap_step_error_m=maximum_step_error,
                inferred_initial_resistance_N_s_m=inferred_initial)


def compare_rate_scaling(fast, slow, ratio):
    maximum_eta_difference = 0.0
    for step, frame in fast["frames"].items():
        for number, reference in frame["atoms"].items():
            actual = slow["frames"][step]["atoms"][number]
            for key in ("xu", "yu", "zu", *(f"c_orientation[{i}]" for i in range(1,5))):
                base.assert_close(actual[key], reference[key], key,
                                  f"rate scaling step {step} atom {number}")
            for key in ("vx", "vy", "vz", *(f"c_momentum[{i}]" for i in range(1,4))):
                base.assert_close(actual[key]*ratio, reference[key], key,
                                  f"rate scaling step {step} atom {number}")
    for step, row in fast["history"].items():
        other = slow["history"][step]
        for key in (*STRESS_COMPONENTS, "sigma_xy_Pa"):
            rheology.assert_number(other[key]*ratio, row[key], f"rate scaling {key}",
                                   relative=2e-7, absolute=2e-12)
        for key in ("eta_Pa_s", "eta_relative"):
            rheology.assert_number(other[key], row[key], f"rate scaling {key}",
                                   relative=2e-7, absolute=2e-12)
        for key in ("stress_power_W", "hydro_dissipation_W"):
            rheology.assert_number(other[key]*ratio**2, row[key], f"rate scaling {key}",
                                   relative=2e-7, absolute=1e-25)
        maximum_eta_difference = max(maximum_eta_difference,
                                    abs(other["eta_Pa_s"]-row["eta_Pa_s"]))
    return dict(shear_rate_ratio=ratio, maximum_viscosity_difference_Pa_s=maximum_eta_difference)


def check_dt(coarse, medium, fine):
    def state(result):
        frame = result["frames"][max(result["frames"])]
        return tuple(atom[key] for _, atom in sorted(frame["atoms"].items())
                     for key in ("xu", "yu", "zu"))
    coarse_error = base.vector_error(state(coarse), state(medium))
    fine_error = base.vector_error(state(medium), state(fine))
    if fine_error>0.75*coarse_error+1e-18:
        raise AssertionError(f"Timestep refinement did not converge: {coarse_error}, {fine_error}")
    if fine_error>1e-10:
        raise AssertionError(f"Pair trajectory refinement difference too large: {fine_error} m")
    eta_coarse = coarse["history"][max(coarse["history"])]["eta_Pa_s"]
    eta_medium = medium["history"][max(medium["history"])]["eta_Pa_s"]
    eta_fine = fine["history"][max(fine["history"])]["eta_Pa_s"]
    if abs(eta_fine-eta_medium)>0.75*abs(eta_medium-eta_coarse)+1e-13:
        raise AssertionError("Pair viscosity did not converge under timestep refinement")
    return dict(position_dt_vs_half_m=coarse_error, position_half_vs_quarter_m=fine_error,
                viscosity_dt_Pa_s=eta_coarse, viscosity_half_Pa_s=eta_medium,
                viscosity_quarter_Pa_s=eta_fine)


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
                rheology.compare_histories(results[case["name"]][1]["history"],
                    result["history"], f"MPI {case['name']}")
            print(f"PASS: {case['name']} ({rank} MPI ranks)", flush=True)
    report = dict(scope="Small spherical-pair numerical validation; not a dense Mari curve",
        epsilon_m=EPSILON, hcut_m=HCUT, radius_m=RADIUS, ranks=ranks,
        checks={}, energy={name:{str(rank):r["energy"] for rank,r in rs.items()}
                          for name,rs in results.items()})
    for rank in ranks:
        summary = {"einstein":rheology.check_einstein(results["sphere-einstein"][rank])}
        for row in results["sphere-einstein"][rank]["history"].values():
            rheology.assert_number(row["sigma_lub_pump_Pa"], 0.0,
                                   "Einstein pump stress", absolute=1e-13)
        for case in inputs[1:3]:
            summary[case["name"]] = check_normal_relaxation(case, results[case["name"]][rank])
        measured_ratio = (summary["contact-gap0p3nm"]["inferred_initial_resistance_N_s_m"]
                        /summary["contact-gap2nm"]["inferred_initial_resistance_N_s_m"])
        expected_ratio = (2e-9+EPSILON)/(0.3e-9+EPSILON)
        rheology.assert_number(measured_ratio, expected_ratio, "Sub-3nm lubrication increase",
                               relative=2e-5, absolute=1e-7)
        summary["sub3nm_resistance_ratio"] = dict(measured=measured_ratio, expected=expected_ratio)
        summary["dt_refinement"] = check_dt(results["equal-oblique"][rank],
            results["equal-half-dt"][rank], results["equal-quarter-dt"][rank])
        summary["shear_rate_scaling"] = compare_rate_scaling(results["equal-oblique"][rank],
            results["equal-shear10"][rank], 10.0)
        # Pure counterrotation has no direct uniform-strain derivative in
        # this modal split. It alters the solved rotations and the other
        # stress channels; its separate direct stress can correctly be zero.
        pump = max(row["lub_pump_dissipation_W"]
                   for row in results["unequal-oblique"][rank]["history"].values())
        if pump<1e-26:
            raise AssertionError("Unequal-sphere shear fixture did not exercise pumping")
        summary["maximum_unequal_pair_pump_dissipation_W"] = pump
        summary["crossing"] = rheology.check_crossing(results["friction-interior"][rank],
            results["friction-crossing"][rank], inputs[-2]["translation"])
        base.check_mpi(results["friction-interior"][rank],
                       results["friction-segmented"][rank], "Mari run/setup segmentation")
        rheology.compare_histories(results["friction-interior"][rank]["history"],
            results["friction-segmented"][rank]["history"], "Mari segmented history")
        report["checks"][str(rank)] = summary
        print(f"PASS: analytic, timestep, rate, pumping, LE and history ({rank} ranks)", flush=True)
    (root/"summary.json").write_text(json.dumps(report, indent=2)+"\n")
    print(f"PASS: report {root/'summary.json'}", flush=True)


if __name__=="__main__":
    main()
