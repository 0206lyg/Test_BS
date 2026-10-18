#!/usr/bin/env python3
"""Compare solver paths from one common, already prepared spherical state.

This is a short numerical-equivalence and performance measurement, not a
steady-state viscosity result or a validation of AR > 1 hydrodynamics. It uses
only the Python standard library. All physical parameters are explicit CLI
arguments; the defaults match the 300 nm spherical suspension example.

  python3 lammps_slurry/tests/validate_stage2.py --lmp /path/lmp \
    --plugin build/lammps-slurry/libslurry.so --mpiexec mpiexec \
    --data /path/prepared.data --output build/stage2-check --ranks 1,2,4

The optional --baseline-plugin must be the previous version of the plugin.
Its run uses no newly introduced fix keywords. MPI timings are measurements
of the current machine only; shared-container timings are not cluster scaling.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import statistics
import subprocess
import time

import validate_implicit as base
import validate_rheology as rheology


COMPONENTS = (*rheology.STRESS_COMPONENTS, "sigma_lub_pump_Pa")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_case(args, name, ranks, plugin, extra, model="mari"):
    directory = args.output / name
    directory.mkdir()
    shutil.copyfile(args.data, directory / "particles.data")
    cutoff = 2*args.radius + args.cutoff
    text = ["plugin load ${slurry_plugin}", "units si", "atom_style ellipsoid",
            "boundary p p p", "atom_modify map array", f"processors 1 {ranks} 1",
            "newton on", "read_data particles.data", "comm_modify vel yes",
            f"pair_style zero {cutoff:.17g}", "pair_coeff * *",
            "neighbor 3e-8 bin", "neigh_modify every 1 delay 0 check yes",
            f"fix shear all deform 0 xy erate {args.shear:.17g} remap v units box",
            f"fix integrate all slurry/implicit {args.viscosity:.17g} {args.shear:.17g} "
            f"{args.cutoff:.17g} {args.epsilon:.17g} {args.kn:.17g} lubrication {model} "
            f"contact_gap {args.contact_gap:.17g} friction {args.mu:.17g} {args.kt:.17g} "
            f"history history.csv 1 0 {extra}",
            f"timestep {args.dt:.17g}",
            "compute orientation all property/atom quatw quati quatj quatk",
            "compute momentum all property/atom angmomx angmomy angmomz",
            "dump trajectory all custom 1 trajectory.dump id xu yu zu vx vy vz "
            "ix iy iz c_orientation[*] c_momentum[*]",
            "dump_modify trajectory sort id format float %.17g",
            "thermo 10", "thermo_style custom step time atoms f_integrate[4]",
            "thermo_modify lost error flush yes format float %.17g",
            f"run {args.steps}"]
    (directory / "in.check").write_text("\n".join(text)+"\n", encoding="ascii")
    command = [*args.mpiexec, "-n", str(ranks), str(args.lmp), "-var", "slurry_plugin",
               str(plugin), "-in", "in.check", "-log", "log.lammps"]
    environment = os.environ.copy()
    environment.setdefault("OMP_NUM_THREADS", "1")
    begin = time.perf_counter()
    with (directory / "stdout.txt").open("w") as output:
        process = subprocess.run(command, cwd=directory, env=environment, stdout=output,
                                 stderr=subprocess.STDOUT, timeout=args.timeout)
    elapsed = time.perf_counter()-begin
    output = (directory / "stdout.txt").read_text(errors="replace")
    if process.returncode:
        raise RuntimeError(f"{name} failed:\n{output[-6000:]}")
    history = rheology.parse_history(directory / "history.csv")
    frames = base.parse_dump(directory / "trajectory.dump")
    if set(history) != set(range(args.steps+1)) or set(frames) != set(history):
        raise AssertionError(f"{name}: incomplete history or trajectory")
    for step, frame in frames.items():
        for number, atom in frame["atoms"].items():
            if not all(math.isfinite(value) for value in atom.values()):
                raise AssertionError(f"{name}: nonfinite particle {number} at step {step}")
            quaternion = [atom[f"c_orientation[{index}]"] for index in range(1,5)]
            if abs(sum(value*value for value in quaternion)-1.0) > 2e-10:
                raise AssertionError(f"{name}: nonunit particle orientation")
    for step, row in history.items():
        rheology.assert_number(row["sigma_xy_Pa"], sum(row[k] for k in COMPONENTS),
                              f"{name} stress sum", relative=1e-10, absolute=1e-11)
        rheology.check_force_residual(row, name)
        if row["coulomb_violation_N"] > 1e-18:
            raise AssertionError(f"{name}: Coulomb bound violated")
        for key in ("lub_squeeze_dissipation_W", "lub_shear_dissipation_W",
                    "lub_pump_dissipation_W"):
            if not math.isfinite(row[key]) or row[key] < 0:
                raise AssertionError(f"{name}: invalid modal power {key}")
    energy = rheology.check_energy(history, name)
    loop_times = re.findall(r"Loop time of ([\deE.+-]+) on", output)
    if not loop_times:
        raise AssertionError(f"{name}: LAMMPS loop timing missing")
    rows = [row for step, row in history.items() if step > 0]
    timings = {key:sum(row[key] for row in rows) for key in rows[0]
               if key.startswith("profile_") and key.endswith("_s")}
    counters = {key:sum(row[key] for row in rows) for key in rows[0]
                if (key.startswith("solver_") and key.endswith("iterations"))
                or key in ("linear_iterations", "newton_iterations", "sparse_factorizations",
                           "sparse_fallbacks", "sparse_preconditioner_solves",
                           "sparse_preconditioner_reuses")}
    if "solver_preconditioner sparse" in extra:
        if counters.get("sparse_factorizations", 0) <= 0:
            raise AssertionError(f"{name}: requested sparse preconditioner was never exercised")
        if counters.get("sparse_fallbacks", 0) != 0:
            raise AssertionError(f"{name}: sparse path fell back in the dense regression fixture")
    report = dict(name=name, ranks=ranks, model=model, options=extra,
                  plugin_sha256=digest(plugin), data_sha256=digest(args.data),
                  elapsed_including_mpi_startup_s=elapsed, loop_wall_s=float(loop_times[-1]),
                  cumulative_phase_s_excluding_setup=timings,
                  cumulative_solver_counts_excluding_setup=counters,
                  median_linear_iterations=statistics.median(row["linear_iterations"] for row in rows),
                  final_eta_relative=history[args.steps]["eta_relative"],
                  maximum_force_residual=max(row["force_residual"] for row in history.values()),
                  maximum_overlap_m=max(row["max_overlap_m"] for row in history.values()),
                  energy=energy)
    if "force_residual_N" in rows[0]:
        report["maximum_force_acceptance_ratio"] = max(
            row["force_residual_N"]/row["force_tolerance_N"] for row in history.values())
    (directory / "summary.json").write_text(json.dumps(report, indent=2)+"\n")
    print(f"PASS {name}: loop={report['loop_wall_s']:.4g} s, "
          f"eta_r={report['final_eta_relative']:.9g}", flush=True)
    return dict(history=history, frames=frames, report=report)


def compare(reference, actual, label):
    """Compare all physical outputs; iteration paths and wall times may differ."""
    rheology.compare_histories(reference["history"], actual["history"], label)
    base.check_mpi(reference, actual, label)
    maxima = dict(position_m=0.0, velocity_m_s=0.0, orientation=0.0,
                  stress_Pa=0.0, relative_viscosity=0.0)
    for step, frame in reference["frames"].items():
        for number, atom in frame["atoms"].items():
            other = actual["frames"][step]["atoms"][number]
            for prefix, fields in (("position_m", ("xu", "yu", "zu")),
                                   ("velocity_m_s", ("vx", "vy", "vz")),
                                   ("orientation", tuple(f"c_orientation[{i}]" for i in range(1,5)))):
                maxima[prefix] = max(maxima[prefix], *(abs(atom[k]-other[k]) for k in fields))
        first = reference["history"][step]
        second = actual["history"][step]
        maxima["stress_Pa"] = max(maxima["stress_Pa"], *(abs(first[k]-second[k]) for k in COMPONENTS))
        maxima["relative_viscosity"] = max(maxima["relative_viscosity"],
                                           abs(first["eta_relative"]-second["eta_relative"]))
    return maxima


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lmp", type=Path, required=True)
    parser.add_argument("--plugin", type=Path, required=True)
    parser.add_argument("--baseline-plugin", type=Path)
    parser.add_argument("--mpiexec", required=True)
    parser.add_argument("--data", type=Path,
                        default=Path(__file__).resolve().parent/"data/stage2_phi055_n64.data",
                        help="one prepared spherical state used for every run; default: bundled N64 phi=0.55")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ranks", default="1,2")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--radius", type=float, default=150e-9)
    parser.add_argument("--epsilon", type=float, default=0.15e-9)
    parser.add_argument("--cutoff", type=float, default=75e-9)
    parser.add_argument("--contact-gap", type=float, default=3e-9)
    parser.add_argument("--viscosity", type=float, default=0.01)
    parser.add_argument("--shear", type=float, default=100.0)
    parser.add_argument("--dt", type=float, default=1e-6)
    parser.add_argument("--mu", type=float, default=0.1)
    parser.add_argument("--kn", type=float, default=1.0)
    parser.add_argument("--kt", type=float, default=80.0)
    parser.add_argument("--spheroid-model", default="", help="optional AR=1 reduction check using this backend")
    parser.add_argument("--replicated-check", action="store_true",
                        help="also compare the former replicated layout at max rank count")
    args = parser.parse_args()
    for key in ("lmp", "plugin", "baseline_plugin", "data"):
        value = getattr(args, key)
        if value is not None:
            value = Path(shutil.which(str(value)) or value).resolve()
            if not value.is_file():
                parser.error(f"Missing {key}: {value}")
            setattr(args, key, value)
    args.mpiexec = shlex.split(args.mpiexec)
    ranks = sorted(set(map(int, args.ranks.split(","))))
    if not ranks or ranks[0] != 1 or args.steps <= 0 or not args.mpiexec:
        parser.error("Positive steps, an MPI launcher, and rank list including 1 are required")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    cases = []
    if args.baseline_plugin:
        cases.append(("baseline-1rank", 1, args.baseline_plugin, "", "mari"))
    cases.append(("nested-root-1rank", 1, args.plugin,
                  "solver_preconditioner nested solver_layout root", "mari"))
    for rank in ranks:
        cases.append((f"sparse-root-{rank}rank", rank, args.plugin,
                      "solver_preconditioner sparse solver_layout root", "mari"))
    if args.replicated_check:
        cases.append((f"sparse-replicated-{ranks[-1]}rank", ranks[-1], args.plugin,
                      "solver_preconditioner sparse solver_layout replicated", "mari"))
    if args.spheroid_model:
        cases.append(("spheroid-sphere-1rank", 1, args.plugin,
                      "solver_preconditioner sparse solver_layout root", args.spheroid_model))
    results = {}
    comparisons = {}
    report = dict(status="running", scope="Same initial spherical state, short trajectory; numerical equivalence and local timing only",
                  steady_state_claim=False, nonspherical_physical_validation=False,
                  cluster_scaling_claim=False, mpi_ranks=ranks, steps=args.steps,
                  transport_environment={key:os.environ.get(key) for key in
                      ("OMP_NUM_THREADS", "MPIR_CVAR_CH4_NETMOD", "FI_PROVIDER",
                       "MPIR_CVAR_CH4_CMA_ENABLE")},
                  parameters={key:getattr(args, key) for key in
                              ("radius", "epsilon", "cutoff", "contact_gap", "viscosity", "shear", "dt", "mu", "kn", "kt")},
                  runs={}, comparisons=comparisons)
    for name, rank, plugin, options, model in cases:
        result = run_case(args, name, rank, plugin, options, model)
        results[name] = result
        report["runs"][name] = result["report"]
        if len(results) > 1:
            first_name = next(iter(results))
            comparisons[name] = compare(results[first_name], result, f"{first_name} vs {name}")
        (args.output / "summary.json").write_text(json.dumps(report, indent=2)+"\n")
    serial = results["sparse-root-1rank"]["report"]["loop_wall_s"]
    report["local_mpi_timing"] = {str(rank):dict(
        speedup=serial/results[f"sparse-root-{rank}rank"]["report"]["loop_wall_s"],
        efficiency=serial/results[f"sparse-root-{rank}rank"]["report"]["loop_wall_s"]/rank)
        for rank in ranks}
    report["status"] = "passed"
    (args.output / "summary.json").write_text(json.dumps(report, indent=2)+"\n")
    print(f"PASS report: {args.output / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
