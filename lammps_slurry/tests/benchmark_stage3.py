#!/usr/bin/env python3
"""Paired wall/CPU benchmark against an immutable stage-2 plugin.

The same N300, phi=0.55 spherical state, physics, tolerances, output cadence and
LAMMPS executable are used for both versions. Timed runs are strictly sequential
and alternate order. No timing sample is silently discarded. Acceptance uses
the ratio of three-run medians for BOTH full-executable wall time and process-
tree CPU time; LAMMPS loop time is a separate diagnostic.

Example (the candidate must already be built):
  python3 lammps_slurry/tests/benchmark_stage3.py --lmp /path/lmp \
    --baseline-plugin /path/stage2/libslurry.so --plugin /path/new/libslurry.so \
    --mpiexec mpiexec --output build/stage3-performance

For an initial 10-step check, --quick-reference may name a prior matching
10-step run directory containing particles.data, trajectory.dump and history.csv.
This mode runs only the candidate and makes no performance acceptance claim.
"""

import argparse
import copy
import json
import math
import os
from pathlib import Path
import re
import resource
import shlex
import shutil
import statistics

import validate_stage2 as validation

STAGE2_SHA256 = "3cfa391323e3e7db0d82ddccddeb6d43bda2941573229fc9921724b26db5c600"
N300_DATA_SHA256 = "8bf24e621ea4c290700f65e0c29ced0170ad7d672ef34971464f9cecc09a3926"


def solver_options(value):
    """Allow solver controls only; never interpolate physics or input commands."""
    tokens = shlex.split(value)
    choices = {"solver_preconditioner": {"auto", "sparse", "nested", "lu", "ilu"},
               "solver_layout": {"root", "replicated"},
               "solver_forcing": {"fixed", "adaptive"},
               "solver_cache": {"on", "off"}}
    seen, index = {}, 0
    while index < len(tokens):
        key = tokens[index]
        if key in seen:
            raise ValueError(f"Duplicate solver option: {key}")
        if key in choices:
            if index+1 >= len(tokens) or tokens[index+1] not in choices[key]:
                raise ValueError(f"Invalid value for {key}")
            seen[key] = tokens[index+1]
            index += 2
        elif key == "solver_ilu":
            if index+3 >= len(tokens):
                raise ValueError("solver_ilu requires DROP FILL REFRESH")
            drop, fill, refresh = (float(item) for item in tokens[index+1:index+4])
            if not (math.isfinite(drop) and 0 < drop < 1 and
                    math.isfinite(fill) and fill >= 1 and fill.is_integer() and
                    math.isfinite(refresh) and refresh >= 1 and refresh.is_integer()):
                raise ValueError("Invalid solver_ilu parameters")
            # Normalize numbers to exclude whitespace or input-language syntax.
            tokens[index+1:index+4] = [str(drop), str(int(fill)), str(int(refresh))]
            seen[key] = tokens[index+1:index+4]
            index += 4
        else:
            raise ValueError(f"Unsupported option {key}; only solver controls are allowed")
    return " ".join(tokens), seen


def compare_physics(reference, actual, label):
    report = validation.compare(reference, actual, label)
    # Small SI powers/energies and angular momenta need unit-aware checks in
    # addition to the original general-purpose trajectory/CSV comparisons.
    report["maximum_power_difference_W"] = 0.0
    report["maximum_energy_difference_J"] = 0.0
    for suffix, output in (("_W", "maximum_power_difference_W"),
                           ("_J", "maximum_energy_difference_J")):
        fields = [key for key in next(iter(reference["history"].values()))
                  if key.endswith(suffix) and key != "power_balance_error_W"]
        global_scale = max((abs(row[key]) for row in reference["history"].values()
                            for key in fields), default=0.0)
        for key in fields:
            field_scale = max(abs(row[key]) for row in reference["history"].values())
            tolerance = 2e-7*max(field_scale, 1e-8*global_scale, 1e-300)
            for step, row in reference["history"].items():
                difference = abs(actual["history"][step][key]-row[key])
                if difference > tolerance:
                    raise AssertionError(f"{label}: {key} differs by {difference}, tolerance {tolerance}")
                report[output] = max(report[output], difference)
    momentum_fields = [f"c_momentum[{index}]" for index in range(1,4)]
    momentum_scale = max(abs(atom[key]) for frame in reference["frames"].values()
                         for atom in frame["atoms"].values() for key in momentum_fields)
    report["maximum_angular_momentum_difference_kg_m2_s"] = 0.0
    for step, frame in reference["frames"].items():
        for number, atom in frame["atoms"].items():
            other = actual["frames"][step]["atoms"][number]
            error = max(abs(other[key]-atom[key]) for key in momentum_fields)
            if error > 2e-7*max(momentum_scale, 1e-300):
                raise AssertionError(f"{label}: angular momentum differs by {error}")
            report["maximum_angular_momentum_difference_kg_m2_s"] = max(
                report["maximum_angular_momentum_difference_kg_m2_s"], error)
    report["maximum_force_tolerance_relative_difference"] = 0.0
    for step, row in reference["history"].items():
        expected = row["force_tolerance_N"]
        error = abs(actual["history"][step]["force_tolerance_N"]/expected-1.0)
        if error > 2e-7:
            raise AssertionError(f"{label}: the accepted force tolerance changed by {error}")
        report["maximum_force_tolerance_relative_difference"] = max(
            report["maximum_force_tolerance_relative_difference"], error)
    return report


def timed_case(args, name, plugin, options):
    expected = args.pinned_plugin_hashes[str(plugin)]
    if validation.digest(plugin) != expected or validation.digest(args.data) != args.pinned_data_hash:
        raise AssertionError("Plugin or common initial state changed during the benchmark")
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    result = validation.run_case(args, name, args.ranks, plugin, options)
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    report = result["report"]
    if report["plugin_sha256"] != expected or report["data_sha256"] != args.pinned_data_hash:
        raise AssertionError("Plugin or common initial state changed during a run")
    report["process_tree_user_cpu_s"] = after.ru_utime-before.ru_utime
    report["process_tree_system_cpu_s"] = after.ru_stime-before.ru_stime
    report["process_tree_cpu_s"] = (report["process_tree_user_cpu_s"]
                                    + report["process_tree_system_cpu_s"])
    if report["process_tree_cpu_s"] <= 0:
        raise AssertionError("Process-tree CPU measurement was not available")
    report["trajectory_sha256"] = validation.digest(args.output/name/"trajectory.dump")
    rows = [row for step, row in result["history"].items() if step > 0]
    report["lu_counts_excluding_setup"] = {
        key:sum(row[key] for row in rows) for key in rows[0] if key.startswith("lu_")}
    report["forcing_counts_excluding_setup"] = {
        key:sum(row[key] for row in rows) for key in rows[0] if key.startswith("forcing_")}
    if any(not math.isfinite(value) or value < 0 for value in report["lu_counts_excluding_setup"].values()):
        raise AssertionError("Invalid LU instrumentation")
    if "solver_preconditioner lu" in options:
        if report["lu_counts_excluding_setup"].get("lu_solves", 0) <= 0:
            raise AssertionError("Requested LU backend was not exercised or not instrumented")
    (args.output/name/"summary.json").write_text(json.dumps(report, indent=2)+"\n")
    return result


def distributions(runs):
    result = {}
    for key in ("loop_wall_s", "elapsed_including_mpi_startup_s", "process_tree_cpu_s"):
        values = [run[key] for run in runs]
        result[key] = dict(samples=values, minimum=min(values), median=statistics.median(values),
                           maximum=max(values), maximum_over_minimum=max(values)/min(values))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lmp", type=Path, required=True)
    parser.add_argument("--plugin", type=Path, required=True)
    parser.add_argument("--baseline-plugin", type=Path)
    parser.add_argument("--expected-baseline-sha256", default=STAGE2_SHA256)
    parser.add_argument("--mpiexec", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path,
        default=Path(__file__).resolve().parent/"data/stage2_phi055_n300.data")
    parser.add_argument("--baseline-options", default="solver_preconditioner sparse solver_layout root")
    parser.add_argument("--candidate-options", default="solver_preconditioner lu solver_layout root")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--ranks", type=int, default=1)
    parser.add_argument("--target-speedup", type=float, default=3.0)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--pin-cpu", type=int,
                        help="optional Linux CPU affinity; unchanged unless explicitly requested")
    parser.add_argument("--quick-reference", type=Path,
                        help="run only a candidate check against a matching cached run; no acceptance claim")
    args = parser.parse_args()
    if args.steps <= 0 or args.repeats < 1 or args.ranks < 1 or args.warmup_steps < 0:
        parser.error("steps, repeats and ranks must be positive; warmup steps must be nonnegative")
    if not args.quick_reference and args.baseline_plugin is None:
        parser.error("--baseline-plugin is required for a paired benchmark")
    if not math.isfinite(args.target_speedup) or args.target_speedup < 3:
        parser.error("The adoption target cannot be less than 3x")
    try:
        args.baseline_options, baseline_controls = solver_options(args.baseline_options)
        args.candidate_options, candidate_controls = solver_options(args.candidate_options)
    except ValueError as error:
        parser.error(str(error))
    for key in ("lmp", "plugin", "baseline_plugin", "data"):
        path = getattr(args, key)
        if path is not None:
            path = Path(shutil.which(str(path)) or path).resolve()
            if not path.is_file():
                parser.error(f"Missing {key}: {path}")
            setattr(args, key, path)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    args.mpiexec = shlex.split(args.mpiexec)
    if not args.mpiexec:
        parser.error("An MPI launcher is required")
    if args.baseline_plugin and validation.digest(args.baseline_plugin) != args.expected_baseline_sha256:
        parser.error("The supplied baseline does not match the expected immutable stage-2 binary")
    if args.pin_cpu is not None:
        if not hasattr(os, "sched_setaffinity") or args.pin_cpu not in os.sched_getaffinity(0):
            parser.error("Requested CPU is outside the available affinity mask")
        os.sched_setaffinity(0, {args.pin_cpu})
    # A one-rank run is also a one-thread run for both plugins and any vendored
    # or linked dense/sparse libraries. This is recorded rather than assumed.
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    args.radius, args.viscosity, args.shear, args.dt = 150e-9, 0.01, 100.0, 1e-6
    args.epsilon, args.cutoff, args.contact_gap = 0.15e-9, 75e-9, 3e-9
    args.kn, args.kt, args.mu = 1.0, 80.0, 0.1
    count_match = re.search(r"^\s*(\d+)\s+atoms\s*$", args.data.read_text(), re.MULTILINE)
    count = int(count_match.group(1)) if count_match else None
    report = dict(status="running", scope=__doc__, particle_count=count, steps=args.steps,
                  mpi_ranks=args.ranks, repeats=args.repeats,
                  candidate_plugin_sha256=validation.digest(args.plugin),
                  baseline_plugin_sha256=validation.digest(args.baseline_plugin) if args.baseline_plugin else None,
                  data_sha256=validation.digest(args.data),
                  candidate_options=args.candidate_options, baseline_options=args.baseline_options,
                  parsed_solver_controls={"baseline":baseline_controls, "candidate":candidate_controls},
                  physical_parameters={key:getattr(args, key) for key in
                      ("radius", "viscosity", "shear", "dt", "epsilon", "cutoff", "contact_gap", "kn", "kt", "mu")},
                  available_affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
                  environment={key:os.environ.get(key) for key in
                      ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                       "MPIR_CVAR_CH4_NETMOD", "MPIR_CVAR_CH4_CMA_ENABLE", "FI_PROVIDER")},
                  cost_definition="Full executable wall includes input reading, setup, simulation, output and MPI launch/cleanup. CPU is RUSAGE_CHILDREN user+system, including reaped descendants. Build, packing and Python result checking are excluded.",
                  cluster_scaling_claim=False, steady_state_claim=False,
                  cpu_accounting_scope="Local launcher and reaped descendants only; not remote MPI-node accounting",
                  force_tolerance_law="1e-28 N + 1e-10 * raw force scale; per-state thresholds compared between versions",
                  warmup_runs={}, runs={}, comparisons={})
    report["canonical_acceptance_scope"] = dict(
        description="N300, phi0.55, AR1 bundled state; immutable stage2 sparse/root baseline; at least 50 steps, three repeats, one rank and one thread",
        expected_data_sha256=N300_DATA_SHA256,
        canonical_data=report["data_sha256"] == N300_DATA_SHA256,
        canonical_baseline=(report["baseline_plugin_sha256"] == STAGE2_SHA256 and
                            baseline_controls == {"solver_preconditioner":"sparse", "solver_layout":"root"}),
        sufficient_steps=args.steps >= 50)
    args.pinned_plugin_hashes = {str(args.plugin):report["candidate_plugin_sha256"]}
    if args.baseline_plugin:
        args.pinned_plugin_hashes[str(args.baseline_plugin)] = report["baseline_plugin_sha256"]
    args.pinned_data_hash = report["data_sha256"]
    summary = args.output/"summary.json"

    def save():
        summary.write_text(json.dumps(report, indent=2)+"\n")

    def checked_run(run_args, name, plugin, options):
        try:
            return timed_case(run_args, name, plugin, options)
        except Exception as error:
            report["status"] = "run_failed"
            report["failure"] = dict(run=name, type=type(error).__name__, message=str(error))
            report["performance_acceptance_claim"] = False
            save()
            raise

    save()
    if args.quick_reference:
        reference_dir = args.quick_reference.resolve()
        reference_summary = json.loads((reference_dir/"summary.json").read_text())
        if reference_summary["plugin_sha256"] != args.expected_baseline_sha256:
            raise AssertionError("Quick reference is not from the required stage-2 baseline")
        report["quick_reference_provenance"] = reference_summary
        if validation.digest(reference_dir/"particles.data") != report["data_sha256"]:
            raise AssertionError("Quick reference uses a different initial data file")
        reference = dict(frames=validation.base.parse_dump(reference_dir/"trajectory.dump"),
                         history=validation.rheology.parse_history(reference_dir/"history.csv"))
        if max(reference["history"]) != args.steps:
            raise AssertionError("Quick reference has a different step count")
        result = checked_run(args, "candidate-quick", args.plugin, args.candidate_options)
        report["runs"]["candidate-quick"] = result["report"]
        report["comparisons"]["candidate-quick"] = compare_physics(reference, result, "Cached stage2 vs candidate")
        report["status"] = "quick_check_passed"
        report["performance_acceptance_claim"] = False
        save()
        print(f"PASS quick physical equivalence; no speedup acceptance claimed: {summary}", flush=True)
        return

    if args.warmup_steps:
        warmup = copy.copy(args)
        warmup.steps = args.warmup_steps
        for variant, plugin, options in (("baseline", args.baseline_plugin, args.baseline_options),
                                         ("candidate", args.plugin, args.candidate_options)):
            result = checked_run(warmup, f"warmup-{variant}", plugin, options)
            report["warmup_runs"][variant] = result["report"]
            save()
    reference = None
    samples = {"baseline":[], "candidate":[]}
    report["execution_order"] = []
    for repeat in range(args.repeats):
        order = ("baseline", "candidate") if repeat % 2 == 0 else ("candidate", "baseline")
        for variant in order:
            name = f"repeat-{repeat+1}-{variant}"
            plugin = args.baseline_plugin if variant == "baseline" else args.plugin
            options = args.baseline_options if variant == "baseline" else args.candidate_options
            result = checked_run(args, name, plugin, options)
            report["runs"][name] = result["report"]
            report["execution_order"].append(name)
            if reference is None:
                reference = result
            else:
                report["comparisons"][name] = compare_physics(reference, result, name)
            samples[variant].append(result["report"])
            save()
    report["distributions"] = {variant:distributions(values) for variant, values in samples.items()}
    report["median_speedups"] = {key:report["distributions"]["baseline"][key]["median"]/
         report["distributions"]["candidate"][key]["median"] for key in
         ("loop_wall_s", "elapsed_including_mpi_startup_s", "process_tree_cpu_s")}
    report["paired_speedups"] = {key:[first[key]/second[key] for first,second in
         zip(samples["baseline"], samples["candidate"])] for key in report["median_speedups"]}
    enough_repeats = args.repeats >= 3
    report["acceptance"] = dict(target_speedup=args.target_speedup, sufficient_repeats=enough_repeats,
        intended_one_rank_scope=args.ranks == 1,
        canonical_scope=all(report["canonical_acceptance_scope"][key] for key in
                            ("canonical_data", "canonical_baseline", "sufficient_steps")),
        wall_pass=report["median_speedups"]["elapsed_including_mpi_startup_s"] >= args.target_speedup,
        cpu_pass=report["median_speedups"]["process_tree_cpu_s"] >= args.target_speedup)
    report["acceptance"]["passed"] = (enough_repeats and args.ranks == 1
        and report["acceptance"]["canonical_scope"]
        and report["acceptance"]["wall_pass"] and report["acceptance"]["cpu_pass"])
    report["status"] = "benchmark_completed"
    report["performance_acceptance_claim"] = report["acceptance"]["passed"]
    save()
    print(f"Benchmark complete. Median speedups: {report['median_speedups']}; "
          f"target met: {report['acceptance']['passed']}; report: {summary}", flush=True)


if __name__ == "__main__":
    main()
