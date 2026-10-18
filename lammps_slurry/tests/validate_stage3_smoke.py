#!/usr/bin/env python3
"""AR2 solver/cache integration and root-broadcast smoke, not physical validation.

The same genuinely nonspherical two-particle state is advanced for five steps
with the immutable stage2 solver and the candidate. Candidate runs on one and
two ranks also check identical trajectories and non-timing diagnostics.
"""
import argparse
import json
import math
from pathlib import Path
import shlex
import shutil

import benchmark_stage3 as benchmark
import validate_stage2 as stage2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lmp", type=Path, required=True)
    parser.add_argument("--plugin", type=Path, required=True)
    parser.add_argument("--baseline-plugin", type=Path, required=True)
    parser.add_argument("--candidate-options", required=True)
    parser.add_argument("--mpiexec", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        args.candidate_options, controls = benchmark.solver_options(args.candidate_options)
    except ValueError as error:
        parser.error(str(error))
    if controls.get("solver_layout") != "root":
        parser.error("This smoke specifically tests solver_layout root")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    args.lmp = Path(shutil.which(str(args.lmp)) or args.lmp).resolve()
    args.plugin = args.plugin.resolve()
    args.baseline_plugin = args.baseline_plugin.resolve()
    args.mpiexec = shlex.split(args.mpiexec)
    if stage2.digest(args.baseline_plugin) != benchmark.STAGE2_SHA256:
        parser.error("The baseline is not the immutable stage2 plugin")
    case = next(case for case in stage2.rheology.cases() if case["name"] == "oblate-friction")
    args.data = args.output/"particles.data"
    stage2.base.write_data(args.data, case)
    args.radius = max(stage2.base.OBLATE)
    args.viscosity, args.shear, args.dt = 0.01, 100.0, 1e-6
    args.epsilon, args.cutoff, args.contact_gap = 0.15e-9, 75e-9, 3e-9
    args.kn, args.kt, args.mu = 1.0, 80.0, 0.1
    args.steps, args.timeout = 5, 120.0
    runs = {}
    comparisons = {}
    for name, ranks, plugin, options in (
        ("baseline-1rank", 1, args.baseline_plugin, "solver_preconditioner sparse solver_layout root"),
        ("candidate-1rank", 1, args.plugin, args.candidate_options),
        ("candidate-2rank", 2, args.plugin, args.candidate_options)):
        runs[name] = stage2.run_case(args, name, ranks, plugin, options, "spheroid")
        rows = list(runs[name]["history"].values())
        if min(row["lubrication_pairs"] for row in rows) < 1:
            raise AssertionError("The spheroid pair was not hydrodynamically active")
        if max(row["spheroid_max_gap_curvature_ratio"] for row in rows) <= 0:
            raise AssertionError("The genuine nonspherical diagnostic was not exercised")
        if min(row["spheroid_completion_added_power_W"] for row in rows) < 0:
            raise AssertionError("Negative completion power")
        runs[name]["report"]["trajectory_sha256"] = stage2.digest(args.output/name/"trajectory.dump")
        if name != "baseline-1rank":
            comparisons[name] = benchmark.compare_physics(runs["baseline-1rank"], runs[name], name)
    first, second = runs["candidate-1rank"], runs["candidate-2rank"]
    if first["report"]["trajectory_sha256"] != second["report"]["trajectory_sha256"]:
        raise AssertionError("Root-layout trajectory differs between one and two ranks")
    for step, row in first["history"].items():
        other = second["history"][step]
        for key, value in row.items():
            same_uninitialized_mean = (step == 0 and
                (key.startswith("interval_mean_") or key == "mean_eta_Pa_s") and
                math.isnan(value) and math.isnan(other[key]))
            if "profile_" not in key and value != other[key] and not same_uninitialized_mean:
                raise AssertionError(f"Root-broadcast diagnostic {key} differs at step {step}")
    report = dict(status="passed", scope=__doc__, physical_validation=False,
                  performance_acceptance_claim=False,
                  aspect_ratio=2, particle_count=2, steps=args.steps,
                  candidate_options=args.candidate_options,
                  baseline_plugin_sha256=stage2.digest(args.baseline_plugin),
                  candidate_plugin_sha256=stage2.digest(args.plugin),
                  data_sha256=stage2.digest(args.data),
                  candidate_one_two_rank_trajectory_identical=True,
                  candidate_one_two_rank_nonprofile_csv_identical=True,
                  comparisons=comparisons,
                  runs={name:result["report"] for name,result in runs.items()})
    (args.output/"summary.json").write_text(json.dumps(report, indent=2)+"\n")
    print(f"PASS AR2 solver and root-broadcast smoke: {args.output/'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
