#!/usr/bin/env python3
"""Five-step AR=2 wiring smoke; this is not a nonspherical physical benchmark.

The fixture checks that the production fix invokes the genuine spheroid kernel
and records finite, force-balanced dynamics and a consistent power ledger.
Independent reference-flow validation and an AR sweep remain separate work.
"""
import argparse
import json
from pathlib import Path
import shlex
import shutil

import validate_stage2 as stage2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lmp", type=Path, required=True)
    parser.add_argument("--plugin", type=Path, required=True)
    parser.add_argument("--mpiexec", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    args.lmp = Path(shutil.which(str(args.lmp)) or args.lmp).resolve()
    args.plugin = args.plugin.resolve()
    args.mpiexec = shlex.split(args.mpiexec)
    case = next(case for case in stage2.rheology.cases() if case["name"] == "oblate-friction")
    args.data = args.output/"particles.data"
    stage2.base.write_data(args.data, case)
    args.radius = max(stage2.base.OBLATE)  # bounding radius for pair_style zero only
    args.viscosity, args.shear, args.dt = 0.01, 100.0, 1e-6
    args.epsilon, args.cutoff, args.contact_gap = 0.15e-9, 75e-9, 3e-9
    args.kn, args.kt, args.mu = 1.0, 80.0, 0.1
    args.steps, args.timeout = 5, 120.0
    result = stage2.run_case(args, "oblate-wiring-1rank", 1, args.plugin,
                           "solver_preconditioner sparse solver_layout root", "spheroid")
    rows = list(result["history"].values())
    if min(row["lubrication_pairs"] for row in rows) < 1:
        raise AssertionError("The spheroid pair was not hydrodynamically active")
    if max(row["spheroid_max_gap_curvature_ratio"] for row in rows) <= 0:
        raise AssertionError("The genuine nonspherical diagnostic was not exercised")
    if min(row["spheroid_completion_added_power_W"] for row in rows) < 0:
        raise AssertionError("Negative completion power")
    report = dict(status="passed", scope=__doc__, physical_validation=False,
                  aspect_ratio=2, particle_semiaxes_m=list(stage2.base.OBLATE),
                  initial_director=list(stage2.base.P0), initial_gap_m=2e-9,
                  steps=args.steps, dt_s=args.dt, shear_rate_s=args.shear,
                  epsilon_m=args.epsilon, cutoff_cap_m=args.cutoff,
                  contact_gap_m=args.contact_gap, friction_coefficient=args.mu,
                  normal_stiffness_N_m=args.kn, tangent_stiffness_N_m=args.kt,
                  run=result["report"],
                  maximum_gap_curvature_ratio=max(row["spheroid_max_gap_curvature_ratio"] for row in rows),
                  maximum_completion_added_power_W=max(row["spheroid_completion_added_power_W"] for row in rows))
    (args.output/"summary.json").write_text(json.dumps(report, indent=2)+"\n")
    print(f"PASS genuine spheroid wiring smoke: {args.output/'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
