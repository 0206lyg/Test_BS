#!/usr/bin/env python3
"""Time-weighted xz viscosity from every-step interval means of slurry/bridge.

Only complete output intervals within the selected strain window are used.
Block means diagnose drift; passing this descriptive test is not a proof of
stationarity or statistical independence, or a resolution-convergence test.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import sys

METRICS = ("eta_Pa_s", "eta_relative", "sigma_xz_Pa", "sigma_solvent_Pa", "sigma_hydro_Pa", "sigma_contact_Pa")
REQUIRED = ("interval_steps", "interval_start_time_s", "interval_end_time_s", "interval_start_strain", "interval_end_strain") + tuple("interval_mean_"+k for k in METRICS)


def read_intervals(path):
    rows = []
    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(line for line in stream if not line.startswith("#"))
        missing = set(REQUIRED)-set(reader.fieldnames or ())
        if missing:
            raise ValueError("Missing every-step interval columns: "+", ".join(sorted(missing)))
        previous_end = None
        for number, raw in enumerate(reader, 2):
            try:
                row = {k: float(v) for k, v in raw.items() if k and v not in (None, "")}
                if row["interval_steps"] <= 0:
                    continue
                if not all(math.isfinite(row[k]) for k in REQUIRED):
                    raise ValueError("nonfinite rheology interval")
                start, end = row["interval_start_time_s"], row["interval_end_time_s"]
                if end <= start or row["interval_end_strain"] <= row["interval_start_strain"]:
                    continue
                if previous_end is not None and start < previous_end-1e-10*max(1., abs(start)):
                    raise ValueError("overlapping or reversed time intervals")
                previous_end = end
                row["_duration"] = end-start
                rows.append(row)
            except (ValueError, KeyError) as error:
                raise ValueError("Malformed history row {}: {}".format(number, error)) from error
    if not rows:
        raise ValueError("No positive-time, positive-strain intervals; viscosity needs nonzero shear and completed steps")
    return rows


def select(rows, start, end):
    eps = 1e-12*max(1., abs(start), abs(end))
    return [row for row in rows if row["interval_start_strain"] >= start-eps and row["interval_end_strain"] <= end+eps]


def mean(rows, field):
    duration = math.fsum(row["_duration"] for row in rows)
    return math.fsum(row["_duration"]*row[field] for row in rows)/duration


def summarize(path, start=5., end=None, blocks=5, stability_tolerance=.1):
    if not math.isfinite(start) or start < 0 or blocks < 2 or not 0 < stability_tolerance < 1:
        raise ValueError("Require start>=0, blocks>=2, 0<stability-tolerance<1")
    rows = read_intervals(path)
    available_end = rows[-1]["interval_end_strain"]
    end = available_end if end is None else end
    if not math.isfinite(end) or end <= start:
        raise ValueError("Averaging end must exceed start; recorded end strain={:.8g}".format(available_end))
    chosen = select(rows, start, min(end, available_end))
    if not chosen:
        raise ValueError("No complete output intervals in strain window [{}, {}]; recorded end={}".format(start, end, available_end))
    left, right = chosen[0]["interval_start_strain"], chosen[-1]["interval_end_strain"]
    edges = [left+(right-left)*i/blocks for i in range(blocks+1)]
    block_records = []
    for i in range(blocks):
        part = select(chosen, edges[i], edges[i+1])
        width = math.fsum(row["interval_end_strain"]-row["interval_start_strain"] for row in part)
        block_records.append({"start_strain": edges[i], "end_strain": edges[i+1],
                              "intervals": len(part), "coverage_fraction": width/(edges[i+1]-edges[i]),
                              "eta_Pa_s": mean(part, "interval_mean_eta_Pa_s") if part else None})
    means = {key: mean(chosen, "interval_mean_"+key) for key in METRICS}
    values = [b["eta_Pa_s"] for b in block_records if b["eta_Pa_s"] is not None]
    enough = blocks >= 5 and all(b["coverage_fraction"] >= .9 for b in block_records)
    relative_spread, relative_drift = None, None
    if len(values) >= 2 and abs(means["eta_Pa_s"]) > 0:
        relative_spread = statistics.stdev(values)/abs(means["eta_Pa_s"])
        half = len(values)//2
        relative_drift = abs(statistics.mean(values[-half:])-statistics.mean(values[:half]))/abs(means["eta_Pa_s"])
    stable = enough and relative_spread is not None and max(relative_spread, relative_drift) <= stability_tolerance
    diagnostics = {}
    for field in ("power_balance_error_W", "force_residual", "mobility_residual", "max_overlap_m"):
        values_field = [abs(row[field]) for row in chosen if field in row and math.isfinite(row[field])]
        if values_field:
            diagnostics["max_abs_"+field] = max(values_field)
    costs = [row["profile_total_s"] for row in chosen if "profile_total_s" in row and math.isfinite(row["profile_total_s"])]
    if costs:
        diagnostics["mean_sampled_solve_seconds"] = statistics.mean(costs)
    return {"history": str(Path(path).resolve()), "requested_window": [start, end],
            "used_window": [left, right], "used_intervals": len(chosen),
            "duration_s": math.fsum(row["_duration"] for row in chosen), "mean": means,
            "blocks": block_records, "block_relative_SD": relative_spread,
            "block_relative_half_window_drift": relative_drift,
            "block_stability_check": "passed" if stable else "failed" if enough else "insufficient_block_coverage",
            "block_stability_tolerance": stability_tolerance,
            "stationarity_certified": False,
            "scope": "Time-weighted selected-window viscosity. Block stability is descriptive; inspect microstructure and repeat with longer time, timestep and spectral resolution as needed.",
            "diagnostics": diagnostics}


def render(report):
    avg = report["mean"]
    lines = ["Strain window {} to {}; {} complete intervals, {:.8g} s".format(
        *report["used_window"], report["used_intervals"], report["duration_s"]),
        "Apparent viscosity = {:.9g} Pa s; relative viscosity = {:.9g}".format(avg["eta_Pa_s"], avg["eta_relative"]),
        "Mean xz stress = {:.9g} Pa (solvent {:.9g}, hydrodynamic {:.9g}, contact {:.9g})".format(
            avg["sigma_xz_Pa"], avg["sigma_solvent_Pa"], avg["sigma_hydro_Pa"], avg["sigma_contact_Pa"]),
        "Block stability check: {} (tolerance {})".format(report["block_stability_check"], report["block_stability_tolerance"]),
        "Selected-window average; steady state is not automatically certified."]
    return "\n".join(lines)+"\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("history", type=Path, nargs="?", default=Path("history.csv"))
    parser.add_argument("--start-strain", type=float, default=5.)
    parser.add_argument("--end-strain", type=float)
    parser.add_argument("--blocks", type=int, default=5)
    parser.add_argument("--stability-tolerance", type=float, default=.1)
    parser.add_argument("--output", type=Path, help="JSON report path")
    args = parser.parse_args()
    try:
        report = summarize(args.history, args.start_strain, args.end_strain, args.blocks, args.stability_tolerance)
        if args.output:
            args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False)+"\n")
        print(render(report), end="")
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, "summarize_bridge.py: {}\n".format(error))

if __name__ == "__main__":
    main()
