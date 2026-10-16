#!/usr/bin/env python3
"""Summarize a selected strain window from every-step interval means in history.csv.

Only complete recorded intervals are used, including within each equal-strain
block. Block SD and SEM describe block-to-block variability; they do not establish
stationarity or statistical independence. Uses the Python standard library only.
"""

import argparse
import csv
import math
from pathlib import Path
import statistics


COMPONENTS = (
    "solvent", "single_free", "single_rotation", "lub_normal", "lub_tangent",
    "contact_normal", "contact_tangent", "lub_pump",
)


def read_intervals(path):
    with path.open(newline="") as stream:
        reader = csv.DictReader(line for line in stream if not line.startswith("#"))
        fields = reader.fieldnames or []
        required = ("interval_steps", "interval_start_strain", "interval_end_strain",
                    "interval_mean_sigma_xy_Pa", "interval_mean_eta_Pa_s",
                    "interval_mean_eta_relative")
        missing = [name for name in required if name not in fields]
        if missing:
            raise ValueError("history.csv is missing columns: " + ", ".join(missing))
        intervals = []
        for number, raw in enumerate(reader, 2):
            try:
                row = {key: float(value) for key, value in raw.items()
                       if key is not None and value not in (None, "")}
                start, end = row["interval_start_strain"], row["interval_end_strain"]
                if row["interval_steps"] <= 0 or end <= start:
                    continue
                if not math.isfinite(start + end):
                    raise ValueError("nonfinite interval strain")
                row["_width"] = end - start
                intervals.append(row)
            except (ValueError, KeyError) as error:
                raise ValueError(f"Malformed CSV row {number}: {error}") from error
    return intervals, fields


def complete_intervals(rows, start, end):
    tolerance = 1e-10 * max(1.0, abs(start), abs(end))
    return [row for row in rows
            if row["interval_start_strain"] >= start - tolerance
            and row["interval_end_strain"] <= end + tolerance]


def weighted_mean(rows, column):
    width = math.fsum(row["_width"] for row in rows)
    if not rows or width <= 0:
        return math.nan
    values = [row[column] for row in rows]
    if not all(math.isfinite(value) for value in values):
        return math.nan
    return math.fsum(row["_width"] * row[column] for row in rows) / width


def fmt(value):
    return f"{value:.9g}" if math.isfinite(value) else "nan"


def summarize(path, start, end, block_count):
    rows, fields = read_intervals(path)
    if not rows:
        return "No positive-strain intervals are available for rheology averaging.\n"
    available_end = max(row["interval_end_strain"] for row in rows)
    end = available_end if end is None else min(end, available_end)
    selected = complete_intervals(rows, start, end) if end > start else []
    if not selected:
        return (f"No complete output intervals in the selected strain window "
                f"[{fmt(start)}, {fmt(end)}]; recorded end strain={fmt(available_end)}.\n")

    columns = ["interval_mean_eta_Pa_s", "interval_mean_eta_relative",
               "interval_mean_sigma_xy_Pa"]
    actual_start = min(row["interval_start_strain"] for row in selected)
    actual_end = max(row["interval_end_strain"] for row in selected)
    covered_width = math.fsum(row["_width"] for row in selected)
    edges = [start + (end - start) * i / block_count for i in range(block_count + 1)]
    blocks = [complete_intervals(selected, edges[i], edges[i + 1])
              for i in range(block_count)]
    block_values = {column: [weighted_mean(block, column) for block in blocks if block]
                    for column in columns}

    lines = [
        f"Selected strain window: [{fmt(start)}, {fmt(end)}]",
        f"Used complete intervals: {len(selected)}; strain range "
        f"[{fmt(actual_start)}, {fmt(actual_end)}]; covered width={fmt(covered_width)}",
        "Means use every-step interval means weighted by interval strain width.",
        "metric,mean,block_SD,block_SEM,nonempty_blocks",
    ]
    for column in columns:
        values = [value for value in block_values[column] if math.isfinite(value)]
        deviation = statistics.stdev(values) if len(values) > 1 else math.nan
        sem = deviation / math.sqrt(len(values)) if values else math.nan
        lines.append(
            f"{column[len('interval_mean_'):]},{fmt(weighted_mean(selected, column))},"
            f"{fmt(deviation)},{fmt(sem)},{len(values)}")

    lines.extend(["", "component,mean_sigma_Pa,mean_eta_Pa_s"])
    for component in COMPONENTS:
        stress = f"interval_mean_sigma_{component}_Pa"
        viscosity = f"interval_mean_eta_{component}_Pa_s"
        if stress in fields:
            lines.append(f"{component},{fmt(weighted_mean(selected, stress))},"
                         f"{fmt(weighted_mean(selected, viscosity)) if viscosity in fields else 'nan'}")

    lines.extend(["", "block,target_start_strain,target_end_strain,used_intervals,"
                  "covered_strain,mean_eta_Pa_s,mean_sigma_xy_Pa"])
    for i, block in enumerate(blocks):
        lines.append(f"{i+1},{fmt(edges[i])},{fmt(edges[i+1])},{len(block)},"
                     f"{fmt(math.fsum(row['_width'] for row in block))},"
                     f"{fmt(weighted_mean(block, columns[0]))},"
                     f"{fmt(weighted_mean(block, columns[2]))}")
    lines.append("Block SD/SEM are descriptive; this selected window is not a stationarity test.")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("history", type=Path, nargs="?", default=Path("history.csv"))
    parser.add_argument("--start-strain", type=float, default=5.0)
    parser.add_argument("--end-strain", type=float)
    parser.add_argument("--blocks", type=int, default=5)
    parser.add_argument("--output", type=Path, help="optional plain-text summary file")
    args = parser.parse_args()
    if (not math.isfinite(args.start_strain) or args.start_strain < 0 or args.blocks < 1
            or (args.end_strain is not None
                and (not math.isfinite(args.end_strain) or args.end_strain <= args.start_strain))):
        parser.error("require start-strain >= 0, end-strain > start-strain and blocks >= 1")
    try:
        result = summarize(args.history, args.start_strain, args.end_strain, args.blocks)
        if args.output:
            args.output.write_text(result)
        print(result, end="")
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, f"summarize_rheology.py: {error}\n")


if __name__ == "__main__":
    main()
