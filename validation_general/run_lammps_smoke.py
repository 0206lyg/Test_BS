#!/usr/bin/env python3
"""Run or analyze short general-hydrodynamics LAMMPS integration cases.

Uses the standard library only. These are execution, force-balance and
stress/power consistency checks, not steady suspension viscosity validation.
The script does not generate or load an orientation table.
"""
import argparse
import csv
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lammps_slurry/tools"))
import bundle_guard

DEFAULT_PHI = {2.: .24, 5.: .10, 10.: .06, 20.: .04}
POWER_FIELDS = ("stress_power_W", "hydro_dissipation_W", "elastic_energy_rate_W",
                "sliding_power_W", "contact_discrete_dissipation_W")
CORE_FIELDS = POWER_FIELDS + ("power_balance_error_W", "force_residual_N",
    "force_tolerance_N", "sigma_xy_Pa", "profile_total_s", "finite_gap_strain_bound_min")


def file_identity(path):
    path = Path(path).expanduser().resolve()
    return {"path": str(path), "sha256": bundle_guard.sha256_file(path),
            "size_bytes": path.stat().st_size}


def source_identity():
    snapshot = bundle_guard.source_snapshot(ROOT)
    return {"root": str(ROOT), "sha256": snapshot["sha256"], "files": snapshot["files"],
            "runner": file_identity(__file__),
            "scope": "source tree observed at report/launch time; compiled-source binding requires a matching build manifest"}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def read_parameters(path):
    result = {}
    if path.is_file():
        for line in path.read_text().splitlines():
            if " = " in line:
                key, value = line.split(" = ", 1)
                result[key] = value
    return result


def numeric_parameter(parameters, name, default=None):
    try:
        value = float(parameters[name])
        return value if math.isfinite(value) else default
    except (KeyError, ValueError):
        return default


def optional_json(path):
    return json.loads(path.read_text()) if path.is_file() else None


def analyze_case(directory, power_tolerance=1e-6):
    directory = Path(directory).expanduser().resolve()
    parameters = read_parameters(directory / "parameters.txt")
    execution = optional_json(directory / "smoke_execution.json")
    history = directory / "history.csv"
    log_path = directory / "log.lammps"
    # console.txt is the manual smoke convention; smoke_console.txt is ours.
    log_paths = [log_path, directory / "smoke_console.txt", directory / "console.txt"]
    log = "\n".join(path.read_text(errors="replace") for path in log_paths if path.is_file())
    errors = list(dict.fromkeys(line for line in log.splitlines() if line.startswith("ERROR")))
    result = {"directory": str(directory), "scope": "short integration smoke; steady viscosity NOT tested",
              "parameters": parameters, "execution": execution, "errors": errors,
              "near_completion_recorded": parameters.get("general_near_completion", "not recorded; pre-option raw model"),
              "artifacts": {name: file_identity(directory / name) for name in
                  ("in.slurry", "parameters.txt", "particles.data", "history.csv", "log.lammps", "final.data")
                  if (directory / name).is_file()},
              "eta_positive_required": False,
              "eta_sign_note": "Elastic relaxation can return stored energy; instantaneous eta need not be positive."}
    rows = []
    parse_errors = []
    if history.is_file():
        with history.open(newline="") as stream:
            for index, row in enumerate(csv.DictReader(stream)):
                try:
                    rows.append({key: float(value) for key, value in row.items()})
                except (TypeError, ValueError) as error:
                    parse_errors.append({"row": index, "error": str(error)})
    result["history_rows"] = len(rows)
    result["history_parse_errors"] = parse_errors
    nonfinite = []
    expected_undefined = set()
    missing_core = sorted(set(CORE_FIELDS) - set(rows[0])) if rows else list(CORE_FIELDS)
    shear = numeric_parameter(parameters, "shear_rate", 100.)
    for index, row in enumerate(rows):
        for field, value in row.items():
            if math.isfinite(value):
                continue
            allowed = (field.startswith("interval_") and row.get("interval_steps", 0) == 0
                       or field == "mean_eta_Pa_s" and row.get("average_duration_s", 0) == 0
                       or "eta" in field and shear == 0
                       or field == "min_gap_m" and row.get("lubrication_pairs", 0) == 0
                          and row.get("contacts", 0) == 0)
            if allowed:
                expected_undefined.add(field)
            else:
                nonfinite.append({"row": index, "field": field, "value": str(value)})
    finite_core = bool(rows) and not missing_core and all(
        math.isfinite(row.get(field, float("nan"))) for row in rows for field in CORE_FIELDS)
    ratios, power_ratios, recomputed_ratios, differences = [], [], [], []
    if finite_core:
        for row in rows:
            tolerance = row["force_tolerance_N"]
            if tolerance > 0:
                ratios.append(abs(row["force_residual_N"])/tolerance)
            # Absolute physical powers, including release of elastic energy,
            # provide a meaningful scale even when net stress power is near 0.
            scale = max(abs(row["stress_power_W"]),
                        sum(abs(row[field]) for field in POWER_FIELDS[1:]), 1e-300)
            power_ratios.append(abs(row["power_balance_error_W"])/scale)
            recomputed = row["stress_power_W"] - sum(row[field] for field in POWER_FIELDS[1:])
            recomputed_ratios.append(abs(recomputed)/scale)
            differences.append(abs(recomputed-row["power_balance_error_W"])/scale)
    expected_steps = numeric_parameter(parameters, "steps")
    last_step = rows[-1].get("step") if rows else None
    if last_step is not None and not math.isfinite(last_step):
        last_step = None
    final_data = (directory / "final.data").is_file()
    loops = re.findall(r"Loop time of ([0-9.eE+\-]+) on (\d+) procs for (\d+) steps", log)
    loop_elapsed = float(loops[-1][0]) if loops else None
    loop_steps = int(loops[-1][2]) if loops else None
    loop_ranks = int(loops[-1][1]) if loops else None
    profile = [row["profile_total_s"] for row in rows if row.get("step", 0) > 0
               and math.isfinite(row.get("profile_total_s", float("nan")))]
    result["timing"] = {
        "invocation_elapsed_seconds": execution.get("lammps_elapsed_seconds") if execution else None,
        "loop_elapsed_seconds": loop_elapsed, "loop_steps": loop_steps, "mpi_ranks_from_log": loop_ranks,
        "loop_seconds_per_step": loop_elapsed/loop_steps if loop_elapsed is not None and loop_steps else None,
        "mean_recorded_step_profile_seconds": sum(profile)/len(profile) if profile else None,
        "note": "LAMMPS loop time excludes initial setup solve; invocation time includes it when recorded."}
    result["force_balance"] = {"maximum_residual_over_tolerance": max(ratios) if ratios else None,
                               "all_positive_tolerances": len(ratios) == len(rows) and bool(rows)}
    result["power_balance"] = {"normalization": "max(abs(Pstress),abs(Phydro)+abs(dUdt)+abs(Pslide)+abs(Pdiscrete),1e-300)",
        "limit": power_tolerance, "maximum_normalized_reported_error": max(power_ratios) if power_ratios else None,
        "maximum_normalized_recomputed_error": max(recomputed_ratios) if recomputed_ratios else None,
        "maximum_normalized_stored_vs_recomputed_difference": max(differences) if differences else None}
    result["nonfinite_fields"] = nonfinite
    result["expected_undefined_fields"] = sorted(expected_undefined)
    result["missing_required_history_fields"] = missing_core
    eta = [row["eta_Pa_s"] for row in rows if math.isfinite(row.get("eta_Pa_s", float("nan")))]
    result["instantaneous_eta_Pa_s"] = {"minimum": min(eta) if eta else None,
                                      "maximum": max(eta) if eta else None,
                                      "meaning": "Transient diagnostic range, not an equilibrium viscosity estimate"}
    checks = {
        "no_lammps_error": not errors,
        "process_succeeded": execution is None or execution.get("lammps_returncode") == 0,
        "binary_content_unchanged": execution is None or execution.get("binary_content_unchanged") is True,
        "general_mode": parameters.get("lubrication_model_resolved", parameters.get("lubrication_model")) == "general",
        "history_finite": finite_core and not nonfinite and not parse_errors,
        "requested_steps_completed": last_step is not None and expected_steps is not None and last_step >= expected_steps,
        "final_data_written": final_data,
        "loop_completed": bool(loops),
        "force_within_solver_tolerance": bool(ratios) and len(ratios) == len(rows) and max(ratios) <= 1.000001,
        "normalized_power_balance": bool(power_ratios) and max(power_ratios+recomputed_ratios) <= power_tolerance,
    }
    result["checks"] = checks
    result["passed"] = all(checks.values())
    result["completed_steps"] = last_step
    return result


def run_logged(command, cwd, logfile, environment):
    print("+ " + " ".join(shlex.quote(str(item)) for item in command), flush=True)
    started = time.monotonic()
    with Path(logfile).open("w") as stream:
        process = subprocess.Popen([str(item) for item in command], cwd=str(cwd), env=environment,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
        for line in process.stdout:
            print(line, end="", flush=True)
            stream.write(line)
            stream.flush()
        code = process.wait()
    return code, time.monotonic()-started


def run_case(args, ar, source, binaries):
    directory = args.workdir / ("AR" + format(ar, ".12g").replace(".", "p"))
    if directory.exists():
        raise ValueError("Case directory exists; preserve it and choose another --workdir: {}".format(directory))
    phi = args.phi if args.phi is not None else DEFAULT_PHI.get(ar, .04)
    prepare_command = [sys.executable, str(ROOT / "lammps_slurry/tools/prepare_case.py"),
        "--output", str(directory), "--packer", str(args.packer), "--lubrication-model", "general",
        "--ar", str(ar), "--phi", str(phi), "--count", str(args.count), "--steps", str(args.steps),
        "--prep-steps", "0", "--output-interval", "1", "--average-start-strain", "0",
        "--seed", str(args.seed), "--timestep", str(args.timestep), "--shear-rate", str(args.shear_rate),
        "--general-tolerance", str(args.general_tolerance), "--general-max-level", str(args.general_max_level),
        "--general-cutoff", str(args.general_cutoff), "--general-normal-log", args.general_normal_log,
        "--general-closure", args.general_closure, "--general-near-completion", args.general_near_completion,
        "--hydro-interaction-range", str(args.hydro_interaction_range)]
    environment = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    # MPI launcher arguments are an argv vector, never shell-evaluated.
    command = ([str(args.mpi_launcher), "-np", str(args.ranks)] if args.mpi_launcher else [])
    command += [str(args.lammps), "-var", "slurry_plugin", '"{}"'.format(args.plugin),
                "-in", "in.slurry", "-log", "log.lammps"]
    record = {"created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "source_at_launch": source, "binaries_at_launch": binaries,
              "prepare_command": prepare_command, "lammps_command": command,
              "ranks_requested": args.ranks, "lammps_returncode": None}
    prepare_log = args.workdir / (directory.name + "_prepare.log")
    code, elapsed = run_logged(prepare_command, ROOT, prepare_log, environment)
    record.update(prepare_returncode=code, prepare_elapsed_seconds=elapsed)
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "smoke_execution.json", record)
    if code == 0:
        code, elapsed = run_logged(command, directory, directory / "smoke_console.txt", environment)
        record.update(lammps_returncode=code, lammps_elapsed_seconds=elapsed)
        record["binary_content_unchanged"] = all(file_identity(value["path"])["sha256"] == value["sha256"]
                                                  for value in binaries.values())
        write_json(directory / "smoke_execution.json", record)
    return analyze_case(directory, args.power_tolerance)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lammps", type=Path)
    parser.add_argument("--plugin", type=Path)
    parser.add_argument("--packer", type=Path)
    parser.add_argument("--workdir", type=Path, help="parent directory for new AR case folders")
    parser.add_argument("--analyze-existing", nargs="+", type=Path, help="read saved runs without executing preparation or LAMMPS")
    parser.add_argument("--output", type=Path, help="JSON report; defaults to WORKDIR/smoke_summary.json")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--ar", nargs="+", type=float, default=[2.])
    selection.add_argument("--all-ar", action="store_true", help="explicitly request AR2/5/10/20; may be expensive")
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--phi", type=float, help="override default smoke fractions: AR2 .24, AR5 .10, AR10 .06, AR20 .04")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--timestep", type=float, default=1e-6)
    parser.add_argument("--shear-rate", type=float, default=100.)
    parser.add_argument("--mpi-launcher", type=Path, help="mpirun/mpiexec executable, launched with -np RANKS")
    parser.add_argument("--ranks", type=int, default=1)
    parser.add_argument("--general-tolerance", type=float, default=.005)
    parser.add_argument("--general-max-level", type=int, default=6)
    parser.add_argument("--general-cutoff", type=float, default=1.)
    parser.add_argument("--general-normal-log", choices=("leading", "derived"), default="leading")
    parser.add_argument("--general-closure", choices=("relative", "pair_additive"), default="relative")
    parser.add_argument("--general-near-completion", choices=("energy", "raw"), default="energy")
    parser.add_argument("--hydro-interaction-range", type=float, default=.5)
    parser.add_argument("--power-tolerance", type=float, default=1e-6)
    args = parser.parse_args()
    if not args.output and not args.workdir:
        parser.error("provide --output for analysis or --workdir for new runs")
    if args.ranks < 1 or args.count < 1 or args.steps < 1:
        parser.error("ranks, count and steps must be positive")
    if not math.isfinite(args.power_tolerance) or args.power_tolerance <= 0:
        parser.error("power-tolerance must be finite and positive")
    if args.ranks > 1 and not args.mpi_launcher and not args.analyze_existing:
        parser.error("--ranks > 1 requires --mpi-launcher")
    source = source_identity()
    binaries = {}
    for name in ("lammps", "plugin", "packer"):
        value = getattr(args, name)
        if value:
            setattr(args, name, value.expanduser().resolve())
            binaries[name] = file_identity(getattr(args, name))
        elif not args.analyze_existing:
            parser.error("new runs require --" + name)
    if args.analyze_existing:
        cases = [analyze_case(directory, args.power_tolerance) for directory in args.analyze_existing]
        identity_scope = "Analysis-time binaries, if supplied, do not establish which binary ran an existing case; see each saved smoke_execution.json when present."
    else:
        if args.workdir is None:
            parser.error("new runs require --workdir")
        args.workdir = args.workdir.expanduser().resolve()
        args.workdir.mkdir(parents=True, exist_ok=True)
        if args.mpi_launcher:
            resolved_launcher = shutil.which(str(args.mpi_launcher))
            args.mpi_launcher = Path(resolved_launcher or args.mpi_launcher).expanduser().resolve()
        ars = [2., 5., 10., 20.] if args.all_ar else args.ar
        if any(not math.isfinite(ar) or ar < 1 for ar in ars):
            parser.error("AR values must be finite and at least 1")
        cases = [run_case(args, ar, source, binaries) for ar in ars]
        identity_scope = "Binary hashes recorded immediately before invocation; source-at-launch is not by itself compiled-source provenance."
    report = {"schema": 1, "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "scope": "LAMMPS integration smoke, not equilibrium bulk-viscosity validation",
              "source_observed": source, "binaries_observed": binaries,
              "identity_scope": identity_scope, "cases": cases,
              "passed": bool(cases) and all(case["passed"] for case in cases)}
    output = args.output or args.workdir / "smoke_summary.json"
    write_json(output, report)
    for case in cases:
        print("{}: {} ({} history rows)".format(case["directory"], "PASS" if case["passed"] else "FAIL",
                                                case["history_rows"]), flush=True)
    print("Report: {}".format(Path(output).resolve()), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError) as error:
        print("Smoke runner failed: {}".format(error), file=sys.stderr)
        sys.exit(2)
