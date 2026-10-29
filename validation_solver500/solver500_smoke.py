#!/usr/bin/env python3
"""Diagnostic short runs from fixed particle data; not a production launcher."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n")


def history_summary(path):
    if not path.exists():
        return {"rows": 0, "last": None}
    with path.open() as f:
        rows = list(csv.DictReader(f))
    parsed = []
    for row in rows:
        out = {}
        for key, value in row.items():
            try:
                value = float(value)
                out[key] = value if math.isfinite(value) else None
            except (TypeError, ValueError):
                out[key] = value
        parsed.append(out)
    keys = ("force_residual", "mobility_residual", "linear_iterations", "newton_iterations",
            "near_pairs", "contacts", "reciprocal_modes", "max_overlap_m",
            "profile_geometry_s", "profile_middle_s", "profile_solve_s", "profile_total_s")
    maxima = {key: max((row[key] for row in parsed if isinstance(row.get(key), (float, int))),
                       default=None) for key in keys}
    balance = [abs(row["power_balance_error_W"])/max(abs(row["hydro_dissipation_W"]),
               abs(row["contact_power_W"]), abs(row["stress_power_W"]), 1e-300)
               for row in parsed if all(isinstance(row.get(k), (float, int)) for k in
               ("power_balance_error_W", "hydro_dissipation_W", "contact_power_W", "stress_power_W"))]
    return {"rows": len(rows), "last": parsed[-1] if parsed else None,
            "maxima": maxima, "maximum_relative_power_balance_error": max(balance, default=None),
            "steady_state_claim": False}


def prepare(source, out, args):
    params = json.loads((source/"parameters.json").read_text())
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(source/"particles.data", out/"particles.data")
    shutil.copy2(source/"parameters.json", out/"source.parameters.json")
    text = (source/"in.bridge").read_text()
    text, count = re.subn(r"(?m)^run \d+\s*$", "run "+str(args.steps), text)
    if count != 1:
        raise ValueError("Expected exactly one run command: "+str(source))
    text, count = re.subn(r"history history\.csv \d+ \S+", "history history.csv 1 "+
                          format(args.average_start_strain, ".17g"), text)
    if count != 1:
        raise ValueError("Expected exactly one history option: "+str(source))
    text = re.sub(r"(?m)^dump trajectory all custom \d+ ", "dump trajectory all custom 1 ", text)
    text = re.sub(r"(?m)^thermo \d+$", "thermo 1", text)
    (out/"in.bridge").write_text(text)
    params.update(output=str(out), steps=args.steps, output_interval=1,
                  average_start_strain=args.average_start_strain,
                  accumulated_strain=abs(params["shear_rate"])*params["timestep"]*args.steps,
                  diagnostic_source=str(source),
                  diagnostic_source_particle_sha256=sha(source/"particles.data"))
    save(out/"parameters.json", params)
    return params


def run_case(out, args, params):
    cmd = [str(args.lammps), "-var", "slurry_plugin", '"'+str(args.plugin)+'"', "-in", "in.bridge"]
    if args.mpi:
        launcher = shutil.which("mpirun") or shutil.which("mpiexec")
        if not launcher:
            raise ValueError("--mpi requires mpirun or mpiexec")
        cmd = [launcher, "-np", "1"]+cmd
    env = dict(os.environ, OMP_NUM_THREADS=str(args.threads), OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1")
    paths = [args.lammps, args.plugin, out/"particles.data", out/"in.bridge"]+args.library
    before = {str(p): sha(p) for p in paths}
    meta = {"purpose": "diagnostic run from fixed particles; not a production-approved launch",
            "source": params["diagnostic_source"], "command": cmd,
            "OMP_NUM_THREADS": args.threads, "OPENBLAS_NUM_THREADS": 1,
            "before_hashes": before, "started_unix_s": time.time(),
            "timeout_s": args.timeout, "status": "running"}
    save(out/"diagnostic_launch.json", meta)
    start = time.monotonic()
    heartbeat = start
    stop_reason = None
    with (out/"diagnostic.stdout.log").open("w") as log:
        proc = subprocess.Popen(cmd, cwd=out, env=env, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)
        try:
            while proc.poll() is None:
                now = time.monotonic()
                if (out/"STOP").exists() or (args.output_root/"STOP").exists():
                    stop_reason = "STOP file requested interruption"
                    break
                if args.timeout > 0 and now-start > args.timeout:
                    stop_reason = "timeout"
                    break
                if now-heartbeat >= 30:
                    print(json.dumps({"run": str(out), "status": "running",
                                      "elapsed_s": now-start}), flush=True)
                    heartbeat = now
                time.sleep(.25)
        except KeyboardInterrupt:
            stop_reason = "keyboard interruption"
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
    after = {str(p): sha(p) for p in paths}
    meta.update(wall_seconds=time.monotonic()-start, returncode=proc.returncode,
                after_hashes=after, inputs_unchanged=before == after, stop_reason=stop_reason,
                history=history_summary(out/"history.csv"))
    log = (out/"diagnostic.stdout.log").read_text(errors="replace")
    meta["error_lines"] = [line for line in log.splitlines() if "ERROR" in line or "failed" in line]
    meta["completed_requested_steps"] = (proc.returncode == 0 and
        meta["history"]["last"] is not None and meta["history"]["last"]["step"] == args.steps and
        (out/"final.data").is_file() and before == after)
    meta["status"] = "completed" if meta["completed_requested_steps"] else "incomplete_or_failed"
    save(out/"diagnostic_launch.json", meta)
    print(json.dumps({"run": str(out), "status": meta["status"], "wall_seconds": meta["wall_seconds"],
                      "returncode": proc.returncode, "history_rows": meta["history"]["rows"],
                      "error_lines": meta["error_lines"]}), flush=True)
    return meta


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, action="append", required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--lammps", type=Path, required=True)
    p.add_argument("--plugin", type=Path, required=True)
    p.add_argument("--library", type=Path, action="append", default=[])
    p.add_argument("--steps", type=int, default=2)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--timeout", type=float, default=600)
    p.add_argument("--average-start-strain", type=float, default=0)
    p.add_argument("--mpi", action="store_true")
    p.add_argument("--prepare-only", action="store_true")
    args = p.parse_args()
    if args.steps < 0 or args.threads < 1 or args.timeout < 0:
        p.error("Invalid steps, threads or timeout")
    args.source = [v.resolve() for v in args.source]
    args.output_root = args.output_root.resolve()
    args.lammps, args.plugin = args.lammps.resolve(), args.plugin.resolve()
    args.library = [v.resolve() for v in args.library]
    if any(c in str(args.plugin) for c in ('"', '\n', '\r')):
        p.error("Plugin path contains unsupported quote or newline")
    if not args.lammps.is_file() or not args.plugin.is_file():
        p.error("Missing executable or plugin")
    args.output_root.mkdir(parents=True, exist_ok=False)
    prepared = []
    for source in args.source:
        out = args.output_root/source.name
        prepared.append((out, prepare(source, out, args)))
    if args.prepare_only:
        print(json.dumps({"prepared": [str(out) for out, _ in prepared]}), flush=True)
        return
    results = []
    for out, params in prepared:
        result = run_case(out, args, params)
        results.append(result)
        save(args.output_root/"results.json", results)
        if result["stop_reason"] == "keyboard interruption" or (args.output_root/"STOP").exists():
            break
    if not all(r["completed_requested_steps"] for r in results) or len(results) != len(prepared):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
