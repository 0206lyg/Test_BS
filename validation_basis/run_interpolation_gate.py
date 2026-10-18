#!/usr/bin/env python3
"""Check an actual complete table against its direct source; Python 3.6 stdlib."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

CASES = ["asymmetric_3nm", "near_face", "oblique", "face_edge", "near_edges",
         "azimuth_seam_left", "azimuth_seam_right", "north_pole", "outer_gap"]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("executable", "basis", "table", "report"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--film-grid", type=int, default=161)
    parser.add_argument("--timeout-seconds", type=float, default=1800.)
    args = parser.parse_args()
    for name in ("executable", "basis", "table", "report"):
        setattr(args, name, getattr(args, name).resolve())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    result = {"schema_version": 1, "gate": "table_interpolation", "passed": False,
              "evaluated_path": "complete_table_vs_direct_matched_source", "limit": 0.01,
              "film_grid": args.film_grid, "cases": [], "not_run": list(CASES),
              "max_error": None, "max_all11_error": None, "max_free_strain_error": None,
              "coverage": "Nine fixed query poses; interpolation agreement only, not independent Stokes or bulk accuracy."}
    stdout, stderr = "", ""
    start = time.monotonic()
    try:
        prepare_path = Path(__file__).resolve().parents[1] / "lammps_slurry" / "tools" / "prepare_case.py"
        spec = importlib.util.spec_from_file_location("slurry_interpolation_prepare", str(prepare_path))
        prepare = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(prepare)
        fingerprint = prepare.fast_source_fingerprint()
        result["source_code_sha256"] = fingerprint
        result["source_fingerprint"] = fingerprint
        for name in ("executable", "basis", "table"):
            result[name + "_sha256"] = sha256(getattr(args, name))
            result[name + "_path"] = str(getattr(args, name))
        command = [str(args.executable), str(args.basis), str(args.table), str(args.film_grid)]
        result["command"] = command
        process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 universal_newlines=True, timeout=args.timeout_seconds)
        stdout, stderr = process.stdout, process.stderr
        result["exit_code"] = process.returncode
        records = [json.loads(line) for line in stdout.splitlines() if line.strip()]
        result["records"] = records
        metadata = [r for r in records if r.get("stage") == "metadata"]
        if len(metadata) != 1 or metadata[0].get("film_grid") != args.film_grid or metadata[0].get("case_count") != len(CASES):
            raise ValueError("Interpolation probe metadata mismatch")
        result["aspect_ratio"] = metadata[0]["aspect_ratio"]
        result["compiled_source_fingerprint"] = metadata[0].get("compiled_source_fingerprint")
        if result["compiled_source_fingerprint"] != fingerprint:
            raise ValueError("Probe was compiled against different or unidentified numerical sources; rebuild it")
        cases = [r for r in records if r.get("stage") == "interpolation"]
        result["cases"] = cases
        names = [r["name"] for r in cases]
        result["not_run"] = [name for name in CASES if name not in names]
        if cases:
            result["max_all11_error"] = max(r["all11_error"] for r in cases)
            result["max_free_strain_error"] = max(r["free_strain_error"] for r in cases)
            result["max_error"] = max(result["max_all11_error"], result["max_free_strain_error"])
        unchanged = prepare.fast_source_fingerprint() == fingerprint and all(
            sha256(getattr(args, name)) == result[name + "_sha256"] for name in ("executable", "basis", "table"))
        result["inputs_unchanged_during_run"] = unchanged
        if not unchanged:
            raise ValueError("Source, basis, table or executable changed during the gate")
        complete = names == CASES and all(r.get("limit") == .01 and
            r["all11_error"] <= .01 and r["free_strain_error"] <= .01 and r["symmetry_error"] <= 1e-10 for r in cases)
        result["passed"] = process.returncode == 0 and complete and all(r.get("passed", True) for r in records)
    except subprocess.TimeoutExpired as error:
        stdout, stderr = error.stdout or "", error.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", "replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", "replace")
        result["error"] = "Interpolation gate timed out; remaining cases were not accepted"
    except Exception as error:
        result["error"] = str(error)
    result["elapsed_seconds"] = time.monotonic() - start
    result["log_path"] = str(args.report.with_suffix(".log"))
    args.report.with_suffix(".log").write_text(stdout + ("\nSTDERR\n" + stderr if stderr else ""), encoding="utf-8")
    args.report.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"passed": result["passed"], "max_error": result["max_error"],
                      "report": str(args.report), "not_run": result["not_run"]}, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
