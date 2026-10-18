#!/usr/bin/env python3
"""Run a fixed, fail-fast physical gate and bind its result to exact inputs.

Only the Python standard library is used. The original two AR10 fixtures stay
unchanged; their older per-case acceptance limits do not override the 5% gate.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

FIXTURE_SHA256 = "8e36e8be1753623c8e3460a76a13ee130444fa3bcbb71472dd9614f9881f8038"
CASE_NAMES = ["offset75_boundary_checked", "tilted150_full17_qualified"]
LIMIT = 0.05


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_prepare(project):
    filename = project / "lammps_slurry" / "tools" / "prepare_case.py"
    spec = importlib.util.spec_from_file_location("slurry_gate_prepare", str(filename))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main():
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", required=True, type=Path)
    parser.add_argument("--basis", required=True, type=Path)
    parser.add_argument("--mode", choices=["direct_basis", "matched_provider", "isolated_only"], required=True)
    parser.add_argument("--film-grid", type=int, default=161)
    parser.add_argument("--fixture", type=Path, default=project / "lammps_slurry" / "tests" / "data" / "independent_pair_reference.txt")
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--matrix-output", type=Path, help="Optional total11 matrices for source-resolution comparison")
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    args = parser.parse_args()
    for name in ("executable", "basis", "fixture", "report"):
        setattr(args, name, getattr(args, name).resolve())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    if args.matrix_output is not None:
        args.matrix_output = args.matrix_output.resolve()
        args.matrix_output.parent.mkdir(parents=True, exist_ok=True)
    log_path = args.report.with_suffix(".log")
    result = {
        "schema_version": 1,
        "gate": "source_physical",
        "evaluated_path": args.mode,
        "limit": LIMIT,
        "film_grid": args.film_grid,
        "passed": False,
        "cases": [],
        "not_run": list(CASE_NAMES) if args.mode != "isolated_only" else [],
        "coverage": "Two fixed AR10 configurations only; no continuous gap, orientation, bulk or other-AR accuracy claim.",
        "minimum_gap_over_c": 0.5 if args.mode != "isolated_only" else None,
        "maximum_gap_over_c": 1.0 if args.mode != "isolated_only" else None,
        "basis_path": str(args.basis),
        "fixture_path": str(args.fixture),
        "executable_path": str(args.executable),
        "log_path": str(log_path),
        "max_all11_error": None,
        "max_free_strain_error": None,
        "max_error": None,
    }
    start = time.monotonic()
    stdout, stderr = "", ""
    try:
        if args.film_grid < 21 or args.film_grid % 2 != 1:
            raise ValueError("film-grid must be odd and at least 21")
        result["basis_sha256"] = sha256(args.basis)
        result["fixture_sha256"] = sha256(args.fixture)
        result["executable_sha256"] = sha256(args.executable)
        if result["fixture_sha256"] != FIXTURE_SHA256:
            raise ValueError("Independent reference fixture hash changed; refusing to validate against altered data")
        prepare = load_prepare(project)
        fingerprint = prepare.fast_source_fingerprint()
        result["source_fingerprint"] = fingerprint
        result["source_code_sha256"] = fingerprint
        command = [str(args.executable), str(args.basis), str(args.fixture), args.mode, str(args.film_grid)]
        if args.matrix_output is not None:
            command.append(str(args.matrix_output))
            result["matrix_output"] = str(args.matrix_output)
        result["command"] = command
        process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 universal_newlines=True, timeout=args.timeout_seconds)
        stdout, stderr = process.stdout, process.stderr
        result["exit_code"] = process.returncode
        records = [json.loads(line) for line in stdout.splitlines() if line.strip()]
        result["records"] = records
        metadata = [record for record in records if record.get("stage") == "metadata"]
        if len(metadata) != 1:
            raise ValueError("Probe did not provide exactly one metadata record")
        result["aspect_ratio"] = metadata[0]["aspect_ratio"]
        result["basis_slot"] = metadata[0].get("basis_slot")
        result["compiled_source_fingerprint"] = metadata[0].get("compiled_source_fingerprint", "unbound")
        if metadata[0].get("evaluated_path") != args.mode or metadata[0].get("film_grid") != args.film_grid:
            raise ValueError("Probe mode/grid metadata does not match requested execution")
        cases = [record for record in records if record.get("stage") == "independent_physical"]
        result["cases"] = cases
        completed = [case["name"] for case in cases]
        result["not_run"] = [name for name in CASE_NAMES if name not in completed] if args.mode != "isolated_only" else []
        if cases:
            result["max_all11_error"] = max(case["all11_error"] for case in cases)
            result["max_free_strain_error"] = max(case["free_strain_error"] for case in cases)
            result["max_error"] = max(result["max_all11_error"], result["max_free_strain_error"])
        if result["compiled_source_fingerprint"] != fingerprint:
            raise ValueError("Probe was not compiled against the current numerical source fingerprint; rebuild the probe")
        unchanged = (prepare.fast_source_fingerprint() == fingerprint and
                     sha256(args.basis) == result["basis_sha256"] and
                     sha256(args.fixture) == result["fixture_sha256"] and
                     sha256(args.executable) == result["executable_sha256"])
        result["inputs_unchanged_during_run"] = unchanged
        if not unchanged:
            raise ValueError("A numerical source or test input changed while the gate ran; result is stale")
        stages_passed = all(record.get("passed", True) for record in records)
        if args.mode == "isolated_only":
            result["coverage"] = "Exact isolated spheroid grand resistance only; no pair or bulk validation."
            complete = any(record.get("stage") == "isolated_units" for record in records)
        else:
            complete = completed == CASE_NAMES and all(
                case.get("limit") == LIMIT and case["all11_error"] <= LIMIT and
                case["free_strain_error"] <= LIMIT for case in cases)
        result["passed"] = process.returncode == 0 and stages_passed and complete
    except subprocess.TimeoutExpired as error:
        stdout = error.stdout or ""
        stderr = error.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", "replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", "replace")
        records = []
        for line in stdout.splitlines():
            try:
                records.append(json.loads(line))
            except ValueError:
                pass  # A timeout may interrupt the last JSON line.
        result["records"] = records
        cases = [record for record in records if record.get("stage") == "independent_physical"]
        result["cases"] = cases
        completed = [case["name"] for case in cases]
        result["not_run"] = [name for name in CASE_NAMES if name not in completed] if args.mode != "isolated_only" else []
        if cases:
            result["max_all11_error"] = max(case["all11_error"] for case in cases)
            result["max_free_strain_error"] = max(case["free_strain_error"] for case in cases)
            result["max_error"] = max(result["max_all11_error"], result["max_free_strain_error"])
        result["error"] = "Physical gate exceeded timeout; incomplete cases were not accepted"
        result["exit_code"] = None
    except Exception as error:
        result["error"] = str(error)
        result.setdefault("exit_code", None)
    result["elapsed_seconds"] = time.monotonic() - start
    log_path.write_text(stdout + ("\nSTDERR\n" + stderr if stderr else ""), encoding="utf-8")
    args.report.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"passed": result["passed"], "mode": args.mode,
                      "max_error": result["max_error"], "report": str(args.report),
                      "not_run": result["not_run"]}, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
