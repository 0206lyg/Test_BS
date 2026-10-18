#!/usr/bin/env python3
"""Run fail-fast source checks and bind their result to exact inputs.

Only the Python standard library is used. The original two AR10 fixtures stay
unchanged; their older per-case acceptance limits do not override the 5% gate.
pair_numerical checks selected AR-parametric raw Stokes and matched-film matrices;
it neither uses the AR10 reference fixture nor certifies pair physical accuracy.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import time

FIXTURE_SHA256 = "8e36e8be1753623c8e3460a76a13ee130444fa3bcbb71472dd9614f9881f8038"
CASE_NAMES = ["offset75_boundary_checked", "tilted150_full17_qualified"]
LIMIT = 0.05
NUMERICAL_CASE_NAMES = ['failed_orientation', 'face_face', 'face_edge', 'tilted', 'edge_edge']
NUMERICAL_COMPONENT_GAPS = [('raw_full17', (0.5, 1.0)), ('matched_total11', (0.001, 0.02))]


def numerical_signatures():
    return [(name, component, gap) for name in NUMERICAL_CASE_NAMES
            for component, gaps in NUMERICAL_COMPONENT_GAPS for gap in gaps]


def numerical_not_run(records):
    completed = {(r.get('name'), r.get('component'), r.get('h_over_c'))
                 for r in records if r.get('stage') == 'pair_numerical'}
    return ['{}/{}/h={}'.format(*signature) for signature in numerical_signatures()
            if signature not in completed]


def numerical_records_passed(records):
    cases = [r for r in records if r.get('stage') == 'pair_numerical']
    signatures = [(r.get('name'), r.get('component'), r.get('h_over_c')) for r in cases]
    if signatures != numerical_signatures():
        return False
    def finite(value):
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    for case in cases:
        minimum = case.get('min_eigenvalue')
        if (case.get('category') != 'numerical' or case.get('passed') is not True or
                case.get('positive_total') is not True or not finite(minimum) or minimum <= 0):
            return False
        if case['component'] == 'raw_full17':
            for value, limit in ((case.get('raw_reciprocity'), 0.01), (case.get('linear_residual'), 1e-8)):
                if not finite(value) or not 0 <= value <= limit:
                    return False
            if case.get('reciprocity_limit') != 0.01 or case.get('residual_limit') != 1e-8:
                return False
    isolated = [r for r in records if r.get('stage') == 'isolated_units']
    return bool(isolated) and all(r.get('passed') is True and r.get('limit') == 0.001 and
        all(finite(r.get(field)) and 0 <= r[field] <= 0.001 for field in ('all11_error', 'raw_reciprocity'))
        for r in isolated)


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


def run_probe_streamed(command, timeout_seconds, log_path, pair_progress=False):
    """Keep the diagnostic log live, and preserve partial output on timeout."""
    chunks = {'stdout': [], 'stderr': []}
    pending_stdout = b''
    timed_out = False
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               start_new_session=True)
    deadline = time.monotonic() + timeout_seconds
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ, 'stdout')
    selector.register(process.stderr, selectors.EVENT_READ, 'stderr')
    def stop_group():
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    def progress(line):
        if not pair_progress:
            return
        try:
            record = json.loads(line.decode('utf-8'))
        except (ValueError, UnicodeError):
            return
        status = ('passed' if record.get('passed') is True else
                  'failed' if record.get('passed') is False else 'started')
        print('PAIR_CHECK name={} component={} h/c={} status={}'.format(
            record.get('name', record.get('slot', record.get('stage', 'unknown'))),
            record.get('component', record.get('stage', 'unknown')),
            record.get('h_over_c', '-'), status), flush=True)
    try:
        with log_path.open('wb') as log:
            while selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    if timed_out:
                        break  # Bound pipe draining even if a descendant retained a pipe.
                    timed_out = True
                    stop_group()
                    deadline = time.monotonic() + 5.0
                    remaining = 5.0
                for key, _ in selector.select(min(remaining, 0.5)):
                    block = os.read(key.fileobj.fileno(), 65536)
                    if not block:
                        selector.unregister(key.fileobj)
                        continue
                    chunks[key.data].append(block)
                    log.write(block)
                    log.flush()
                    if key.data == 'stdout':
                        pending_stdout += block
                        while b'\n' in pending_stdout:
                            line, pending_stdout = pending_stdout.split(b'\n', 1)
                            progress(line)
        if pending_stdout:
            progress(pending_stdout)
        try:
            process.wait(timeout=5.0 if timed_out else max(0.001, deadline-time.monotonic()))
        except subprocess.TimeoutExpired:
            timed_out = True
            stop_group()
            process.wait(timeout=5.0)
    finally:
        if process.poll() is None:
            stop_group()
            process.wait(timeout=5.0)
        selector.close()
        process.stdout.close()
        process.stderr.close()
    stdout = b''.join(chunks['stdout']).decode('utf-8', 'replace')
    stderr = b''.join(chunks['stderr']).decode('utf-8', 'replace')
    if timed_out:
        raise subprocess.TimeoutExpired(command, timeout_seconds, output=stdout, stderr=stderr)
    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


def main():
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", required=True, type=Path)
    parser.add_argument("--basis", required=True, type=Path)
    parser.add_argument("--mode", choices=["direct_basis", "matched_provider", "isolated_only", "pair_numerical"], required=True)
    parser.add_argument("--film-grid", type=int, default=161)
    parser.add_argument("--fixture", type=Path, default=project / "lammps_slurry" / "tests" / "data" / "independent_pair_reference.txt")
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--matrix-output", type=Path, help="Optional total11 matrices for source-resolution comparison")
    parser.add_argument("--timeout-seconds", type=float, default=None,
                        help="Default: 21600 for pair_numerical; 600 for other modes")
    args = parser.parse_args()
    if args.timeout_seconds is None:
        args.timeout_seconds = 21600.0 if args.mode == 'pair_numerical' else 600.0
    for name in ("executable", "basis", "fixture", "report"):
        setattr(args, name, getattr(args, name).resolve())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    if args.matrix_output is not None:
        args.matrix_output = args.matrix_output.resolve()
        args.matrix_output.parent.mkdir(parents=True, exist_ok=True)
    log_path = args.report.with_suffix(".log")
    log_path.write_text('', encoding='utf-8')
    pair_numerical = args.mode == 'pair_numerical'
    independent_mode = args.mode in ('direct_basis', 'matched_provider')
    result = {
        "schema_version": 1,
        "gate": "source_numerical" if pair_numerical else "source_physical",
        "gate_category": "physical" if independent_mode else "numerical",
        "evaluated_path": args.mode,
        "limit": None if pair_numerical else LIMIT,
        "film_grid": args.film_grid,
        "timeout_seconds": args.timeout_seconds,
        "passed": False,
        "cases": [],
        "not_run": numerical_not_run([]) if pair_numerical else (list(CASE_NAMES) if independent_mode else []),
        "coverage": "Two fixed AR10 configurations only; no continuous gap, orientation, bulk or other-AR accuracy claim.",
        "minimum_gap_over_c": 0.001 if pair_numerical else (0.5 if independent_mode else None),
        "maximum_gap_over_c": 1.0 if independent_mode or pair_numerical else None,
        "basis_path": str(args.basis),
        "fixture_path": str(args.fixture),
        "executable_path": str(args.executable),
        "log_path": str(log_path),
        "max_all11_error": None,
        "max_free_strain_error": None,
        "max_error": None,
    }
    if pair_numerical:
        result['coverage'] = ('Five AR-parametric poses: raw full17 reciprocity/residual/SPD at h/c=0.5,1; '
                              'matched total11 SPD at h/c=0.001,0.02. Numerical checks only; '
                              'no independent pair accuracy, continuous-domain or bulk validation.')
        result['numerical_limits'] = {'raw_reciprocity': 0.01, 'linear_residual': 1e-8,
                                     'isolated_all11_error': 0.001, 'positive_total': True}
    result['fixture_used'] = independent_mode
    start = time.monotonic()
    stdout, stderr = "", ""
    try:
        if not math.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0:
            raise ValueError('timeout-seconds must be finite and positive')
        if args.film_grid < 21 or args.film_grid % 2 != 1:
            raise ValueError("film-grid must be odd and at least 21")
        result["basis_sha256"] = sha256(args.basis)
        result["fixture_sha256"] = sha256(args.fixture) if not pair_numerical else None
        result["executable_sha256"] = sha256(args.executable)
        if not pair_numerical and result["fixture_sha256"] != FIXTURE_SHA256:
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
        process = run_probe_streamed(command, args.timeout_seconds, log_path,
                                     pair_progress=pair_numerical)
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
        cases = [record for record in records if record.get("stage") == ("pair_numerical" if pair_numerical else "independent_physical")]
        result["cases"] = cases
        completed = [case["name"] for case in cases]
        result["not_run"] = numerical_not_run(records) if pair_numerical else ([name for name in CASE_NAMES if name not in completed] if independent_mode else [])
        if cases and not pair_numerical:
            result["max_all11_error"] = max(case["all11_error"] for case in cases)
            result["max_free_strain_error"] = max(case["free_strain_error"] for case in cases)
            result["max_error"] = max(result["max_all11_error"], result["max_free_strain_error"])
        if result["compiled_source_fingerprint"] != fingerprint:
            raise ValueError("Probe was not compiled against the current numerical source fingerprint; rebuild the probe")
        unchanged = (prepare.fast_source_fingerprint() == fingerprint and
                     sha256(args.basis) == result["basis_sha256"] and
                     (pair_numerical or sha256(args.fixture) == result["fixture_sha256"]) and
                     sha256(args.executable) == result["executable_sha256"])
        result["inputs_unchanged_during_run"] = unchanged
        if not unchanged:
            raise ValueError("A numerical source or test input changed while the gate ran; result is stale")
        stages_passed = all(record.get("passed", True) for record in records)
        if pair_numerical:
            complete = numerical_records_passed(records)
            if metadata[0].get('gate_category') != 'numerical':
                raise ValueError('Pair numerical probe did not declare its numerical scope')
        elif args.mode == "isolated_only":
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
        cases = [record for record in records if record.get("stage") == ("pair_numerical" if pair_numerical else "independent_physical")]
        result["cases"] = cases
        completed = [case["name"] for case in cases]
        result["not_run"] = numerical_not_run(records) if pair_numerical else ([name for name in CASE_NAMES if name not in completed] if independent_mode else [])
        if cases and not pair_numerical:
            result["max_all11_error"] = max(case["all11_error"] for case in cases)
            result["max_free_strain_error"] = max(case["free_strain_error"] for case in cases)
            result["max_error"] = max(result["max_all11_error"], result["max_free_strain_error"])
        result["error"] = "Source gate exceeded timeout; incomplete cases were not accepted"
        result["exit_code"] = None
    except Exception as error:
        result["error"] = str(error)
        result.setdefault("exit_code", None)
    result["elapsed_seconds"] = time.monotonic() - start
    args.report.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"passed": result["passed"], "mode": args.mode,
                      "max_error": result["max_error"], "report": str(args.report),
                      "not_run": result["not_run"]}, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
