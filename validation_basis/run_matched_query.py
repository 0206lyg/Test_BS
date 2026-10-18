#!/usr/bin/env python3
"""Capture one AR10 source matrix with exact build/input provenance.

Successful query completion is not an accuracy gate.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from run_physical_gate import load_prepare, sha256


def coefficients_sha256(path):
    """Bind identical serialized models while excluding only runtime phi count."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for line in stream:
            if line.startswith(b"FOURIER"):
                tokens = line.split()
                if len(tokens) == 3 and tokens[0] == b"FOURIER" and tokens[1].isdigit() and tokens[2].isdigit():
                    line = b"FOURIER <runtime_phi> " + tokens[2] + b"\n"
            digest.update(line)
    return digest.hexdigest()


def verify_reference_overlay(path, project, production_fingerprint):
    manifest = json.loads(path.read_text())
    if (manifest.get("schema") != "reference_only_bound_override_v1" or
            manifest.get("reference_only") is not True or manifest.get("physics_modified") is not False or
            manifest.get("production_files_modified") is not False or
            manifest.get("base_production_fingerprint") != production_fingerprint or
            manifest.get("old_matrix_entry_limit") != 16000000 or
            manifest.get("reference_matrix_entry_limit") != 24000000):
        raise ValueError("Invalid or stale reference-only loader manifest")
    root = project.parent.resolve()
    checked = []
    for item in manifest["files"]:
        base, copy = (root / item["base_path"]).resolve(), (root / item["copy_path"]).resolve()
        base.relative_to(root)
        copy.relative_to(root)
        if sha256(base) != item["base_sha256"] or sha256(copy) != item["copy_sha256"]:
            raise ValueError("Reference-only loader file hash mismatch")
        original, changed = base.read_bytes(), copy.read_bytes()
        if item["byte_identical"]:
            if changed != original:
                raise ValueError("A byte-identical reference include changed")
        elif (changed != original.replace(b"16000000", b"24000000") or
              original.count(b"16000000") != item["bound_replacements"]):
            raise ValueError("Reference loader differs beyond the authorized allocation bound")
        checked += [(base, item["base_sha256"]), (copy, item["copy_sha256"])]
    fingerprint = hashlib.sha256(json.dumps({"base": production_fingerprint,
        "files": {r["copy_path"]: r["copy_sha256"] for r in manifest["files"]}}, sort_keys=True).encode()).hexdigest()
    if fingerprint != manifest["overlay_fingerprint"]:
        raise ValueError("Reference loader overlay fingerprint mismatch")
    return manifest, checked, fingerprint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("executable", "basis", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--gap", required=True, type=float)
    parser.add_argument("--film-grid", type=int, default=161)
    parser.add_argument("--mode", choices=["matched_provider", "direct_basis"], default="matched_provider")
    parser.add_argument("--epsilon", type=float, default=0.)
    parser.add_argument("--angles", type=float, nargs=3, metavar=("THETA_I", "THETA_J", "PHI"))
    parser.add_argument("--reference-overlay-manifest", type=Path)
    args = parser.parse_args()
    for name in ("executable", "basis", "output"):
        setattr(args, name, getattr(args, name).resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = {"schema_version": 1, "query_completed": False, "physical_accuracy_tested": False}
    start = time.monotonic()
    try:
        project = Path(__file__).resolve().parents[1]
        prepare = load_prepare(project)
        fingerprint = prepare.fast_source_fingerprint()
        expected_compiled = fingerprint
        overlay_files = []
        if args.reference_overlay_manifest is not None:
            if args.mode != "direct_basis":
                raise ValueError("Reference-only loader cannot evaluate/promote the production matched model")
            args.reference_overlay_manifest = args.reference_overlay_manifest.resolve()
            manifest, overlay_files, expected_compiled = verify_reference_overlay(args.reference_overlay_manifest, project, fingerprint)
            result.update(reference_only=True, reference_overlay=manifest,
                          base_production_fingerprint=fingerprint,
                          reference_overlay_manifest_path=str(args.reference_overlay_manifest),
                          reference_overlay_manifest_sha256=sha256(args.reference_overlay_manifest))
        inputs = {"basis_sha256": sha256(args.basis), "executable_sha256": sha256(args.executable)}
        result["basis_coefficients_sha256"] = coefficients_sha256(args.basis)
        command = [str(args.executable), str(args.basis), format(args.gap, ".17g"), str(args.film_grid)]
        if args.angles is not None:
            if args.mode != "matched_provider" or args.epsilon < 0. or not all(math.isfinite(x) for x in [args.epsilon] + args.angles):
                raise ValueError("Custom pose requires matched_provider and finite nonnegative regularization")
            command += [args.mode] + [format(x, ".17g") for x in [args.epsilon] + args.angles]
        elif args.epsilon != 0.:
            raise ValueError("Nonzero regularization requires explicit --angles")
        elif args.mode == "direct_basis":
            command.append(args.mode)
        completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   universal_newlines=True, timeout=600.)
        result.update(inputs)
        result.update(command=command, source_fingerprint=expected_compiled, exit_code=completed.returncode,
                      basis_path=str(args.basis), executable_path=str(args.executable))
        if completed.returncode:
            raise ValueError("Query failed: " + completed.stderr.strip())
        data = json.loads(completed.stdout)
        result.update(data)
        if data.get("compiled_source_fingerprint") != expected_compiled:
            raise ValueError("Compiled source fingerprint is stale or unbound")
        if (data.get("h_over_c") != args.gap or data.get("film_grid") != args.film_grid or
                data.get("evaluated_path") != args.mode or data.get("epsilon_over_c") != args.epsilon):
            raise ValueError("Query geometry/grid metadata mismatch")
        if args.angles is not None:
            ti, tj, phi = args.angles
            expected_axes = [[math.sin(ti), 0., math.cos(ti)],
                             [math.sin(tj)*math.cos(phi), math.sin(tj)*math.sin(phi), math.cos(tj)]]
            axes = data.get("axes", [])
            if len(axes) != 2 or any(len(row) != 3 for row in axes) or any(abs(axes[i][j]-expected_axes[i][j]) > 2e-14 for i in range(2) for j in range(3)):
                raise ValueError("Custom-pose axes mismatch")
        matrix = data.get("reduced11", [])
        if len(matrix) != 11 or any(len(row) != 11 or any(not math.isfinite(x) for x in row) for row in matrix):
            raise ValueError("Invalid total11 matrix")
        unchanged = (prepare.fast_source_fingerprint() == fingerprint and sha256(args.basis) == inputs["basis_sha256"] and
                     sha256(args.executable) == inputs["executable_sha256"])
        if args.reference_overlay_manifest is not None:
            unchanged = (unchanged and all(sha256(path) == expected for path, expected in overlay_files) and
                         sha256(args.reference_overlay_manifest) == result["reference_overlay_manifest_sha256"])
        result["inputs_unchanged_during_run"] = unchanged
        if not unchanged:
            raise ValueError("Source or input changed during the query")
        result["query_completed"] = True
    except Exception as error:
        result["error"] = str(error)
    result["elapsed_seconds"] = time.monotonic() - start
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"query_completed": result["query_completed"], "output": str(args.output), "error": result.get("error")}))
    return 0 if result["query_completed"] else 2


if __name__ == "__main__":
    sys.exit(main())
