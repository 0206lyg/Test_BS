#!/usr/bin/env python3
"""Create particles.data and in.slurry directly from sbatch/CLI parameters.

Random sequential insertion uses the Perram--Wertheim contact function only
to reject initial overlaps. It is not a surface-gap approximation and is not
used by the dynamics or lubrication kernel. Particles have uniform random
quaternions. A standalone C++ packer then grows and geometrically relaxes the
particles. Its steps are not physical dynamics or strain equilibration.
"""

import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile


DEFAULT_PACKER = Path(__file__).resolve().parents[2] / "build" / "lammps-slurry" / "slurry_pack"
DEFAULT_FAST_GENERATOR = DEFAULT_PACKER.with_name("slurry_precompute_fast")
DEFAULT_FAST_FILM_GRID = 161
DEFAULT_FAST_NPHI = 48
DEFAULT_FAST_GATE_EXECUTABLE = DEFAULT_PACKER.with_name("slurry_basis_physical_probe")
DEFAULT_FAST_GATE_RUNNER = Path(__file__).resolve().parents[2] / "validation_basis" / "run_physical_gate.py"
DEFAULT_BASIS_GENERATOR = DEFAULT_PACKER.with_name("slurry_generate_basis")
DEFAULT_INTERPOLATION_EXECUTABLE = DEFAULT_PACKER.with_name("slurry_table_interpolation_probe")
DEFAULT_INTERPOLATION_RUNNER = Path(__file__).resolve().parents[2] / "validation_basis" / "run_interpolation_gate.py"
FAST_GRID_POLICY = "curvature-angle16-plus-physical-nodes_log-gap-sqrt2_join2.5_v1"


def offline_executable(args, name):
    return getattr(args, "packer", DEFAULT_PACKER).expanduser().resolve().with_name(name)


def load_json_object(path):
    try:
        with path.open() as stream:
            record = json.load(stream)
    except (OSError, ValueError) as error:
        raise ValueError("Cannot read JSON record {}: {}".format(path, error))
    if not isinstance(record, dict):
        raise ValueError("Expected a JSON object in {}".format(path))
    return record


def write_json_atomic(path, record):
    with tempfile.NamedTemporaryFile(mode="w", prefix=".record-", dir=str(path.parent), delete=False) as stream:
        pending = Path(stream.name)
        json.dump(record, stream, indent=2, sort_keys=True)
        stream.write("\n")
    try:
        os.replace(str(pending), str(path))
    finally:
        if pending.exists():
            pending.unlink()


def source_gate_summary(path, args, source, fingerprint, film_grid, mode="matched_provider"):
    """A check of the serialized basis alone cannot qualify the matched model."""
    gate = load_json_object(path)
    expected = {"gate": "source_physical", "evaluated_path": mode,
                "basis_sha256": file_sha256(source), "source_fingerprint": fingerprint,
                "compiled_source_fingerprint": fingerprint,
                "film_grid": film_grid,
                "fixture_sha256": file_sha256(Path(__file__).resolve().parents[1] / "tests/data/independent_pair_reference.txt")}
    for key, value in expected.items():
        if gate.get(key) != value:
            raise ValueError("Source physical gate {} mismatch; rerun the matched-provider check".format(key))
    if not math.isclose(float(gate.get("aspect_ratio", 0)), args.ar, rel_tol=1e-10):
        raise ValueError("Source physical gate AR mismatch")
    limit, error = gate.get("limit"), gate.get("max_all11_error")
    free_error = gate.get("max_free_strain_error")
    if (not isinstance(limit, (int, float)) or not isinstance(error, (int, float)) or
            not isinstance(free_error, (int, float)) or not math.isfinite(free_error) or free_error < 0 or
            not math.isfinite(limit) or not math.isfinite(error) or not 0 < limit <= 0.05 or
            error < 0 or max(error, free_error) > limit or gate.get("passed") is not True):
        raise ValueError("Source physical gate has not passed the 5% limit; full automatic table generation stopped")
    return {"status": "passed", "max_error": max(error, free_error), "limit": limit,
            "reference_sha": gate.get("fixture_sha256"), "report_sha256": file_sha256(path),
            "report": str(path), "evaluated_path": mode,
            "compiled_source_fingerprint": fingerprint,
            "scope": "Reported independent reference cases only; no bulk or universal accuracy claim"}


def resolve_source_gate(args, source, fingerprint, film_grid, destination):
    if not math.isclose(args.ar, 10.0, rel_tol=1e-10):
        return resolve_generic_source_gate(args, source, fingerprint, film_grid, destination)
    supplied = getattr(args, "finite_gap_fast_source_gate", None)
    report = supplied.expanduser().resolve() if supplied is not None else destination
    reports = (("direct_basis", report.with_suffix(".direct.json")), ("matched_provider", report))
    if supplied is None:
        executable = (getattr(args, "finite_gap_fast_gate_executable", None) or offline_executable(args, "slurry_basis_physical_probe")).expanduser().resolve()
        if not DEFAULT_FAST_GATE_RUNNER.is_file() or not executable.is_file() or not os.access(str(executable), os.X_OK):
            raise ValueError("Full automatic table generation requires the compiled source-accuracy gate; rebuild or provide --finite-gap-fast-source-gate")
        for mode, output in reports:
            command = [sys.executable, str(DEFAULT_FAST_GATE_RUNNER), "--executable", str(executable),
                       "--basis", str(source), "--mode", mode, "--film-grid", str(film_grid),
                       "--report", str(output)]
            completed = subprocess.run(command)
            if completed.returncode != 0:
                raise ValueError("{} source physical gate failed (report {}); full table was not generated".format(mode, output))
    checks = [source_gate_summary(path, args, source, fingerprint, film_grid, mode) for mode, path in reports]
    return {"status": "passed", "max_error": max(check["max_error"] for check in checks),
            "limit": min(check["limit"] for check in checks), "reference_sha": checks[1]["reference_sha"],
            "checks": checks, "scope": "Direct near basis and matched-provider reference cases; bulk remains untested"}


def resolve_generic_source_gate(args, source, fingerprint, film_grid, destination):
    """AR-parametric numerical checks do not substitute for independent pair data."""
    supplied = getattr(args, "finite_gap_fast_source_gate", None)
    report = supplied.expanduser().resolve() if supplied is not None else destination
    if supplied is None:
        executable = (getattr(args, "finite_gap_fast_gate_executable", None) or offline_executable(args, "slurry_basis_physical_probe")).expanduser().resolve()
        if not executable.is_file() or not os.access(str(executable), os.X_OK):
            raise ValueError("Generic-AR table generation requires the isolated spheroid numerical probe; rebuild offline tools")
        completed = subprocess.run([sys.executable, str(DEFAULT_FAST_GATE_RUNNER), "--executable", str(executable),
                                    "--basis", str(source), "--mode", "isolated_only", "--film-grid", str(film_grid),
                                    "--report", str(report)])
        if completed.returncode != 0:
            raise ValueError("AR={} isolated spheroid numerical gate failed; no full table was generated".format(args.ar))
    raw = load_json_object(report)
    for key, expected in (("evaluated_path", "isolated_only"), ("basis_sha256", file_sha256(source)),
                          ("source_fingerprint", fingerprint), ("compiled_source_fingerprint", fingerprint), ("film_grid", film_grid)):
        if raw.get(key) != expected:
            raise ValueError("Generic-AR numerical gate {} identity mismatch".format(key))
    if not math.isclose(float(raw.get("aspect_ratio", 0)), args.ar, rel_tol=1e-10):
        raise ValueError("Generic-AR numerical gate AR mismatch")
    isolated = [record for record in raw.get("records", []) if record.get("stage") == "isolated_units"]
    if raw.get("passed") is not True or not isolated or any(record.get("passed") is not True for record in isolated):
        raise ValueError("Generic-AR isolated numerical gate did not pass")
    for record in isolated:
        for field in ("all11_error", "raw_reciprocity"):
            value = record.get(field)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= .001:
                raise ValueError("Generic-AR isolated {} exceeds the 0.1% numerical limit".format(field))
    numerical = {"status": "passed", "isolated_checks": isolated, "report": str(report),
                 "report_sha256": file_sha256(report), "boundary_checks": [],
                 "compiled_source_fingerprint": fingerprint,
                 "scope": "Analytic isolated response and available basis boundary diagnostics; no independent pair comparison"}
    generation = Path(str(source) + ".pipeline.json")
    if generation.is_file():
        provenance = load_json_object(generation)
        if provenance.get("basis_sha256") != file_sha256(source):
            raise ValueError("Generated basis provenance has a stale basis identity")
        basis_header = read_basis_header(source, args.ar)
        labels = ("near",) if basis_header["format"] == "SLURRY_HYBRID_V3" else ("far", "near")
        numerical["active_source_policy"] = basis_header["source_policy"]
        for label in labels:
            path = Path(str(source) + "." + label + "-generation.json")
            record = load_json_object(path)
            expected_sha = provenance.get("generation_reports", {}).get(label)
            if expected_sha != file_sha256(path):
                raise ValueError("Generated {} basis boundary report identity mismatch".format(label))
            if not math.isclose(float(record.get("aspect_ratio", 0)), args.ar, rel_tol=1e-10):
                raise ValueError("Generated basis boundary report AR mismatch")
            error = record.get("offgrid_physical11_weighted_relative")
            basis_error = record.get("offgrid_basis_weighted_relative")
            if (not isinstance(error, (int, float)) or not isinstance(basis_error, (int, float)) or
                    not math.isfinite(error) or not math.isfinite(basis_error) or
                    not 0 <= error <= .001 or not 0 <= basis_error <= .01 or record.get("loader_roundtrip_pass") is not True):
                raise ValueError("Generated {} basis boundary numerical check failed; full table was not generated".format(label))
            numerical["boundary_checks"].append({"basis": label, "physical11_relative_error": error,
                                                  "physical11_limit": .001, "basis_relative_error": basis_error,
                                                  "basis_limit": .01, "report_sha256": file_sha256(path)})
    args.finite_gap_fast_source_numerical_gate = numerical
    return {"status": "not_run", "reason": "Fixed independent pair references cover AR10 only",
            "scope": "AR={} basis passed available numerical checks; independent pair accuracy remains unverified".format(args.ar)}


def read_fast_manifest(table, args, source=None, fingerprint=None, expected_settings=None):
    path = Path(str(table) + ".manifest.json")
    if not path.is_file():
        provenance = Path(str(table) + ".json")
        profile = load_json_object(provenance).get("profile", "unverified") if provenance.is_file() else "unverified"
        return {"schema": 1, "ar": args.ar, "geometry": "equal_size_oblate",
                "source_gate": {"status": "unverified"},
                "table": {"sha256": file_sha256(table), "settings": {"origin": "explicit_file", "profile": profile}},
                "interpolation_gate": {"status": "unverified"},
                "quality": "smoke_only" if profile == "smoke" else "unverified"}
    manifest = load_json_object(path)
    if manifest.get("schema") != 1 or manifest.get("geometry") != "equal_size_oblate":
        raise ValueError("Unsupported fast-table manifest schema/geometry")
    if not math.isclose(float(manifest.get("ar", 0)), args.ar, rel_tol=1e-10):
        raise ValueError("Fast-table manifest AR mismatch")
    if manifest.get("table", {}).get("sha256") != file_sha256(table):
        raise ValueError("Fast-table manifest SHA mismatch; table identity is stale")
    if source is not None and manifest.get("basis_sha256") != file_sha256(source):
        raise ValueError("Fast-table manifest basis identity mismatch")
    if fingerprint is not None and manifest.get("source_code_sha256") != fingerprint:
        raise ValueError("Fast-table manifest numerical source identity mismatch")
    if expected_settings is not None and manifest.get("table", {}).get("settings") != expected_settings:
        raise ValueError("Fast-table manifest generation settings mismatch")
    for label, maximum in (("source_gate", 0.05), ("interpolation_gate", 0.01)):
        gate = manifest.get(label)
        if not isinstance(gate, dict) or gate.get("status") not in ("passed", "failed", "not_run", "unverified"):
            raise ValueError("Fast-table manifest has no valid {} status".format(label))
        if gate["status"] == "passed":
            limit, error = gate.get("limit"), gate.get("max_error")
            if (not isinstance(limit, (int, float)) or not isinstance(error, (int, float)) or
                    not math.isfinite(limit) or not math.isfinite(error) or not 0 < limit <= maximum or
                    not 0 <= error <= limit):
                raise ValueError("Fast-table manifest {} pass conflicts with its numerical result".format(label))
    if manifest.get("table", {}).get("settings", {}).get("profile") == "smoke" and any(
            manifest[key].get("status") == "passed" for key in ("source_gate", "interpolation_gate")):
        raise ValueError("A coarse smoke table cannot be marked physically or interpolation qualified")
    if fingerprint is not None:
        checks = list(manifest.get("source_gate", {}).get("checks", []))
        checks += [manifest.get("source_numerical_gate", {}), manifest.get("interpolation_gate", {})]
        for check in checks:
            if check.get("status") == "passed" and check.get("compiled_source_fingerprint") != fingerprint:
                raise ValueError("Fast-table validation report is not bound to the currently compiled numerical sources")
    return manifest


def run_interpolation_gate(args, source, table, fingerprint, film_grid):
    executable = (getattr(args, "finite_gap_fast_interpolation_executable", None) or offline_executable(args, "slurry_table_interpolation_probe")).expanduser().resolve()
    if not DEFAULT_INTERPOLATION_RUNNER.is_file() or not executable.is_file() or not os.access(str(executable), os.X_OK):
        raise ValueError("Full automatic table activation requires slurry_table_interpolation_probe; rebuild the offline validation tools")
    report = Path(str(table) + ".interpolation-gate.json")
    completed = subprocess.run([sys.executable, str(DEFAULT_INTERPOLATION_RUNNER), "--executable", str(executable),
                               "--basis", str(source), "--table", str(table), "--film-grid", str(film_grid),
                               "--report", str(report)])
    gate = load_json_object(report)
    expected = {"gate": "table_interpolation", "evaluated_path": "complete_table_vs_direct_matched_source",
                "basis_sha256": file_sha256(source), "table_sha256": file_sha256(table),
                "source_fingerprint": fingerprint, "compiled_source_fingerprint": fingerprint, "film_grid": film_grid}
    for key, value in expected.items():
        if gate.get(key) != value:
            raise ValueError("Interpolation gate {} identity mismatch; table was not activated".format(key))
    if not math.isclose(float(gate.get("aspect_ratio", 0)), args.ar, rel_tol=1e-10):
        raise ValueError("Interpolation gate AR mismatch")
    error, limit = gate.get("max_error"), gate.get("limit")
    passed = (completed.returncode == 0 and gate.get("passed") is True and
              isinstance(error, (int, float)) and isinstance(limit, (int, float)) and
              math.isfinite(error) and math.isfinite(limit) and 0 <= error <= limit <= 0.01 and limit > 0)
    return {"status": "passed" if passed else "failed", "max_error": error, "limit": limit,
            "report_sha256": file_sha256(report), "report": str(report),
            "compiled_source_fingerprint": fingerprint,
            "scope": "Fixed off-grid pair interpolation checks; independent source and bulk checks are separate"}


def read_basis_header(path, requested_ar):
    """Check identity before an expensive offline solve; C++ validates payloads."""
    with path.open() as stream:
        magic = stream.readline().strip()
        parameters = stream.readline().split()
        anchor_line = stream.readline().split() if magic in ("SLURRY_HYBRID_V2", "SLURRY_HYBRID_V3") else None
    if magic not in ("SLURRY_SCATTERING_V1", "SLURRY_SCATTERING_V2", "SLURRY_HYBRID_V1", "SLURRY_HYBRID_V2", "SLURRY_HYBRID_V3") or len(parameters) != 2:
        raise ValueError("Invalid finite-gap scattering basis header")
    try:
        basis_ar, degree = float(parameters[0]), int(parameters[1])
    except ValueError:
        raise ValueError("Invalid finite-gap scattering basis AR/order")
    if not math.isfinite(basis_ar) or basis_ar < 1 or not 1 <= degree <= 30:
        raise ValueError("Invalid finite-gap scattering basis AR/order")
    if not math.isclose(basis_ar, requested_ar, rel_tol=1e-10):
        raise ValueError("Scattering basis AR={} does not match requested AR={}; no fallback".format(basis_ar, requested_ar))
    anchors = []
    if anchor_line is not None:
        try:
            if len(anchor_line) < 2 or anchor_line[0] != "ANCHORS":
                raise ValueError("missing ANCHORS marker")
            count = int(anchor_line[1])
            anchors = list(map(float, anchor_line[2:]))
            if not 2 <= count <= 32 or len(anchors) != count or not all(math.isfinite(x) and x > 0 for x in anchors):
                raise ValueError("invalid anchor values/count")
            if any(y <= x for x, y in zip(anchors, anchors[1:])):
                raise ValueError("anchors are not increasing")
            if magic == "SLURRY_HYBRID_V3" and (count != 2 or any(
                    not math.isclose(x, y, rel_tol=1e-12) for x, y in zip(anchors, (0.5, 1.0)))):
                raise ValueError("V3 requires exactly the two anchors 0.5 and 1")
        except ValueError as error:
            raise ValueError("Invalid {} anchors: {}".format(magic, error))
    if magic == "SLURRY_HYBRID_V3":
        policy = {"finite_gap": "direct near basis at the actual effective gap for heff/c >= 0.5",
                  "near_contact": "same first residual segment as V2 below heff/c = 0.5",
                  "active_basis_slot": "near", "far_slot_used": False}
    else:
        policy = {"finite_gap": "legacy source policy for {}".format(magic),
                  "far_slot_used": magic in ("SLURRY_HYBRID_V1", "SLURRY_HYBRID_V2")}
    return {"format": magic, "aspect_ratio": basis_ar, "degree": degree, "anchors_over_c": anchors,
            "source_policy": policy,
            "payload_validation": "C++ provider checks all matrices when loaded"}


def basis_generation_settings(args):
    """Initial geometric resolution, followed by the unchanged numerical gates.

    Confocal source depth stays at c/2. Its smallest surface separation is
    a-sqrt(a*a-3*c*c/4), asymptotically 3*c*c/(8*a). Thus meridional ring
    resolution grows with AR and isolated-kernel azimuth resolution with AR^2.
    Runtime pair quadrature resolves the fixed h/c matching gaps, hence AR.
    These are starting resolutions, not an accuracy certificate for every AR.
    """
    scale = max(1.0, float(args.ar) / 10.0)
    automatic = {
        "source_rings": int(math.ceil(96 * scale / 16)) * 16,
        "source_phi": int(math.ceil(128 * scale / 32)) * 32,
        "fit_phi": int(math.ceil(4096 * scale * scale / 128)) * 128,
    }
    settings = {}
    for key, default in automatic.items():
        value = getattr(args, "finite_gap_fast_basis_" + key, None)
        settings[key] = default if value is None else value
        integer(settings[key], "finite-gap-fast-basis-" + key.replace("_", "-"), minimum=4)
    settings["check_phi"] = 2 * settings["fit_phi"]
    near = getattr(args, "finite_gap_fast_basis_degree", 12)
    far = getattr(args, "finite_gap_fast_far_degree", 8)
    for degree in (near, far):
        integer(degree, "finite-gap basis degree")
        if degree > 30:
            raise ValueError("Finite-gap basis degree must not exceed 30")
        coefficients = 3 * (degree + 1) ** 2 - 1
        if settings["source_rings"] * (2 * (degree + 2) + 1) * 3 * coefficients > 16000000:
            raise ValueError("Requested AR/basis resolution exceeds the generator coefficient storage limit")
    if (settings["source_rings"] > 512 or settings["source_phi"] > 1024 or
            settings["source_rings"] * settings["source_phi"] > 100000 or
            settings["check_phi"] > 1048576):
        raise ValueError("Requested AR/basis resolution exceeds the generator dimensions; no generation started")
    settings["degrees"] = {"near": near, "far": far}
    return settings


def check_generated_basis_report(path, ar):
    """Do not install a generated source that already fails its own diagnostics."""
    record = load_json_object(path)
    failures = []
    if not math.isclose(float(record.get("aspect_ratio", 0)), ar, rel_tol=1e-10):
        failures.append("AR mismatch")
    for field, limit in (("isolated_all11_energy_error", .001), ("isolated_reciprocity", .001),
                         ("offgrid_physical11_weighted_relative", .001),
                         ("offgrid_basis_weighted_relative", .01)):
        value = record.get(field)
        if (not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= limit):
            failures.append("{}={} (limit {})".format(field, value, limit))
    if record.get("loader_roundtrip_pass") is not True:
        failures.append("serialized loader roundtrip did not pass")
    if failures:
        raise ValueError("AR={} generated near basis failed: {}. Report: {}. "
                         "No basis was installed and no table was generated".format(ar, "; ".join(failures), path))
    return record


def resolve_fast_basis(args):
    source = args.finite_gap_fast_source
    if source is None:
        return None
    source = source.expanduser().resolve()
    settings = (basis_generation_settings(args) if not source.exists() or
                Path(str(source) + ".pipeline.json").is_file() else None)
    if not source.exists():
        generator = (getattr(args, "finite_gap_fast_basis_generator", None) or offline_executable(args, "slurry_generate_basis")).expanduser().resolve()
        if not generator.is_file() or not os.access(str(generator), os.X_OK):
            raise ValueError("Missing basis {} and executable AR-basis generator {}; build slurry_generate_basis or supply a matching basis".format(source, generator))
        near_degree, far_degree = settings["degrees"]["near"], settings["degrees"]["far"]
        source_rings, source_phi = settings["source_rings"], settings["source_phi"]
        source.parent.mkdir(parents=True, exist_ok=True)
        print("Generating actual AR={} scattering bases (far degree {}, near degree {}, source rings {}, "
              "runtime phi {}, fit/check phi {}/{}); physical gate follows separately".format(
              args.ar, far_degree, near_degree, source_rings, source_phi,
              settings["fit_phi"], settings["check_phi"]), flush=True)
        with tempfile.TemporaryDirectory(prefix=".basis-build-", dir=str(source.parent)) as temporary:
            far, hybrid = Path(temporary) / "far.ssb", Path(temporary) / "matched.ssb"
            commands = []
            for degree, output, label in ((far_degree, far, "far"), (near_degree, hybrid, "near")):
                command = [str(generator), "--ar", format_float(args.ar), "--degree", str(degree),
                           "--source-rings", str(source_rings), "--source-phi", str(source_phi),
                           "--fit-phi", str(settings["fit_phi"]), "--check-phi", str(settings["check_phi"]),
                           "--output", str(output), "--report", str(source) + "." + label + "-generation.json"]
                if label == "near":
                    command += ["--far-basis", str(far), "--hybrid-version", "3", "--anchor-gaps", "0.5,1"]
                commands.append(command)
                try:
                    subprocess.run(command, check=True)
                except subprocess.CalledProcessError as error:
                    raise ValueError("AR-basis {} generation failed with exit {}; no table was generated".format(label, error.returncode))
            header = read_basis_header(hybrid, args.ar)
            if header["format"] != "SLURRY_HYBRID_V3":
                raise ValueError("Basis generator did not produce the requested V3 source; rebuild the offline generator")
            # V3 uses only the near slot; do not reject its inactive far slot.
            check_generated_basis_report(Path(str(source) + ".near-generation.json"), args.ar)
            os.replace(str(far), str(source) + ".far.ssb")
            os.replace(str(hybrid), str(source))
            write_json_atomic(Path(str(source) + ".pipeline.json"), {
                "schema": 1, "ar": args.ar, "geometry": "equal_size_oblate", "basis_sha256": file_sha256(source),
                "source_code_sha256": fast_source_fingerprint(), "generator_sha256": file_sha256(generator),
                "degrees": {"near": near_degree, "far": far_degree}, "commands": commands,
                "source_format": header["format"], "source_policy": header["source_policy"],
                "source_anchors_over_c": header["anchors_over_c"],
                "source_rings": source_rings, "source_phi": source_phi,
                "fit_phi": settings["fit_phi"], "check_phi": settings["check_phi"],
                "resolution_policy": "confocal_geometry_scaled_v1",
                "generation_reports": {label: file_sha256(Path(str(source) + "." + label + "-generation.json")) for label in ("far", "near")},
                "source_gate": {"status": "not_run"}})
    if not source.is_file():
        raise ValueError("Fast-table basis is not a regular file: {}".format(source))
    pipeline = Path(str(source) + ".pipeline.json")
    if pipeline.is_file():
        record = load_json_object(pipeline)
        # Old pipeline records used fixed quadrature, but did not store these keys.
        record.setdefault("fit_phi", 4096)
        record.setdefault("check_phi", 8192)
        expected = settings
        if record.get("basis_sha256") != file_sha256(source):
            raise ValueError("Generated basis changed without matching provenance; select a coherent basis file")
        for key, value in expected.items():
            if record.get(key) != value:
                raise ValueError("Existing generated basis {} does not match requested {}; select matching settings or a new source filename".format(key, value))
    args.finite_gap_fast_basis_header = read_basis_header(source, args.ar)
    args.finite_gap_fast_source = source
    return source


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fast_source_fingerprint():
    """Invalidate automatic caches when numerical headers change, even before rebuild."""
    project = Path(__file__).resolve().parents[1]
    paths = list((project / "src").glob("*.h"))
    paths += list((project / "vendor" / "gr_re2").glob("*.h"))
    paths.append(project / "tools" / "precompute_fast.cpp")
    paths += list((project / "tools").glob("basis_*.h"))
    paths += list((project / "tools").glob("slurry_generate_basis.cpp"))
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path.relative_to(project)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_sha256(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def read_fast_table_header(path, args, a):
    """Check coverage and complete payload before starting particle preparation.

    The C++ provider separately validates the matrix values at load/evaluation.
    These checks prevent a wrong AR, truncated cache or uncovered gap range from
    silently selecting an old model or failing only after expensive packing.
    """
    with path.open("rb") as stream:
        try:
            magic = stream.readline().decode("ascii").strip()
            values = stream.readline().decode("ascii").split()
            if magic not in ("SLURRY_SPD_LOGFOURIER_V1", "SLURRY_SPD_LOGFOURIER_V2") or len(values) != 6:
                raise ValueError("Invalid finite-gap fast table header")
            table_ar, lower, upper = map(float, values[:3])
            ntheta, nphi, ngap = map(int, values[3:])
            if not (4 <= ntheta <= 128 and 4 <= nphi <= 256 and 4 <= ngap <= 256):
                raise ValueError("Fast table node counts exceed supported interpolation limits")
            if nphi % 2:
                raise ValueError("Matrix-log/Fourier table requires an even number of phi nodes")
            joins = []
            if magic == "SLURRY_SPD_LOGFOURIER_V2":
                line = stream.readline().decode("ascii").split()
                if not line:
                    raise ValueError("Missing V2 table join list")
                count = int(line[0])
                joins = list(map(float, line[1:]))
                if not 0 <= count <= 32 or len(joins) != count:
                    raise ValueError("Invalid V2 table join count")
            nodes = [list(map(float, stream.readline().decode("ascii").split()))
                     for _ in range(3)]
            if stream.readline().decode("ascii").strip() != "DATA":
                raise ValueError("Fast table is missing its DATA marker")
            payload_offset = stream.tell()
        except (UnicodeError, OverflowError) as error:
            raise ValueError("Malformed finite-gap fast table header: {}".format(error))
    if not all(math.isfinite(x) for x in (table_ar, lower, upper)) or lower <= 0 or upper <= lower:
        raise ValueError("Fast table has invalid AR/gap bounds")
    for label, row, count in zip(("theta", "phi", "gap"), nodes, (ntheta, nphi, ngap)):
        if len(row) != count or not all(math.isfinite(x) for x in row):
            raise ValueError("Fast table has invalid {} nodes".format(label))
        if not all(y > x for x, y in zip(row, row[1:])):
            raise ValueError("Fast table {} nodes must increase strictly".format(label))
    phi_step = 2 * math.pi / nphi
    if not (math.isclose(nodes[0][0], 0.0, abs_tol=1e-12) and
            math.isclose(nodes[0][-1], math.pi / 2, rel_tol=0, abs_tol=1e-12)):
        raise ValueError("Fast table theta nodes must cover [0, pi/2]")
    if any(not math.isclose(value, -math.pi + i * phi_step, rel_tol=1e-10, abs_tol=1e-10)
           for i, value in enumerate(nodes[1])):
        raise ValueError("Fast table phi nodes must be uniform over [-pi, pi)")
    if not (math.isclose(nodes[2][0], lower, rel_tol=1e-10) and
            math.isclose(nodes[2][-1], upper, rel_tol=1e-10)):
        raise ValueError("Fast table gap nodes disagree with its bounds")
    if magic == "SLURRY_SPD_LOGFOURIER_V1":
        joins = [2.5] if any(math.isclose(x, 2.5, rel_tol=0, abs_tol=1e-12) for x in nodes[2]) else []
    elif (any(not math.isfinite(x) or x < lower or x > upper for x in joins) or
          any(y <= x for x, y in zip(joins, joins[1:])) or
          any(not any(math.isclose(x, y, rel_tol=1e-12, abs_tol=1e-14) for y in nodes[2]) for x in joins)):
        raise ValueError("V2 table joins must increase and be exact in-range gap nodes")
    if not math.isclose(table_ar, args.ar, rel_tol=1e-10):
        raise ValueError("Fast table AR={} does not match requested AR={}; no fallback".format(table_ar, args.ar))
    selected_nphi = getattr(args, "finite_gap_fast_nphi", DEFAULT_FAST_NPHI)
    if nphi != selected_nphi:
        raise ValueError("Fast table n_phi={} does not match requested {}; select --finite-gap-fast-nphi explicitly for another trial resolution".format(nphi, selected_nphi))
    c = a / args.ar
    epsilon_over_c = args.lubrication_epsilon_nm * 1e-9 / c
    required_upper = args.finite_gap_range * args.ar + epsilon_over_c
    if lower > epsilon_over_c * (1 + 1e-10) or upper < required_upper * (1 - 1e-10):
        raise ValueError("Fast table does not cover the requested regularization and outer range; regenerate it")
    payload_bytes = ntheta * ntheta * nphi * ngap * 66 * 8
    if payload_bytes > (1 << 29) * 8:
        raise ValueError("Fast table exceeds the runtime payload size limit")
    if path.stat().st_size != payload_offset + payload_bytes:
        raise ValueError("Fast table binary payload is truncated or has extra bytes")
    return {"format": magic, "ar": table_ar, "h_min_over_c": lower, "h_max_over_c": upper,
            "joins_over_c": joins,
            "n_theta": ntheta, "n_phi": nphi, "n_gap": ngap}


def resolve_fast_table(args, a):
    """Reuse a complete table, or generate it only for explicit fast selection."""
    requested_table = args.finite_gap_fast_table
    requested_table = requested_table.expanduser().resolve() if requested_table is not None else None
    source = resolve_fast_basis(args)
    table = requested_table
    generator = (args.finite_gap_fast_generator or offline_executable(args, "slurry_precompute_fast")).expanduser().resolve()
    film_grid = getattr(args, "finite_gap_fast_film_grid", DEFAULT_FAST_FILM_GRID)
    integer(film_grid, "finite-gap-fast-film-grid", minimum=21)
    nphi = getattr(args, "finite_gap_fast_nphi", DEFAULT_FAST_NPHI)
    smoke = getattr(args, "finite_gap_fast_smoke", False)
    if smoke and nphi == DEFAULT_FAST_NPHI:
        nphi = 8
        args.finite_gap_fast_nphi = nphi
    if smoke and nphi != 8:
        raise ValueError("Coarse end-to-end smoke requires n_phi=8; it is not a selectable physical-resolution profile")
    integer(nphi, "finite-gap-fast-nphi", minimum=8)
    if nphi > 256 or nphi % 2:
        raise ValueError("finite-gap-fast-nphi must be even and between 8 and 256")
    threads = getattr(args, "finite_gap_fast_threads", 1)
    integer(threads, "finite-gap-fast-threads")
    profile = "smoke" if smoke else "full"
    epsilon_over_c = args.lubrication_epsilon_nm * 1e-9 / (a / args.ar)
    numerical_fingerprint = fast_source_fingerprint()
    settings = {"profile": profile, "film_grid": film_grid, "n_phi": nphi,
                "angle_intervals": 3 if smoke else 16,
                "grid_policy": "coarse_end_to_end_smoke_v1" if smoke else FAST_GRID_POLICY,
                "epsilon_over_c": epsilon_over_c, "range_over_a": args.finite_gap_range}
    source_format = args.finite_gap_fast_basis_header["format"] if source is not None else None
    table_format = "SLURRY_SPD_LOGFOURIER_V2" if source_format in ("SLURRY_HYBRID_V2", "SLURRY_HYBRID_V3") else "SLURRY_SPD_LOGFOURIER_V1"
    if source is not None:
        settings["source_format"] = source_format
        settings["table_format"] = table_format
        settings["source_anchors_over_c"] = args.finite_gap_fast_basis_header["anchors_over_c"]
        settings["source_policy"] = args.finite_gap_fast_basis_header["source_policy"]
        if source_format in ("SLURRY_HYBRID_V2", "SLURRY_HYBRID_V3"):
            settings["grid_policy"] = ("coarse_end_to_end_smoke_with_source_joins_v2" if smoke else
                                       "curvature-angle16-plus-physical-nodes_log-gap-sqrt2_explicit-source-joins_v2")
    automatic_cache = table is None
    generated = False
    if table is None:
        if source is None:
            raise ValueError("finite_gap_fast requires --finite-gap-fast-table or --finite-gap-fast-source; no fallback")
        if not generator.is_file() or not os.access(str(generator), os.X_OK):
            raise ValueError("Missing executable fast-table generator: {}".format(generator))
        identity = "|".join((table_format, file_sha256(source), file_sha256(generator),
                             numerical_fingerprint, format_float(args.ar),
                             json.dumps(settings, sort_keys=True, separators=(",", ":"))))
        args.finite_gap_fast_cache_source_fingerprint = numerical_fingerprint
        key = hashlib.sha256(identity.encode("ascii")).hexdigest()[:24]
        table = source.parent / ".fast_cache" / ("ar{}_{}.sft".format(format_float(args.ar), key))
    if not table.exists():
        if source is None:
            raise ValueError("Missing fast table {}; provide --finite-gap-fast-source to generate it".format(table))
        if not generator.is_file() or not os.access(str(generator), os.X_OK):
            raise ValueError("Missing executable fast-table generator: {}".format(generator))
        table.parent.mkdir(parents=True, exist_ok=True)
        if not smoke and args.finite_gap_fast_basis_header["format"] not in ("SLURRY_HYBRID_V1", "SLURRY_HYBRID_V2", "SLURRY_HYBRID_V3"):
            raise ValueError("Full automatic table generation requires an actual hybrid source bundle")
        source_gate = ({"status": "not_run", "reason": "Coarse end-to-end smoke is not a physical accuracy test"}
                       if smoke else resolve_source_gate(args, source, numerical_fingerprint, film_grid,
                                                        Path(str(table) + ".source-gate.json")))
        print("Generating finite_gap_fast cache for AR={} at {}".format(args.ar, table), flush=True)
        with tempfile.TemporaryDirectory(prefix=".fast-build-", dir=str(table.parent)) as temporary:
            pending = Path(temporary) / "table.sft"
            command = [str(generator), "--basis", str(source), "--output", str(pending),
                       "--ar", format_float(args.ar), "--epsilon-over-c", format_float(epsilon_over_c),
                       "--range", format_float(args.finite_gap_range), "--film-grid", str(film_grid),
                       "--phi-nodes", str(nphi), "--threads", str(threads),
                       "--source-fingerprint", numerical_fingerprint,
                       "--checkpoint-dir", str(table) + ".checkpoints"]
            if not smoke:
                command += ["--angle-intervals", str(settings["angle_intervals"])]
            if smoke:
                command += ["--smoke", "1"]
            try:
                subprocess.run(command, check=True)
            except subprocess.CalledProcessError as error:
                raise ValueError("Fast-table generation failed with exit {}; no fallback".format(error.returncode))
            header = read_fast_table_header(pending, args, a)
            if header["format"] != table_format:
                raise ValueError("Generated table format does not match the selected source bridge version")
            if table_format == "SLURRY_SPD_LOGFOURIER_V2":
                expected_joins = [x for x in args.finite_gap_fast_basis_header["anchors_over_c"]
                                  if header["h_min_over_c"] <= x <= header["h_max_over_c"]]
                if len(header["joins_over_c"]) != len(expected_joins) or any(
                        not math.isclose(x, y, rel_tol=1e-12) for x, y in zip(header["joins_over_c"], expected_joins)):
                    raise ValueError("Generated V2 table joins do not match the selected source anchors")
            if fast_source_fingerprint() != numerical_fingerprint:
                raise ValueError("Numerical source changed during table generation; incomplete result will not be activated")
            generator_record = Path(str(pending) + ".json")
            saved_generator_record = Path(str(table) + ".json")
            manifest = {"schema": 1, "ar": args.ar, "geometry": "equal_size_oblate",
                        "basis_sha256": file_sha256(source), "source_code_sha256": numerical_fingerprint,
                        "source_gate": source_gate,
                        "source_numerical_gate": getattr(args, "finite_gap_fast_source_numerical_gate", {"status": "not_run"}),
                        "table": {"sha256": file_sha256(pending), "settings": settings, "header": header,
                                  "generator_sha256": file_sha256(generator), "threads_requested": threads},
                        "interpolation_gate": {"status": "not_run"},
                        "quality": "smoke_only" if smoke else "source_checks_recorded_interpolation_not_checked"}
            if generator_record.is_file():
                manifest["table"]["generator_report_sha256"] = file_sha256(generator_record)
                os.replace(str(generator_record), str(saved_generator_record))
            write_json_atomic(Path(str(table) + ".manifest.json"), manifest)
            os.replace(str(pending), str(table))
            generated = True
    if not table.is_file():
        raise ValueError("Fast table is not a regular file: {}".format(table))
    read_fast_table_header(table, args, a)
    if automatic_cache and not Path(str(table) + ".manifest.json").is_file():
        raise ValueError("Automatic fast cache has no manifest; use a newly generated cache or an explicit unverified table")
    args.finite_gap_fast_manifest = read_fast_manifest(table, args, source, numerical_fingerprint,
                                                      settings if source is not None else None)
    if automatic_cache and not smoke:
        gate_name = "source_gate" if math.isclose(args.ar, 10.0, rel_tol=1e-10) else "source_numerical_gate"
        if args.finite_gap_fast_manifest.get(gate_name, {}).get("status") != "passed":
            raise ValueError("Automatic full cache has not passed its required {}".format(gate_name))
    if (automatic_cache or generated) and not smoke and args.finite_gap_fast_manifest["interpolation_gate"]["status"] != "passed":
        args.finite_gap_fast_manifest["interpolation_gate"] = run_interpolation_gate(args, source, table, numerical_fingerprint, film_grid)
        passed = args.finite_gap_fast_manifest["interpolation_gate"]["status"] == "passed"
        physical_passed = args.finite_gap_fast_manifest["source_gate"]["status"] == "passed"
        args.finite_gap_fast_manifest["quality"] = (("reference_cases_passed_bulk_unverified" if physical_passed else
                                                     "numerical_and_interpolation_checks_passed_pair_physics_unverified")
                                                    if passed else "interpolation_failed")
        write_json_atomic(Path(str(table) + ".manifest.json"), args.finite_gap_fast_manifest)
        if not passed:
            raise ValueError("Complete-table interpolation gate failed; table preserved for diagnosis and runtime preparation stopped")
    args.finite_gap_fast_table = table
    return table


def positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite positive number")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return float(value)


def integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def quaternion(rng):
    """Uniform Haar measure on SO(3), stored as LAMMPS (w,x,y,z)."""
    u, v, w = (rng.random() for _ in range(3))
    return (math.sqrt(u) * math.cos(2 * math.pi * w),
            math.sqrt(1 - u) * math.sin(2 * math.pi * v),
            math.sqrt(1 - u) * math.cos(2 * math.pi * v),
            math.sqrt(u) * math.sin(2 * math.pi * w))


def shape_matrix(quat, axial_ratio):
    """Q/a^2 for an ellipsoid with semiaxes (a,a,c), c/a=axial_ratio."""
    w, x, y, z = quat
    nx, ny, nz = 2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x*x + y*y)
    d = axial_ratio**2 - 1
    return (1 + d*nx*nx, 1 + d*ny*ny, 1 + d*nz*nz,
            d*nx*ny, d*nx*nz, d*ny*nz)


def quadratic_inverse(matrix, displacement):
    xx, yy, zz, xy, xz, yz = matrix
    cx = yy*zz - yz*yz
    cy = xx*zz - xz*xz
    cz = xx*yy - xy*xy
    cxy = xz*yz - xy*zz
    cxz = xy*yz - xz*yy
    cyz = xy*xz - xx*yz
    determinant = xx*cx + xy*cxy + xz*cxz
    if determinant <= 0 or not math.isfinite(determinant):
        raise ValueError("Initial-overlap test has a non-positive shape matrix")
    x, y, z = displacement
    return (cx*x*x + cy*y*y + cz*z*z +
            2*(cxy*x*y + cxz*x*z + cyz*y*z)) / determinant


def contact_function(matrix_i, matrix_j, displacement):
    """Maximum PW contact function: >1 separated, =1 tangent, <1 overlap."""
    difference = tuple(y-x for x, y in zip(matrix_i, matrix_j))

    def value(lam):
        matrix = tuple(x + lam*d for x, d in zip(matrix_i, difference))
        return lam * (1-lam) * quadratic_inverse(matrix, displacement)

    lower, upper = 0.0, 1.0
    ratio = (math.sqrt(5)-1)/2
    left, right = upper-ratio*(upper-lower), lower+ratio*(upper-lower)
    fl, fr = value(left), value(right)
    for _ in range(44):
        if fl < fr:
            lower, left, fl = left, right, fr
            right = lower + ratio*(upper-lower)
            fr = value(right)
        else:
            upper, right, fr = right, left, fl
            left = upper - ratio*(upper-lower)
            fl = value(left)
    return max(fl, fr)


def minimum_image(position_i, position_j, box):
    return tuple((x-y) - box*math.floor((x-y)/box + 0.5)
                 for x, y in zip(position_i, position_j))


def place_particles(count, a, c, box, seed, attempts_per_particle):
    rng = random.Random(seed)
    particles = []
    attempts = 0
    for particle_index in range(count):
        for _ in range(attempts_per_particle):
            attempts += 1
            position = tuple(rng.random()*box for _ in range(3))
            quat = quaternion(rng)
            matrix = shape_matrix(quat, c/a)
            for other in particles:
                displacement = tuple(v/a for v in minimum_image(position, other[0], box))
                distance_squared = sum(v*v for v in displacement)
                if distance_squared >= 4:
                    continue
                if (distance_squared <= 4*(c/a)**2 or
                        contact_function(matrix, other[2], displacement) <= 1 + 1e-10):
                    break
            else:
                particles.append((position, quat, matrix))
                break
        else:
            raise ValueError(
                f"Random insertion stopped at {particle_index}/{count} particles. "
                "No overlapping fallback was written. Change the seed or increase "
                "initialization.maximum_attempts_per_particle; dense states require "
                "a separate preparation method.")

    # Independently traverse the completed configuration before writing it.
    minimum_contact = None
    checked_pairs = 0
    for first, second in itertools.combinations(particles, 2):
        dr = tuple(v/a for v in minimum_image(first[0], second[0], box))
        if sum(v*v for v in dr) >= 4:
            continue
        checked_pairs += 1
        contact = contact_function(first[2], second[2], dr)
        if contact <= 1 + 1e-10:
            raise ValueError("Internal error: completed placement contains an overlap")
        minimum_contact = contact if minimum_contact is None else min(minimum_contact, contact)
    return particles, attempts, checked_pairs, minimum_contact


def format_float(value):
    return f"{value:.17g}"


def write_data(path, particles, a, c, density, box, shear):
    lines = ["Grown oblate spheroids; SI units; a=b, symmetry axis local z", "",
             f"{len(particles)} atoms", f"{len(particles)} ellipsoids", "1 atom types", "",
             f"0 {format_float(box)} xlo xhi", f"0 {format_float(box)} ylo yhi",
             f"0 {format_float(box)} zlo zhi", "0 0 0 xy xz yz", "", "Atoms # ellipsoid", ""]
    for number, (position, _, _) in enumerate(particles, 1):
        lines.append(f"{number} 1 1 {format_float(density)} " + " ".join(map(format_float, position)))
    lines.extend(["", "Ellipsoids", ""])
    for number, (_, quat, _) in enumerate(particles, 1):
        # read_data takes full axis lengths, while the plugin sees semiaxes.
        lines.append(f"{number} " + " ".join(map(format_float, (2*a, 2*a, 2*c, *quat))))
    lines.extend(["", "Velocities", ""])
    for number, (position, _, _) in enumerate(particles, 1):
        # Last three values are angular momentum for atom_style ellipsoid.
        lines.append(f"{number} {format_float(shear*position[1])} 0 0 0 0 0")
    path.write_text("\n".join(lines) + "\n")


def verify_packing(particles, a, box, tolerance):
    """Independent Python PW check of every potentially overlapping pair."""
    worst = 0.0
    for first, second in itertools.combinations(particles, 2):
        dr = tuple(v/a for v in minimum_image(first[0], second[0], box))
        if sum(v*v for v in dr) >= 4:
            continue
        contact = contact_function(first[2], second[2], dr)
        worst = max(worst, 1-math.sqrt(max(0, contact)))
    if worst > tolerance*1.05 + 1e-12:
        raise ValueError(f"Packing verification failed: PW overlap {worst:.9g} > {tolerance:.9g}")
    return worst


def read_packing(path, count, a, c, box):
    rows = path.read_text().splitlines()
    header = rows[0].split()
    if len(header) != 5 or int(header[0]) != count:
        raise ValueError("Packer output has an incorrect particle count")
    for actual, expected in zip(map(float, header[1:]), (box, a, c, 1.0)):
        if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=0):
            raise ValueError("Packer output did not reach the requested final geometry")
    if len(rows) != count+1:
        raise ValueError("Packer output has missing/extra particles")
    particles = []
    for row in rows[1:]:
        values = tuple(map(float, row.split()))
        if len(values) != 7 or not all(math.isfinite(v) for v in values):
            raise ValueError("Packer output contains malformed coordinates")
        position, quat = values[:3], values[3:]
        if not all(0 <= x < box for x in position):
            raise ValueError("Packer output lies outside the periodic box")
        if not math.isclose(sum(q*q for q in quat), 1, abs_tol=1e-10, rel_tol=0):
            raise ValueError("Packer output contains a nonunit quaternion")
        particles.append((position, quat, shape_matrix(quat, c/a)))
    return particles


def resolve_lubrication(args, a, resolve_tables=True):
    """Resolve hydro lengths independently of mechanical rough-contact onset."""
    requested = args.lubrication_model
    spherical = math.isclose(args.ar, 1.0, rel_tol=1e-12, abs_tol=0)
    model = ("mari" if spherical else "spheroid") if requested == "auto" else requested
    fast_options = (args.finite_gap_fast_table, args.finite_gap_fast_source, args.finite_gap_fast_generator,
                    getattr(args, "finite_gap_fast_source_gate", None), getattr(args, "finite_gap_fast_gate_executable", None),
                    getattr(args, "finite_gap_fast_basis_generator", None),
                    getattr(args, "finite_gap_fast_interpolation_executable", None))
    if model != "finite_gap_fast" and (any(value is not None for value in fast_options) or
            getattr(args, "finite_gap_fast_film_grid", DEFAULT_FAST_FILM_GRID) != DEFAULT_FAST_FILM_GRID or
            getattr(args, "finite_gap_fast_nphi", DEFAULT_FAST_NPHI) != DEFAULT_FAST_NPHI or
            getattr(args, "finite_gap_fast_threads", 1) != 1 or getattr(args, "finite_gap_fast_smoke", False) or
            getattr(args, "finite_gap_fast_basis_degree", 12) != 12 or getattr(args, "finite_gap_fast_far_degree", 8) != 8 or
            getattr(args, "finite_gap_fast_basis_source_rings", 96) != 96 or
            getattr(args, "finite_gap_fast_basis_source_phi", 128) != 128):
        raise ValueError("finite-gap-fast options require --lubrication-model finite_gap_fast; no implicit model change")
    requested_finite = model == "finite_gap"
    if model == "finite_gap_fast":
        if args.finite_gap_basis is not None or args.finite_gap_table is not None:
            raise ValueError("finite_gap_fast uses --finite-gap-fast-table/source, not --finite-gap-basis/table")
        if args.lubrication_cutoff_nm is not None:
            raise ValueError("finite_gap_fast uses --finite-gap-range; --lubrication-cutoff-nm would be ambiguous")
        if args.finite_gap_film_grid != 161:
            raise ValueError("finite_gap_fast does not evaluate a film grid at runtime")
        if spherical:
            raise ValueError("This explicit finite_gap_fast trial is non-spherical; select mari for AR=1")
        if args.ar < 1:
            raise ValueError("finite_gap_fast supports oblate spheroids with AR>1; prolate shapes are not supported")
    elif requested_finite:
        if (args.finite_gap_basis is None) == (args.finite_gap_table is None):
            raise ValueError("finite_gap requires exactly one --finite-gap-basis or --finite-gap-table")
        source = (args.finite_gap_basis or args.finite_gap_table).expanduser().resolve()
        if not source.is_file():
            raise ValueError("Missing finite-gap model file: {}".format(source))
        if args.lubrication_cutoff_nm is not None:
            raise ValueError("finite_gap uses --finite-gap-range; --lubrication-cutoff-nm would be ambiguous")
        if args.finite_gap_film_grid < 21:
            raise ValueError("--finite-gap-film-grid must be at least 21")
        if args.finite_gap_table is not None and args.finite_gap_film_grid != 161:
            raise ValueError("--finite-gap-film-grid applies to a hybrid basis, not a fixed table")
        if args.finite_gap_basis is not None:
            if not spherical:
                args.finite_gap_basis_header = read_basis_header(source, args.ar)
        if args.ar < 1:
            raise ValueError("finite_gap supports oblate spheroids with AR>=1; prolate shapes are not supported")
        if spherical:
            model = "mari"  # Preserve the complete existing sphere path and scaling.
    elif args.finite_gap_basis is not None or args.finite_gap_table is not None or args.finite_gap_range != 0.5 or args.finite_gap_film_grid != 161:
        raise ValueError("finite-gap options require --lubrication-model finite_gap")
    if model == "mari" and not spherical:
        raise ValueError("lubrication-model mari currently requires AR=1; "
                         "use spheroid for the local-curvature extension or legacy for the old model")
    if model in ("spheroid", "finite_gap", "finite_gap_fast") and (args.lubrication_epsilon_relative != 0.001 or
                                args.lubrication_cutoff_relative != 0.5):
        raise ValueError("spheroid lubrication uses physical lengths; use --lubrication-epsilon-nm "
                         "and --lubrication-cutoff-nm instead of Mari relative-length overrides")
    if model not in ("spheroid", "finite_gap", "finite_gap_fast") and args.lubrication_epsilon_nm != 0.15:
        raise ValueError("--lubrication-epsilon-nm applies only to the spheroid model; "
                         "use --lubrication-epsilon-relative for mari or --roughness-nm for legacy")
    if model == "mari":
        regularization = args.lubrication_epsilon_relative*a
    elif model in ("spheroid", "finite_gap", "finite_gap_fast"):
        regularization = args.lubrication_epsilon_nm*1e-9
    else:
        regularization = args.roughness_nm*1e-9
    if args.lubrication_cutoff_nm is not None:
        cutoff = args.lubrication_cutoff_nm*1e-9
    else:
        cutoff = (args.lubrication_cutoff_relative*a if model == "mari"
                  else args.finite_gap_range*a if model in ("finite_gap", "finite_gap_fast")
                  else 75e-9 if model == "spheroid" else 7.5e-9)
    if cutoff <= regularization:
        raise ValueError("resolved lubrication cutoff must exceed its regularization length")
    if args.contact_gap_nm*1e-9 >= cutoff:
        raise ValueError("contact-gap-nm must be smaller than the resolved lubrication cutoff")
    if model == "finite_gap_fast" and resolve_tables:
        resolve_fast_table(args, a)
    return model, regularization, cutoff


def prepare(args):
    output = args.output.expanduser().resolve()
    if output.exists():
        raise ValueError("Output path already exists: {}; choose a new run directory".format(output))
    for key in ("ar", "phi", "thickness_nm", "density", "roughness_nm", "viscosity",
                "normal_stiffness", "tangential_stiffness", "lubrication_epsilon_relative",
                "lubrication_epsilon_nm", "lubrication_cutoff_relative", "timestep", "initial_phi",
                "growth_fraction", "overlap_tolerance", "solver_ilu_drop", "finite_gap_range"):
        positive(getattr(args, key), key)
    if args.lubrication_cutoff_nm is not None:
        positive(args.lubrication_cutoff_nm, "lubrication_cutoff_nm")
    for key in ("count", "output_interval", "max_relax", "solver_ilu_fill", "solver_ilu_refresh"):
        integer(getattr(args, key), key)
    integer(args.steps, "steps", minimum=0)
    integer(args.prep_steps, "prep-steps", minimum=0)
    integer(args.seed, "seed", minimum=0)
    for key in ("contact_gap_nm", "friction", "average_start_strain"):
        value = getattr(args, key)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    if not math.isfinite(args.shear_rate):
        raise ValueError("shear-rate must be finite")
    if args.phi >= 1:
        raise ValueError("phi must be below 1")
    # AR = D/t = a/c; full thickness is fixed, equatorial axes vary with AR.
    c = args.thickness_nm*0.5e-9
    a = args.ar*c
    volume = 4*math.pi*a*a*c/3
    box = (args.count*volume/args.phi)**(1/3)
    lubrication_model, regularization, cutoff = resolve_lubrication(args, a, resolve_tables=False)
    contact_gap = args.contact_gap_nm*1e-9
    interaction_range = max(cutoff, contact_gap)
    skin = 0.2*c
    required_length = 2*(2*a+interaction_range+skin)
    if box <= required_length:
        required_count = math.floor(args.phi*required_length**3/volume)+1
        raise ValueError(f"For this AR/phi the periodic initializer requires N >= {required_count}; "
                         f"increase --count (currently {args.count}).")
    packer = args.packer.expanduser().resolve()
    if not packer.is_file():
        raise ValueError(f"Missing geometry packer {packer}; rerun build_slurry_cpu.sbatch")
    if lubrication_model == "finite_gap_fast":
        resolve_fast_table(args, a)
    output.mkdir(parents=True, exist_ok=False)
    finite_gap_metadata = {}
    finite_gap_argument = ""
    if lubrication_model == "finite_gap_fast":
        source = args.finite_gap_fast_table
        filename = "finite_gap_fast.sft"
        shutil.copyfile(str(source), str(output/filename))
        write_json_atomic(output / "finite_gap_manifest.json", args.finite_gap_fast_manifest)
        finite_gap_metadata = {"finite_gap_file": filename,
            "finite_gap_sha256": file_sha256(output/filename),
            "finite_gap_source": str(source), "finite_gap_range_over_a": args.finite_gap_range,
            "finite_gap_fast_table_header": read_fast_table_header(source, args, a),
            "finite_gap_fast_phi_nodes_selected": args.finite_gap_fast_nphi,
            "finite_gap_fast_provenance": ("Manifest binds numerical sources, basis, executable, grid settings and separate validation reports"
                if "source_code_sha256" in args.finite_gap_fast_manifest else
                "Explicit table without pipeline manifest: file SHA identifies this input; source and interpolation accuracy are unverified"),
            "finite_gap_runtime_scope": "Equal-sized oblate spheroids matching the selected AR table; no old-model fallback",
            "finite_gap_runtime_evaluation": "Matrix-log/Fourier table; normal analytic conditioning; no runtime Stokes or film-grid solve",
            "finite_gap_source_gate": args.finite_gap_fast_manifest["source_gate"],
            "finite_gap_source_policy": args.finite_gap_fast_manifest["table"].get("settings", {}).get("source_policy", "unverified explicit table"),
            "finite_gap_source_numerical_gate": args.finite_gap_fast_manifest.get("source_numerical_gate", {"status": "unverified"}),
            "finite_gap_interpolation_gate": args.finite_gap_fast_manifest["interpolation_gate"],
            "finite_gap_manifest_file": "finite_gap_manifest.json",
            "finite_gap_validation_status": "Source and interpolation reports are separate; preparation does not certify either or bulk rheology"}
        finite_gap_argument = " finite_gap_fast_table {} finite_gap_range {}".format(filename, format_float(args.finite_gap_range))
        generator_record = Path(str(source) + ".json")
        if generator_record.is_file():
            shutil.copyfile(str(generator_record), str(output / "finite_gap_generation.json"))
    elif lubrication_model == "finite_gap":
        kind = "basis" if args.finite_gap_basis is not None else "table"
        source = (args.finite_gap_basis or args.finite_gap_table).expanduser().resolve()
        filename = "finite_gap_basis.ssb" if kind == "basis" else "finite_gap_table.sfg"
        shutil.copyfile(str(source), str(output/filename))
        finite_gap_metadata = {"finite_gap_file": filename,
            "finite_gap_sha256": hashlib.sha256((output/filename).read_bytes()).hexdigest(),
            "finite_gap_source": str(source), "finite_gap_range_over_a": args.finite_gap_range,
            "finite_gap_film_grid": args.finite_gap_film_grid, "finite_gap_anchor_cache": "exact canonical pose only",
            "finite_gap_runtime_scope": "Explicit experimental model; unsupported gaps/orientations fail without fallback",
            "finite_gap_passivity": "Global velocity LDLT plus full five-strain Schur check including solvent and isolated bodies"}
        finite_gap_argument = " finite_gap_{} {} finite_gap_range {}".format(kind, filename, format_float(args.finite_gap_range))
        if kind == "basis":
            finite_gap_metadata["finite_gap_source_policy"] = args.finite_gap_basis_header["source_policy"]
            finite_gap_argument += " finite_gap_film_grid {}".format(args.finite_gap_film_grid)

    print(f"Lubrication: requested={args.lubrication_model}, resolved={lubrication_model}; "
          f"regularization={regularization/1e-9:.12g} nm, cutoff={cutoff/1e-9:.12g} nm; "
          f"independent mechanical contact gap={args.contact_gap_nm:g} nm", flush=True)
    if lubrication_model == "spheroid":
        print("Spheroid local-curvature closure: pair cutoff=min(physical cap, 0.5*Rlocal); "
              "Rlocal is the mean of the two local harmonic curvature radii. "
              "Non-spherical quantitative validation remains pending.", flush=True)

    if lubrication_model in ("finite_gap", "finite_gap_fast"):
        print("Experimental finite-gap model: reference file copied into the run directory; "
              "unsupported geometry/gaps and failed global passivity checks stop explicitly. "
              "No bulk-rheology or performance certification is implied.", flush=True)
    if lubrication_model == "finite_gap_fast":
        manifest = args.finite_gap_fast_manifest
        print("Finite-gap table: AR={}, profile={}, source_physical={}, source_numerical={}, interpolation={}".format(
              args.ar, manifest["table"].get("settings", {}).get("profile", "unverified"),
              manifest["source_gate"]["status"], manifest.get("source_numerical_gate", {}).get("status", "unverified"),
              manifest["interpolation_gate"]["status"]), flush=True)

    initial_phi = min(args.initial_phi, args.phi)
    scale = (initial_phi/args.phi)**(1/3)
    particles, attempts, _, _ = place_particles(args.count, a*scale, c*scale, box, args.seed, 20000)
    seed_path, packed_path = output/"packing.seed", output/"packing.final"
    with seed_path.open("w") as stream:
        stream.write(f"{args.count} " + " ".join(map(format_float, (box, a, c, scale))) + "\n")
        for position, quat, _ in particles:
            stream.write(" ".join(map(format_float, (*position, *quat))) + "\n")
    command = [str(packer), "--input", str(seed_path), "--output", str(packed_path),
               "--history", str(output/"packing_history.csv"),
               "--growth-fraction", format_float(args.growth_fraction),
               "--overlap-tolerance", format_float(args.overlap_tolerance),
               "--max-relax", str(args.max_relax)]
    print(f"Growing N={args.count}, AR={args.ar:g}, phi={initial_phi:g} -> {args.phi:g}", flush=True)
    # Preserve live progress in Slurm stdout and a separate preparation log.
    with (output/"packing.log").open("w") as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        if process.wait():
            raise ValueError(f"Packing did not reach the target; see {output/'packing_history.csv'}. "
                             "No in.slurry was written and shear will not start.")
    particles = read_packing(packed_path, args.count, a, c, box)
    worst = verify_packing(particles, a, box, args.overlap_tolerance)
    write_data(output/"particles.data", particles, a, c, args.density, box,
               0.0 if args.prep_steps else args.shear_rate)
    fmt = format_float
    lubrication_options = f"lubrication {lubrication_model}" + finite_gap_argument
    if lubrication_model == "mari":
        lubrication_options += f" lubrication_delta {fmt(args.lubrication_epsilon_relative)}"
        if args.lubrication_cutoff_nm is None:
            lubrication_options += f" lubrication_range {fmt(args.lubrication_cutoff_relative)}"
    contact_options = (f"contact_gap {fmt(contact_gap)} "
                       f"friction {fmt(args.friction)} {fmt(args.tangential_stiffness)}")
    solver_options = (f"solver_preconditioner {args.solver_preconditioner} "
                      f"solver_layout {args.solver_layout} "
                      f"solver_forcing {args.solver_forcing} solver_cache {args.solver_cache} "
                      f"solver_ilu {fmt(args.solver_ilu_drop)} {args.solver_ilu_fill} {args.solver_ilu_refresh}")
    preparation = ""
    if args.prep_steps:
        preparation = f'''# Relax the finite rough-contact layer without shear or frictional memory.
fix prepare all slurry/implicit {fmt(args.viscosity)} 0 {fmt(cutoff)} {fmt(regularization)} {fmt(args.normal_stiffness)} {lubrication_options} {solver_options} contact_gap {fmt(contact_gap)} friction 0 {fmt(args.tangential_stiffness)} history preparation_history.csv {args.output_interval} 0
thermo {args.output_interval}
thermo_style custom step time atoms f_prepare[1] f_prepare[2] f_prepare[3] f_prepare[4] f_prepare[5] f_prepare[6] f_prepare[7] f_prepare[8]
thermo_modify flush yes lost error
print "Zero-shear preparation: {args.prep_steps} steps; finite rough-contact relaxation; no production strain"
run {args.prep_steps}
write_data prepared.data nocoeff
unfix prepare
reset_timestep 0
'''
    text = f'''# Generated directly from run_slurry_cpu.sbatch / prepare_case.py CLI.
# Athermal overdamped dynamics; geometry packing finished before this physical run.
# Coupled resistance, rough-surface normal contact and Coulomb sliding; zero inertia.
# Stress decomposition and interval means are recorded in history.csv.
# AR={fmt(args.ar)} phi={fmt(args.phi)} t={fmt(args.thickness_nm)} nm rho={fmt(args.density)} kg/m3
# Lubrication requested={args.lubrication_model}, resolved={lubrication_model}; epsilon/floor={fmt(regularization)} m; cutoff={fmt(cutoff)} m
# Solver preconditioner={args.solver_preconditioner}; MPI layout={args.solver_layout}
# Solver forcing={args.solver_forcing}; cache={args.solver_cache}; ILUT drop/fill/refresh={fmt(args.solver_ilu_drop)}/{args.solver_ilu_fill}/{args.solver_ilu_refresh}
plugin load ${{slurry_plugin}}
units si
dimension 3
boundary p p p
atom_style ellipsoid
atom_modify map array
newton on
read_data particles.data
comm_modify vel yes
pair_style zero {fmt(2*a+interaction_range)}
pair_coeff * *
neighbor {fmt(skin)} bin
neigh_modify every 1 delay 0 check yes
timestep {fmt(args.timestep)}
{preparation}# Production starts at step/time/strain zero, with a fresh tangential contact history.
fix shear all deform 0 xy erate {fmt(args.shear_rate)} remap v units box
fix integrate all slurry/implicit {fmt(args.viscosity)} {fmt(args.shear_rate)} {fmt(cutoff)} {fmt(regularization)} {fmt(args.normal_stiffness)} {lubrication_options} {solver_options} {contact_options} history history.csv {args.output_interval} {fmt(args.average_start_strain)}
variable strain equal {fmt(abs(args.shear_rate))}*time
compute shape all property/atom shapex shapey shapez quatw quati quatj quatk angmomx angmomy angmomz tqx tqy tqz
dump trajectory all custom {args.output_interval} trajectory.lammpstrj id type x y z vx vy vz fx fy fz c_shape[*]
dump_modify trajectory sort id format float %.17g
thermo {args.output_interval}
thermo_style custom step time atoms v_strain f_integrate[1] f_integrate[2] f_integrate[3] f_integrate[4] f_integrate[5] f_integrate[6] f_integrate[7] f_integrate[8]
thermo_modify flush yes lost error
fix diagnostics all ave/time {args.output_interval} 1 {args.output_interval} f_integrate[1] f_integrate[2] f_integrate[3] f_integrate[4] f_integrate[5] f_integrate[6] f_integrate[7] f_integrate[8] file diagnostics.dat ave one title1 "# Instantaneous overdamped solver diagnostics; overlaps/gaps in m" title2 "# step lubrication_pairs overlapping_pairs max_overlap_m scaled_relative_residual linear_iterations newton_iterations predicted_contacts min_candidate_gap_m"
print "Integrator: slurry/implicit; dt={fmt(args.timestep)} s; steps={args.steps}; duration={fmt(args.steps*args.timestep)} s; total strain={fmt(abs(args.shear_rate)*args.steps*args.timestep)}"
print "Lubrication: {lubrication_model}; epsilon/floor={fmt(regularization)} m; cutoff={fmt(cutoff)} m; mechanical contact gap={fmt(contact_gap)} m"
print "Solver preconditioner: {args.solver_preconditioner}; MPI layout: {args.solver_layout}"
print "Solver forcing: {args.solver_forcing}; cache: {args.solver_cache}; ILUT drop/fill/refresh: {fmt(args.solver_ilu_drop)}/{args.solver_ilu_fill}/{args.solver_ilu_refresh}"
run {args.steps}
write_data final.data nocoeff
'''
    (output/"in.slurry").write_text(text)
    # A generated record, not another input/configuration layer.
    metadata = dict(vars(args))
    metadata.update(finite_gap_metadata)
    metadata.update({"a_m": a, "b_m": a, "c_m": c, "box_length_m": box,
                     "particle_volume_m3": volume, "particle_mass_kg": args.density*volume,
                     "volume_fraction_actual": args.count*volume/box**3,
                     "lubrication_model_requested": args.lubrication_model,
                     "lubrication_model_resolved": lubrication_model,
                     "lubrication_reference_radius_m": a if lubrication_model == "mari" else None,
                     "lubrication_regularization_m": regularization,
                     "lubrication_cutoff_m": cutoff,
                     "lubrication_epsilon_scaling": ("epsilon_relative * mean pair radius" if lubrication_model == "mari"
                                                     else "absolute physical length, independent of aspect ratio" if lubrication_model in ("spheroid", "finite_gap", "finite_gap_fast")
                                                     else "absolute legacy floor"),
                     "lubrication_cutoff_scaling": ("cutoff_relative * mean pair radius"
                                                    if lubrication_model == "mari" and args.lubrication_cutoff_nm is None
                                                    else "min(absolute cap, 0.5*Rlocal); Rlocal=(2/trace(Ci)+2/trace(Cj))/2"
                                                    if lubrication_model == "spheroid"
                                                    else "lambda * mean equatorial semiaxis; C2 fade over final 20 percent" if lubrication_model in ("finite_gap", "finite_gap_fast")
                                                    else "absolute cutoff"),
                     "lubrication_gap_rule": ("max(h,0)+epsilon" if lubrication_model in ("mari", "spheroid", "finite_gap", "finite_gap_fast")
                                              else "max(h,roughness)"),
                     "lubrication_validation_scope": ("Local-curvature closure; non-spherical pair and bulk validation deferred"
                                                        if lubrication_model == "spheroid" else
                                                        "Mari sphere kernel" if lubrication_model == "mari" else
                                                        "AR-matched precomputed matrix-log/Fourier model; source/interpolation/runtime gates reported separately" if lubrication_model == "finite_gap_fast" else
                                                        "Explicit experimental independent-reference finite-gap model; narrow-gap and bulk scope enforced by provider" if lubrication_model == "finite_gap" else "Legacy spheroid model"),
                     "lubrication_relative_options_use": ("Mari pair-radius scaling" if lubrication_model == "mari" else
                                                           "Not used; physical epsilon and explicit model range apply" if lubrication_model in ("spheroid", "finite_gap", "finite_gap_fast") else
                                                           "Not used; legacy absolute roughness/cutoff apply"),
                     "packing_initial_phi_actual": initial_phi,
                     "packing_seed_attempts": attempts, "packing_maximum_PW_overlap": worst,
                     "simulated_time_s": args.steps*args.timestep,
                     "accumulated_strain": abs(args.shear_rate)*args.steps*args.timestep,
                     "preparation_time_s": args.prep_steps*args.timestep,
                     "density_source": ("ACS Nano DOI 10.1021/acsnano.6c10201 PDF p.3: 2.4 g/cm3"
                                        if args.density == 2400.0 else "User override; default ACS Nano density is 2400 kg/m3"),
                     "medium": "User-specified Newtonian medium; default 0.01 Pa s is an assumed CMC90k value",
                     "preparation": "Geometric growth, then optional zero-shear frictionless rough-contact relaxation; production time/strain and tangential history reset afterwards",
                     "roughness_use": ("Ignored by Mari lubrication; hydro epsilon is set independently by lubrication_epsilon_relative"
                                       if lubrication_model == "mari" else "Ignored by spheroid lubrication; hydro epsilon is set independently by lubrication_epsilon_nm"
                                       if lubrication_model in ("spheroid", "finite_gap", "finite_gap_fast") else
                                       "Legacy lubrication coefficient floor only; not a minimum surface gap"),
                     "contact_gap_use": "Mechanical contact onset hc; normal compression max(hc-h,0), not a forced minimum gap",
                     "average_window": "Selected production strain window; stationarity is not assumed or automatically established",
                     "integrator": "slurry/implicit: athermal zero-inertia coupled resistance solve with first-order frozen-geometry rough contact and tangential friction",
                     "density_role": "Stored physical property and mass/angular-momentum representation; inertial acceleration is omitted",
                     "scope": "Athermal overdamped model stress decomposition; no adhesion, adhesive rolling, Brownian noise or far-field many-body hydrodynamics"})
    (output/"parameters.txt").write_text("".join(f"{key} = {value}\n" for key, value in metadata.items()))
    print(f"Prepared {output}\nN={args.count}, AR={args.ar:g}, phi={args.phi:g}, L={box:.12g} m", flush=True)
    print(f"dt={args.timestep:g} s, steps={args.steps}, time={args.steps*args.timestep:g} s, "
          f"strain={abs(args.shear_rate)*args.steps*args.timestep:g}; coupled overdamped dynamics.", flush=True)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new run directory")
    parser.add_argument("--packer", type=Path, default=DEFAULT_PACKER)
    parser.add_argument("--ar", type=float, default=2.0)
    parser.add_argument("--phi", type=float, default=0.24)
    parser.add_argument("--shear-rate", type=float, default=100.0)
    parser.add_argument("--thickness-nm", type=float, default=300.0, help="full polar thickness")
    parser.add_argument("--density", type=float, default=2400.0, help="kg/m^3")
    parser.add_argument("--roughness-nm", type=float, default=3.0, help="legacy lubrication gap floor only; ignored by mari and spheroid")
    parser.add_argument("--contact-gap-nm", type=float, default=3.0, help="mechanical roughness-contact onset; independent of lubrication regularization")
    parser.add_argument("--friction", type=float, default=0.1, help="Coulomb sliding coefficient")
    parser.add_argument("--tangential-stiffness", type=float, default=80.0, help="N/m; tangential contact spring")
    parser.add_argument("--count", type=int, default=300)
    parser.add_argument("--viscosity", type=float, default=0.01, help="Newtonian medium, Pa s")
    parser.add_argument("--normal-stiffness", type=float, default=1.0, help="N/m; finite repulsive spring")
    parser.add_argument("--lubrication-model", choices=("auto", "legacy", "mari", "spheroid", "finite_gap", "finite_gap_fast"), default="auto",
                        help="auto: Mari sphere model at AR=1, local-curvature spheroid extension otherwise")
    parser.add_argument("--lubrication-epsilon-relative", type=float, default=0.001,
                        help="Mari hydro epsilon / sphere radius; independent of contact gap")
    parser.add_argument("--lubrication-epsilon-nm", type=float, default=0.15,
                        help="spheroid physical hydro regularization in nm; independent of AR and contact gap")
    parser.add_argument("--lubrication-cutoff-relative", type=float, default=0.5,
                        help="Mari outer cutoff / sphere radius; overridden by --lubrication-cutoff-nm")
    parser.add_argument("--lubrication-cutoff-nm", type=float, default=None,
                        help="absolute outer range/cap; defaults: legacy 7.5 nm, spheroid 75 nm (also capped by 0.5*Rlocal), mari relative")
    parser.add_argument("--finite-gap-basis", type=Path, default=None,
                        help="explicit scattering basis matching the selected oblate AR; copied and hashed in the run directory")
    parser.add_argument("--finite-gap-table", type=Path, default=None,
                        help="explicit fixed-geometry reference table; unsupported poses/gaps fail without fallback")
    parser.add_argument("--finite-gap-fast-table", type=Path, default=None,
                        help="explicit precomputed matrix-log/Fourier trial table/cache; absent files require a generation source")
    parser.add_argument("--finite-gap-fast-source", type=Path, default=None,
                        help="matching basis; if missing, generate real far/near AR bases before source checks and table generation")
    parser.add_argument("--finite-gap-fast-basis-generator", type=Path, default=None,
                        help="AR basis generator; default build/lammps-slurry/slurry_generate_basis")
    parser.add_argument("--finite-gap-fast-basis-degree", type=int, default=12,
                        help="near basis order for missing-basis generation; candidate setting, not accuracy certification")
    parser.add_argument("--finite-gap-fast-far-degree", type=int, default=8,
                        help="far basis order for missing-basis generation; actual source solve for the requested AR")
    parser.add_argument("--finite-gap-fast-basis-source-rings", type=int, default=None,
                        help="override AR-scaled source rings; automatic: 96 at AR<=10, 192 at AR20")
    parser.add_argument("--finite-gap-fast-basis-source-phi", type=int, default=None,
                        help="override AR-scaled runtime quadrature; automatic: 128 at AR<=10, 256 at AR20")
    parser.add_argument("--finite-gap-fast-basis-fit-phi", type=int, default=None,
                        help="override AR^2-scaled kernel quadrature; automatic: 4096 at AR<=10, 16384 at AR20; check uses twice this")
    parser.add_argument("--finite-gap-fast-generator", type=Path, default=None,
                        help="offline generator executable; default build/lammps-slurry/slurry_precompute_fast")
    parser.add_argument("--finite-gap-fast-film-grid", type=int, default=DEFAULT_FAST_FILM_GRID,
                        help="offline fast-table source film grid; default 161; included in automatic cache identity")
    parser.add_argument("--finite-gap-fast-nphi", type=int, default=DEFAULT_FAST_NPHI,
                        help="selected fast-table azimuth resolution; default 48; must match explicit table n_phi")
    parser.add_argument("--finite-gap-fast-threads", type=int, default=1,
                        help="offline table generation threads; independent of MPI runtime ranks")
    parser.add_argument("--finite-gap-fast-smoke", action="store_true",
                        help="coarse end-to-end connection test only; never a physical/rheology qualification")
    parser.add_argument("--finite-gap-fast-source-gate", type=Path, default=None,
                        help="AR10 matched report plus sibling .direct.json; other AR isolated-only report; otherwise run checks")
    parser.add_argument("--finite-gap-fast-gate-executable", type=Path, default=None,
                        help="source physical gate executable; default build/lammps-slurry/slurry_basis_physical_probe")
    parser.add_argument("--finite-gap-fast-interpolation-executable", type=Path, default=None,
                        help="complete-table interpolation probe; default build/lammps-slurry/slurry_table_interpolation_probe")
    parser.add_argument("--finite-gap-film-grid", type=int, default=161,
                        help="hybrid finite-film grid per direction; default 161, minimum 21; change only for convergence checks")
    parser.add_argument("--finite-gap-range", type=float, default=0.5,
                        help="experimental outer gap / equatorial semiaxis; final 20 percent is smoothly windowed")
    parser.add_argument("--solver-preconditioner", choices=("auto", "sparse", "nested", "lu", "ilu"), default="ilu",
                        help="implicit linear solver backend; lu uses full-Jacobian SparseLU, ilu uses IncompleteLUT, nested retains the previous inner CG")
    parser.add_argument("--solver-layout", choices=("root", "replicated"), default="root",
                        help="root solves once and shares the result; replicated retains the previous MPI layout")
    parser.add_argument("--solver-forcing", choices=("fixed", "adaptive"), default="fixed",
                        help="inner linear-solve forcing; does not change the final force tolerance")
    parser.add_argument("--solver-cache", choices=("off", "on"), default="on",
                        help="reuse eligible solver preconditioner factors between solves")
    parser.add_argument("--solver-ilu-drop", type=float, default=1e-3,
                        help="positive ILUT drop tolerance; applies when the ilu backend is used")
    parser.add_argument("--solver-ilu-fill", type=int, default=4,
                        help="positive ILUT fill factor; applies when the ilu backend is used")
    parser.add_argument("--solver-ilu-refresh", type=int, default=40,
                        help="positive iteration threshold for refreshing the ILUT factor")
    parser.add_argument("--timestep", type=float, default=1e-6, help="s; overdamped step, check temporal convergence")
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--output-interval", type=int, default=100)
    parser.add_argument("--average-start-strain", type=float, default=5.0, help="start of the reported averaging window, not an automatic steady-state test")
    parser.add_argument("--prep-steps", type=int, default=1000, help="optional zero-shear rough-contact relaxation; 0 disables")
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--initial-phi", type=float, default=0.05)
    parser.add_argument("--growth-fraction", type=float, default=0.02, help="relative phi increase per attempt")
    parser.add_argument("--overlap-tolerance", type=float, default=1e-6, help="dimensionless PW contact-scaling deficit")
    parser.add_argument("--max-relax", type=int, default=20000, help="maximum geometric iterations per growth attempt")
    args = parser.parse_args()
    try:
        prepare(args)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(2, f"prepare_case.py: {error}\n")


if __name__ == "__main__":
    main()


