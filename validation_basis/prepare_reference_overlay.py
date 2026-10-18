#!/usr/bin/env python3
"""Prepare the optional L20 audit loader without changing production files.

Only the matrix allocation guard changes, from 16M to 24M entries. The generated
manifest is verified by run_matched_query.py and cannot authorize a production
matched-provider query. This script uses only the Python standard library; it
prints compile commands but does not compile or run the expensive reference.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import sys

from run_physical_gate import load_prepare, sha256


def prepare_overlay(project, output, compiler):
    project = project.resolve()
    audit_root = (project / "validation_basis").resolve()
    output = output.resolve()
    relative_output = output.relative_to(audit_root)
    if not relative_output.parts:
        raise ValueError("Use a subdirectory of validation_basis for the reference overlay")
    prepare = load_prepare(project)
    production_fingerprint = prepare.fast_source_fingerprint()
    specs = [
        ("lammps_slurry/tools/slurry_generate_basis.cpp", 2),
        ("lammps_slurry/src/slurry_scattering_math.h", 1),
        ("lammps_slurry/src/slurry_scattering_provider.h", 0),
    ]
    prepared = []
    for relative_path, count in specs:
        source = project / relative_path
        original = source.read_bytes()
        if count and original.count(b"16000000") != count:
            raise ValueError("Allocation guard changed; review this audit override: " + relative_path)
        changed = original.replace(b"16000000", b"24000000") if count else original
        prepared.append((source, original, changed, count))
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for source, original, changed, count in prepared:
        target = output / source.name
        target.write_bytes(changed)
        records.append(dict(base_path=str(source.relative_to(project.parent)),
                            copy_path=str(target.relative_to(project.parent)),
                            base_sha256=hashlib.sha256(original).hexdigest(),
                            copy_sha256=sha256(target), byte_identical=(count == 0),
                            bound_replacements=count,
                            difference="allocation guard only" if count else "byte-identical include"))
    if (prepare.fast_source_fingerprint() != production_fingerprint or
            any(source.read_bytes() != original for source, original, _, _ in prepared)):
        raise ValueError("Production source changed during overlay creation")
    fingerprint = hashlib.sha256(json.dumps({"base": production_fingerprint,
        "files": {record["copy_path"]: record["copy_sha256"] for record in records}},
        sort_keys=True).encode()).hexdigest()
    manifest = dict(schema="reference_only_bound_override_v1", reference_only=True,
                    physics_modified=False, production_files_modified=False,
                    old_matrix_entry_limit=16000000, reference_matrix_entry_limit=24000000,
                    expected_L20_density_entries=22844160,
                    base_production_fingerprint=production_fingerprint,
                    overlay_fingerprint=fingerprint, files=records,
                    purpose="Optional direct-source order-convergence audit; not a production model")
    common = [compiler, "-O1", "-std=c++17", "-DEIGEN_DONT_PARALLELIZE",
              "-I", str(output), "-I", str(project / "lammps_slurry/src"),
              "-I", str(project / "lammps_slurry/tools"),
              "-I", str(project / "lammps_slurry/vendor/eigen3")]
    commands = {
        "generator": common + [str(output / "slurry_generate_basis.cpp"),
                                "-o", str(output / "slurry_reference_generate_basis")],
        "query": common + ['-DSLURRY_NUMERICAL_SOURCE_FINGERPRINT="' + fingerprint + '"',
                            str(project / "lammps_slurry/tests/query_matched_source.cpp"),
                            "-o", str(output / "slurry_reference_query")],
    }
    manifest["compile_argv"] = commands
    manifest["compile_commands"] = {key: " ".join(shlex.quote(part) for part in command)
                                     for key, command in commands.items()}
    generation = {}
    for degree in (16, 20):
        command = [str(output / "slurry_reference_generate_basis"), "--ar", "10",
                   "--degree", str(degree), "--source-rings", "128", "--source-phi", "256",
                   "--source-order", str(degree+2), "--source-depth", "0.5",
                   "--target-rings", str(degree+3), "--target-phi", str(2*degree+5),
                   "--fit-rings", "256", "--fit-phi", "4096", "--check-rings", "263",
                   "--check-phi", "8192", "--rcond", "1e-11",
                   "--output", str(output / ("ar10_L%d_phi256.ssb" % degree)),
                   "--report", str(output / ("ar10_L%d_generation.json" % degree))]
        generation["L%d_phi256" % degree] = " ".join(shlex.quote(part) for part in command)
    manifest["reference_generation_commands"] = generation
    manifest["reference_generation_scope"] = (
        "Plain direct-source bases with the same near-body coefficients as the audited hybrid bundles. "
        "For quadrature refinement, copy each file and change only its single FOURIER runtime-count "
        "token from 256 to 512, retaining the order token. Do not change coefficients. "
        "Use these only with direct_basis. Serialized whole-bundle hashes differ from historical hybrid bundles.")
    manifest_path = output / "provenance.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return dict(manifest=str(manifest_path), overlay_fingerprint=fingerprint,
                base_production_fingerprint=production_fingerprint,
                compile_commands=manifest["compile_commands"],
                reference_generation_commands=generation,
                reference_generation_scope=manifest["reference_generation_scope"],
                note="Compile these commands, then use run_matched_query.py with --mode direct_basis and --reference-overlay-manifest. No accuracy gate has run.")


def main():
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=project / "validation_basis/reference_L20")
    parser.add_argument("--compiler", default="c++")
    args = parser.parse_args()
    try:
        result = prepare_overlay(project, args.output_dir, args.compiler)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
