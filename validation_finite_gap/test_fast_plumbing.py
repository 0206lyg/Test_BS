#!/usr/bin/env python3
"""Activation/cache routing checks only; synthetic tables are NOT physical tests.

Uses the Python standard library. Optional --packer runs preparation with the
real packer; it does not run LAMMPS or validate the table's resistance values.
"""
import argparse
from array import array
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("prepare_case", str(ROOT / "lammps_slurry/tools/prepare_case.py"))
prepare_case = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare_case)
PACKER = None


def synthetic_table(path, ar=10.0, nphi=8, joins=None):
    # Complete binary framing with zeros: suitable for routing tests ONLY.
    phi = " ".join(str(-math.pi + 2 * math.pi * i / nphi) for i in range(nphi))
    gaps = sorted(set([.001, .1, 1, 5.001] + (joins or [])))
    magic = "SLURRY_SPD_LOGFOURIER_V2" if joins is not None else "SLURRY_SPD_LOGFOURIER_V1"
    join_line = "{} {}\n".format(len(joins), " ".join(str(x) for x in joins)) if joins is not None else ""
    text = ("{}\n{} 0.001 5.001 4 {} {}\n{}"
            "0 {} {} {}\n{}\n{}\nDATA\n").format(
                magic, ar, nphi, len(gaps), join_line, math.pi / 6, math.pi / 3, math.pi / 2, phi,
                " ".join(str(x) for x in gaps))
    with path.open("wb") as stream:
        stream.write(text.encode("ascii"))
        array("d", [0.0] * (4 * 4 * len(gaps) * nphi * 66)).tofile(stream)


def arguments(table=None, **overrides):
    values = dict(lubrication_model="finite_gap_fast", ar=10.0,
                  finite_gap_fast_table=table, finite_gap_fast_source=None, finite_gap_fast_generator=None,
                  finite_gap_fast_nphi=8,
                  finite_gap_basis=None, finite_gap_table=None, finite_gap_range=0.5, finite_gap_film_grid=161,
                  lubrication_cutoff_nm=None, lubrication_epsilon_relative=0.001,
                  lubrication_cutoff_relative=0.5, lubrication_epsilon_nm=0.15,
                  roughness_nm=3.0, contact_gap_nm=3.0)
    values.update(overrides)
    if values["lubrication_model"] != "finite_gap_fast" and "finite_gap_fast_nphi" not in overrides:
        values["finite_gap_fast_nphi"] = prepare_case.DEFAULT_FAST_NPHI
    return SimpleNamespace(**values)


class FastPlumbing(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="slurry-fast-routing-")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.table = self.directory / "synthetic.sft"
        synthetic_table(self.table)

    def test_explicit_activation_and_lengths(self):
        model, epsilon, cutoff = prepare_case.resolve_lubrication(arguments(self.table), 1.5e-6)
        self.assertEqual(model, "finite_gap_fast")
        self.assertAlmostEqual(epsilon / 1e-9, 0.15)
        self.assertAlmostEqual(cutoff / 1e-9, 750.0)
        with self.assertRaisesRegex(ValueError, "does not evaluate a film grid at runtime"):
            prepare_case.resolve_lubrication(arguments(self.table, finite_gap_film_grid=81), 1.5e-6)

    def test_no_auto_activation_or_fallback(self):
        with self.assertRaisesRegex(ValueError, "require --lubrication-model finite_gap_fast"):
            prepare_case.resolve_lubrication(arguments(self.table, lubrication_model="auto"), 1.5e-6)
        self.assertEqual(prepare_case.resolve_lubrication(arguments(lubrication_model="auto"), 1.5e-6)[0], "spheroid")

    def test_wrong_table_ar_rejected_without_another_ar_run(self):
        synthetic_table(self.table, ar=11.0)
        with self.assertRaisesRegex(ValueError, "does not match requested AR"):
            prepare_case.resolve_lubrication(arguments(self.table), 1.5e-6)

    def test_truncated_payload_rejected(self):
        self.table.write_bytes(self.table.read_bytes()[:-1])
        with self.assertRaisesRegex(ValueError, "payload"):
            prepare_case.resolve_lubrication(arguments(self.table), 1.5e-6)

    def test_failed_residual_format_rejected(self):
        payload = self.table.read_bytes()
        for old_magic in (b"SLURRY_FAST_RESIDUAL_V1", b"SLURRY_SPD_TABLE_V1"):
            self.table.write_bytes(payload.replace(b"SLURRY_SPD_LOGFOURIER_V1", old_magic, 1))
            with self.assertRaisesRegex(ValueError, "Invalid finite-gap fast table header"):
                prepare_case.resolve_lubrication(arguments(self.table), 1.5e-6)

    def test_v2_explicit_joins_and_basis_anchors(self):
        original = self.table.read_bytes()
        payload = original.replace(b"SLURRY_SPD_LOGFOURIER_V1", b"SLURRY_SPD_LOGFOURIER_V2", 1)
        payload = payload.replace(b"5.001 4 8 4\n", b"5.001 4 8 4\n1 1\n", 1)
        self.table.write_bytes(payload)
        header = prepare_case.read_fast_table_header(self.table, arguments(self.table), 1.5e-6)
        self.assertEqual(header["joins_over_c"], [1.0])
        self.assertEqual(header["format"], "SLURRY_SPD_LOGFOURIER_V2")
        self.table.write_bytes(payload.replace(b"\n1 1\n", b"\n1 0.5\n", 1))
        with self.assertRaisesRegex(ValueError, "exact in-range gap nodes"):
            prepare_case.read_fast_table_header(self.table, arguments(self.table), 1.5e-6)
        basis = self.directory / "v2_header_only.ssb"
        basis.write_text("SLURRY_HYBRID_V2\n10 12\nANCHORS 3 0.5 1 2.5\n")
        self.assertEqual(prepare_case.read_basis_header(basis, 10)["anchors_over_c"], [.5, 1, 2.5])
        basis.write_text("SLURRY_HYBRID_V3\n10 12\nANCHORS 2 0.5 1\n")
        header = prepare_case.read_basis_header(basis, 10)
        self.assertEqual(header["anchors_over_c"], [.5, 1])
        self.assertEqual(header["source_policy"]["active_basis_slot"], "near")
        self.assertFalse(header["source_policy"]["far_slot_used"])
        basis.write_text("SLURRY_HYBRID_V3\n10 12\nANCHORS 3 0.5 1 2.5\n")
        with self.assertRaisesRegex(ValueError, "V3 requires exactly"):
            prepare_case.read_basis_header(basis, 10)

    def test_odd_phi_count_rejected(self):
        self.table.write_bytes(self.table.read_bytes().replace(b"5.001 4 8 4", b"5.001 4 9 4", 1))
        with self.assertRaisesRegex(ValueError, "even number of phi nodes"):
            prepare_case.resolve_lubrication(arguments(self.table), 1.5e-6)

    def test_manual_phi_resolution_must_match_selection(self):
        synthetic_table(self.table, nphi=24)
        with self.assertRaisesRegex(ValueError, "n_phi=24 does not match requested 48"):
            prepare_case.resolve_lubrication(arguments(self.table, finite_gap_fast_nphi=48), 1.5e-6)
        model, _, _ = prepare_case.resolve_lubrication(arguments(self.table, finite_gap_fast_nphi=24), 1.5e-6)
        self.assertEqual(model, "finite_gap_fast")

    def test_uncovered_gap_range_rejected(self):
        with self.assertRaisesRegex(ValueError, "does not cover"):
            prepare_case.resolve_lubrication(arguments(self.table, finite_gap_range=0.6), 1.5e-6)

    def test_missing_table_never_falls_back(self):
        with self.assertRaisesRegex(ValueError, "Missing fast table"):
            prepare_case.resolve_lubrication(arguments(self.directory / "missing.sft"), 1.5e-6)

    def test_generation_cache_reused(self):
        source = self.directory / "basis.ssb"
        source.write_text("SLURRY_HYBRID_V3\n10 12\nANCHORS 2 0.5 1\n")  # Routing-only fake.
        synthetic_table(self.table, joins=[.5, 1])
        generator = self.directory / "fake_generator.py"
        counter = self.directory / "generator_calls.txt"
        recorded_arguments = self.directory / "generator_arguments.json"
        generator.write_text("#!" + sys.executable + "\n" +
            "import json, pathlib, shutil, sys\n" +
            "counter=pathlib.Path(" + repr(str(counter)) + ")\n" +
            "counter.write_text(counter.read_text()+'called\\n' if counter.exists() else 'called\\n')\n" +
            "pathlib.Path(" + repr(str(recorded_arguments)) + ").write_text(json.dumps(sys.argv))\n" +
            "shutil.copyfile(" + repr(str(self.table)) + ",sys.argv[sys.argv.index('--output')+1])\n")
        generator.chmod(0o755)
        first = arguments(finite_gap_fast_source=source, finite_gap_fast_generator=generator,
                          finite_gap_fast_smoke=True, finite_gap_fast_threads=8)
        second = arguments(finite_gap_fast_source=source, finite_gap_fast_generator=generator,
                           finite_gap_fast_smoke=True)
        prepare_case.resolve_lubrication(first, 1.5e-6)
        prepare_case.resolve_lubrication(second, 1.5e-6)
        self.assertEqual(first.finite_gap_fast_table, second.finite_gap_fast_table)
        self.assertEqual(counter.read_text(), "called\n")
        command = json.loads(recorded_arguments.read_text())
        self.assertEqual(command[command.index("--threads") + 1], "8")
        self.assertEqual(command[command.index("--smoke") + 1], "1")
        self.assertEqual(first.finite_gap_fast_manifest["source_gate"]["status"], "not_run")
        self.assertEqual(first.finite_gap_fast_manifest["interpolation_gate"]["status"], "not_run")
        settings = first.finite_gap_fast_manifest["table"]["settings"]
        self.assertEqual(settings["source_format"], "SLURRY_HYBRID_V3")
        self.assertEqual(settings["table_format"], "SLURRY_SPD_LOGFOURIER_V2")
        self.assertEqual(settings["source_anchors_over_c"], [.5, 1])
        self.assertFalse(settings["source_policy"]["far_slot_used"])
        self.assertEqual(first.finite_gap_fast_table.read_bytes(), self.table.read_bytes())
        refined = arguments(finite_gap_fast_source=source, finite_gap_fast_generator=generator,
                            finite_gap_fast_film_grid=321, finite_gap_fast_smoke=True)
        prepare_case.resolve_lubrication(refined, 1.5e-6)
        self.assertNotEqual(refined.finite_gap_fast_table, first.finite_gap_fast_table)
        changed_headers = arguments(finite_gap_fast_source=source, finite_gap_fast_generator=generator,
                                    finite_gap_fast_smoke=True)
        with mock.patch.object(prepare_case, "fast_source_fingerprint", return_value="changed-numerical-headers"):
            prepare_case.resolve_lubrication(changed_headers, 1.5e-6)
        self.assertNotEqual(changed_headers.finite_gap_fast_table, first.finite_gap_fast_table)
        self.assertEqual(counter.read_text(), "called\ncalled\ncalled\n")

    def test_basis_ar_is_read_not_hardcoded(self):
        source = self.directory / "identity_only.ssb"
        source.write_text("SLURRY_HYBRID_V1\n2 12\n")
        direct = arguments(lubrication_model="finite_gap", ar=2.0, finite_gap_basis=source)
        self.assertEqual(prepare_case.resolve_lubrication(direct, 0.3e-6)[0], "finite_gap")
        with self.assertRaisesRegex(ValueError, "does not match requested AR"):
            prepare_case.read_basis_header(source, 10.0)

    def test_missing_basis_runs_real_generation_stages_in_order(self):
        # Fake executable tests orchestration only; it contains no Stokes solve.
        source = self.directory / "generated.ssb"
        generator = self.directory / "fake_basis_generator.py"
        calls = self.directory / "basis_calls.txt"
        generator.write_text("#!" + sys.executable + "\n" +
            "import json,pathlib,sys\n" +
            "a=sys.argv; get=lambda key:a[a.index(key)+1]\n" +
            "hybrid='--far-basis' in a\n" +
            "assert not hybrid or pathlib.Path(get('--far-basis')).is_file()\n" +
            "c=pathlib.Path(" + repr(str(calls)) + ")\n" +
            "c.write_text((c.read_text() if c.exists() else '')+get('--degree')+'\\n')\n" +
            "assert not hybrid or (get('--hybrid-version')=='3' and get('--anchor-gaps')=='0.5,1')\n" +
            "pathlib.Path(get('--output')).write_text(('SLURRY_HYBRID_V3' if hybrid else 'SLURRY_SCATTERING_V2')+'\\n'+get('--ar')+' '+get('--degree')+'\\n'+('ANCHORS 2 0.5 1\\n' if hybrid else ''))\n" +
            "pathlib.Path(get('--report')).write_text(json.dumps({'routing_only':True,'aspect_ratio':float(get('--ar'))}))\n")
        generator.chmod(0o755)
        selected = arguments(finite_gap_fast_source=source, finite_gap_fast_basis_generator=generator)
        self.assertEqual(prepare_case.resolve_fast_basis(selected), source)
        self.assertEqual(calls.read_text(), "8\n12\n")
        self.assertEqual(prepare_case.read_basis_header(source, 10)["degree"], 12)
        record = json.loads(Path(str(source) + ".pipeline.json").read_text())
        self.assertEqual(record["source_gate"]["status"], "not_run")
        self.assertEqual(record["source_rings"], 96)
        self.assertEqual(record["source_format"], "SLURRY_HYBRID_V3")
        self.assertFalse(record["source_policy"]["far_slot_used"])
        self.assertEqual(record["basis_sha256"], prepare_case.file_sha256(source))

    def test_other_ar_numerical_gate_does_not_claim_pair_physics(self):
        # Synthetic probe report tests status routing, not AR2 model accuracy.
        source = self.directory / "ar2_identity_only.ssb"
        source.write_text("SLURRY_HYBRID_V3\n2 12\nANCHORS 2 0.5 1\n")
        near_report = Path(str(source) + ".near-generation.json")
        far_report = Path(str(source) + ".far-generation.json")
        near_report.write_text(json.dumps({"aspect_ratio": 2, "offgrid_physical11_weighted_relative": .0001,
                                          "offgrid_basis_weighted_relative": .001, "loader_roundtrip_pass": True}))
        # V3 keeps this slot for serialization, but it does not supply resistance.
        far_report.write_text(json.dumps({"aspect_ratio": 2, "loader_roundtrip_pass": False}))
        Path(str(source) + ".pipeline.json").write_text(json.dumps({
            "basis_sha256": prepare_case.file_sha256(source), "generation_reports": {
                "near": prepare_case.file_sha256(near_report), "far": prepare_case.file_sha256(far_report)}}))
        fingerprint = prepare_case.fast_source_fingerprint()
        report = self.directory / "synthetic_numerical_gate.json"
        record = {"evaluated_path": "isolated_only", "aspect_ratio": 2.0,
                  "basis_sha256": prepare_case.file_sha256(source), "source_fingerprint": fingerprint,
                  "compiled_source_fingerprint": fingerprint, "film_grid": 161, "passed": True,
                  "records": [{"stage": "isolated_units", "passed": True,
                               "all11_error": .0001, "raw_reciprocity": .0001}]}
        report.write_text(json.dumps(record))
        selected = arguments(ar=2.0, finite_gap_fast_source_gate=report)
        physical = prepare_case.resolve_source_gate(selected, source, fingerprint, 161, report)
        self.assertEqual(physical["status"], "not_run")
        self.assertEqual(selected.finite_gap_fast_source_numerical_gate["status"], "passed")
        self.assertEqual([r["basis"] for r in selected.finite_gap_fast_source_numerical_gate["boundary_checks"]], ["near"])
        self.assertFalse(selected.finite_gap_fast_source_numerical_gate["active_source_policy"]["far_slot_used"])
        record["compiled_source_fingerprint"] = "stale"
        report.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "compiled_source_fingerprint identity mismatch"):
            prepare_case.resolve_source_gate(selected, source, fingerprint, 161, report)

    def test_full_generation_stops_at_failed_source_gate(self):
        source = self.directory / "identity_only.ssb"
        source.write_text("SLURRY_HYBRID_V1\n10 12\n")
        generator = self.directory / "must_not_run"
        generator.write_text("#!/bin/sh\nexit 99\n")
        generator.chmod(0o755)
        selected = arguments(finite_gap_fast_source=source, finite_gap_fast_generator=generator)
        with mock.patch.object(prepare_case, "resolve_source_gate", side_effect=ValueError("physical gate failed")) as gate:
            with self.assertRaisesRegex(ValueError, "physical gate failed"):
                prepare_case.resolve_lubrication(selected, 1.5e-6)
            gate.assert_called_once()
        self.assertFalse(any(source.parent.glob(".fast_cache/*.sft")))

    def test_manifest_detects_table_identity_mismatch(self):
        manifest = {"schema": 1, "ar": 10, "geometry": "equal_size_oblate",
                    "source_code_sha256": prepare_case.fast_source_fingerprint(),
                    "table": {"sha256": "0" * 64, "settings": {}},
                    "source_gate": {"status": "unverified"}, "interpolation_gate": {"status": "unverified"}}
        Path(str(self.table) + ".manifest.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "manifest SHA mismatch"):
            prepare_case.resolve_lubrication(arguments(self.table), 1.5e-6)

    def test_launcher_preserves_environment_and_accepts_explicit_flags(self):
        launcher = ROOT / "lammps_slurry/run_slurry_cpu.sbatch"
        subprocess.run(["bash", "-n", str(launcher)], check=True)
        prefix = launcher.read_text().split("readonly SLURRY_LAUNCHER_RELEASE")[0]
        snippet = (prefix + '\nprintf "%s\\n" "$LUBRICATION_MODEL" "$FINITE_GAP_FAST_TABLE"\n'
                   + 'printf "%s\\n" "${SLURRY_PREPARE_OVERRIDES[@]}"\n')
        import os
        environment = dict(os.environ, LUBRICATION_MODEL="finite_gap_fast", FINITE_GAP_FAST_TABLE="table with spaces.sft")
        result = subprocess.run(["bash", "-s", "--", "--lubrication-model", "finite_gap_fast", "--finite-gap-fast-source", "basis.ssb"],
                                input=snippet, universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=environment, check=True)
        self.assertEqual(result.stdout.splitlines(), ["finite_gap_fast", "table with spaces.sft",
                         "--lubrication-model", "finite_gap_fast", "--finite-gap-fast-source", "basis.ssb"])

    def test_prepared_input_and_metadata(self):
        if PACKER is None:
            self.skipTest("Pass --packer to exercise actual particle preparation")
        output = self.directory / "prepared"
        subprocess.run([sys.executable, str(ROOT / "lammps_slurry/tools/prepare_case.py"),
                        "--ar", "10", "--phi", "0.0001", "--initial-phi", "0.0001", "--count", "2",
                        "--prep-steps", "0", "--steps", "0", "--packer", str(PACKER), "--output", str(output),
                        "--lubrication-model", "finite_gap_fast", "--finite-gap-fast-table", str(self.table),
                        "--finite-gap-fast-nphi", "8"],
                       check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        self.assertIn("lubrication finite_gap_fast finite_gap_fast_table finite_gap_fast.sft finite_gap_range 0.5", (output / "in.slurry").read_text())
        metadata = (output / "parameters.txt").read_text()
        self.assertIn("lubrication_model_resolved = finite_gap_fast", metadata)
        self.assertIn("finite_gap_fast_phi_nodes_selected = 8", metadata)
        self.assertIn("finite_gap_sha256 = " + prepare_case.file_sha256(self.table), metadata)
        manifest = json.loads((output / "finite_gap_manifest.json").read_text())
        self.assertEqual(manifest["source_gate"]["status"], "unverified")
        self.assertEqual(manifest["interpolation_gate"]["status"], "unverified")
        self.assertEqual((output / "finite_gap_fast.sft").read_bytes(), self.table.read_bytes())

    def test_invalid_particle_box_rejected_before_expensive_generation(self):
        output = self.directory / "must_not_be_prepared"
        source = self.directory / "must_not_be_generated.ssb"
        result = subprocess.run([sys.executable, str(ROOT / "lammps_slurry/tools/prepare_case.py"),
                                 "--ar", "10", "--phi", "0.24", "--count", "2", "--output", str(output),
                                 "--lubrication-model", "finite_gap_fast", "--finite-gap-fast-source", str(source)],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("periodic initializer requires N", result.stderr)
        self.assertFalse(source.exists())
        self.assertFalse(output.exists())

    def test_direct_finite_gap_default_grid(self):
        if PACKER is None:
            self.skipTest("Pass --packer to exercise actual particle preparation")
        basis = ROOT / "lammps_slurry/models/ar10_matched.ssb"
        output = self.directory / "prepared_direct"
        subprocess.run([sys.executable, str(ROOT / "lammps_slurry/tools/prepare_case.py"),
                        "--ar", "10", "--phi", "0.0001", "--initial-phi", "0.0001", "--count", "2",
                        "--prep-steps", "0", "--steps", "0", "--packer", str(PACKER), "--output", str(output),
                        "--lubrication-model", "finite_gap", "--finite-gap-basis", str(basis)],
                       check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        self.assertIn("finite_gap_film_grid 161", (output / "in.slurry").read_text())
        self.assertIn("finite_gap_film_grid = 161", (output / "parameters.txt").read_text())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packer", type=Path)
    options, remaining = parser.parse_known_args()
    PACKER = options.packer
    unittest.main(argv=[sys.argv[0]] + remaining)
