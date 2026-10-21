#!/usr/bin/env python3
"""Anchor identity/qualification contracts only; no physical solver is run."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gate = load("anchor_gate_runner", ROOT / "validation_basis/run_physical_gate.py")
prepare = load("anchor_gate_prepare", ROOT / "lammps_slurry/tools/prepare_case.py")


def numerical_records(anchors):
    """Synthetic records exercise acceptance logic, not Stokes accuracy."""
    records = [{"stage": "isolated_units", "passed": True, "limit": .001,
                "all11_error": .0001, "raw_reciprocity": .0001}]
    for name, component, gap in gate.numerical_signatures(anchors):
        record = {"stage": "pair_numerical", "category": "numerical", "name": name,
                  "component": component, "h_over_c": gap, "passed": True,
                  "positive_total": True, "min_eigenvalue": 1.0}
        if component == "raw_full17":
            record.update(raw_reciprocity=.0001, linear_residual=1e-12,
                          reciprocity_limit=.01, residual_limit=1e-8)
        else:
            effective = max(gap, gate.HARD_FLOOR_OVER_C)
            record.update(gap_rule="max(h,h_min)", h_min_over_c=gate.HARD_FLOOR_OVER_C,
                          h_eff_over_c=effective, matrix_dimension=17,
                          free_strain_min_eigenvalue=1.0,
                          evaluation_branch="direct_stokes" if effective >= anchors[0] else "matched_asymptotic")
        records.append(record)
    return records


class AnchorGateContracts(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="slurry-anchor-contract-")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.source = self.directory / "header_only.ssb"
        self.source.write_text("SLURRY_HYBRID_V4\n20 12\nANCHORS 2 0.125 0.25\n")

    def test_records_follow_selected_anchors_and_reject_old_signature(self):
        selected = [.125, .25]
        records = numerical_records(selected)
        self.assertTrue(gate.numerical_records_passed(records, selected))
        self.assertFalse(gate.numerical_records_passed(records, [.5, 1.]))
        self.assertFalse(gate.numerical_records_passed(numerical_records([.5, 1.]), selected))
        for changed in (records[:-1], records + [records[-1]]):
            self.assertFalse(gate.numerical_records_passed(changed, selected))

    def test_branch_identity_is_not_falsely_called_near(self):
        selected = [.01, .04]
        records = numerical_records(selected)
        self.assertTrue(gate.numerical_records_passed(records, selected))
        direct = next(r for r in records if r.get("component") == "matched_full17" and r["h_over_c"] == .02)
        self.assertEqual(direct["evaluation_branch"], "direct_stokes")
        direct["evaluation_branch"] = "matched_asymptotic"
        self.assertFalse(gate.numerical_records_passed(records, selected))

    def test_unbound_or_invalid_anchor_lists_fail_closed(self):
        records = numerical_records([.125, .25])
        for invalid in (None, [], [.125], [.25, .125], [.1, .1], [0, 1], [True, 2], [float("nan"), 1]):
            with self.subTest(anchors=invalid):
                self.assertFalse(gate.numerical_records_passed(records, invalid))
                with self.assertRaises(ValueError):
                    prepare.requested_anchor_gaps(SimpleNamespace(finite_gap_fast_anchor_gaps=invalid if invalid is not None else []))

    def test_existing_bundle_anchors_are_authoritative(self):
        args = SimpleNamespace(ar=20., finite_gap_fast_source=self.source)
        self.assertEqual(prepare.resolve_fast_basis(args), self.source)
        self.assertEqual(args.finite_gap_fast_basis_header["anchors_over_c"], [.125, .25])
        args.finite_gap_fast_anchor_gaps = [.5, 1.]
        with self.assertRaisesRegex(ValueError, "Existing basis source anchors"):
            prepare.resolve_fast_basis(args)

    def test_missing_basis_requires_explicit_anchors_before_generator(self):
        args = SimpleNamespace(ar=20., finite_gap_fast_source=self.directory / "absent.ssb")
        with mock.patch.object(prepare.subprocess, "run") as process:
            with self.assertRaisesRegex(ValueError, "requires explicit.*anchor-gaps"):
                prepare.resolve_fast_basis(args)
            process.assert_not_called()
        self.assertFalse(args.finite_gap_fast_source.exists())

    def test_cached_numerical_anchor_identity_must_match_table_settings(self):
        anchors = [.125, .25]
        records = numerical_records(anchors)
        table = self.directory / "manifest_only.sft"
        table.write_bytes(b"Synthetic manifest identity fixture; not a resistance table")
        numerical = {"status": "passed", "category": "numerical", "evaluated_path": "pair_numerical",
                     "source_anchors_over_c": anchors, "physical_accuracy_qualified": False,
                     "isolated_checks": records[:1], "pair_checks": records[1:]}
        manifest = {"schema": 1, "ar": 20., "geometry": "equal_size_oblate",
                    "basis_sha256": prepare.file_sha256(self.source),
                    "table": {"sha256": prepare.file_sha256(table), "settings": {"source_anchors_over_c": anchors}},
                    "source_gate": {"status": "not_run"}, "interpolation_gate": {"status": "unverified"},
                    "source_numerical_gate": numerical}
        path = Path(str(table) + ".manifest.json")
        for defect in (None, "table_anchors", "missing_gate_anchors", "claims_physical"):
            current = copy.deepcopy(manifest)
            if defect == "table_anchors":
                current["table"]["settings"]["source_anchors_over_c"] = [.5, 1.]
            elif defect == "missing_gate_anchors":
                current["source_numerical_gate"].pop("source_anchors_over_c")
            elif defect == "claims_physical":
                current["source_numerical_gate"]["physical_accuracy_qualified"] = True
            path.write_text(json.dumps(current))
            with self.subTest(defect=defect):
                if defect is None:
                    self.assertEqual(prepare.read_fast_manifest(table, SimpleNamespace(ar=20.), source=self.source)
                                     ["source_numerical_gate"]["status"], "passed")
                else:
                    with self.assertRaisesRegex(ValueError, "Cached numerical gate"):
                        prepare.read_fast_manifest(table, SimpleNamespace(ar=20.), source=self.source)

    def test_cutoff_override_is_already_rejected_for_both_finite_paths(self):
        for mode in ("finite_gap", "finite_gap_fast"):
            args = SimpleNamespace(lubrication_model=mode, ar=20.,
                finite_gap_fast_table=None, finite_gap_fast_source=None, finite_gap_fast_generator=None,
                finite_gap_basis=self.source if mode == "finite_gap" else None, finite_gap_table=None,
                lubrication_cutoff_nm=75.)
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "cutoff-nm would be ambiguous"):
                prepare.resolve_lubrication(args, 3e-6)

    def test_launcher_forwards_two_anchor_values_without_launching(self):
        launcher = ROOT / "lammps_slurry/run_slurry_cpu.sbatch"
        subprocess.run(["bash", "-n", str(launcher)], check=True)
        prefix = launcher.read_text().split("readonly SLURRY_LAUNCHER_RELEASE")[0]
        script = prefix + '\nprintf "%s\\n" "${SLURRY_PREPARE_OVERRIDES[@]}"\n'
        result = subprocess.run(["bash", "-s", "--", "--finite-gap-fast-anchor-gaps", ".125", ".25"],
                                input=script, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines(), ["--finite-gap-fast-anchor-gaps", ".125", ".25"])
        missing = subprocess.run(["bash", "-s", "--", "--finite-gap-fast-anchor-gaps", ".125"],
                                 input=script, capture_output=True, text=True)
        self.assertEqual(missing.returncode, 2)

    def test_custom_generation_preserves_anchor_choice_and_unqualified_scope(self):
        source = self.directory / "generated.ssb"
        generator = self.directory / "fake_generator"
        generator.write_text("#!/bin/sh\nexit 99\n")
        generator.chmod(0o755)
        args = SimpleNamespace(ar=20., finite_gap_fast_source=source,
                               finite_gap_fast_basis_generator=generator,
                               finite_gap_fast_anchor_gaps=[.125, .25])
        calls = []

        def synthesize(command, **kwargs):
            calls.append(command)
            output = Path(command[command.index("--output") + 1])
            report = Path(command[command.index("--report") + 1])
            if "--anchor-gaps" in command:
                anchors = command[command.index("--anchor-gaps") + 1]
                output.write_text("SLURRY_HYBRID_V4\n20 12\nANCHORS 2 " + anchors.replace(",", " ") + "\n")
            else:
                output.write_text("SLURRY_SCATTERING_V2\n20 8\n")
            report.write_text(json.dumps({"aspect_ratio": 20., "isolated_all11_energy_error": 0.,
                "isolated_reciprocity": 0., "offgrid_physical11_weighted_relative": 0.,
                "offgrid_basis_weighted_relative": 0., "loader_roundtrip_pass": True}))

        with mock.patch.object(prepare.subprocess, "run", side_effect=synthesize):
            prepare.resolve_fast_basis(args)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[1][calls[1].index("--anchor-gaps") + 1], "0.125,0.25")
        provenance = json.loads(Path(str(source) + ".pipeline.json").read_text())
        self.assertEqual(provenance["source_anchors_over_c"], [.125, .25])
        self.assertEqual(provenance["source_gate"], {"status": "not_run", "physical_accuracy_qualified": False})

    def test_runner_binds_custom_anchors_without_physical_qualification(self):
        anchors = [.125, .25]
        fingerprint = prepare.fast_source_fingerprint()
        metadata = {"stage": "metadata", "evaluated_path": "pair_numerical", "gate_category": "numerical",
                    "aspect_ratio": 20., "film_grid": 161, "hybrid_version": 4,
                    "model_revision": "JANOSCHEK_MATCHED_HARDFLOOR_FULL17_V4", "gap_rule": "max(h,h_min)",
                    "compiled_source_fingerprint": fingerprint, "source_anchors_over_c": anchors,
                    "physical_accuracy_qualified": False}
        executable = self.directory / "synthetic_probe.py"
        report = self.directory / "report.json"
        for defect in (None, "wrong_metadata_anchors", "wrong_record_anchors", "claims_physical"):
            records = numerical_records([.5, 1.] if defect == "wrong_record_anchors" else anchors)
            current = copy.deepcopy(metadata)
            if defect == "wrong_metadata_anchors":
                current["source_anchors_over_c"] = [.5, 1.]
            elif defect == "claims_physical":
                current["physical_accuracy_qualified"] = True
            payload = "\n".join(json.dumps(r) for r in [current] + records)
            executable.write_text("#!" + sys.executable + "\nprint(" + repr(payload) + ")\n")
            executable.chmod(0o755)
            result = subprocess.run([sys.executable, str(ROOT / "validation_basis/run_physical_gate.py"),
                "--executable", str(executable), "--basis", str(self.source), "--mode", "pair_numerical",
                "--report", str(report)], capture_output=True, text=True)
            saved = json.loads(report.read_text())
            with self.subTest(defect=defect):
                self.assertEqual(result.returncode, 0 if defect is None else 2, result.stderr)
                self.assertEqual(saved["passed"], defect is None)
                self.assertEqual(saved["source_anchors_over_c"], anchors)
                self.assertIs(saved["physical_accuracy_qualified"], False)
                self.assertEqual(saved["gate_category"], "numerical")


if __name__ == "__main__":
    unittest.main()
