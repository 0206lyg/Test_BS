"""Planner/finalizer contracts only: no Stokes solves, compiles or job submission.

Any manufactured records below are orchestration fixtures, never evidence of
physical accuracy. External processes are forbidden except explicitly mocked
failed invocations used to check independent AR failure handling.
"""
from copy import deepcopy
import csv
import itertools
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import reference_table_campaign as campaign
import run_campaign_reference as worker


class ReferenceCampaignContract(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="campaign_contract_only_")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.output = self.root / "campaign"
        self.no_process = mock.patch.object(campaign.subprocess, "run",
            side_effect=AssertionError("Contract tests must not run external processes"))
        self.no_process.start()
        self.addCleanup(self.no_process.stop)

    def plan(self, *extra):
        args = campaign.parser().parse_args([
            "plan", "--output-dir", str(self.output), *map(str, extra)])
        path, record = campaign.make_plan(args)
        return args, path, record

    def finalizer_args(self, path):
        tools = []
        for name in ("generator", "checker", "query-tool"):
            tool = self.root / name
            tool.write_bytes(b"test fixture only; not executable")
            tools += ["--" + name, str(tool)]
        return campaign.parser().parse_args(["finalize", "--campaign", str(path), *tools])

    def report_missing_context(self, command, folder, name):
        """Mock process launch, run only the real lightweight reporting code."""
        import assess_campaign_accuracy as assessor
        command = [str(value) for value in command]
        self.assertEqual(name, "accuracy_assessment")
        self.assertEqual(Path(command[1]).name, "assess_campaign_accuracy.py")
        for forbidden in ("run_reference.py", "run_campaign_reference.py", "slurry_generate_basis",
                          "run_reference_table_pipeline.py", "sbatch"):
            self.assertNotIn(forbidden, " ".join(command))
        manifest = Path(command[2])
        report = assessor.assess_campaign(json.loads(manifest.read_text()), manifest.parent)
        output = Path(command[command.index("--output") + 1])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report))
        rows = list(assessor.flat_rows(report))
        csv_path = Path(command[command.index("--csv") + 1])
        with csv_path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return dict(command=command, returncode=0, process_launch_mocked=True)

    def assert_all_case_geometry(self, plan):
        assessment = json.loads((self.output / "accuracy_input.json").read_text())
        self.assertEqual([g["id"] for g in assessment["groups"]], [g["id"] for g in plan["groups"]])
        for expected_group, actual_group in zip(plan["groups"], assessment["groups"]):
            self.assertEqual(len(actual_group["cases"]), len(expected_group["cases"]))
            for expected, actual in zip(expected_group["cases"], actual_group["cases"]):
                self.assertEqual(actual["id"], expected["id"])
                self.assertEqual(actual["geometry"], dict(aspect_ratio=expected_group["aspect_ratio"],
                    c_m=plan["settings"]["c_nm"]*1e-9, theta_i=expected["theta_i"],
                    theta_j=expected["theta_j"], phi=expected["phi"],
                    h_over_c=expected["gap_nm"]/plan["settings"]["c_nm"]))

    def test_default_plan_has_704_tasks_and_complete_actual_angular_grids(self):
        _, _, plan = self.plan()
        self.assertEqual(plan["task_count"], 704)
        self.assertEqual(plan["settings"]["aspect_ratios"], [2., 5., 10., 20.])
        self.assertFalse(plan["physical_accuracy_qualified"])
        self.assertEqual([task["task_id"] for task in plan["tasks"]], list(range(704)))
        expected_grid = set(itertools.product(plan["settings"]["theta_nodes"],
            plan["settings"]["theta_nodes"], plan["settings"]["phi_nodes"]))
        self.assertEqual(len(expected_grid), 128)
        for group in plan["groups"]:
            grid = [plan["tasks"][index] for index in group["grid_task_ids"]]
            self.assertEqual(len(grid), 128)
            self.assertEqual({(t["theta_i"], t["theta_j"], t["phi"]) for t in grid}, expected_grid)
            self.assertTrue(all(t["aspect_ratio"] == group["aspect_ratio"] for t in grid))
            self.assertTrue(all(t["gap_nm"] == 150. and t["role"] == "grid" for t in grid))
            self.assertEqual(len(group["cases"]), 24)
            validation = [t for t in plan["tasks"] if t["aspect_ratio"] == group["aspect_ratio"] and t["role"] == "validation"]
            self.assertEqual(len(validation), 48)

    def test_heldout_cases_change_gap_or_angle_and_have_two_distinct_resolutions(self):
        _, _, plan = self.plan()
        seen_ids = set()
        for group in plan["groups"]:
            training = {(t["theta_i"], t["theta_j"], t["phi"], t["gap_nm"])
                for t in (plan["tasks"][index] for index in group["grid_task_ids"])}
            training_angles = {row[:3] for row in training}
            kinds = set()
            for case in group["cases"]:
                self.assertNotIn(case["query_id"], seen_ids)
                seen_ids.add(case["query_id"])
                pose = tuple(case[k] for k in ("theta_i", "theta_j", "phi"))
                self.assertNotIn((*pose, case["gap_nm"]), training)
                kind = case["heldout_kind"]
                kinds.add(kind)
                if kind in ("angle", "both"):
                    self.assertNotIn(pose, training_angles)
                if kind in ("gap", "both"):
                    self.assertLess(case["gap_nm"], 150.)
                if kind == "gap":
                    self.assertIn(pose, training_angles)
                coarse, fine = [plan["tasks"][index] for index in case["reference_task_ids"]]
                self.assertEqual((coarse["resolution_label"], fine["resolution_label"]), ("coarse", "fine"))
                self.assertLess(coarse["resolution"]["order"], fine["resolution"]["order"])
                for task in (coarse, fine):
                    self.assertEqual(task["query_id"], case["query_id"])
                    self.assertEqual(task["aspect_ratio"], group["aspect_ratio"])
                    for key in ("theta_i", "theta_j", "phi", "gap_nm"):
                        self.assertEqual(task[key], case[key])
            self.assertEqual(kinds, {"gap", "angle", "both"})

    def test_identical_plan_is_reused_without_rewriting_and_changed_settings_are_rejected(self):
        args, path, original = self.plan()
        before = path.read_bytes()
        _, reused = campaign.make_plan(args)
        self.assertEqual(reused, original)
        self.assertEqual(path.read_bytes(), before)
        args.c_nm *= 2
        with self.assertRaisesRegex(ValueError, "different|changed|identity|new output"):
            campaign.make_plan(args)
        self.assertEqual(path.read_bytes(), before)

    def test_persisted_task_geometry_cannot_change_under_same_configuration_hash(self):
        args, path, plan = self.plan()
        plan["tasks"][0]["theta_i"] += .001
        path.write_text(json.dumps(plan))
        before = path.read_bytes()
        with self.assertRaisesRegex(ValueError, "different|changed|identity|modified|task|plan"):
            campaign.make_plan(args)
        self.assertEqual(path.read_bytes(), before)

    def test_actual_planner_tasks_and_controls_are_accepted_by_worker(self):
        _, _, plan = self.plan("--max-refinements", "3")
        values = worker.controls(plan)
        self.assertEqual(values["solver"], "out-of-core-dense")
        self.assertEqual(values["max_memory_gb"], 28.)
        for group in plan["groups"]:
            selected = [group["grid_task_ids"][0], *group["cases"][0]["reference_task_ids"]]
            for task_id in selected:
                task, physical = worker.select_task(plan, task_id)
                self.assertEqual(task["task_id"], task_id)
                self.assertEqual(physical["aspect_ratio"], group["aspect_ratio"])
                worker.resolution(task["resolution"], worker.PAIR_RESOLUTION)
        self.assertEqual(plan["defaults"]["max_refinements"], 3)

    def write_selected_report(self, task, campaign_path):
        folder = self.output / task["output_subdir"]
        folder.mkdir(parents=True, exist_ok=True)
        report_path = folder / "reference.json"
        report = {key: task[key] for key in ("aspect_ratio", "theta_i", "theta_j", "phi")}
        report.update(h_nm=task["gap_nm"], length_unit_nm=task["c_nm"],
            order=task["resolution"]["order"], level=task["resolution"]["level"],
            test_fixture_only=True, physical_accuracy_qualified=False)
        report_path.write_text(json.dumps(report))
        bound = dict(task=deepcopy(task), campaign_sha256=campaign.sha(campaign_path),
            test_fixture_only=True)
        state = dict(identity=bound, identity_sha256=campaign.identity(bound),
            accepted_for_development=True, physical_accuracy_qualified=False,
            status="complete", selected_pair_report={"path": str(report_path),
                "sha256": campaign.sha(report_path)}, test_fixture_only=True)
        state_path = folder / "task.json"
        state_path.write_text(json.dumps(state))
        return report_path, report, state_path, state

    def test_selected_report_is_bound_to_exact_file_task_and_campaign(self):
        _, path, plan = self.plan()
        task = plan["tasks"][0]
        report_path, report, state_path, state = self.write_selected_report(task, path)
        selected = campaign.task_report(self.output, task, path)
        self.assertEqual(selected["state"], "available")
        self.assertTrue(selected["accepted"])
        report_path.write_text(json.dumps(dict(report, extra="changed bytes")))
        self.assertEqual(campaign.task_report(self.output, task, path)["state"], "invalid")
        report_path.write_text(json.dumps(report))
        for change in ("task", "campaign", "identity_digest", "physical_flag"):
            with self.subTest(change=change):
                altered = deepcopy(state)
                if change == "task":
                    altered["identity"]["task"]["gap_nm"] /= 2
                    altered["identity_sha256"] = campaign.identity(altered["identity"])
                elif change == "campaign":
                    altered["identity"]["campaign_sha256"] = "b" * 64
                    altered["identity_sha256"] = campaign.identity(altered["identity"])
                elif change == "identity_digest":
                    altered["identity_sha256"] = "b" * 64
                else:
                    altered["physical_accuracy_qualified"] = True
                state_path.write_text(json.dumps(altered))
                self.assertEqual(campaign.task_report(self.output, task, path)["state"], "invalid")

    def test_report_geometry_size_or_resolution_mismatch_is_rejected_even_with_correct_sha(self):
        _, path, plan = self.plan()
        task = plan["tasks"][0]
        report_path, report, state_path, state = self.write_selected_report(task, path)
        for key, value in (("aspect_ratio", 99.), ("theta_i", .002), ("theta_j", .003),
                           ("phi", 0.), ("h_nm", 75.), ("length_unit_nm", 300.),
                           ("order", 9), ("level", 2)):
            with self.subTest(field=key):
                report_path.write_text(json.dumps(dict(report, **{key: value})))
                state["selected_pair_report"]["sha256"] = campaign.sha(report_path)
                state_path.write_text(json.dumps(state))
                self.assertEqual(campaign.task_report(self.output, task, path)["state"], "invalid")

    def test_missing_pair_report_is_unresolved_not_available(self):
        _, path, plan = self.plan()
        task = plan["tasks"][0]
        _, _, state_path, state = self.write_selected_report(task, path)
        state.pop("selected_pair_report")
        state.update(status="unresolved", accepted_for_development=False)
        state_path.write_text(json.dumps(state))
        selected = campaign.task_report(self.output, task, path)
        self.assertEqual(selected["state"], "no_pair_report")
        self.assertFalse(selected.get("accepted", False))

    def test_incomplete_pipeline_cache_cannot_be_reused_as_a_complete_table(self):
        tools = []
        for name in ("generator", "checker"):
            tool = self.root / name
            tool.write_bytes(b"test-only tool identity")
            tools.append(tool)
        inputs = [{"sha256": "a" * 64}]
        pipeline = self.root / "incomplete_pipeline.json"
        pipeline.write_text(json.dumps(dict(workflow_completed=True,
            physical_accuracy_qualified=False, input_reports=inputs,
            executables={str(tool): campaign.sha(tool) for tool in tools}, outputs={})))
        self.assertFalse(campaign.verified_pipeline(pipeline, inputs, tools))

    def test_complete_pipeline_cache_is_invalidated_by_hash_or_units_changes(self):
        _, _, plan = self.plan("--ars", "2")
        tools = []
        for name in ("generator", "checker"):
            tool = self.root / name
            tool.write_bytes(b"test-only tool identity")
            tools.append(tool)
        inputs = [{"sha256": "a" * 64}]
        folder = self.root / "manufactured_cache_fixture"
        folder.mkdir()
        (folder / "reference.source17").write_bytes(b"test-only manufactured source")
        (folder / "reference.sft").write_bytes(b"test-only manufactured table")
        source = dict(physical_accuracy_qualified=False, test_fixture_only=True,
            source_text_sha256=campaign.sha(folder / "reference.source17"))
        generation = dict(status="provisional_numerical_transport_passed",
            layout="complete_actual_source_grid", physical_accuracy_qualified=False,
            model_revision="JANOSCHEK_DISSIPATIVE_INCREMENT_FULL17_V1",
            source_fnv1a64="a" * 16, numerical_source_fingerprint="b" * 64,
            floor_m=1e-9, c_m=150e-9, hmax_over_c=1., test_fixture_only=True)
        runtime = dict(runtime_layout="complete_actual_source_grid", pipeline_algebra_passed=True,
            poses_checked=1, physical_accuracy_qualified=False,
            model_revision=generation["model_revision"], source_fnv1a64=generation["source_fnv1a64"],
            numerical_source_fingerprint=generation["numerical_source_fingerprint"], test_fixture_only=True)
        for name, data in (("reference.source17.manifest.json", source),
                           ("generation.json", generation), ("runtime.json", runtime)):
            (folder / name).write_text(json.dumps(data))
        outputs = {path.name: dict(bytes=path.stat().st_size, sha256=campaign.sha(path))
            for path in folder.iterdir()}
        record = dict(schema="provisional_reference_table_pipeline_v1", status="complete",
            layout="full-grid", matched_model="dissipative-increment", workflow_completed=True,
            physical_accuracy_qualified=False, input_reports=inputs, test_fixture_only=True,
            stages=[dict(stage=name, returncode=0) for name in ("import", "generate", "runtime")],
            executables={str(tool): campaign.sha(tool) for tool in tools}, outputs=outputs)
        path = folder / "pipeline.json"
        path.write_text(json.dumps(record))
        self.assertTrue(campaign.verified_pipeline(path, inputs, tools, plan["settings"]))
        self.assertFalse(campaign.verified_pipeline(path, [{"sha256": "c" * 64}], tools, plan["settings"]))
        for setting, value in (("floor_nm", 2.), ("c_nm", 200.), ("anchor_over_c", .5)):
            self.assertFalse(campaign.verified_pipeline(path, inputs, tools,
                dict(plan["settings"], **{setting: value})))
        (folder / "reference.sft").write_bytes(b"changed table")
        self.assertFalse(campaign.verified_pipeline(path, inputs, tools, plan["settings"]))

    def test_query_reuse_is_bound_to_table_bytes_and_missing_validation_stays_unresolved(self):
        _, path, plan = self.plan("--ars", "2")
        args = self.finalizer_args(path)
        attempt = self.output / "AR2" / "table_attempts" / "attempt_0001"
        attempt.mkdir(parents=True)
        (attempt / "pipeline.json").write_text('{"test_fixture_only":true}')
        table = attempt / "reference.sft"
        table.write_bytes(b"test-only table version one")
        query_outputs = []

        def available_grid(root, task, *extra, **kwargs):
            if task["role"] != "grid":
                return dict(task_id=task["task_id"], state="missing")
            return dict(task_id=task["task_id"], state="available", accepted=True,
                path=str(root / ("fixture_%06d.json" % task["task_id"])), sha256="a" * 64)

        def simulated_process(command, folder, name):
            command = [str(x) for x in command]
            if name == "query":
                output = Path(command[command.index("--output") + 1])
                output.write_text('{"test_fixture_only":true,"physical_accuracy_qualified":false}')
                query_outputs.append(output)
            elif name == "accuracy_assessment":
                output = Path(command[command.index("--output") + 1])
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(json.dumps(dict(test_fixture_only=True, physical_accuracy_qualified=False,
                    groups=[dict(id="AR2", counts={"PASS": 0, "FAIL": 0, "UNRESOLVED": 24})])))
            else:
                self.fail("Unexpected process stage " + name)
            return dict(command=command, returncode=0, test_fixture_only=True)

        with mock.patch.object(campaign, "task_report", side_effect=available_grid), \
             mock.patch.object(campaign, "verified_pipeline", return_value=True), \
             mock.patch.object(campaign, "run_logged", side_effect=simulated_process):
            first = campaign.finalize(args)
            campaign.finalize(args)
            self.assertEqual(len(query_outputs), 1)
            table.write_bytes(b"test-only table version two")
            last = campaign.finalize(args)
        self.assertEqual(len(query_outputs), 2)
        self.assertNotEqual(query_outputs[0], query_outputs[1])
        self.assertTrue(query_outputs[0].is_file())
        for summary in (first, last):
            self.assertFalse(summary["physical_accuracy_qualified"])
            self.assertEqual(summary["groups"][0]["accuracy_status"], "UNRESOLVED")
            self.assertEqual(summary["groups"][0]["accuracy_counts"]["PASS"], 0)
        assessment = json.loads((self.output / "accuracy_input.json").read_text())
        for case in assessment["groups"][0]["cases"]:
            self.assertTrue(all(not Path(reference).exists() for reference in case["references"]))

    def test_missing_references_produce_all_unresolved_rows_without_scientific_processes(self):
        _, path, plan = self.plan()
        with mock.patch.object(campaign, "run_logged", side_effect=self.report_missing_context) as reporting:
            summary = campaign.finalize(self.finalizer_args(path))
        self.assertEqual(reporting.call_count, 1)
        self.assertFalse(summary["physical_accuracy_qualified"])
        self.assertEqual(summary["task_counts"], {"available": 0, "missing": 704,
            "invalid": 0, "no_pair_report": 0})
        self.assertEqual(len(summary["groups"]), 4)
        for group in summary["groups"]:
            self.assertEqual(group["table_status"], "waiting_for_references")
            self.assertEqual(group["accuracy_status"], "UNRESOLVED")
            self.assertEqual(len(group["missing_or_rejected_grid_tasks"]), 128)
            self.assertFalse(group["physical_accuracy_qualified"])
            self.assertEqual(group["accuracy_counts"], {"PASS": 0, "FAIL": 0, "UNRESOLVED": 24})
        self.assert_all_case_geometry(plan)
        with (self.output / "accuracy" / "accuracy.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 96)
        self.assertEqual({row["status"] for row in rows}, {"UNRESOLVED"})
        self.assertEqual({float(row["AR"]) for row in rows}, {2., 5., 10., 20.})
        stored = json.loads((self.output / "campaign_summary.json").read_text())
        self.assertEqual(stored, summary)

    def test_each_ar_generation_failure_is_retained_and_other_ars_continue(self):
        _, path, plan = self.plan()
        args = self.finalizer_args(path)
        invocations = []

        def available(root, task, *extra, **kwargs):
            return dict(task_id=task["task_id"], state="available", accepted=True,
                path=str(root / ("fixture_%06d.json" % task["task_id"])),
                sha256="a" * 64, test_fixture_only=True)

        def fail(command, folder, name):
            invocations.append([str(x) for x in command])
            if name == "accuracy_assessment":
                return self.report_missing_context(command, folder, name)
            self.assertEqual(name, "table_generation")
            return dict(command=invocations[-1], returncode=2, test_fixture_only=True)

        with mock.patch.object(campaign, "task_report", side_effect=available), \
             mock.patch.object(campaign, "run_logged", side_effect=fail):
            summary = campaign.finalize(args)
        self.assertEqual(len(invocations), 5)
        self.assertEqual([g["id"] for g in summary["groups"]], ["AR2", "AR5", "AR10", "AR20"])
        self.assertTrue(all(g["table_status"] == "generation_failed" for g in summary["groups"]))
        self.assertTrue(all(g["accuracy_status"] == "UNRESOLVED" for g in summary["groups"]))
        self.assert_all_case_geometry(plan)
        self.assertEqual(sum(g["accuracy_counts"]["UNRESOLVED"] for g in summary["groups"]), 96)
        self.assertEqual(Path(invocations[-1][1]).name, "assess_campaign_accuracy.py")
        for command in invocations[:-1]:
            self.assertIn("run_reference_table_pipeline.py", " ".join(command))
            self.assertIn("--layout", command)
            self.assertEqual(command[command.index("--layout") + 1], "full-grid")
            self.assertEqual(command[command.index("--matched-model") + 1], "dissipative-increment")
            self.assertNotIn("slurry_generate_basis", " ".join(command))
            self.assertNotIn("sbatch", command)


if __name__ == "__main__":
    unittest.main()
