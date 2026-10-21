#!/usr/bin/env python3
"""Qualify independent reference17 refinements and test one-pose matching.

Example (three independent discretization levels per gap):
  python compare_reference.py ref_*.json --anchor-gap-over-c 0.1 \
      --operator-helper /tmp/operator_export --output comparison.json

The helper exports production isolated/Janoschek operators; it is NOT a
reference solver.  Refinement changes are error estimates, not rigorous bounds.
No eigenvalues are clipped and a failed source cannot become qualified by
symmetrization, table encoding, or a successful algebra test.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Any

import numpy as np
from scipy.linalg import solve_triangular

try:
    from .geometry import canonical_geometry, strain_basis
except ImportError:
    from geometry import canonical_geometry, strain_basis

CONVENTION = "slurry_full17_c_eta_1_v1"
BLOCKS = {
    "force": np.array([0, 1, 2, 6, 7, 8]),
    "torque_over_a": np.array([3, 4, 5, 9, 10, 11]),
    "intrinsic_stresslet_over_a": np.arange(12, 17),
}
MOMENT_METHODS = {"physical_traction", "reciprocal_work", "combined_layer_farfield_moments"}
PANEL_SCHEMA = "independent_ar20_panel_bie_v1"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def symmetric(a: np.ndarray) -> np.ndarray:
    return (a + a.T) / 2


def matrix(value: Any) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (17, 17) or not np.isfinite(result).all():
        raise ValueError("A finite 17 by 17 raw resistance matrix is required")
    return result


def finite_number(value: Any) -> float | None:
    if isinstance(value, (float, int)) and not isinstance(value, bool) and np.isfinite(value):
        return float(value)
    return None


def digest_identity(value: Any) -> str | None:
    """A provenance identity must be an actual, nonempty SHA256 digest."""
    if (isinstance(value, str) and len(value) == 64 and
            all(character in '0123456789abcdefABCDEF' for character in value)):
        return value.lower()
    return None


def operator_identity(data: dict) -> tuple[dict | None, list[str]]:
    """Identify the numerical operator, excluding driver/storage bookkeeping.

    Panel-BIE reports cannot override their required identity with an external
    declaration. Generic independent reference17 files instead declare:
      reference_operator_identity = {
        "solver_family": "external_solver_name",
        "implementation_sha256": "...", "geometry_sha256": "...",
        "parameters": { ... optional operator parameters ... }
      }
    Their top-level full17 convention remains mandatory. If a generic report
    supplies library_sha256, it also becomes part of its operator identity.
    """
    failures = []
    convention = data.get("convention_id")
    if convention != CONVENTION:
        failures.append("operator_identity_convention_missing_or_different")
    if data.get("schema") == PANEL_SCHEMA:
        hashes = data.get("source_hashes")
        hashes = hashes if isinstance(hashes, dict) else {}
        identity = {"kind": "panel_bie", "convention_id": convention}
        required = {
            "library_sha256": data.get("library_sha256"),
            "backend_sha256": hashes.get("backend.py"),
            "geometry_sha256": hashes.get("geometry.py"),
        }
        for field, value in required.items():
            digest = digest_identity(value)
            if digest is None:
                failures.append(f"panel_operator_{field}_missing_or_invalid")
            identity[field] = digest
        for field in ("alpha", "beta"):
            value = finite_number(data.get(field))
            if value is None:
                failures.append(f"panel_operator_{field}_missing_or_nonfinite")
            identity[field] = value
    else:
        declaration = data.get("reference_operator_identity")
        if not isinstance(declaration, dict):
            return None, failures + ["external_reference_operator_identity_missing"]
        family = declaration.get("solver_family")
        if not isinstance(family, str) or not family.strip():
            failures.append("external_operator_solver_family_missing")
        identity = {"kind": "external_reference", "solver_family": family,
                    "convention_id": convention}
        for field in ("implementation_sha256", "geometry_sha256"):
            digest = digest_identity(declaration.get(field))
            if digest is None:
                failures.append(f"external_operator_{field}_missing_or_invalid")
            identity[field] = digest
        parameters = declaration.get("parameters", {})
        try:
            if not isinstance(parameters, dict):
                raise ValueError("operator parameters must be an object")
            # Canonical JSON rejects NaN and nonserializable declarations.
            identity["parameters"] = json.loads(json.dumps(parameters, sort_keys=True, allow_nan=False))
        except (TypeError, ValueError):
            failures.append("external_operator_parameters_invalid")
        if "library_sha256" in data:
            digest = digest_identity(data["library_sha256"])
            if digest is None:
                failures.append("external_operator_library_sha256_invalid")
            identity["library_sha256"] = digest
    return (None if failures else identity), failures


def operator_identity_gate(refs: list["Reference"]) -> dict:
    """Require one compatible operator for the latest three refinements."""
    identities = []
    failures = []
    records = []
    for reference in refs:
        identity, missing = operator_identity(reference.data)
        identities.append(identity)
        records.append({"path": str(reference.path), "identity": identity,
                        "failures": missing})
        failures.extend(missing)
    if len(refs) < 3:
        failures.append("fewer_than_three_operator_identities")
    if all(identity is not None for identity in identities) and identities:
        signatures = {json.dumps(identity, sort_keys=True, allow_nan=False) for identity in identities}
        if len(signatures) != 1:
            failures.append("latest_three_operator_identities_differ")
    return {"passed": not failures, "scope": "latest_three_reference_sources",
            "failures": sorted(set(failures)), "sources": records}


def energy_delta(delta: np.ndarray, reference: np.ndarray) -> float:
    """sup |z.T delta z|/(z.T reference z), requiring reference SPD."""
    chol = np.linalg.cholesky(reference)
    left = solve_triangular(chol, symmetric(delta), lower=True)
    normalized = solve_triangular(chol, left.T, lower=True).T
    return float(np.max(np.abs(np.linalg.eigvalsh(symmetric(normalized)))))


def energy_scaled_reciprocity(raw: np.ndarray, reference: np.ndarray) -> float:
    """Antisymmetric response norm, retaining weak modes hidden by Frobenius."""
    chol = np.linalg.cholesky(reference)
    left = solve_triangular(chol, raw-raw.T, lower=True)
    normalized = solve_triangular(chol, left.T, lower=True).T
    return float(np.linalg.norm(normalized, ord=2))


def discretization_signature(data: dict) -> str | None:
    if data.get("discretization_sha256"):
        details = {"mesh_sha256": data["discretization_sha256"]}
    else:
        # A user-supplied resolution label alone is not evidence of refinement.
        # Quadrature-tolerance changes on one mesh are also not spatial
        # refinement; record those separately in each reference report.
        if not (finite_number(data.get("node_count")) or 0) > 0 or not (finite_number(data.get("order")) or 0) > 0:
            return None
        keys = ("order", "level", "focus", "rim_levels", "rim_width", "contact_levels",
                "contact_radius", "node_count", "patch_count")
        details = {key: data[key] for key in keys if key in data}
    return json.dumps(details, sort_keys=True) if details else None


def block_errors(delta: np.ndarray, reference: np.ndarray) -> dict:
    """Energy-whiten inputs; scale output components by sqrt(reference diagonal).

    The relative spectral norm compares a whole F/T/S response block.  It is
    not an elementwise relative error and does not replace the full energy
    test, which detects losses hidden by a dominant squeeze response.
    """
    chol = np.linalg.cholesky(reference)
    scales = np.sqrt(np.diag(reference))
    out = {}
    for name, rows in BLOCKS.items():
        ref_block = solve_triangular(chol, reference[rows].T, lower=True).T / scales[rows, None]
        difference = solve_triangular(chol, delta[rows].T, lower=True).T / scales[rows, None]
        reference_norm = np.linalg.norm(ref_block, ord=2)
        out[name] = {
            "energy_input_scaled_operator_absolute": float(np.linalg.norm(difference, ord=2)),
            "energy_input_scaled_operator_relative": float(np.linalg.norm(difference, ord=2) / reference_norm),
            "unweighted_relative_frobenius": float(np.linalg.norm(delta[rows]) / np.linalg.norm(reference[rows])),
        }
    return out


def free_strain(g: np.ndarray) -> np.ndarray:
    return symmetric(g[12:, 12:] - g[12:, :12] @ np.linalg.solve(g[:12, :12], g[:12, 12:]))


def prescribed_maps(gap: float) -> tuple[np.ndarray, np.ndarray]:
    pose = canonical_geometry(gap)
    common = np.zeros((17, 6))
    held = np.zeros((17, 5))
    held[12:] = np.eye(5)
    # Origin midway between the two closest surface points, not body centers.
    midpoint = np.array([0.0, 0.0, gap / 2])
    es = strain_basis()
    for body, center in enumerate(pose.centers):
        arm = center - midpoint
        common[6*body:6*body+3, :3] = np.eye(3)
        for k in range(3):
            common[6*body:6*body+3, 3+k] = np.cross(np.eye(3)[k], arm) / pose.a
        common[6*body+3:6*body+6, 3:] = np.eye(3)
        for k in range(5):
            held[6*body:6*body+3, k] = -(es[k] @ arm) / pose.a
    return common, held


def response_metrics(predicted: np.ndarray, reference: np.ndarray, gap: float) -> dict:
    delta = predicted - reference
    common, held = prescribed_maps(gap)
    result = {
        "full17_max_relative_energy_error": energy_delta(delta, reference),
        "blocks": block_errors(delta, reference),
        "common_motion_max_relative_energy_error": energy_delta(common.T @ delta @ common, common.T @ reference @ common),
        "fixed_body_total_dipole_max_relative_energy_error": energy_delta(held.T @ delta @ held, held.T @ reference @ held),
        "fixed_body_intrinsic_stress_relative_frobenius_error": float(np.linalg.norm(delta[12:] @ held) / np.linalg.norm(reference[12:] @ held)),
        "fixed_body_definition": "U=Omega=0; uinf=E(x-contact_midpoint), so center-nonaffine U=-E(xi-contact_midpoint)",
    }
    if np.linalg.eigvalsh(symmetric(predicted))[0] > 0:
        ref_free, pred_free = free_strain(reference), free_strain(predicted)
        result["force_torque_free_strain_max_relative_energy_error"] = energy_delta(pred_free-ref_free, ref_free)
        vref = -np.linalg.solve(reference[:12, :12], reference[:12, 12:])
        vpred = -np.linalg.solve(predicted[:12, :12], predicted[:12, 12:])
        result["force_torque_free_velocity_relative_frobenius_error"] = float(np.linalg.norm(vpred-vref) / max(np.linalg.norm(vref), 1e-300))
    return result


@dataclass
class Reference:
    path: Path
    data: dict
    gap: float
    level: int | None
    raw: np.ndarray
    total: np.ndarray
    report: dict


def read_reference(path: Path, args: argparse.Namespace) -> Reference:
    data = json.loads(path.read_text())
    geometry = data.get("geometry_metadata", data)
    if not isinstance(geometry, dict):
        geometry = data
    gap = finite_number(geometry.get("h_over_c"))
    if gap is None or gap <= 0:
        raise ValueError(f"{path}: positive h_over_c required")
    expected = canonical_geometry(gap).metadata()
    for key in ("aspect_ratio", "theta_i", "theta_j", "phi"):
        actual = finite_number(geometry.get(key))
        if actual is None or abs(actual-expected[key]) > 1e-13:
            raise ValueError(f"{path}: not the exact recorded AR20 pose ({key})")
    raw = matrix(data["raw_full17"])
    total = symmetric(raw)
    validation = dict(data.get("validation", {}))
    if "boundary_report" in data:
        boundary_path = (path.parent / data["boundary_report"]).resolve()
        boundary = json.loads(boundary_path.read_text())
        values = [finite_number(boundary.get(part, {}).get("max_scaled_error")) for part in ("global", "gap_patch")]
        values = [v for v in values if v is not None]
        if values:
            validation.setdefault("boundary_max_scaled_error", max(values))
        validation.setdefault("boundary_report_sha256", sha(boundary_path))
        # A file called boundary_report alone does not prove held-out points.
    boundary_error = finite_number(validation.get("boundary_max_scaled_error"))
    surface_error = finite_number(validation.get("exact_surface_position_max_error_over_gap"))
    residual = finite_number(validation.get("linear_relative_residual", data.get("linear_residual")))
    reciprocity = float(np.linalg.norm(raw-raw.T) / max(np.linalg.norm(raw), 1e-300))
    mineig = float(np.linalg.eigvalsh(total)[0])
    scaled_reciprocity = energy_scaled_reciprocity(raw, total) if mineig > 0 else None
    moments = dict(data.get("moments", {}))
    reasons = []
    if data.get("convention_id") != CONVENTION:
        reasons.append("missing_or_different_full17_convention")
    if data.get("independent_reference") is not True:
        reasons.append("source_not_declared_independent_reference")
    if data.get("shares_production_solver") is not False:
        reasons.append("production_solver_independence_not_declared")
    if data.get("synthetic") is True:
        reasons.append("synthetic_diagnostic_is_not_physical_reference")
    if data.get("isolated_body") is not None:
        reasons.append("isolated_body_diagnostic_is_not_pair_reference")
    if moments.get("method") not in MOMENT_METHODS:
        reasons.append("physical_moment_extraction_not_identified")
    if moments.get("isolated_spheroid_validation_passed") is not True:
        reasons.append("isolated_spheroid_moment_validation_missing_or_failed")
    if validation.get("off_grid_boundary") is not True:
        reasons.append("independent_boundary_points_not_verified")
    is_panel_backend = data.get("schema") == PANEL_SCHEMA
    if is_panel_backend and data.get("status") != "complete":
        reasons.append("panel_reference_process_not_complete")
    if (validation.get("boundary_evaluation_valid") is False or
            (is_panel_backend and validation.get("boundary_evaluation_valid") is not True)):
        reasons.append("boundary_evaluator_failed_or_clean_execution_not_verified")
    if boundary_error is None or not 0 <= boundary_error <= args.boundary_tolerance:
        reasons.append("boundary_error_missing_or_above_tolerance")
    if surface_error is None or not 0 <= surface_error <= args.surface_tolerance:
        reasons.append("surface_position_error_over_gap_missing_or_above_tolerance")
    if residual is None or not 0 <= residual <= args.linear_tolerance:
        reasons.append("linear_residual_missing_or_above_tolerance")
    if reciprocity > args.reciprocity_tolerance:
        reasons.append("raw_reciprocity_above_tolerance")
    if scaled_reciprocity is None or scaled_reciprocity > args.reciprocity_tolerance:
        reasons.append("energy_scaled_raw_reciprocity_missing_or_above_tolerance")
    if not mineig > 0:
        reasons.append("symmetric_reference_not_positive_definite")
    flux = finite_number(moments.get("max_scaled_flux_error"))
    if moments.get("method") == "combined_layer_farfield_moments" and flux is None:
        reasons.append("combined_layer_flux_missing_or_nonfinite")
    if flux is not None and not 0 <= flux <= args.boundary_tolerance:
        reasons.append("combined_layer_flux_above_tolerance")
    level = data.get("resolution_level")
    if not isinstance(level, int) or isinstance(level, bool):
        level = None
        reasons.append("resolution_level_missing")
    identity, identity_failures = operator_identity(data)
    reasons.extend(identity_failures)
    report = {
        "path": str(path), "sha256": sha(path), "resolution_level": level,
        "method": data.get("method", data.get("schema")),
        "library_sha256": data.get("library_sha256"),
        "solver_contract": data.get("solver_contract"),
        "operator_identity": identity, "operator_identity_failures": identity_failures,
        "raw_reciprocity": reciprocity, "minimum_symmetric_eigenvalue": mineig,
        "energy_scaled_raw_reciprocity": scaled_reciprocity,
        "discretization_signature": discretization_signature(data),
        "quadrature_tolerance": data.get("quadrature_tolerance"),
        "boundary_max_scaled_error": boundary_error, "linear_relative_residual": residual,
        "boundary_evaluation_valid": validation.get("boundary_evaluation_valid"),
        "exact_surface_position_max_error_over_gap": surface_error,
        "moment_method": moments.get("method"), "max_scaled_flux_error": flux,
        "source_gates_passed": not reasons, "source_gate_failures": reasons,
        "source_claimed_qualified": data.get("qualified", data.get("qualified_reference", False)),
    }
    return Reference(path, data, gap, level, raw, total, report)


def qualify_group(refs: list[Reference], args: argparse.Namespace) -> dict:
    # Preserve input order when metadata is incomplete; incomplete levels never qualify.
    if all(r.level is not None for r in refs):
        refs.sort(key=lambda r: r.level)
    levels = [r.level for r in refs]
    distinct_levels = None not in levels and len(set(levels)) == len(levels)
    signatures = [r.report.get("discretization_signature") for r in refs]
    distinct_discretizations = None not in signatures and len(set(signatures)) == len(signatures)
    changes = []
    for previous, current in zip(refs[:-1], refs[1:]):
        item = {"previous": str(previous.path), "current": str(current.path)}
        if previous.report["minimum_symmetric_eigenvalue"] > 0 and current.report["minimum_symmetric_eigenvalue"] > 0:
            metrics = response_metrics(current.total, previous.total, current.gap)
            item.update(metrics)
            item["refinement_gate_passed"] = metrics["full17_max_relative_energy_error"] <= args.reference_budget and all(
                b["energy_input_scaled_operator_relative"] <= args.reference_budget for b in metrics["blocks"].values())
        else:
            item["refinement_gate_passed"] = False
            item["failure"] = "nonpositive_reference_prevents_energy_comparison"
        changes.append(item)
    recent_changes = changes[-2:]
    recent_sources = refs[-3:]
    identity_gate = operator_identity_gate(recent_sources)
    qualified = (len(refs) >= 3 and distinct_levels and distinct_discretizations
                 and identity_gate["passed"]
                 and all(r.report["source_gates_passed"] for r in recent_sources)
                 and all(c["refinement_gate_passed"] for c in recent_changes))
    latest = refs[-1]
    libraries = {digest_identity(r.data.get("library_sha256")) for r in refs}
    uncertainty = None
    block_uncertainty = None
    if len(recent_changes) == 2 and all("full17_max_relative_energy_error" in c for c in recent_changes):
        recent_deltas = [current.total-previous.total for previous, current in zip(refs[-3:-1], refs[-2:])]
        # These estimates enter a budget whose reference is the latest matrix.
        # Earlier pairwise refinement metrics above deliberately use that
        # pair's previous matrix and must not be reused with a different norm.
        uncertainty = max(energy_delta(delta, latest.total) for delta in recent_deltas)
        block_uncertainty = {name: max(block_errors(delta, latest.total)[name]["energy_input_scaled_operator_relative"]
                                      for delta in recent_deltas) for name in BLOCKS}
    return {
        "h_over_c": latest.gap, "gap_nm": 150*latest.gap,
        "references": [r.report for r in refs], "refinements": changes,
        "qualified_reference": bool(qualified), "distinct_resolution_levels": bool(distinct_levels),
        "distinct_discretizations": bool(distinct_discretizations),
        "operator_identity_gate": identity_gate,
        "same_library_across_inputs": len(libraries) == 1 and None not in libraries,
        "refinement_provenance_note": ("Observed changes include a library change, not solely spatial resolution."
                                       if len(libraries) > 1 else "Library identity reported above; no independent physical qualification inferred."),
        "estimated_energy_uncertainty": uncertainty, "estimated_block_uncertainty": block_uncertainty,
        "uncertainty_interpretation": "maximum of last two observed refinement changes, measured against latest reference; not a rigorous error bound",
        "latest_path": str(latest.path),
    }


def helper_export(helper: Path, gap: float, total: np.ndarray | None = None) -> dict:
    with tempfile.TemporaryDirectory(prefix="ar20_operator_") as temporary:
        command = [str(helper), "--gap-over-c", repr(gap)]
        if total is not None:
            path = Path(temporary) / "total17.txt"
            np.savetxt(path, total, fmt="%.17g")
            command.extend(["--total-in", str(path)])
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode:
            raise RuntimeError(f"operator helper failed ({result.returncode}): {result.stderr.strip()}")
        data = json.loads(result.stdout)
        actual_gap = finite_number(data.get("h_over_c"))
        if (data.get("convention_id") != CONVENTION or actual_gap is None or
                abs(actual_gap-gap) > 1e-13):
            raise ValueError("Operator helper convention or gap mismatch")
        expected = canonical_geometry(gap).metadata()
        for key in ("aspect_ratio", "theta_i", "theta_j", "phi"):
            actual = finite_number(data.get(key))
            if actual is None or abs(actual-expected[key]) > 1e-13:
                raise ValueError(f"Operator helper is not the recorded AR20 pose ({key})")
        return data


def compare(args: argparse.Namespace) -> dict:
    refs = [read_reference(Path(p).resolve(), args) for p in args.references]
    groups: dict[float, list[Reference]] = {}
    for ref in refs:
        found = next((gap for gap in groups if abs(gap-ref.gap) < 1e-13), ref.gap)
        groups.setdefault(found, []).append(ref)
    summaries = {gap: qualify_group(group, args) for gap, group in sorted(groups.items())}
    result = {
        "schema": "independent_ar20_reference_and_matching_comparison_v1", "qualified": False,
        "scope": "Only the recorded AR20 pose and explicitly supplied effective gaps; no other AR or full table",
        "convention_id": CONVENTION,
        "conventions": {
            "input": "Ui-uinf(xi),a(Omegai-Omegainf),Uj-uinf(xj),a(Omegaj-Omegainf),aE5; c=eta=1,a=20",
            "output": "Fsource_i,Tsource_i/a,Fsource_j,Tsource_j/a,sum intrinsic S/a; physical hydrodynamic forces=-Fsource",
            "stresslet": "minus STF symmetric source dipole about each body center; not the global-origin force dipole",
            "combined_layer": "density is not traction; for D=+3r_i r_j(r.n)/(4pi r^5), M=STF integral[alpha sym(r_body tensor q)+2beta sym(n tensor q)], S=-M",
        },
        "thresholds": {
            "reference_refinement_budget": args.reference_budget, "total_model_budget": args.model_budget,
            "boundary_max_scaled": args.boundary_tolerance, "raw_reciprocity": args.reciprocity_tolerance,
            "energy_scaled_raw_reciprocity": args.reciprocity_tolerance,
            "surface_position_error_over_gap": args.surface_tolerance,
            "linear_relative_residual": args.linear_tolerance, "encoding_energy": args.encoding_tolerance,
            "minimum_consecutive_refinements": 2, "minimum_heldout_gaps": args.minimum_heldout,
        },
        "reference_groups": list(summaries.values()), "qualification_failures": [],
    }
    if args.anchor_gap_over_c is None:
        result["qualified"] = all(s["qualified_reference"] for s in summaries.values())
        result["qualification_scope"] = "Reference matrices only; matching not requested"
        if not result["qualified"]:
            result["qualification_failures"].append("one_or_more_reference_groups_unqualified")
        return result
    if not args.operator_helper:
        raise ValueError("--operator-helper is required for constant-residual matching")
    helper = Path(args.operator_helper).resolve()
    anchor_gap = next((gap for gap in groups if abs(gap-args.anchor_gap_over_c) < 1e-13), None)
    if anchor_gap is None:
        raise ValueError("Requested anchor gap is absent from reference inputs")
    anchor_refs = groups[anchor_gap]
    anchor = anchor_refs[-1]
    anchor_summary = summaries[anchor_gap]
    anchor_operators = helper_export(helper, anchor_gap)
    residual = anchor.total - matrix(anchor_operators["isolated17"]) - matrix(anchor_operators["leading17"])
    anchor_deltas = [current.total-previous.total for previous, current in zip(anchor_refs[-3:-1], anchor_refs[-2:])]
    result["matching"] = {
        "anchor_gap_over_c": anchor_gap, "anchor_file": str(anchor.path),
        "anchor_qualified": anchor_summary["qualified_reference"],
        "formula": "Gmodel(h)=Gisolated(h)+Janoschek17(h)+[Gref(anchor)-Gisolated(anchor)-Janoschek17(anchor)]",
        "model_limitations": ["Janoschek Appendix A intentionally omits the K33 subleading normal-normal logarithm; the implemented paper-scope leading model is not an all-divergent-terms expansion", "finite terms are not fully recovered by a constant anchor residual", "the full17 strain/stresslet extension requires independent physical validation beyond the paper's local six-coordinate resistance", "no claim of validity beyond the explicitly compared pose and gaps"],
        "operator_helper": str(helper), "operator_helper_sha256": sha(helper),
        "model_domain_for_gate": args.model_domain, "queries": [],
    }
    heldout_count = 0
    model_pass = True
    for gap, group in sorted(groups.items()):
        reference = group[-1]
        summary = summaries[gap]
        query = {"h_over_c": gap, "gap_nm": 150*gap, "reference_qualified": summary["qualified_reference"]}
        is_anchor = abs(gap-anchor_gap) < 1e-13
        counts = not is_anchor and (args.model_domain == "all" or gap < anchor_gap)
        query.update(is_anchor_reconstruction=is_anchor, counts_for_model_gate=counts,
                     role="anchor_reconstruction_not_independent_test" if is_anchor else ("heldout_model_gap" if counts else "above_anchor_continuation_diagnostic_only"))
        operators = helper_export(helper, gap)
        prediction = symmetric(matrix(operators["isolated17"]) + matrix(operators["leading17"]) + residual)
        query["minimum_predicted_eigenvalue"] = float(np.linalg.eigvalsh(prediction)[0])
        query["predicted_positive"] = query["minimum_predicted_eigenvalue"] > 0
        if reference.report["minimum_symmetric_eigenvalue"] > 0:
            metrics = response_metrics(prediction, reference.total, gap)
            query["model_errors"] = metrics
            # Propagate observed anchor changes in the QUERY energy metric.
            # Summing anchor-relative percentages would use the wrong metric.
            propagated = max((energy_delta(d, reference.total) for d in anchor_deltas), default=None)
            query["estimated_anchor_change_in_query_energy_metric"] = propagated
            query_uncertainty = summary["estimated_energy_uncertainty"]
            if propagated is not None and query_uncertainty is not None:
                budget = metrics["full17_max_relative_energy_error"] + propagated + query_uncertainty
                query["estimated_total_energy_budget_used"] = budget
                block_budgets = {}
                for name in BLOCKS:
                    anchor_block = max(block_errors(d, reference.total)[name]["energy_input_scaled_operator_relative"] for d in anchor_deltas)
                    block_budgets[name] = metrics["blocks"][name]["energy_input_scaled_operator_relative"] + anchor_block + summary["estimated_block_uncertainty"][name]
                query["estimated_total_block_budgets_used"] = block_budgets
                error_pass = budget <= args.model_budget and all(v <= args.model_budget for v in block_budgets.values())
            else:
                query["estimated_total_energy_budget_used"] = None
                query["estimated_total_block_budgets_used"] = None
                error_pass = False
            query["budget_interpretation"] = "discrepancy plus observed reference/anchor refinement estimates; conservative accounting, not a rigorous bound"
        else:
            error_pass = False
            query["comparison_failure"] = "reference_not_SPD"
        encoding_pass = False
        if query["predicted_positive"]:
            try:
                encoded = helper_export(helper, gap, prediction)
                reconstructed = matrix(encoded["decoded_total17"])
                error = energy_delta(reconstructed-prediction, prediction)
                encoding_pass = encoded["encoded_values"] == 153 and error <= args.encoding_tolerance
                query["encoding"] = {"independent_values": encoded["encoded_values"], "maximum_relative_energy_error": error, "passed": encoding_pass,
                                     "scope": "one-node153-entry storage/restore only, not table interpolation"}
            except (RuntimeError, ValueError) as exc:
                query["encoding"] = {"passed": False, "failure": str(exc)}
        query["qualified_comparison"] = bool(counts and summary["qualified_reference"] and anchor_summary["qualified_reference"] and query["predicted_positive"] and encoding_pass and error_pass)
        if counts:
            heldout_count += 1
            model_pass &= query["qualified_comparison"]
        result["matching"]["queries"].append(query)
    result["matching"]["heldout_gap_count"] = heldout_count
    result["qualified"] = bool(anchor_summary["qualified_reference"] and heldout_count >= args.minimum_heldout and model_pass)
    if not anchor_summary["qualified_reference"]:
        result["qualification_failures"].append("matching_anchor_unqualified")
    if heldout_count < args.minimum_heldout:
        result["qualification_failures"].append("insufficient_heldout_gaps")
    if not model_pass:
        result["qualification_failures"].append("one_or_more_heldout_comparisons_failed_or_unqualified")
    result["qualification_scope"] = "Only explicitly compared held-out gaps in model_domain, using qualified independent reference groups"
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("references", nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--anchor-gap-over-c", type=float)
    parser.add_argument("--operator-helper")
    parser.add_argument("--reference-budget", type=float, default=.002)
    parser.add_argument("--model-budget", type=float, default=.01)
    parser.add_argument("--boundary-tolerance", type=float, default=.001)
    parser.add_argument("--surface-tolerance", type=float, default=.001)
    parser.add_argument("--reciprocity-tolerance", type=float, default=.001)
    parser.add_argument("--linear-tolerance", type=float, default=1e-8)
    parser.add_argument("--encoding-tolerance", type=float, default=1e-9)
    parser.add_argument("--minimum-heldout", type=int, default=1)
    parser.add_argument("--model-domain", choices=("below-anchor", "all"), default="below-anchor")
    args = parser.parse_args()
    positive = (args.reference_budget, args.model_budget, args.boundary_tolerance, args.surface_tolerance,
                args.reciprocity_tolerance, args.linear_tolerance, args.encoding_tolerance)
    if any(not np.isfinite(v) or v <= 0 for v in positive) or args.minimum_heldout < 1:
        parser.error("Tolerances must be finite and positive; minimum-heldout must be at least1")
    started = time.perf_counter()
    try:
        report = compare(args)
    except (ValueError, RuntimeError, KeyError, OSError, np.linalg.LinAlgError) as exc:
        report = {"schema": "independent_ar20_reference_and_matching_comparison_v1", "qualified": False, "failure": str(exc)}
    report["wall_seconds"] = time.perf_counter()-started
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"output": str(destination), "qualified": report["qualified"], "failure": report.get("failure"), "wall_seconds": report["wall_seconds"]}))
    return 0 if report["qualified"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
