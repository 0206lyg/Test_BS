#!/usr/bin/env python3
"""Compare step-0 responses at exactly matching initial geometry; no pass threshold."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite(value, label):
    value = float(value)
    require(math.isfinite(value), "Nonfinite "+label)
    return value


def norm(values):
    return math.sqrt(math.fsum(v*v for v in values))


def first_frame(path):
    with path.open() as f:
        require(f.readline().strip() == "ITEM: TIMESTEP", "Missing trajectory first frame")
        require(int(f.readline()) == 0, "First trajectory frame is not step 0")
        require(f.readline().strip() == "ITEM: NUMBER OF ATOMS", "Missing trajectory atom count")
        count = int(f.readline())
        box_header = f.readline().split()
        require(box_header[:3] == ["ITEM:", "BOX", "BOUNDS"], "Missing trajectory box")
        bounds = [[finite(v, "box bound") for v in f.readline().split()] for _ in range(3)]
        header = f.readline().split()
        require(header[:2] == ["ITEM:", "ATOMS"], "Missing trajectory atom fields")
        fields = header[2:]
        atoms = {}
        for _ in range(count):
            tokens = f.readline().split()
            require(len(tokens) == len(fields), "Incomplete trajectory first frame")
            values = [finite(v, "first-frame atom field") for v in tokens]
            row = dict(zip(fields, values))
            ident = int(row["id"])
            require(ident == row["id"] and ident not in atoms, "Invalid/duplicate atom ID")
            atoms[ident] = row
    return {"count": count, "box_header": box_header, "bounds": bounds,
            "fields": fields, "atoms": atoms}


def read_configuration(path):
    lines = [line.strip().split() for line in path.read_text().splitlines()
             if line.strip() and not line.lstrip().startswith("#")]
    fixes = [tokens for tokens in lines if tokens[:4] == ["fix", "hydro", "all", "slurry/bridge"]]
    require(len(fixes) == 1, "Expected one slurry/bridge fix")
    tokens = fixes[0]
    physics = {name: finite(value, name) for name, value in
               zip(("viscosity", "shear_rate", "gap_floor", "normal_stiffness"), tokens[4:8])}
    for name, count in (("contact", 3), ("spectral", 1), ("spectral_tolerance", 1)):
        require(name in tokens, "Missing physical option "+name)
        index = tokens.index(name)+1
        physics[name] = [finite(value, name) for value in tokens[index:index+count]]
        require(len(physics[name]) == count, "Incomplete physical option "+name)
    dt = [tokens for tokens in lines if tokens[0] == "timestep"]
    require(len(dt) == 1, "Expected one timestep command")
    physics["timestep"] = finite(dt[0][1], "timestep")
    deformation = [tokens for tokens in lines if tokens[:4] == ["fix", "shear", "all", "deform"]]
    require(len(deformation) == 1, "Expected one deform fix")
    physics["deformation_tokens"] = deformation[0]
    index = tokens.index("tolerance")+1 if "tolerance" in tokens else None
    # `tokens` still refers to the hydro line, since comprehensions have local scope.
    tolerance = finite(tokens[index], "solver tolerance") if index is not None else None
    iteration_index = tokens.index("iterations")+1 if "iterations" in tokens else None
    iterations = int(tokens[iteration_index]) if iteration_index is not None else None
    return physics, {"configured_solver_relative_tolerance": tolerance,
                     "configured_linear_iteration_limit": iterations}


def read_case(run):
    launch = json.loads((run/"diagnostic_launch.json").read_text())
    particle_hash = sha(run/"particles.data")
    input_hash = sha(run/"in.bridge")
    for name, expected in (("particles.data", particle_hash), ("in.bridge", input_hash)):
        observed = [value for path, value in launch["before_hashes"].items() if Path(path).name == name]
        require(observed == [expected], "Input changed since launch: "+name)
    if "after_hashes" in launch:
        require(launch["before_hashes"] == launch["after_hashes"], "Binary/input hashes changed during execution")
    with (run/"history.csv").open() as f:
        rows = [row for row in csv.DictReader(f) if row["step"] == "0"]
    require(len(rows) == 1, "Need exactly one completed step-0 history row")
    row = rows[0]
    physics, tolerances = read_configuration(run/"in.bridge")
    for key in ("solver_relative_residual", "force_residual", "mobility_residual", "newton_iterations",
                "linear_iterations", "profile_total_s", "profile_solve_s"):
        tolerances[key] = finite(row[key], "history "+key)
    return {"run": str(run), "particle_sha256": particle_hash, "input_sha256": input_hash,
            "frame": first_frame(run/"trajectory.lammpstrj"), "history": row,
            "physics": physics, "tolerances": tolerances,
            "provenance": {"recorded_launch_status": launch.get("status"),
                           "recorded_returncode": launch.get("returncode"),
                           "finalized_run": launch.get("status") == "completed",
                           "before_after_hash_verification_available": "after_hashes" in launch,
                           "binary_sha256_at_launch": {path: value for path, value in launch["before_hashes"].items()
                                                       if Path(path).name not in ("in.bridge", "particles.data")},
                           "trajectory_sha256_when_read": sha(run/"trajectory.lammpstrj"),
                           "history_sha256_when_read": sha(run/"history.csv")}}


def compare_vectors(first, second, fields, units):
    ids = sorted(first)
    a = [first[ident][field] for ident in ids for field in fields]
    b = [second[ident][field] for ident in ids for field in fields]
    delta = [y-x for x, y in zip(a, b)]
    an, bn, dn = norm(a), norm(b), norm(delta)
    where = max(range(len(delta)), key=lambda index: abs(delta[index]))
    per_particle = [norm([second[ident][field]-first[ident][field] for field in fields]) for ident in ids]
    return {"fields": fields, "units": units, "baseline_l2": an, "candidate_l2": bn,
            "difference_l2": dn, "relative_l2_to_baseline": dn/an if an else (0. if dn == 0. else None),
            "relative_l2_to_larger_response": dn/max(an, bn) if max(an, bn) else 0.,
            "zero_baseline": an == 0., "maximum_absolute_component_difference": abs(delta[where]),
            "maximum_absolute_component_location": {"atom_id": ids[where//len(fields)], "field": fields[where % len(fields)]},
            "maximum_per_particle_vector_difference": max(per_particle)}


def compare_scalar(a, b):
    a, b = finite(a, "baseline scalar"), finite(b, "candidate scalar")
    delta = b-a
    return {"baseline": a, "candidate": b, "signed_difference": delta, "absolute_difference": abs(delta),
            "relative_difference_to_baseline": abs(delta)/abs(a) if a else (0. if delta == 0. else None),
            "relative_difference_to_larger_response": abs(delta)/max(abs(a), abs(b)) if max(abs(a), abs(b)) else 0.}


def compare(baseline, candidate):
    require(baseline["particle_sha256"] == candidate["particle_sha256"], "Fixed particles.data SHA256 differs")
    require(baseline["physics"] == candidate["physics"], "Physical input parameters differ")
    a, b = baseline["frame"], candidate["frame"]
    require(a["count"] == b["count"] and set(a["atoms"]) == set(b["atoms"]), "Particle ID/count differs")
    require(a["box_header"] == b["box_header"] and a["bounds"] == b["bounds"], "Step-0 periodic box differs")
    geometry = ["type", "x", "y", "z"]+["c_shape[{}]".format(i) for i in range(1, 4)]
    qfields = ["c_shape[{}]".format(i) for i in range(4, 8)]
    for ident in a["atoms"]:
        aa, bb = a["atoms"][ident], b["atoms"][ident]
        require(all(aa[key] == bb[key] for key in geometry), "Step-0 geometry differs at atom "+str(ident))
        qa, qb = [aa[key] for key in qfields], [bb[key] for key in qfields]
        require(qa == qb or qa == [-v for v in qb], "Step-0 orientation differs at atom "+str(ident))
    groups = {"translational_velocity": (["vx", "vy", "vz"], "m/s"),
              "hydrodynamic_force_on_particle": (["f_hydro[{}]".format(i) for i in range(1, 4)], "N"),
              "hydrodynamic_torque_on_particle": (["f_hydro[{}]".format(i) for i in range(4, 7)], "N m"),
              "hydrodynamic_stresslet_on_fluid_STF5": (["f_hydro[{}]".format(i) for i in range(7, 12)], "N m"),
              "contact_force": (["f_hydro[{}]".format(i) for i in range(12, 15)], "N"),
              "contact_torque": (["f_hydro[{}]".format(i) for i in range(15, 18)], "N m"),
              "angular_momentum": (["c_shape[{}]".format(i) for i in range(8, 11)], "kg m^2/s")}
    vectors = {name: compare_vectors(a["atoms"], b["atoms"], fields, units)
               for name, (fields, units) in groups.items()}
    scalar_keys = ("eta_Pa_s", "eta_relative", "sigma_solvent_Pa", "sigma_hydro_Pa", "sigma_contact_Pa", "sigma_xz_Pa",
                   "hydro_dissipation_W", "contact_power_W", "stress_power_W", "power_balance_error_W")
    scalars = {key: compare_scalar(baseline["history"][key], candidate["history"][key]) for key in scalar_keys}
    power_balance = {}
    for name, case in (("baseline", baseline), ("candidate", candidate)):
        scale = max(abs(finite(case["history"][key], key)) for key in
                    ("hydro_dissipation_W", "contact_power_W", "stress_power_W"))
        error = finite(case["history"]["power_balance_error_W"], "power balance")
        power_balance[name] = {"error_W": error, "physical_power_scale_W": scale,
                               "relative_error_to_physical_power": abs(error)/max(scale, 1e-300)}
    return {"status": "comparison_available", "step": 0, "particle_count": a["count"],
            "fixed_particle_sha256": baseline["particle_sha256"], "exact_box_and_geometry_match": True,
            "physical_input_parameters": baseline["physics"], "vector_comparisons": vectors,
            "scalar_comparisons": scalars, "power_balance_diagnostics": power_balance,
            "power_balance_ratio_note": "Relative change of an already tiny balance error is ill-conditioned near zero; use its absolute error or error normalized by physical power",
            "baseline": {key: baseline[key] for key in ("run", "input_sha256", "tolerances", "provenance")},
            "candidate": {key: candidate[key] for key in ("run", "input_sha256", "tolerances", "provenance")},
            "acceptance_threshold": None, "acceptance_decision": None,
            "scope": "Same-state step-0 numerical response comparison; configured solver tolerances are context, not an error bound on observables",
            "steady_state_claim": False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    code = 0
    try:
        result = compare(read_case(args.baseline.resolve()), read_case(args.candidate.resolve()))
    except (OSError, ValueError, KeyError, IndexError) as error:
        result = {"status": "comparison_unavailable", "reason": str(error),
                  "baseline": str(args.baseline.resolve()), "candidate": str(args.candidate.resolve()),
                  "acceptance_decision": None}
        code = 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+"\n")
    print(json.dumps({"output": str(args.output.resolve()), "status": result["status"],
                      "reason": result.get("reason"), "acceptance_decision": None}), flush=True)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
