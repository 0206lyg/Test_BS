#!/usr/bin/env python3
"""Summarize the actual, matched-packing LAMMPS comparison (not steady rheology)."""
from pathlib import Path
import csv
import hashlib
import json
import statistics
import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
LOCAL = REPO.parent / "slurry_local_test"
NAMES = ("ar20_base200", "ar20_sliding200", "ar20_strong200",
         "ar20_phi30_base", "ar20_phi30_strong")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def order(path):
    section = path.read_text().split("Ellipsoids")[1].split("Velocities")[0]
    normals = []
    for line in section.splitlines():
        values = line.split()
        if len(values) != 8:
            continue
        w, x, y, z = map(float, values[4:])
        normals.append((2*(x*z+w*y), 2*(y*z-w*x), 1-2*(x*x+y*y)))
    n = np.asarray(normals)
    tensor = n.T @ n / len(n)
    values, vectors = np.linalg.eigh(tensor)
    return {"nematic_S": float((3*values[-1]-1)/2),
            "director": vectors[:, -1].tolist(), "normal_second_moment": tensor.tolist()}


def power_defect(row):
    denominator = max(abs(float(row["stress_power_W"])),
                      abs(float(row["hydro_dissipation_W"])) + abs(float(row["contact_power_W"])),
                      1e-300)
    return abs(float(row["power_balance_error_W"])) / denominator


report = {
    "comparison": "Same production random-growth packing at each phi; no attraction. Base, sliding-only, sliding+strong-rolling.",
    "lammps_commit": "8de817dd79bfe4525d5d39246a212d833e6dee07",
    "lammps_release": "stable_30Sep2026",
    "build": {"optimization": "-O1", "MPI": "serial stubs", "mobility": "mesh, builtin FFT",
              "preconditioner": "Eigen reference backend", "threads_phi24": 2, "threads_phi30": 1},
    "limits": ["Short initial transient; no steady-state or jamming-fraction determination.",
               "N108 cubic boxes are small (L/a approximately 4.2-4.6).",
               "Random-growth initialization can develop strong nematic order during packing.",
               "No runtime claim for Negishi FFTW/MKL or MPI production builds."],
    "runs": {}, "comparisons": {}, "binary_sha256": {}, "source_sha256": {}
}
report["release_at_summary"] = json.loads((REPO / "release.json").read_text())
report["prepare_tool_sha256_at_summary"] = sha(REPO / "lammps_slurry/tools/prepare_bridge.py")
report["input_length_policy"] = "All cases use explicit physical rolling/twisting lengths; equivalent to the new particle/roughness modes at AR20."
for name, path in (("lammps", LOCAL / "lammps_pinned_build/lmp"),
                   ("liblammps", LOCAL / "lammps_pinned_build/liblammps.so"),
                   ("plugin", LOCAL / "pinned_plugin/build/libslurry.so")):
    report["binary_sha256"][name] = sha(path)
for path in sorted((REPO / "lammps_slurry/src").glob("*")):
    if path.suffix in (".cpp", ".h"):
        report["source_sha256"][str(path.relative_to(REPO))] = sha(path)

all_rows = {}
for name in NAMES:
    folder = ROOT / name
    parameters = json.loads((folder / "parameters.json").read_text())
    rows = list(csv.DictReader((folder / "history.csv").open()))
    console = (folder / "console.log").read_text()
    complete = int(rows[-1]["step"]) == parameters["steps"] and "ERROR:" not in console
    take = lambda key: [float(row[key]) for row in rows]
    tail = rows[len(rows)//2:]
    tail50 = rows[-50:]
    record = {
        "complete": complete, "recorded_steps": int(rows[-1]["step"]),
        "recorded_strain": int(rows[-1]["step"])*parameters["timestep"]*abs(parameters["shear_rate"]),
        "phi": parameters["phi"], "AR": parameters["ar"], "count": parameters["count"],
        "box_length_over_a": parameters["box_length_m"] / parameters["a_m"],
        "friction": parameters["friction"], "rotation_contact": parameters["rotation_contact"],
        "kn_N_per_m": parameters["normal_stiffness"], "kt_N_per_m": parameters["tangential_stiffness"],
        "timestep_s": parameters["timestep"], "shear_rate_s_inv": parameters["shear_rate"],
        "initial_order": order(folder / "particles.data"),
        "eta_relative_initial": take("eta_relative")[0], "eta_relative_final": take("eta_relative")[-1],
        "eta_relative_mean_last_half": statistics.mean(float(row["eta_relative"]) for row in tail),
        "last_half_step_range": [int(tail[0]["step"]), int(tail[-1]["step"])],
        "eta_relative_mean_last_50_or_available": statistics.mean(float(row["eta_relative"]) for row in tail50),
        "last_50_or_available_step_range": [int(tail50[0]["step"]), int(tail50[-1]["step"])],
        "last_50_or_available_count": len(tail50),
        "contacts_initial": take("contacts")[0], "contacts_final": take("contacts")[-1],
        "contacts_maximum": max(take("contacts")), "rolling_yield_final": take("rolling_yield_contacts")[-1],
        "contact_degree_final": 2*take("contacts")[-1]/parameters["count"],
        "contacts_mean_last_50_or_available": statistics.mean(float(row["contacts"]) for row in tail50),
        "contact_degree_mean_last_50_or_available": 2*statistics.mean(float(row["contacts"]) for row in tail50)/parameters["count"],
        "twisting_yield_final": take("twisting_yield_contacts")[-1],
        "solver_relative_residual_maximum": max(take("solver_relative_residual")),
        "relative_power_defect_maximum": max(map(power_defect, rows)),
        "minimum_gap_nm": 1e9*min(take("min_gap_m")),
        "profile_total_s_mean": statistics.mean(take("profile_total_s")),
        "artifact_sha256": {file.name: sha(file) for file in sorted(folder.iterdir())
                            if file.is_file() and file.name in ("particles.data", "in.bridge", "parameters.json",
                                "history.csv", "console.log", "final.data", "packing_clearance.json")}
    }
    if (folder / "final.data").is_file():
        record["final_order"] = order(folder / "final.data")
    report["runs"][name] = record
    all_rows[name] = rows

for base, other in (("ar20_base200", "ar20_sliding200"), ("ar20_base200", "ar20_strong200"),
                    ("ar20_sliding200", "ar20_strong200"), ("ar20_phi30_base", "ar20_phi30_strong")):
    assert sha(ROOT/base/"particles.data") == sha(ROOT/other/"particles.data")
    a, b = report["runs"][base], report["runs"][other]
    indexed_a = {int(row["step"]): row for row in all_rows[base]}
    indexed_b = {int(row["step"]): row for row in all_rows[other]}
    shared = sorted(set(indexed_a) & set(indexed_b))
    tail = shared[len(shared)//2:]
    tail50 = shared[-50:]
    last = shared[-1]
    report["comparisons"][base+"__"+other] = {
        "matched_initial_particles_sha256": sha(ROOT/base/"particles.data"),
        "matched_last_step": last,
        "final_relative_viscosity_ratio": float(indexed_b[last]["eta_relative"]) / float(indexed_a[last]["eta_relative"]),
        "last_half_mean_relative_viscosity_ratio": statistics.mean(float(indexed_b[k]["eta_relative"]) for k in tail)
            / statistics.mean(float(indexed_a[k]["eta_relative"]) for k in tail),
        "last_50_or_available_mean_relative_viscosity_ratio": statistics.mean(float(indexed_b[k]["eta_relative"]) for k in tail50)
            / statistics.mean(float(indexed_a[k]["eta_relative"]) for k in tail50),
    }
report["all_complete"] = all(run["complete"] for run in report["runs"].values())
(ROOT / "runtime_validation.json").write_text(json.dumps(report, indent=2, sort_keys=True)+"\n")
print(json.dumps({"all_complete": report["all_complete"], "comparisons": report["comparisons"]}, indent=2))
