#!/usr/bin/env python3
"""Reproduce/analyze one fixed moving N108 AR20 periodic suspension check.

The three cases share byte-identical initial geometry: 20 dt steps, 40 dt/2
steps, and a run-0 response with doubled middle spectral resolution. This is
an integration, finite-resolution and time-step check; it is not a steady-state
viscosity result or a comparison against an exact Stokes reference.

Examples:
  python3 validation_bridge/run_bulk_check.py --analyze-existing /path/bridge-runs --output result.json
  python3 validation_bridge/run_bulk_check.py --run /new/path --lammps /path/lmp --plugin /path/libslurry.so
Only the Python standard library is required.
"""
import argparse
import csv
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = Path(__file__).resolve().parent/"fixtures/random108_ar20.data"
CASES = ("random108", "random108_halfdt", "random108_spectral4")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def norm(values):
    return math.sqrt(math.fsum(x*x for x in values))


def relative(delta, reference):
    denominator = norm(reference)
    return norm(delta)/denominator if denominator > 0 else None


def source_identity():
    files = list((ROOT/"lammps_slurry/src").glob("*"))
    files += list((ROOT/"lammps_slurry/vendor/gr_re2").glob("*.h"))
    files += [ROOT/"lammps_slurry/CMakeLists.txt", Path(__file__).resolve()]
    hashes = {str(p.relative_to(ROOT)): sha256(p) for p in sorted(files) if p.is_file()}
    canonical = json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()
    return {"observed_source_sha256": hashlib.sha256(canonical).hexdigest(), "files": hashes,
            "scope": "Observed source identity; executable compilation provenance requires the build manifest."}


def read_input(path):
    text = Path(path).read_text()
    def scalar(pattern):
        found = re.search(pattern, text, re.MULTILINE)
        if not found:
            raise ValueError("Input missing required value: "+pattern)
        return float(found.group(1))
    return {"timestep_s": scalar(r"^timestep\s+(\S+)"),
            "steps": int(scalar(r"^run\s+(\d+)")),
            "shear_rate_per_s": scalar(r"^fix\s+shear\s+all\s+deform\s+0\s+xz\s+erate\s+(\S+)"),
            "spectral": scalar(r"^fix\s+hydro\s+all\s+slurry/bridge\s+.*\bspectral\s+(\S+)")}


def read_history(path):
    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        rows = []
        for raw in reader:
            row = {k: float(v) for k, v in raw.items() if k and v not in (None, "")}
            # Initial averaging columns legitimately contain NaN before a step
            # is integrated. They are omitted from the strict JSON output.
            if not all(math.isfinite(row[k]) for k in ("step", "time_s", "strain", "sigma_xz_Pa", "eta_Pa_s")):
                raise ValueError("Nonfinite physical output in "+str(path))
            rows.append(row)
    if not rows:
        raise ValueError("No history rows in "+str(path))
    if len({int(r["step"]) for r in rows}) != len(rows):
        raise ValueError("Duplicate history timestep")
    return rows


def read_dump(path):
    frames = []
    with Path(path).open() as stream:
        while True:
            line = stream.readline()
            if not line:
                break
            if line.rstrip() != "ITEM: TIMESTEP":
                raise ValueError("Malformed dump timestep header")
            step = int(stream.readline())
            if stream.readline().rstrip() != "ITEM: NUMBER OF ATOMS":
                raise ValueError("Malformed dump atom-count header")
            count = int(stream.readline())
            header = stream.readline().split()
            if header[:3] != ["ITEM:", "BOX", "BOUNDS"]:
                raise ValueError("Malformed dump box header")
            bounds = [list(map(float, stream.readline().split())) for _ in range(3)]
            if "xy" in header:
                xy, xz, yz = bounds[0][2], bounds[1][2], bounds[2][2]
                xlo = bounds[0][0]-min(0., xy, xz, xy+xz)
                xhi = bounds[0][1]-max(0., xy, xz, xy+xz)
                ylo = bounds[1][0]-min(0., yz)
                yhi = bounds[1][1]-max(0., yz)
            else:
                xy = xz = yz = 0.
                xlo, xhi = bounds[0][:2]
                ylo, yhi = bounds[1][:2]
            zlo, zhi = bounds[2][:2]
            cell = [[xhi-xlo, xy, xz], [0., yhi-ylo, yz], [0., 0., zhi-zlo]]
            labels = stream.readline().split()
            if labels[:2] != ["ITEM:", "ATOMS"]:
                raise ValueError("Malformed dump particle header")
            labels = labels[2:]
            atoms = {}
            for _ in range(count):
                values = list(map(float, stream.readline().split()))
                if len(values) != len(labels) or not all(math.isfinite(v) for v in values):
                    raise ValueError("Incomplete/nonfinite dump particle record")
                atom = dict(zip(labels, values))
                tag = int(atom["id"])
                if tag in atoms:
                    raise ValueError("Duplicate particle ID")
                atoms[tag] = atom
            frames.append({"step": step, "cell_m": cell, "origin_m": [xlo, ylo, zlo], "atoms": atoms})
    if not frames:
        raise ValueError("No complete trajectory frames in "+str(path))
    return frames


def periodic_difference(first, second, cell):
    d = [a-b for a, b in zip(first, second)]
    sz = d[2]/cell[2][2]
    sy = (d[1]-cell[1][2]*sz)/cell[1][1]
    sx = (d[0]-cell[0][1]*sy-cell[0][2]*sz)/cell[0][0]
    center = [round(s) for s in (sx, sy, sz)]
    candidates = []
    for delta in itertools.product((-1, 0, 1), repeat=3):
        image = [center[k]+delta[k] for k in range(3)]
        r = [d[k]-sum(cell[k][j]*image[j] for j in range(3)) for k in range(3)]
        candidates.append(r)
    return min(candidates, key=norm)


def quaternion_distance(a, b):
    a, b = [v/norm(a) for v in a], [v/norm(b) for v in b]
    sign = 1. if math.fsum(x*y for x, y in zip(a, b)) >= 0 else -1.
    difference = [x-sign*y for x, y in zip(a, b)]
    addition = [x+sign*y for x, y in zip(a, b)]
    return 4*math.atan2(norm(difference), norm(addition))


def group(frame, indices):
    return [frame["atoms"][tag]["f_hydro[{}]".format(k)] for tag in sorted(frame["atoms"]) for k in indices]


def compare_frames(first, second):
    if set(first["atoms"]) != set(second["atoms"]):
        raise ValueError("Particle IDs differ between trajectories")
    if max(abs(x-y) for a, b in zip(first["cell_m"], second["cell_m"]) for x, y in zip(a, b)) > 1e-12:
        raise ValueError("Endpoint simulation boxes differ; trajectories must have equal physical duration")
    positions, angles = [], []
    for tag in sorted(first["atoms"]):
        a, b = first["atoms"][tag], second["atoms"][tag]
        positions.append(norm(periodic_difference([a[k] for k in "xyz"], [b[k] for k in "xyz"], second["cell_m"])))
        angles.append(quaternion_distance([a["c_shape[{}]".format(k)] for k in range(4, 8)],
                                          [b["c_shape[{}]".format(k)] for k in range(4, 8)]))
    result = {"position_max_difference_m": max(positions), "position_rms_difference_m": norm(positions)/math.sqrt(len(positions)),
              "orientation_max_difference_rad": max(angles), "orientation_rms_difference_rad": norm(angles)/math.sqrt(len(angles))}
    for name, indices in (("hydro_force", range(1, 4)), ("hydro_torque", range(4, 7)), ("stresslet", range(7, 12)),
                          ("contact_force", range(12, 15)), ("contact_torque", range(15, 18))):
        a, b = group(first, indices), group(second, indices)
        result[name+"_relative_L2_difference"] = relative([x-y for x, y in zip(a, b)], b)
    return result


def dump_consistency(frames, history, shear):
    hist = {int(row["step"]): row for row in history}
    max_force, max_torque, max_sigma_error, max_power_error, max_q_error = 0., 0., 0., 0., 0.
    max_force_relative, max_torque_relative = 0., 0.
    for frame in frames:
        row = hist.get(frame["step"])
        if row is None:
            continue
        volume = math.prod(frame["cell_m"][i][i] for i in range(3))
        # STF5 xz coefficient is f_hydro[10]; Cauchy stress is minus on-fluid S/V.
        sigma_h = -math.fsum(atom["f_hydro[10]"] for atom in frame["atoms"].values())/(math.sqrt(2)*volume)
        max_sigma_error = max(max_sigma_error, abs(sigma_h-row["sigma_hydro_Pa"]))
        exact_power = row["sigma_xz_Pa"]*shear*volume
        max_power_error = max(max_power_error, abs(exact_power-row["stress_power_W"]))
        fh, fc = group(frame, range(1, 4)), group(frame, range(12, 15))
        th, tc = group(frame, range(4, 7)), group(frame, range(15, 18))
        f_residual = [a+b for a, b in zip(fh, fc)]
        t_residual = [a+b for a, b in zip(th, tc)]
        max_force = max(max_force, norm(f_residual))
        max_torque = max(max_torque, norm(t_residual))
        fscale, tscale = max(norm(fh), norm(fc)), max(norm(th), norm(tc))
        if fscale > 0:
            max_force_relative = max(max_force_relative, norm(f_residual)/fscale)
        if tscale > 0:
            max_torque_relative = max(max_torque_relative, norm(t_residual)/tscale)
        for atom in frame["atoms"].values():
            max_q_error = max(max_q_error, abs(math.fsum(atom["c_shape[{}]".format(k)]**2 for k in range(4, 8))-1.))
    return {"max_force_balance_L2_N": max_force, "max_torque_balance_L2_N_m": max_torque,
            "max_relative_force_balance_L2": max_force_relative, "max_relative_torque_balance_L2": max_torque_relative,
            "max_stresslet_to_hydro_sigma_error_Pa": max_sigma_error,
            "max_sigma_gamma_volume_to_power_error_W": max_power_error, "max_quaternion_squared_norm_error": max_q_error}


def case_report(directory):
    directory = Path(directory)
    settings = read_input(directory/"in.bridge")
    rows = read_history(directory/"history.csv")
    frames = read_dump(directory/"trajectory.lammpstrj")
    if int(rows[-1]["step"]) != settings["steps"] or frames[-1]["step"] != settings["steps"] or not (directory/"final.data").is_file():
        raise ValueError("Run is incomplete: {} (expected step {}, history {}, dump {})".format(directory, settings["steps"], rows[-1]["step"], frames[-1]["step"]))
    if not math.isclose(rows[-1]["time_s"], settings["steps"]*settings["timestep_s"], rel_tol=1e-12, abs_tol=1e-20):
        raise ValueError("Run time does not agree with timestep/steps")
    intervals = [r for r in rows if r["interval_steps"] > 0]
    duration = math.fsum(r["interval_end_time_s"]-r["interval_start_time_s"] for r in intervals)
    average = None
    if intervals:
        average = {name: math.fsum((r["interval_end_time_s"]-r["interval_start_time_s"])*r["interval_mean_"+name] for r in intervals)/duration
                   for name in ("eta_Pa_s", "eta_relative", "sigma_solvent_Pa", "sigma_hydro_Pa", "sigma_contact_Pa", "sigma_xz_Pa")}
    power_relative = [abs(r["power_balance_error_W"])/max(abs(r["stress_power_W"]), abs(r["hydro_dissipation_W"]), abs(r["contact_power_W"]), 1e-300) for r in rows]
    initial, final = rows[0], rows[-1]
    costs = [r["profile_total_s"] for r in rows[1:]]
    cell0 = frames[0]["cell_m"]
    box_volume = math.prod(cell0[k][k] for k in range(3))
    first_atoms = frames[0]["atoms"]
    particle_volume = math.fsum(math.pi/6*math.prod(atom["c_shape[{}]".format(k)] for k in range(1, 4)) for atom in first_atoms.values())
    aspect_ratios = [atom["c_shape[1]"]/atom["c_shape[3]"] for atom in first_atoms.values()]
    report = {"path": str(directory.resolve()), "settings": settings, "particle_count": len(first_atoms),
              "volume_fraction": particle_volume/box_volume, "aspect_ratio_min": min(aspect_ratios), "aspect_ratio_max": max(aspect_ratios),
              "initial_fixture_sha256": sha256(directory/"particles.data"),
              "input_sha256": sha256(directory/"in.bridge"), "history_sha256": sha256(directory/"history.csv"),
              "trajectory_sha256": sha256(directory/"trajectory.lammpstrj"),
              "initial_eta_Pa_s": initial["eta_Pa_s"], "final_eta_Pa_s": final["eta_Pa_s"],
              "final_time_s": final["time_s"], "final_strain": final["strain"], "selected_window_average": average,
              "max_relative_power_balance_error": max(power_relative),
              "max_force_residual_nondimensional_raw": max(r["force_residual"] for r in rows),
              "max_mobility_residual_nondimensional_raw": max(r["mobility_residual"] for r in rows),
              "minimum_hydrodynamic_dissipation_W": min(r["hydro_dissipation_W"] for r in rows),
              "max_overlap_m": max(r["max_overlap_m"] for r in rows),
              "max_contacts": max(int(r["contacts"]) for r in rows), "max_near_pairs": max(int(r["near_pairs"]) for r in rows),
              "reciprocal_modes_initial": int(initial["reciprocal_modes"]),
              "profile_initial_solution_seconds": initial["profile_total_s"],
              "profile_post_step_mean_seconds": statistics.mean(costs) if costs else None,
              "profile_post_step_median_seconds": statistics.median(costs) if costs else None,
              "profile_all_solution_seconds": math.fsum(r["profile_total_s"] for r in rows),
              "timing_scope": "Measured geometry + middle construction + coupled solve. Includes setup and final pending solution; not packing, MPI launch or output I/O.",
              "independent_dump_checks": dump_consistency(frames, rows, settings["shear_rate_per_s"])}
    return report, frames, rows


def execution_identity(directory, report, observed):
    """Distinguish recorded run binaries from files merely present today."""
    path = Path(directory)/"launch.json"
    if not path.is_file():
        return {"status": "missing_launch_record", "execution_binary_verified": False,
                "reason": "No per-case launch identity is present. A current plugin hash does not identify the binary that produced these files."}
    record = json.loads(path.read_text())
    if record.get("execution_provenance_schema") != 1:
        return {"status": "unsupported_launch_record", "execution_binary_verified": False,
                "launch_record_sha256": sha256(path),
                "reason": "This record does not bind the completed run's binaries and result artifacts."}
    if record.get("returncode") != 0 or record.get("binary_stable_during_run") is not True:
        raise ValueError("Launch record is failed, incomplete or reports changed binaries: "+str(path))
    expected_artifacts = record.get("artifact_sha256", {})
    for name in ("particles.data", "in.bridge", "history.csv", "trajectory.lammpstrj", "final.data"):
        if expected_artifacts.get(name) != sha256(Path(directory)/name):
            raise ValueError("Result artifact differs from recorded run: "+str(Path(directory)/name))
    binaries = {}
    for key in ("plugin", "lammps"):
        before, after = record.get(key+"_sha256"), record.get("finished_"+key+"_sha256")
        if not isinstance(before, str) or len(before) != 64 or after != before:
            raise ValueError("Incomplete/mismatched run binary hashes in "+str(path))
        binaries[key] = {"sha256": before,
                         "matches_currently_observed_binary": observed.get(key, {}).get("sha256") == before if key in observed else None}
    return {"status": "completed_run_identity_and_artifacts_verified", "execution_binary_verified": True,
            "launch_record_sha256": sha256(path), "binaries": binaries,
            "openmp_threads": record.get("openmp_threads"),
            "compiled_source_identity_verified": False,
            "scope": "Pre/post-run binary hashes and output artifact hashes match. Source compilation provenance remains a separate build-manifest check."}


def analyze(base, plugin=None, lammps=None, threads=8):
    cases = {name: case_report(Path(base)/name) for name in CASES}
    reports = {name: value[0] for name, value in cases.items()}
    hashes = {r["initial_fixture_sha256"] for r in reports.values()}
    if len(hashes) != 1:
        raise ValueError("Cases do not share byte-identical initial particles")
    coarse, fine, spectral = (reports[name] for name in CASES)
    if coarse["settings"]["steps"] != 20 or fine["settings"]["steps"] != 40:
        raise ValueError("This fixed bulk audit requires 20 coarse steps and 40 refined steps")
    if not math.isclose(coarse["final_time_s"], fine["final_time_s"], rel_tol=1e-12):
        raise ValueError("Coarse and refined runs have different physical duration")
    if not math.isclose(coarse["settings"]["timestep_s"], 2*fine["settings"]["timestep_s"], rel_tol=1e-12):
        raise ValueError("Refined run must use half the coarse timestep")
    if spectral["settings"]["steps"] != 0 or not math.isclose(coarse["settings"]["timestep_s"], spectral["settings"]["timestep_s"], rel_tol=1e-12):
        raise ValueError("Spectral comparison requires run 0 at the same contact timestep")
    if len({r["settings"]["shear_rate_per_s"] for r in reports.values()}) != 1:
        raise ValueError("Compared cases use different shear rates")
    if not math.isclose(coarse["settings"]["spectral"], fine["settings"]["spectral"], rel_tol=1e-12):
        raise ValueError("Timestep comparison changed spectral resolution")
    dt_comparison = compare_frames(cases[CASES[0]][1][-1], cases[CASES[1]][1][-1])
    dt_comparison["endpoint_eta_relative_difference"] = abs(coarse["final_eta_Pa_s"]-fine["final_eta_Pa_s"])/abs(fine["final_eta_Pa_s"])
    dt_comparison["window_average_eta_relative_difference"] = abs(coarse["selected_window_average"]["eta_Pa_s"]-fine["selected_window_average"]["eta_Pa_s"])/abs(fine["selected_window_average"]["eta_Pa_s"])
    spectral_comparison = compare_frames(cases[CASES[0]][1][0], cases[CASES[2]][1][0])
    spectral_comparison["initial_eta_relative_difference"] = abs(coarse["initial_eta_Pa_s"]-spectral["initial_eta_Pa_s"])/abs(spectral["initial_eta_Pa_s"])
    provenance = {"source": source_identity(), "declared_openmp_threads": threads, "thread_count_scope": "Caller-specified for existing runs; set in launch environment by --run."}
    for key, path in (("plugin", plugin), ("lammps", lammps)):
        if path is not None:
            provenance[key] = {"path": str(Path(path).resolve()), "sha256": sha256(path), "scope": "Binary observed at analysis; compare run launch identity for execution provenance."}
    execution = {name: execution_identity(Path(base)/name, reports[name], provenance) for name in CASES}
    verified = [value for value in execution.values() if value["execution_binary_verified"]]
    if len(verified) == len(CASES):
        for key in ("plugin", "lammps"):
            if len({value["binaries"][key]["sha256"] for value in verified}) != 1:
                raise ValueError("Cases used different recorded "+key+" binaries")
    provenance["case_execution_identity"] = execution
    provenance["all_execution_binary_identities_verified"] = len(verified) == len(CASES)
    provenance["currently_observed_hashes_are_execution_evidence"] = False
    report = {"schema": 1, "case": "one_fixed_initial_suspension", "provenance": provenance,
              "cases": reports, "timestep_comparison": dt_comparison, "middle_resolution_comparison": spectral_comparison,
              "stationarity_certified": False, "accuracy_against_exact_Stokes_claimed": False,
              "interpretation": ["Endpoint and average differences compare two finite-resolution runs, not exact-reference errors.",
                                 "A 20 microsecond trajectory at 100/s reaches strain 0.002 and does not measure steady-state viscosity.",
                                 "Contact power may be negative while pre-existing elastic compression relaxes; hydrodynamic dissipation must remain nonnegative.",
                                 "Timing includes the reported simultaneous force/contact solve, not the older prescribed-velocity near-kernel-only benchmark."]}
    # No implicit replacement of nonfinite data: a physical nonfinite metric
    # must fail the audit rather than silently serialize NaN as JSON.
    json.dumps(report, allow_nan=False)
    return report


def input_text(dt, steps, spectral):
    return '''plugin load ${slurry_plugin}
units si
dimension 3
boundary p p p
atom_style ellipsoid
atom_modify map array
read_data particles.data
comm_modify vel yes
pair_style zero 1.5e-7
pair_coeff * *
neighbor 3e-8 bin
neigh_modify every 1 delay 0 check yes
timestep DT
fix shear all deform 0 xz erate 100 remap v units box
fix hydro all slurry/bridge .01 100 1e-9 1 contact 3e-9 .1 80 spectral SPECTRAL spectral_tolerance 1e-8 tolerance 1e-6 iterations 1000 history history.csv 1 0
compute shape all property/atom shapex shapey shapez quatw quati quatj quatk angmomx angmomy angmomz
dump trajectory all custom 1 trajectory.lammpstrj id type x y z vx vy vz c_shape[*] f_hydro[*]
dump_modify trajectory sort id format float %.17g
thermo 1
thermo_style custom step time atoms f_hydro[1] f_hydro[2] f_hydro[3] f_hydro[4]
thermo_modify flush yes lost error
run STEPS
write_data final.data nocoeff
'''.replace("DT", "{:.17g}".format(dt)).replace("SPECTRAL", "{:.17g}".format(spectral)).replace("STEPS", str(steps))


def run_cases(base, fixture, lammps, plugin, threads=8, mpi=False):
    base, fixture = Path(base).resolve(), Path(fixture).resolve()
    if base.exists():
        raise ValueError("Validation output already exists; refusing to overwrite: "+str(base))
    if any(c in str(plugin) for c in ('"', '\n', '\r')):
        raise ValueError("Plugin path cannot contain a quote or line break")
    launcher = (shutil.which("mpirun") or shutil.which("mpiexec")) if mpi else None
    if mpi and launcher is None:
        raise ValueError("--mpi requires mpirun or mpiexec in PATH")
    identity = {"mpi": mpi, "mpi_launcher": launcher, "openmp_threads": threads, "fixture_sha256": sha256(fixture), "plugin_sha256": sha256(plugin), "lammps_sha256": sha256(lammps), "source": source_identity()}
    base.mkdir(parents=True)
    (base/"launch_identity.json").write_text(json.dumps(identity, indent=2, sort_keys=True)+"\n")
    for name, dt, steps, spectral in ((CASES[0], 1e-6, 20, 2), (CASES[1], .5e-6, 40, 2), (CASES[2], 1e-6, 0, 4)):
        case = base/name
        case.mkdir()
        shutil.copyfile(fixture, case/"particles.data")
        (case/"in.bridge").write_text(input_text(dt, steps, spectral))
        command = [str(Path(lammps).resolve()), "-var", "slurry_plugin", '"'+str(Path(plugin).resolve())+'"', "-in", "in.bridge"]
        if launcher:
            command = [launcher, "-np", "1"]+command
        begun = time.monotonic()
        run_identity = {"execution_provenance_schema": 1, "command": command, "returncode": None,
                        "binary_stable_during_run": False, **identity}
        (case/"launch.json").write_text(json.dumps(run_identity, indent=2, sort_keys=True)+"\n")
        with (case/"console.log").open("w") as stream:
            environment = dict(os.environ, OMP_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS="1")
            process = subprocess.run(command, cwd=str(case), env=environment, stdout=stream, stderr=subprocess.STDOUT)
        finished_plugin, finished_lammps = sha256(plugin), sha256(lammps)
        stable = finished_plugin == identity["plugin_sha256"] and finished_lammps == identity["lammps_sha256"]
        artifacts = {name: sha256(case/name) for name in ("particles.data", "in.bridge", "history.csv", "trajectory.lammpstrj", "final.data") if (case/name).is_file()}
        run_identity.update({"returncode": process.returncode, "elapsed_wall_seconds": time.monotonic()-begun,
                             "binary_stable_during_run": stable, "finished_plugin_sha256": finished_plugin,
                             "finished_lammps_sha256": finished_lammps, "artifact_sha256": artifacts})
        (case/"launch.json").write_text(json.dumps(run_identity, indent=2, sort_keys=True)+"\n")
        if process.returncode:
            raise RuntimeError("LAMMPS case failed: {}; inspect console.log".format(case))
        if not stable:
            raise RuntimeError("Executable changed during validation; results are not accepted")
    return base


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--analyze-existing", type=Path)
    mode.add_argument("--run", type=Path, help="new output directory; runs the three fixed cases")
    parser.add_argument("--fixture", type=Path, default=FIXTURE)
    parser.add_argument("--lammps", type=Path)
    parser.add_argument("--plugin", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--mpi", action="store_true", help="launch each LAMMPS run through mpirun/mpiexec -np 1")
    parser.add_argument("--threads", type=int, default=8, help="OpenMP threads for --run; declared setting for existing runs")
    parser.add_argument("--states-output", type=Path, help="optional compact first/last per-particle state JSON")
    args = parser.parse_args()
    try:
        if args.threads < 1:
            parser.error("--threads must be positive")
        if args.run:
            if args.lammps is None or args.plugin is None:
                parser.error("--run requires --lammps and --plugin")
            base = run_cases(args.run, args.fixture, args.lammps, args.plugin, args.threads, args.mpi)
        else:
            base = args.analyze_existing
        report = analyze(base, args.plugin, args.lammps, args.threads)
        output = args.output or Path(base)/"bulk_check.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False)+"\n")
        if args.states_output:
            states = {}
            for name in CASES:
                frames = read_dump(Path(base)/name/"trajectory.lammpstrj")
                states[name] = {"first": frames[0], "last": frames[-1]}
            args.states_output.parent.mkdir(parents=True, exist_ok=True)
            args.states_output.write_text(json.dumps(states, sort_keys=True, separators=(",", ":"), allow_nan=False)+"\n")
        print("Wrote {}. Moving integration and resolution check; no steady-state claim.".format(output))
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        parser.exit(2, "run_bulk_check.py: {}\n".format(error))

if __name__ == "__main__":
    main()
