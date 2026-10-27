#!/usr/bin/env python3
"""Local integration fixture only: two almost touching compressive spheroids.

These runs deliberately start with 0.1 nm elastic overlap to guarantee contact
coverage; they are not a packing/viscosity/jamming experiment.
"""
import argparse
import csv
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess

p = argparse.ArgumentParser()
p.add_argument("--repo", type=Path, required=True)
p.add_argument("--lmp", type=Path, required=True)
p.add_argument("--plugin", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--timestep", type=float, default=1e-6)
p.add_argument("--backend", choices=("direct", "mesh"), default="direct")
args = p.parse_args()
for name in ("repo", "lmp", "plugin", "output"):
    setattr(args, name, getattr(args, name).expanduser().resolve())
if not math.isfinite(args.timestep) or args.timestep <= 0:
    p.error("--timestep must be finite and positive")
if any(character in str(args.plugin) for character in ('"', '\n', '\r')):
    p.error("plugin path cannot contain quotes or newlines")
spec = importlib.util.spec_from_file_location("prepare_bridge", args.repo/"lammps_slurry/tools/prepare_bridge.py")
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)
args.output.mkdir(parents=True, exist_ok=True)
results = []
for ar in (1, 2, 5):
    folder = args.output/("ar"+str(ar))
    folder.mkdir(exist_ok=True)
    a, c = prepare.particle_dimensions(8., ar)
    box = 6*a
    n = (math.sqrt(.5), 0., -math.sqrt(.5))
    def support(angle):
        co, si = math.cos(angle), math.sin(angle)
        local_n = (co*n[0]-si*n[2], 0., si*n[0]+co*n[2])
        norm = math.sqrt(a*a*local_n[0]**2+c*c*local_n[2]**2)
        lx, lz = a*a*local_n[0]/norm, c*c*local_n[2]/norm
        return (co*lx+si*lz, 0., -si*lx+co*lz)
    first, second = support(0.), support(.13)
    displacement = tuple(first[k]+second[k]-1e-10*n[k] for k in range(3))
    center = (box/2,)*3
    particles = [
        (tuple(center[k]-.5*displacement[k] for k in range(3)), (1.,0.,0.,0.), None),
        (tuple(center[k]+.5*displacement[k] for k in range(3)), (math.cos(.065),0.,math.sin(.065),0.), None),
    ]
    prepare.write_data(folder/"particles.data", particles, a, c, box, 2400., 100.)
    (folder/"in.bridge").write_text(f'''# deterministic contact integration fixture, not production packing
plugin load "{args.plugin}"
units si
dimension 3
boundary p p p
atom_style ellipsoid
atom_modify map array
read_data particles.data
comm_modify vel yes
pair_style zero {c:.17g}
pair_coeff * *
neighbor {.2*c:.17g} bin
neigh_modify every 1 delay 0 check yes
timestep {args.timestep:.17g}
fix shear all deform 0 xz erate 100 remap v units box
fix hydro all slurry/bridge .01 100 8e-6 .001 100 contact 10 80 rolling 10 80 shape spheroid mobility_backend {args.backend} spectral 2 spectral_tolerance 1e-8 tolerance 1e-6 iterations 1000 history history.csv 1 0
thermo 1
thermo_style custom step time atoms f_hydro[1] f_hydro[2] f_hydro[3] f_hydro[4]
thermo_modify flush yes lost error
run 5
write_data final.data nocoeff
''')
    env = dict(os.environ, OMP_NUM_THREADS="1", OMP_DYNAMIC="FALSE")
    with (folder/"stdout.log").open("w") as stream:
        result = subprocess.run([str(args.lmp), "-in", "in.bridge"], cwd=folder, env=env, stdout=stream, stderr=subprocess.STDOUT)
    record = {"ar": ar, "returncode": result.returncode, "a_m": a, "c_m": c, "reference_radius_m": 4e-6, "regularization_m": 4e-9, "timestep_s": args.timestep, "mobility_backend": args.backend}
    if result.returncode == 0:
        with (folder/"history.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        record["rows"] = len(rows)
        record["max_contacts"] = max(float(row["contacts"]) for row in rows)
        record["max_solver_relative_residual"] = max(float(row["solver_relative_residual"]) for row in rows)
        record["min_endpoint_gap_m"] = min(float(row["actual_endpoint_min_gap_m"]) for row in rows)
        record["max_gap_consistency_m"] = max(float(row["contact_gap_consistency_m"]) for row in rows)
        record["max_rolling_elastic_energy_J"] = max(float(row["rolling_elastic_energy_J"]) for row in rows)
        # At step zero, interval/selected-window means intentionally are NaN:
        # no physical interval has elapsed yet. Require finite outputs afterward.
        evolved_rows = [row for row in rows if int(row["step"]) > 0]
        record["all_finite"] = all(math.isfinite(float(value)) for row in evolved_rows for value in row.values())
        record["pass"] = (len(rows)>=5 and record["max_contacts"]>=1 and record["all_finite"] and record["max_solver_relative_residual"]<=1e-6)
    else:
        record["pass"] = False
    results.append(record)
    print(json.dumps(record), flush=True)
    (args.output/"summary.json").write_text(json.dumps(results, indent=2)+"\n")
raise SystemExit(0 if all(record["pass"] for record in results) else 1)
