#!/usr/bin/env python3
"""Prepare a table-free compact-bridge LAMMPS run (standard library only).

AR=a/c; full thickness=2c. The hydrodynamic floor max(h,1 nm) and the
mechanical contact onset are independent. Packing is geometry preparation,
not physical relaxation; its accepted surface gaps must satisfy the same
mechanical contact onset. Spheroids and flat circular cylinders (flake) share
the seed/growth/relaxation workflow and the same shear driver. This module
never loads a resistance table.
"""
import argparse
import itertools
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_PACKER = PROJECT.parent / "build/lammps-slurry/slurry_pack"


def particle_volume(a, c, shape="spheroid"):
    """Physical volume; a is radius and c is half the full thickness."""
    if shape == "spheroid":
        return 4*math.pi*a*a*c/3
    if shape == "flake":
        return 2*math.pi*a*a*c
    raise ValueError("shape must be spheroid or flake")


def write_packing_geometry(path, particles, a, c, box, scale=1.):
    with path.open("w") as stream:
        stream.write("{} {} {} {} {}\n".format(len(particles), box, a, c, scale))
        for x, q, _ in particles:
            stream.write(" ".join("{:.17g}".format(v) for v in (*x, *q))+"\n")


def certify_packing(args, particles, a, c, box, output):
    """Check actual surface clearance with the production gap implementation."""
    packer = args.packer.expanduser().resolve()
    if not packer.is_file() or not os.access(str(packer), os.X_OK):
        raise ValueError("Missing executable geometry packer: {}".format(packer))
    geometry, report_path = output/"packing.validation", output/"packing_clearance.json"
    write_packing_geometry(geometry, particles, a, c, box)
    target = args.contact_gap_nm*1e-9
    command = [str(packer), "--input", str(geometry), "--shape", args.shape,
               "--contact-gap-m", str(target),
               "--certify-only", "1", "--report", str(report_path)]
    process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             universal_newlines=True)
    (output/"packing_validation.log").write_text(process.stdout)
    if process.returncode:
        raise ValueError("Actual surface-gap certification failed; shear input was not written. "
                         "See packing_validation.log")
    try:
        report = json.loads(report_path.read_text())
    except (OSError, ValueError) as error:
        raise ValueError("Missing or malformed actual surface-gap certificate") from error
    if not isinstance(report, dict) or report.get("certified") is not True:
        raise ValueError("Geometry packer did not certify the requested contact gap")
    # A spheroid certificate must never certify a cylinder whose rim extends
    # outside that spheroid. Older spheroid-only certificates remain readable.
    reported_shape = report.get("particle_shape", "spheroid")
    if reported_shape != args.shape:
        raise ValueError("Geometry packer did not certify the requested particle shape")
    for field in ("target_gap_m", "minimum_surface_gap_m", "max_clearance_deficit_m",
                  "nominal_volume_fraction"):
        value = report.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("Geometry packer returned invalid clearance data: "+field)
    if not math.isclose(report["target_gap_m"], target, rel_tol=1e-12, abs_tol=1e-30):
        raise ValueError("Geometry packer did not certify the requested contact gap")
    if (report["minimum_surface_gap_m"] < target or report["max_clearance_deficit_m"] != 0. or
            not math.isclose(report["nominal_volume_fraction"], args.phi, rel_tol=1e-10)):
        raise ValueError("Geometry packer returned a contradictory clearance certificate")
    checks = report.get("checked_pair_images")
    if isinstance(checks, bool) or not isinstance(checks, int) or checks < 1:
        raise ValueError("Geometry packer did not verify periodic image clearances")
    return report

# Geometric PW routines retained from prepare_case.py. No finite-gap/table
# pipeline dependency: old offline scripts are not imported by this run path.
def quaternion(rng):
    u, v, w = (rng.random() for _ in range(3))
    return (math.sqrt(u)*math.cos(2*math.pi*w),
            math.sqrt(1-u)*math.sin(2*math.pi*v),
            math.sqrt(1-u)*math.cos(2*math.pi*v),
            math.sqrt(u)*math.sin(2*math.pi*w))


def shape_matrix(q, ratio):
    w, x, y, z = q
    n = (2*(x*z+w*y), 2*(y*z-w*x), 1-2*(x*x+y*y))
    d = ratio*ratio-1
    return (1+d*n[0]**2, 1+d*n[1]**2, 1+d*n[2]**2,
            d*n[0]*n[1], d*n[0]*n[2], d*n[1]*n[2])


def quadratic_inverse(m, r):
    xx, yy, zz, xy, xz, yz = m
    cx, cy, cz = yy*zz-yz*yz, xx*zz-xz*xz, xx*yy-xy*xy
    cxy, cxz, cyz = xz*yz-xy*zz, xy*yz-xz*yy, xy*xz-xx*yz
    det = xx*cx+xy*cxy+xz*cxz
    if not det > 0:
        raise ValueError("Nonpositive packing shape determinant")
    x, y, z = r
    return (cx*x*x+cy*y*y+cz*z*z+2*(cxy*x*y+cxz*x*z+cyz*y*z))/det


def contact_function(a, b, r):
    def value(t):
        return t*(1-t)*quadratic_inverse(tuple(x+t*(y-x) for x, y in zip(a, b)), r)
    lo, hi = 0., 1.
    rho = (math.sqrt(5)-1)/2
    left, right = hi-rho*(hi-lo), lo+rho*(hi-lo)
    fl, fr = value(left), value(right)
    for _ in range(48):
        if fl < fr:
            lo, left, fl = left, right, fr
            right = lo+rho*(hi-lo)
            fr = value(right)
        else:
            hi, right, fr = right, left, fl
            left = hi-rho*(hi-lo)
            fl = value(left)
    return max(fl, fr)


def image_displacements(first, second, box, radius):
    """Every orthorhombic image within the physical search sphere."""
    d = tuple(y-x for x, y in zip(first, second))
    bounds = [(math.ceil((-radius-v)/box), math.floor((radius-v)/box)) for v in d]
    for n in itertools.product(*(range(lo, hi+1) for lo, hi in bounds)):
        r = tuple(d[k]+box*n[k] for k in range(3))
        if sum(v*v for v in r) <= radius*radius*(1+1e-14):
            yield n, r


def verify_packing(particles, a, box, tolerance):
    worst, checked = 0., 0
    for i, first in enumerate(particles):
        for j in range(i, len(particles)):
            second = particles[j]
            for image, dr in image_displacements(first[0], second[0], box, 2*a):
                if i == j and image == (0, 0, 0):
                    continue
                if i == j and next((n for n in image if n), 0) < 0:
                    continue
                checked += 1
                f = contact_function(first[2], second[2], tuple(v/a for v in dr))
                worst = max(worst, 1-math.sqrt(max(0., f)))
    if worst > tolerance*1.05+1e-12:
        raise ValueError("Packing contains a periodic-image overlap: PW deficit={:.8g}".format(worst))
    return worst, checked


def aligned_particles(count, a, c, box, gap):
    # Search grid sizes, not particle configurations. Include vacancies if the
    # requested count is prime. All selected particles have symmetry axis z.
    nxmax = int(math.floor(box/(2*a+gap)))
    nzmax = int(math.floor(box/(2*c+gap)))
    candidates = []
    for nx in range(1, nxmax+1):
        for ny in range(1, nxmax+1):
            nz = int(math.ceil(count/(nx*ny)))
            if nz <= nzmax:
                spacing = (box/(nx*a), box/(ny*a), box/(nz*c))
                candidates.append(((nx*ny*nz-count, max(spacing)/min(spacing)), (nx, ny, nz)))
    if not candidates:
        raise ValueError("The requested cubic box cannot fit this aligned grid. "
                         "Lower --phi or increase --count; random growth is a separate initialization.")
    dims = min(candidates)[1]
    q = (1., 0., 0., 0.)
    shape = shape_matrix(q, c/a)
    positions = [tuple((index[k]+.5)*box/dims[k] for k in range(3))
                 for index in itertools.product(*(range(n) for n in dims))]
    # Spread vacancies through the grid instead of leaving one slab empty.
    chosen = [positions[int(math.floor(i*len(positions)/count))] for i in range(count)]
    return [(x, q, shape) for x in chosen], {"lattice": dims, "vacancies": len(positions)-count}


def random_particles(args, a, c, box, output):
    bounding_radius = math.hypot(a, c) if args.shape == "flake" else a
    if box <= 4*bounding_radius+2*args.contact_gap_nm*1e-9:
        raise ValueError("The geometry-only random packer needs L>4*bounding_radius+2*contact_gap. "
                         "This is not the hydrodynamic 3a image limit; use --initialization aligned "
                         "for smaller compatible boxes, or increase --count/lower --phi.")
    packer = args.packer.expanduser().resolve()
    if not packer.is_file() or not os.access(str(packer), os.X_OK):
        raise ValueError("Missing executable geometry packer: {}".format(packer))
    scale = (min(args.initial_phi, args.phi)/args.phi)**(1/3)
    # Preserve the latest spheroid seed proposal sequence. For flake this PW
    # test concerns the inscribed spheroid only; the C++ packer first relaxes
    # all actual-cylinder overlaps at the initial scale, then starts growth.
    # The seed is never treated as a cylinder clearance certificate.
    rng, particles, attempts = random.Random(args.seed), [], 0
    for index in range(args.count):
        for _ in range(20000):
            attempts += 1
            x = tuple(rng.random()*box for _ in range(3))
            q = quaternion(rng)
            m = shape_matrix(q, c/a)
            for old in particles:
                dr = tuple((v-u)-box*math.floor((v-u)/box+.5) for u, v in zip(x, old[0]))
                if sum(v*v for v in dr) >= (2*a*scale)**2:
                    continue
                if contact_function(m, old[2], tuple(v/(a*scale) for v in dr)) <= 1+1e-10:
                    break
            else:
                particles.append((x, q, m))
                break
        else:
            raise ValueError("Random seed insertion failed at {}/{}; lower --initial-phi".format(index, args.count))
    seed, final = output/"packing.seed", output/"packing.final"
    write_packing_geometry(seed, particles, a, c, box, scale)
    command = [str(packer), "--input", str(seed), "--output", str(final), "--shape", args.shape,
               "--history", str(output/"packing_history.csv"),
               "--growth-fraction", str(args.growth_fraction),
               "--overlap-tolerance", str(args.overlap_tolerance), "--max-relax", str(args.max_relax),
               "--contact-gap-m", str(args.contact_gap_nm*1e-9)]
    with (output/"packing.log").open("w") as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        process.stdout.close()
        if process.wait():
            raise ValueError("Geometry packing failed; shear input was not written. See packing.log")
    rows = final.read_text().splitlines()
    header = rows[0].split()
    if len(header) != 5 or int(header[0]) != args.count or len(rows) != args.count+1:
        raise ValueError("Malformed geometry packer output")
    if any(not math.isclose(v, e, rel_tol=1e-12) for v, e in zip(map(float, header[1:]), (box, a, c, 1.))):
        raise ValueError("Geometry packer did not reach requested dimensions")
    particles = []
    for row in rows[1:]:
        values = tuple(map(float, row.split()))
        if len(values) != 7 or not all(math.isfinite(v) for v in values):
            raise ValueError("Malformed geometry packer particle")
        x, q = values[:3], values[3:]
        if not all(0 <= v < box for v in x) or not math.isclose(sum(v*v for v in q), 1., abs_tol=1e-10):
            raise ValueError("Invalid coordinate or quaternion in geometry packer output")
        particles.append((x, q, shape_matrix(q, c/a)))
    return particles, {"seed_attempts": attempts, "initial_scale": scale,
                       "seed_proposal": "random positions/quaternions with inscribed-spheroid PW rejection",
                       "growth_shape": args.shape}


def write_data(path, particles, a, c, box, density, shear, shape="spheroid"):
    f = lambda v: "{:.17g}".format(v)
    # LAMMPS atom_style ellipsoid stores axes and orientation for both shapes.
    # Its density-to-mass conversion uses ellipsoid volume; multiply only the
    # storage density so a cylinder retains the requested physical density.
    storage_density = density*(1.5 if shape == "flake" else 1.)
    description = ("Compact bridge oblate spheroids; SI units; local symmetry axis z"
                   if shape == "spheroid" else
                   "Flat circular cylinders; ellipsoid axes are storage proxies; SI units; symmetry axis z")
    lines = [description, "",
             "{} atoms".format(len(particles)), "{} ellipsoids".format(len(particles)), "1 atom types", ""]
    lines += ["0 {} {}lo {}hi".format(f(box), axis, axis) for axis in "xyz"]
    lines += ["0 0 0 xy xz yz", "", "Atoms # ellipsoid", ""]
    for i, (x, q, _) in enumerate(particles, 1):
        lines.append("{} 1 1 {} ".format(i, f(storage_density))+" ".join(map(f, x)))
    lines += ["", "Ellipsoids", ""]
    for i, (_, q, _) in enumerate(particles, 1):
        lines.append(str(i)+" "+" ".join(map(f, (2*a, 2*a, 2*c, *q))))
    lines += ["", "Velocities", ""]
    for i, (x, _, _) in enumerate(particles, 1):
        lines.append("{} {} 0 0 0 0 0".format(i, f(shear*x[2])))
    path.write_text("\n".join(lines)+"\n")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True, help="new run directory")
    p.add_argument("--packer", type=Path, default=DEFAULT_PACKER)
    p.add_argument("--shape", choices=("spheroid", "flake"), default="spheroid",
                   help="physical particle shape; flake is a continuous flat circular cylinder")
    p.add_argument("--initialization", choices=("random", "aligned"), default="random")
    p.add_argument("--mobility-backend", choices=("mesh", "direct"), default="mesh",
                   help="periodic mobility implementation; direct is a reference/debug option")
    p.add_argument("--rough-hydro", type=int, choices=(0, 1), default=0,
                   help="spheroid-only rough-surface excess tangential resistance; default off")
    p.add_argument("--rolling-length-mode", choices=("roughness", "particle"), default="roughness",
                   help="automatic rolling torque arm: rough footprint radius or particle major radius; "
                        "a positive --rolling-length-nm overrides this choice")
    for name, default, help_text in (
            ("rough-height-nm", 0., "cap height per surface; 0 derives contact-gap/2"),
            ("rough-width-nm", 0., "cap footprint radius; 0 derives 10 times the cap height"),
            ("rough-coverage", .2, "surface area fraction covered by caps, between 0 and 1"),
            ("rolling-friction", 0., "dimensionless rolling Coulomb coefficient; 0 disables rolling"),
            ("twisting-friction", 0., "dimensionless twisting Coulomb coefficient; 0 disables twisting"),
            ("rolling-length-nm", 0., "rolling torque arm; 0 uses --rolling-length-mode"),
            ("twisting-length-nm", 0., "twisting torque arm; 0 derives the rough footprint radius"),
            ("rolling-stiffness", 0., "N m/rad; 0 derives normal-stiffness times rolling length squared"),
            ("twisting-stiffness", 0., "N m/rad; 0 derives tangential-stiffness times twisting length squared")):
        p.add_argument("--"+name, type=float, default=default, help=help_text)
    for name, default in (("ar", 20.), ("phi", .24), ("thickness-nm", 300.), ("density", 2400.),
                          ("viscosity", .01), ("shear-rate", 100.), ("gap-floor-nm", 1.),
                          ("contact-gap-nm", 3.), ("friction", .1), ("normal-stiffness", 100.),
                          ("tangential-stiffness", 80.), ("timestep", 1e-6),
                          ("average-start-strain", 5.), ("initial-phi", .05),
                          ("growth-fraction", .02), ("overlap-tolerance", 1e-6), ("tolerance", 1e-6), ("spectral", 2.), ("spectral-tolerance", 1e-8)):
        p.add_argument("--"+name, type=float, default=default)
    for name, default in (("count", 108), ("steps", 100000), ("output-interval", 100),
                          ("seed", 1729), ("max-relax", 20000), ("iterations", 1000)):
        p.add_argument("--"+name, type=int, default=default)
    return p


def surface_models(args):
    """Resolve optional, explicitly phenomenological spheroid surface models."""
    names = ("rough_height_nm", "rough_width_nm", "rough_coverage", "rolling_friction",
             "twisting_friction", "rolling_length_nm", "twisting_length_nm",
             "rolling_stiffness", "twisting_stiffness")
    for name in names:
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            raise ValueError(name+" must be finite and nonnegative")
    if args.rough_coverage > 1:
        raise ValueError("rough_coverage must be between 0 and 1")
    rotation = args.rolling_friction > 0 or args.twisting_friction > 0
    enabled = bool(args.rough_hydro) or rotation
    metadata = {
        "rough_hydrodynamics": {"enabled": bool(args.rough_hydro)},
        "rotation_contact": {"enabled": rotation},
        "surface_parameter_basis": "exploratory model settings; not measured graphite parameters"}
    if not enabled:
        return "", metadata
    if args.shape != "spheroid":
        raise ValueError("rough hydrodynamics and rolling/twisting contact are spheroid-only; "
                         "for flake set --rough-hydro 0 --rolling-friction 0 --twisting-friction 0")
    height = (args.rough_height_nm if args.rough_height_nm > 0 else args.contact_gap_nm/2)*1e-9
    if height <= 0:
        raise ValueError("enabled surface models require a positive rough height; "
                         "automatic height requires contact_gap_nm > 0")
    width = args.rough_width_nm*1e-9 if args.rough_width_nm > 0 else 10*height
    if width < 2*height:
        raise ValueError("rough_width_nm must be at least twice the resolved rough height")
    if args.rolling_length_nm > 0:
        rolling_length = args.rolling_length_nm*1e-9
        rolling_length_source = "explicit_nm"
    elif args.rolling_length_mode == "particle":
        # Current packing uses identical oblate spheroids. For an equal-size
        # pair the reduced major radius 2*a_i*a_j/(a_i+a_j) is simply a.
        rolling_length = args.ar*args.thickness_nm*.5e-9
        rolling_length_source = "particle_major_radius"
    else:
        rolling_length = width
        rolling_length_source = "rough_footprint_radius"
    twisting_length = args.twisting_length_nm*1e-9 if args.twisting_length_nm > 0 else width
    kr = (args.rolling_stiffness if args.rolling_stiffness > 0 else
          args.normal_stiffness*rolling_length**2)
    kw = (args.twisting_stiffness if args.twisting_stiffness > 0 else
          args.tangential_stiffness*twisting_length**2)
    for name, coefficient, stiffness in (("rolling", args.rolling_friction, kr),
                                          ("twisting", args.twisting_friction, kw)):
        if coefficient > 0 and (not math.isfinite(stiffness) or stiffness <= 0):
            raise ValueError(name+" stiffness must resolve to a finite positive value")
    f = lambda value: "{:.17g}".format(value)
    options = []
    if args.rough_hydro:
        options += ["rough_hydro", f(height), f(width), f(args.rough_coverage)]
    if rotation:
        options += ["rotation_contact", f(args.rolling_friction), f(args.twisting_friction),
                    f(rolling_length), f(twisting_length), f(kr), f(kw)]
    metadata["rough_hydrodynamics"].update({
        "height_per_surface_m": height, "footprint_radius_m": width,
        "coverage": args.rough_coverage,
        "hydrodynamic_reach_m": 3*height,
        "microscopic_excess_cutoff_m": height,
        "gap_floor_m": args.gap_floor_nm*1e-9,
        "microstate_rule": "exclude overlapping cap pairs and cap penetration into the opposite base; "
                           "floor only nonnegative microscopic fluid gaps; no density renormalization",
        "gap_rule": "actual smooth-body gap and cap geometry; microscopic fluid gap uses gap_floor; "
                    "contact onset is not subtracted from the smooth lubrication gap",
        "model": "finite-patch hard-core cap-pair tangential and rotational excess; "
                 "positive shifted inverse microgap cutoff; normal baseline unchanged; "
                 "dense-roughness correlations are an unresolved closure approximation"})
    metadata["rotation_contact"].update({
        "rolling_friction": args.rolling_friction, "twisting_friction": args.twisting_friction,
        "rolling_length_mode": args.rolling_length_mode,
        "rolling_length_source": rolling_length_source,
        "rolling_length_m": rolling_length, "twisting_length_m": twisting_length,
        "rolling_stiffness_Nm_per_rad": kr, "twisting_stiffness_Nm_per_rad": kw,
        "model": "elastic Coulomb rolling and twisting torques, active only at contact"})
    return " "+" ".join(options), metadata


def prepare(args):
    for name in ("ar", "phi", "thickness_nm", "density", "viscosity", "gap_floor_nm", "normal_stiffness",
                 "tangential_stiffness", "timestep", "initial_phi", "growth_fraction", "overlap_tolerance", "tolerance", "spectral", "spectral_tolerance"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(name+" must be finite and positive")
    for name in ("contact_gap_nm", "friction", "average_start_strain"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            raise ValueError(name+" must be finite and nonnegative")
    if not math.isfinite(args.shear_rate) or args.ar < 1 or args.phi >= 1 or args.tolerance >= 1 or args.spectral_tolerance >= 1:
        raise ValueError("Require finite shear, AR>=1, 0<phi<1 and 0<tolerance<1")
    for name in ("count", "output_interval", "max_relax", "iterations"):
        if getattr(args, name) < 1:
            raise ValueError(name+" must be a positive integer")
    if min(args.steps, args.seed) < 0:
        raise ValueError("steps and seed must be nonnegative")
    surface_options, surface_metadata = surface_models(args)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise ValueError("Run directory already exists: {}; choose a new path".format(output))
    c, a = args.thickness_nm*.5e-9, args.ar*args.thickness_nm*.5e-9
    volume = particle_volume(a, c, args.shape)
    box = (args.count*volume/args.phi)**(1/3)
    output.mkdir(parents=True, exist_ok=False)
    if args.initialization == "aligned":
        particles, packing = aligned_particles(args.count, a, c, box, args.contact_gap_nm*1e-9)
    else:
        particles, packing = random_particles(args, a, c, box, output)
    # PW is an ellipsoid criterion. Cylinder acceptance is based exclusively
    # on the following true-cylinder periodic surface-gap certificate.
    worst, checked = (verify_packing(particles, a, box, args.overlap_tolerance)
                      if args.shape == "spheroid" else (None, None))
    clearance = certify_packing(args, particles, a, c, box, output)
    write_data(output/"particles.data", particles, a, c, box, args.density, args.shear_rate, args.shape)
    f = lambda v: "{:.17g}".format(v)
    # Pair zero only supplies LAMMPS communication. Hydro/contact images are
    # enumerated in the fix, independently of this neighbor list's radius.
    text = '''# {model_description}
plugin load ${slurry_plugin}
units si
dimension 3
boundary p p p
atom_style ellipsoid
atom_modify map array
read_data particles.data
comm_modify vel yes
pair_style zero {communication}
pair_coeff * *
neighbor {skin} bin
neigh_modify every 1 delay 0 check yes
timestep {dt}
fix shear all deform 0 xz erate {rate} remap v units box
fix hydro all slurry/bridge {eta} {rate} {floor} {kn} contact {hc} {mu} {kt} shape {particle_shape} mobility_backend {backend} spectral {spectral} spectral_tolerance {spectral_tolerance} tolerance {tolerance} iterations {iterations}{surface_options} history history.csv {interval} {average}
compute shape all property/atom shapex shapey shapez quatw quati quatj quatk angmomx angmomy angmomz
dump trajectory all custom {interval} trajectory.lammpstrj id type x y z vx vy vz c_shape[*] f_hydro[*]
dump_modify trajectory sort id format float %.17g
thermo {interval}
thermo_style custom step time atoms f_hydro[1] f_hydro[2] f_hydro[3] f_hydro[4]
thermo_modify flush yes lost error
run {steps}
write_data final.data nocoeff
'''.format(model_description=("Table-free periodic ellipsoid mobility + compact bridge v4" if args.shape == "spheroid"
                             else "Flat circular cylinders: Reynolds face films + approximate rim; matched-spheroid periodic mobility"),
           communication=f(c), skin=f(.2*c), dt=f(args.timestep), rate=f(args.shear_rate), eta=f(args.viscosity),
           floor=f(args.gap_floor_nm*1e-9), kn=f(args.normal_stiffness), hc=f(args.contact_gap_nm*1e-9),
           mu=f(args.friction), kt=f(args.tangential_stiffness), particle_shape=args.shape, backend=args.mobility_backend,
           spectral=f(args.spectral), spectral_tolerance=f(args.spectral_tolerance), tolerance=f(args.tolerance),
           iterations=args.iterations, interval=args.output_interval, average=f(args.average_start_strain), steps=args.steps,
           surface_options=surface_options,
           slurry_plugin="{slurry_plugin}")
    (output/"in.bridge").write_text(text)
    metadata = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    metadata.update(surface_metadata)
    metadata.update({"model": ("compact_bridge_v4_periodic" if args.shape == "spheroid" else
                               "cylinder_reynolds_rim_periodic"),
                     "particle_shape": args.shape, "shear_plane": "xz", "a_m": a, "c_m": c,
                     "box_length_m": box, "particle_volume_m3": volume, "hydro_gap_rule": "max(actual_gap, gap_floor)",
                     "particle_mass_kg": args.density*volume,
                     "lammps_storage": ("ellipsoid" if args.shape == "spheroid" else
                                        "ellipsoid axes/orientation proxy; density multiplied by 1.5 for cylinder mass"),
                     "middle_self_geometry": ("spheroid" if args.shape == "spheroid" else
                                              "matched spheroid approximation with the same a and c"),
                     "near_support_center_m": (3*a if args.shape == "spheroid" else 2*math.hypot(a,c)+a),
                     "periodic_images": "all images in geometric support; self images included",
                     "packing": packing, "packing_max_PW_overlap": worst, "packing_image_checks": checked,
                     "packing_clearance": clearance,
                     "normal_contact_law": "linear spring; normal_stiffness in N/m; no Hertz conversion",
                     "contact_gap_use": "pair contact onset; initial actual surface gaps are certified at or above this gap",
                     "step_geometry": "endpoint actual-gap correction with frozen-geometry implicit force solves",
                     "spectral_tolerance_scope": "Fourier truncation only; not a total mobility or mesh error bound",
                     "mesh_numerics": ({"window_support": 10, "gaussian_alpha": .6,
                                         "quadrature_order": "automatic from wave-number/particle-size bound",
                                         "grid_oversampling": 1., "grid_dimensions": "automatic radix-2/3 smooth dimensions",
                                         "maximum_grid_nodes": 16777216,
                                         "settings_origin": "compiled SlurryPeriodic::Settings defaults",
                                         "actual_grid_and_quadrature": "available through Mobility diagnostic accessors"}
                                        if args.mobility_backend == "mesh" else None),
                     "accumulated_strain": abs(args.shear_rate)*args.timestep*args.steps,
                     "average_window": "selected strain window; no automatic steady-state certification",
                     "preparation": "geometric contact-clearance preparation; no physical equilibration"})
    (output/"parameters.json").write_text(json.dumps(metadata, indent=2, sort_keys=True)+"\n")
    print("Prepared {}: shape={}, N={}, AR={}, phi={}, L/a={:.6g}; no table generation".format(
        output, args.shape, args.count, args.ar, args.phi, box/a), flush=True)
    return metadata


def main():
    p = parser()
    args = p.parse_args()
    try:
        prepare(args)
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        p.exit(2, "prepare_bridge.py: {}\n".format(error))

if __name__ == "__main__":
    main()
