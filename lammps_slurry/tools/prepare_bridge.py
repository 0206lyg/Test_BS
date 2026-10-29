#!/usr/bin/env python3
"""Prepare a table-free compact-bridge LAMMPS run (standard library only).

AR=a/c. The nominal diameter is the cubic-volume mean and remains fixed as AR
changes. Lubrication uses max(h,0)+epsilon*(r_i+r_j)/2, with epsilon dimensionless;
this regularization is not a mechanical roughness layer. Sliding and rolling
springs act only at actual surface contact. Initial packing certifies physical
nonoverlap and is geometric preparation, not physical equilibration. This
module never loads a resistance table.
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


def size_scale(particle):
    return particle[3] if len(particle) > 3 else 1.


def population(args):
    """Same-AR species with nearly equal solid volumes and exact mean volume."""
    if args.size_distribution == "monodisperse":
        return [{"type": 1, "count": args.count, "scale": 1.}]
    q = args.size_ratio
    small = max(1, min(args.count-1, int(math.floor(args.count*q**3/(1+q**3)+.5))))
    large = args.count-small
    scale = (args.count/(small+large*q**3))**(1/3)
    return [{"type": 1, "count": small, "scale": scale},
            {"type": 2, "count": large, "scale": q*scale}]


def mixed_seed_slots(species, rng):
    """Balance both species in coarse cells; randomize cells and insertion order.

    Cells constrain only the dilute seed proposals, not the relaxed packing.
    Random positions and orientations avoid introducing a crystalline lattice.
    """
    count = sum(p["count"] for p in species)
    side = max(1, int(round((count/16.)**(1/3))))
    cells = list(itertools.product(range(side), repeat=3))
    rng.shuffle(cells)
    slots = []
    used = large_used = 0
    large_total = species[1]["count"]
    for k, cell in enumerate(cells):
        stop = (k+1)*count//len(cells)
        large_stop = int(math.floor(stop*large_total/count+.5))
        nlarge = large_stop-large_used
        scales = ([species[0]["scale"]]*(stop-used-nlarge)+
                  [species[1]["scale"]]*nlarge)
        rng.shuffle(scales)
        slots.extend((cell, side, scale) for scale in scales)
        used, large_used = stop, large_stop
    rng.shuffle(slots)
    return slots


def mixing_diagnostics(particles, box, cells_per_axis=2):
    """Report finite-cell concentration variation without asserting equilibrium."""
    scales = sorted(set(size_scale(p) for p in particles))
    bins = {cell: [0]*len(scales) for cell in itertools.product(range(cells_per_axis), repeat=3)}
    for p in particles:
        cell = tuple(min(cells_per_axis-1, int(x/box*cells_per_axis)) for x in p[0])
        bins[cell][scales.index(size_scale(p))] += 1
    records = []
    for cell, counts in bins.items():
        volumes = [count*scale**3 for count, scale in zip(counts, scales)]
        total = sum(volumes)
        records.append({"cell": list(cell), "counts": counts,
                        "solid_volume_fractions_by_size": [v/total for v in volumes] if total else None})
    return {"cells_per_axis": cells_per_axis,
            "cell_counts_by_size": records,
            "interpretation": "geometric mixing diagnostic; not physical equilibration"}


def write_packing_geometry(path, particles, a, c, box, scale=1.):
    with path.open("w") as stream:
        stream.write("{} {} {} {} {}\n".format(len(particles), box, a, c, scale))
        variable = any(len(p) > 3 for p in particles)
        for particle in particles:
            x, q = particle[:2]
            row = (*x, *q, size_scale(particle)) if variable else (*x, *q)
            stream.write(" ".join("{:.17g}".format(v) for v in row)+"\n")


def certify_packing(args, particles, a, c, box, output):
    """Check actual surface clearance with the production gap implementation."""
    packer = args.packer.expanduser().resolve()
    if not packer.is_file() or not os.access(str(packer), os.X_OK):
        raise ValueError("Missing executable geometry packer: {}".format(packer))
    geometry, report_path = output/"packing.validation", output/"packing_clearance.json"
    write_packing_geometry(geometry, particles, a, c, box)
    target = 0.
    command = [str(packer), "--input", str(geometry), "--shape", args.shape,
               "--contact-gap-m", str(target),
               "--certify-only", "1", "--report", str(report_path)]
    if any(len(p) > 3 for p in particles):
        command += ["--particle-scales", "1"]
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
            for image, dr in image_displacements(first[0], second[0], box, a*(size_scale(first)+size_scale(second))):
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
    species = population(args)
    maximum_scale = max(p["scale"] for p in species)
    bounding_radius = (math.hypot(a, c) if args.shape == "flake" else a)*maximum_scale
    if box <= 4*bounding_radius:
        raise ValueError("The geometry-only random packer needs L>4*bounding_radius. "
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
    slots = mixed_seed_slots(species, rng) if len(species) == 2 else [(None, 1, 1.)]*args.count
    for index, (cell, side, particle_scale) in enumerate(slots):
        for _ in range(20000):
            attempts += 1
            x = tuple((rng.random()+(cell[k] if cell else 0))*box/side for k in range(3))
            q = quaternion(rng)
            m = tuple(v*particle_scale**2 for v in shape_matrix(q, c/a))
            for old in particles:
                dr = tuple((v-u)-box*math.floor((v-u)/box+.5) for u, v in zip(x, old[0]))
                if sum(v*v for v in dr) >= (a*(particle_scale+size_scale(old))*scale)**2:
                    continue
                if contact_function(m, old[2], tuple(v/(a*scale) for v in dr)) <= 1+1e-10:
                    break
            else:
                particles.append((x, q, m, particle_scale) if len(species) == 2 else (x, q, m))
                break
        else:
            raise ValueError("Random seed insertion failed at {}/{}; lower --initial-phi".format(index, args.count))
    seed, final = output/"packing.seed", output/"packing.final"
    write_packing_geometry(seed, particles, a, c, box, scale)
    command = [str(packer), "--input", str(seed), "--output", str(final), "--shape", args.shape,
               "--history", str(output/"packing_history.csv"),
               "--growth-fraction", str(args.growth_fraction),
               "--overlap-tolerance", str(args.overlap_tolerance), "--max-relax", str(args.max_relax),
               "--contact-gap-m", "0"]
    if len(species) == 2:
        command += ["--particle-scales", "1"]
    seed_mixing = mixing_diagnostics(particles, box)
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
        if len(values) != (8 if len(species) == 2 else 7) or not all(math.isfinite(v) for v in values):
            raise ValueError("Malformed geometry packer particle")
        x, q = values[:3], values[3:7]
        particle_scale = values[7] if len(values) == 8 else 1.
        if not any(math.isclose(particle_scale, p["scale"], rel_tol=1e-12) for p in species):
            raise ValueError("Geometry packer changed particle size")
        if not all(0 <= v < box for v in x) or not math.isclose(sum(v*v for v in q), 1., abs_tol=1e-10):
            raise ValueError("Invalid coordinate or quaternion in geometry packer output")
        m = tuple(v*particle_scale**2 for v in shape_matrix(q, c/a))
        particles.append((x, q, m, particle_scale) if len(species) == 2 else (x, q, m))
    for p in species:
        if sum(math.isclose(size_scale(x), p["scale"], rel_tol=1e-12) for x in particles) != p["count"]:
            raise ValueError("Geometry packer changed species counts")
    return particles, {"seed_attempts": attempts, "initial_scale": scale,
                       "seed_mixing": seed_mixing, "final_mixing": mixing_diagnostics(particles, box),
                       "species_assignment": "stratified randomized dilute seed" if len(species) == 2 else "single size",
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
    scales = sorted(set(size_scale(p) for p in particles))
    lines = [description, "",
             "{} atoms".format(len(particles)), "{} ellipsoids".format(len(particles)), "{} atom types".format(len(scales)), ""]
    lines += ["0 {} {}lo {}hi".format(f(box), axis, axis) for axis in "xyz"]
    lines += ["0 0 0 xy xz yz", "", "Atoms # ellipsoid", ""]
    for i, particle in enumerate(particles, 1):
        x = particle[0]
        atom_type = scales.index(size_scale(particle))+1
        lines.append("{} {} 1 {} ".format(i, atom_type, f(storage_density))+" ".join(map(f, x)))
    lines += ["", "Ellipsoids", ""]
    for i, particle in enumerate(particles, 1):
        q, s = particle[1], size_scale(particle)
        lines.append(str(i)+" "+" ".join(map(f, (2*a*s, 2*a*s, 2*c*s, *q))))
    lines += ["", "Velocities", ""]
    for i, particle in enumerate(particles, 1):
        x = particle[0]
        lines.append("{} {} 0 0 0 0 0".format(i, f(shear*x[2])))
    path.write_text("\n".join(lines)+"\n")


def parser():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--output", type=Path, required=True, help="new run directory")
    p.add_argument("--packer", type=Path, default=DEFAULT_PACKER)
    p.add_argument("--shape", choices=("spheroid",), default="spheroid",
                   help="oblate spheroid; flat cylinders do not support this contact model")
    p.add_argument("--size-distribution", choices=("monodisperse", "bidisperse"), default="bidisperse")
    p.add_argument("--size-ratio", type=float, default=1.4, help="large/small linear size; equal solid-volume populations")
    p.add_argument("--initialization", choices=("random", "aligned"), default="random")
    p.add_argument("--mobility-backend", choices=("nearfield", "mesh", "direct"), default="nearfield",
                   help="nearfield: isolated spheroid drag plus local lubrication; mesh/direct retain legacy periodic hydrodynamics")
    p.add_argument("--contact-scheme", choices=("imex", "coupled"), default="imex",
                   help="nearfield contact integration: current-load frozen caps, or fully coupled endpoint Coulomb reference")
    for name, default, help_text in (
            ("ar", 2., "equatorial/polar semiaxis ratio, 1 <= AR <= 5"),
            ("nominal-diameter-um", 8., "cubic-volume mean diameter D in micrometers; fixed across AR"),
            ("lubrication-epsilon", .001, "dimensionless additive gap regularization, normalized by D/2"),
            ("friction", 10., "sliding Coulomb coefficient; 10 is a strong-friction trial, not a graphite fit"),
            ("rolling-friction", 10., "rolling Coulomb coefficient; 0 disables rolling resistance"),
            ("normal-stiffness", 30., "normal spring stiffness in N/m"),
            ("tangential-stiffness", 24., "sliding spring stiffness in N/m"),
            ("rolling-stiffness", 24., "rolling quasi-force spring stiffness in N/m; not N m/rad")):
        p.add_argument("--"+name, type=float, default=default, help=help_text)
    for name, default in (("phi", .24), ("density", 2400.),
                          ("viscosity", .01), ("shear-rate", 100.), ("timestep", 2.5e-6),
                          ("average-start-strain", 5.), ("initial-phi", .05),
                          ("growth-fraction", .02), ("overlap-tolerance", 1e-6),
                          ("tolerance", 1e-6), ("spectral", 2.), ("spectral-tolerance", 1e-8)):
        p.add_argument("--"+name, type=float, default=default)
    for name, default in (("count", 1000), ("steps", 200000), ("output-interval", 40),
                          ("seed", 1729), ("max-relax", 20000), ("iterations", 1000)):
        p.add_argument("--"+name, type=int, default=default)
    return p


def particle_dimensions(nominal_diameter_um, ar):
    """Return oblate semiaxes at constant volume-equivalent diameter."""
    radius = nominal_diameter_um*.5e-6
    return radius*ar**(1/3), radius*ar**(-2/3)


def validate_args(args):
    """Validate physical settings before creating a run directory."""
    if not math.isfinite(args.size_ratio) or not 1 < args.size_ratio <= 2:
        raise ValueError("Require 1 < size-ratio <= 2")
    if args.size_distribution == "bidisperse" and args.count < 2:
        raise ValueError("Bidisperse requires at least two particles")
    if args.size_distribution == "bidisperse" and args.mobility_backend != "nearfield":
        raise ValueError("Bidisperse requires nearfield mobility")
    for name in ("ar", "phi", "nominal_diameter_um", "density", "viscosity",
                 "lubrication_epsilon", "normal_stiffness", "tangential_stiffness",
                 "rolling_stiffness", "timestep", "initial_phi", "growth_fraction",
                 "overlap_tolerance", "tolerance", "spectral", "spectral_tolerance"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(name+" must be finite and positive")
    for name in ("friction", "rolling_friction", "average_start_strain"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            raise ValueError(name+" must be finite and nonnegative")
    if (not math.isfinite(args.shear_rate) or not 1 <= args.ar <= 5 or args.phi >= 1 or
            args.tolerance >= 1 or args.spectral_tolerance >= 1):
        raise ValueError("Require finite shear, 1<=AR<=5, 0<phi<1 and 0<tolerance<1")
    if args.shape != "spheroid":
        raise ValueError("The contact-overlap model requires spheroid particles")
    for name in ("count", "output_interval", "max_relax", "iterations"):
        if getattr(args, name) < 1:
            raise ValueError(name+" must be a positive integer")
    if min(args.steps, args.seed) < 0:
        raise ValueError("steps and seed must be nonnegative")


def prepare(args):
    validate_args(args)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise ValueError("Run directory already exists: {}; choose a new path".format(output))
    a, c = particle_dimensions(args.nominal_diameter_um, args.ar)
    species = population(args)
    nominal_diameter = args.nominal_diameter_um*1e-6
    reference_radius = nominal_diameter/2
    regularization = args.lubrication_epsilon*reference_radius
    volume = particle_volume(a, c, args.shape)
    box = (args.count*volume/args.phi)**(1/3)
    output.mkdir(parents=True, exist_ok=False)
    if args.initialization == "aligned":
        largest = max(p["scale"] for p in species)
        particles, packing = aligned_particles(args.count, a*largest, c*largest, box, 0.)
        if len(species) == 2:
            scales = [p["scale"] for p in species for _ in range(p["count"])]
            random.Random(args.seed).shuffle(scales)
            particles = [(p[0], p[1], tuple(v*s*s for v in shape_matrix(p[1], c/a)), s)
                         for p, s in zip(particles, scales)]
        packing["final_mixing"] = mixing_diagnostics(particles, box)
    else:
        particles, packing = random_particles(args, a, c, box, output)
    worst, checked = verify_packing(particles, a, box, args.overlap_tolerance)
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
fix hydro all slurry/bridge {eta} {rate} {diameter} {epsilon} {kn} contact {mu} {kt} rolling {mur} {kr} shape spheroid mobility_backend {backend} contact_scheme {contact_scheme} spectral {spectral} spectral_tolerance {spectral_tolerance} tolerance {tolerance} iterations {iterations} history history.csv {interval} {average}
compute shape all property/atom shapex shapey shapez quatw quati quatj quatk angmomx angmomy angmomz
dump trajectory all custom {interval} trajectory.lammpstrj id type x y z vx vy vz c_shape[*] f_hydro[*]
dump_modify trajectory sort id format float %.17g
thermo {interval}
thermo_style custom step time atoms f_hydro[1] f_hydro[2] f_hydro[3] f_hydro[4]
thermo_modify flush yes lost error
run {steps}
write_data final.data nocoeff
'''.format(model_description=("Isolated spheroid drag + near lubrication + implicit contact" if args.mobility_backend == "nearfield"
                             else "Periodic spheroid mobility + Mari gap regularization + sliding/rolling contact"),
           communication=f(c*max(p["scale"] for p in species)), skin=f(.2*c*min(p["scale"] for p in species)), dt=f(args.timestep), rate=f(args.shear_rate), eta=f(args.viscosity),
           diameter=f(nominal_diameter), epsilon=f(args.lubrication_epsilon), kn=f(args.normal_stiffness),
           mu=f(args.friction), kt=f(args.tangential_stiffness), mur=f(args.rolling_friction),
           kr=f(args.rolling_stiffness), backend=args.mobility_backend, contact_scheme=args.contact_scheme,
           spectral=f(args.spectral), spectral_tolerance=f(args.spectral_tolerance), tolerance=f(args.tolerance),
           iterations=args.iterations, interval=args.output_interval, average=f(args.average_start_strain), steps=args.steps,
           slurry_plugin="{slurry_plugin}")
    (output/"in.bridge").write_text(text)
    metadata = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    metadata.update({"model": ("nearfield_lubrication_spheroid_drag" if args.mobility_backend == "nearfield"
                               else "mari_singh_spheroid_periodic"),
                     "particle_shape": args.shape, "shear_plane": "xz", "a_m": a, "c_m": c,
                     "box_length_m": box, "particle_volume_m3": volume,
                     "nominal_diameter_definition": "cubic-volume mean diameter: mean(V)=pi*D^3/6",
                     "size_populations": [dict(p, a_m=a*p["scale"], c_m=c*p["scale"],
                          volume_equivalent_diameter_m=nominal_diameter*p["scale"],
                          particle_volume_m3=volume*p["scale"]**3,
                          particle_mass_kg=args.density*volume*p["scale"]**3,
                          solid_volume_fraction=p["count"]*p["scale"]**3/args.count) for p in species],
                     "nominal_diameter_m": nominal_diameter,
                     "reference_radius_m": reference_radius,
                     "computational_length_m": reference_radius,
                     "particle_size_constraint": "mean of particle volume radius cubed equals (D/2)^3",
                     "full_axes_m": [2*a, 2*a, 2*c],
                     "axis_rule": "a=(D/2)*AR^(1/3); c=(D/2)*AR^(-2/3)",
                     "reference_geometry_fields": "a_m, c_m, full_axes_m, particle_volume_m3 and particle_mass_kg refer to the cubic-volume mean particle; size_populations holds actual species sizes",
                     "lubrication_normalization": "dimensionless surface gap 2*h/(r_i+r_j); epsilon is dimensionless",
                     "lubrication_regularization_m": regularization,
                     "lubrication_regularization_m_scope": "reference value; actual pair value uses arithmetic mean volume radii",
                     "pair_size_parameters": [
                         {"types": [p["type"], q["type"]],
                          "lubrication_reference_radius_m": reference_radius*(p["scale"]+q["scale"])/2,
                          "regularization_length_m": regularization*(p["scale"]+q["scale"])/2,
                          "rolling_length_m": 2*reference_radius*p["scale"]*q["scale"]/(p["scale"]+q["scale"]),
                          "rolling_angular_stiffness_Nm_per_rad": args.rolling_stiffness*(2*reference_radius*p["scale"]*q["scale"]/(p["scale"]+q["scale"]))**2}
                         for i, p in enumerate(species) for q in species[i:]],
                     "hydro_gap_rule": "max(actual_gap,0)+lubrication_epsilon*(r_i+r_j)/2",
                     "contact_activation": "actual surface gap < 0; no positive-gap contact shell",
                     "friction_parameter_basis": "mu_s=mu_r=10 defaults are a strong-friction trial, not measured graphite or a proven saturation limit",
                     "rolling_contact": {
                         "enabled": args.rolling_friction > 0,
                         "rolling_stiffness_N_per_m": args.rolling_stiffness,
                         "equivalent_angular_stiffness_Nm_per_rad": args.rolling_stiffness*reference_radius**2,
                         "equivalent_angular_stiffness_scope": "reference particle; actual pair values in pair_size_parameters",
                         "reference_radius_m": reference_radius,
                         "lever_rule": "2*r_i*r_j/(r_i+r_j); angular stiffness kr*pair_radius^2",
                         "model": "tangential rolling displacement spring with Coulomb-limited quasi-force; no twisting spring"},
                     "particle_mass_kg": args.density*volume,
                     "lammps_storage": "ellipsoid", "middle_self_geometry": "spheroid",
                     "many_body_far_field": args.mobility_backend != "nearfield",
                     "near_support_center_m": ((2*a+.5*reference_radius)*max(p["scale"] for p in species) if args.mobility_backend == "nearfield" else 3*a),
                     "nearfield_numerics": ({"rigid_unknowns": 6*args.count,
                         "linear_solver": ("sparse 6N Newton; FGMRES with checked current-Jacobian residual; "
                                           + ("reused SPD contact-Jacobian LDLT preconditioner and SPD direct fallback"
                                              if args.contact_scheme == "imex" else "hydro-LDLT preconditioner and direct LU fallback")),
                         "isolated_hydrodynamics": "spheroid drag, Jeffery strain coupling and stresslet",
                         "pair_hydrodynamics": "Mari spheres; local-curvature asymptotic spheroid extension with positive completion",
                         "maximum_surface_cutoff_m": .5*reference_radius*max(p["scale"] for p in species),
                         "pair_surface_cutoff_rule": "0.5*(r_i+r_j)/2, additionally bounded by local curvature for nonspheres",
                         "fft_calls_per_step": 0} if args.mobility_backend == "nearfield" else None),
                     "periodic_images": "all images in geometric support; self images included",
                     "packing": packing, "packing_max_PW_overlap": worst, "packing_image_checks": checked,
                     "packing_clearance": clearance,
                     "normal_contact_law": "linear spring; normal_stiffness in N/m; no Hertz conversion",
                     "contact_stiffness_role": "numerical penalty; validate overlap and stiffness dependence for each case",
                     "maximum_overlap_fraction_criterion": .05,
                     "overlap_fraction_reference": "minimum particle semiaxis; sphere radius for AR1",
                     "initial_surface_gap_target_m": 0.,
                     "step_geometry": (("IMEX implicit springs with current-load frozen friction caps and endpoint normal-gap correction"
                                        if args.contact_scheme == "imex" else
                                        "coupled backward Euler contact histories with endpoint gap correction")
                                       if args.mobility_backend == "nearfield" else
                                       "explicit Heun predictor-corrector; two linear current/predicted-state force solves"),
                     "friction_cap_time": ("current physical normal load; immutable through endpoint corrections"
                                           if args.mobility_backend == "nearfield" and args.contact_scheme == "imex"
                                           else "force-stage normal load"),
                     "spectral_tolerance_scope": ("unused by nearfield backend" if args.mobility_backend == "nearfield"
                         else "Fourier truncation only; not a total mobility or mesh error bound"),
                     "mesh_numerics": ({"window_support": 10, "gaussian_alpha": .6,
                                         "quadrature_order": "automatic from wave-number/particle-size bound",
                                         "grid_oversampling": 1., "grid_dimensions": "automatic radix-2/3 smooth dimensions",
                                         "maximum_grid_nodes": 16777216,
                                         "settings_origin": "compiled SlurryPeriodic::Settings defaults",
                                         "actual_grid_and_quadrature": "available through Mobility diagnostic accessors"}
                                        if args.mobility_backend == "mesh" else None),
                     "accumulated_strain": abs(args.shear_rate)*args.timestep*args.steps,
                     "average_window": "selected strain window; no automatic steady-state certification",
                     "preparation": "geometric physical-nonoverlap preparation; no physical equilibration"})
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
