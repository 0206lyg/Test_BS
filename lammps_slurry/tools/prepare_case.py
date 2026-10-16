#!/usr/bin/env python3
"""Create particles.data and in.slurry directly from sbatch/CLI parameters.

Random sequential insertion uses the Perram--Wertheim contact function only
to reject initial overlaps. It is not a surface-gap approximation and is not
used by the dynamics or lubrication kernel. Particles have uniform random
quaternions. A standalone C++ packer then grows and geometrically relaxes the
particles. Its steps are not physical dynamics or strain equilibration.
"""

import argparse
import itertools
import math
from pathlib import Path
import random
import subprocess
import sys


DEFAULT_PACKER = Path(__file__).resolve().parents[2] / "build" / "lammps-slurry" / "slurry_pack"


def positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite positive number")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return float(value)


def integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def quaternion(rng):
    """Uniform Haar measure on SO(3), stored as LAMMPS (w,x,y,z)."""
    u, v, w = (rng.random() for _ in range(3))
    return (math.sqrt(u) * math.cos(2 * math.pi * w),
            math.sqrt(1 - u) * math.sin(2 * math.pi * v),
            math.sqrt(1 - u) * math.cos(2 * math.pi * v),
            math.sqrt(u) * math.sin(2 * math.pi * w))


def shape_matrix(quat, axial_ratio):
    """Q/a^2 for an ellipsoid with semiaxes (a,a,c), c/a=axial_ratio."""
    w, x, y, z = quat
    nx, ny, nz = 2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x*x + y*y)
    d = axial_ratio**2 - 1
    return (1 + d*nx*nx, 1 + d*ny*ny, 1 + d*nz*nz,
            d*nx*ny, d*nx*nz, d*ny*nz)


def quadratic_inverse(matrix, displacement):
    xx, yy, zz, xy, xz, yz = matrix
    cx = yy*zz - yz*yz
    cy = xx*zz - xz*xz
    cz = xx*yy - xy*xy
    cxy = xz*yz - xy*zz
    cxz = xy*yz - xz*yy
    cyz = xy*xz - xx*yz
    determinant = xx*cx + xy*cxy + xz*cxz
    if determinant <= 0 or not math.isfinite(determinant):
        raise ValueError("Initial-overlap test has a non-positive shape matrix")
    x, y, z = displacement
    return (cx*x*x + cy*y*y + cz*z*z +
            2*(cxy*x*y + cxz*x*z + cyz*y*z)) / determinant


def contact_function(matrix_i, matrix_j, displacement):
    """Maximum PW contact function: >1 separated, =1 tangent, <1 overlap."""
    difference = tuple(y-x for x, y in zip(matrix_i, matrix_j))

    def value(lam):
        matrix = tuple(x + lam*d for x, d in zip(matrix_i, difference))
        return lam * (1-lam) * quadratic_inverse(matrix, displacement)

    lower, upper = 0.0, 1.0
    ratio = (math.sqrt(5)-1)/2
    left, right = upper-ratio*(upper-lower), lower+ratio*(upper-lower)
    fl, fr = value(left), value(right)
    for _ in range(44):
        if fl < fr:
            lower, left, fl = left, right, fr
            right = lower + ratio*(upper-lower)
            fr = value(right)
        else:
            upper, right, fr = right, left, fl
            left = upper - ratio*(upper-lower)
            fl = value(left)
    return max(fl, fr)


def minimum_image(position_i, position_j, box):
    return tuple((x-y) - box*math.floor((x-y)/box + 0.5)
                 for x, y in zip(position_i, position_j))


def place_particles(count, a, c, box, seed, attempts_per_particle):
    rng = random.Random(seed)
    particles = []
    attempts = 0
    for particle_index in range(count):
        for _ in range(attempts_per_particle):
            attempts += 1
            position = tuple(rng.random()*box for _ in range(3))
            quat = quaternion(rng)
            matrix = shape_matrix(quat, c/a)
            for other in particles:
                displacement = tuple(v/a for v in minimum_image(position, other[0], box))
                distance_squared = sum(v*v for v in displacement)
                if distance_squared >= 4:
                    continue
                if (distance_squared <= 4*(c/a)**2 or
                        contact_function(matrix, other[2], displacement) <= 1 + 1e-10):
                    break
            else:
                particles.append((position, quat, matrix))
                break
        else:
            raise ValueError(
                f"Random insertion stopped at {particle_index}/{count} particles. "
                "No overlapping fallback was written. Change the seed or increase "
                "initialization.maximum_attempts_per_particle; dense states require "
                "a separate preparation method.")

    # Independently traverse the completed configuration before writing it.
    minimum_contact = None
    checked_pairs = 0
    for first, second in itertools.combinations(particles, 2):
        dr = tuple(v/a for v in minimum_image(first[0], second[0], box))
        if sum(v*v for v in dr) >= 4:
            continue
        checked_pairs += 1
        contact = contact_function(first[2], second[2], dr)
        if contact <= 1 + 1e-10:
            raise ValueError("Internal error: completed placement contains an overlap")
        minimum_contact = contact if minimum_contact is None else min(minimum_contact, contact)
    return particles, attempts, checked_pairs, minimum_contact


def format_float(value):
    return f"{value:.17g}"


def write_data(path, particles, a, c, density, box, shear):
    lines = ["Grown oblate spheroids; SI units; a=b, symmetry axis local z", "",
             f"{len(particles)} atoms", f"{len(particles)} ellipsoids", "1 atom types", "",
             f"0 {format_float(box)} xlo xhi", f"0 {format_float(box)} ylo yhi",
             f"0 {format_float(box)} zlo zhi", "0 0 0 xy xz yz", "", "Atoms # ellipsoid", ""]
    for number, (position, _, _) in enumerate(particles, 1):
        lines.append(f"{number} 1 1 {format_float(density)} " + " ".join(map(format_float, position)))
    lines.extend(["", "Ellipsoids", ""])
    for number, (_, quat, _) in enumerate(particles, 1):
        # read_data takes full axis lengths, while the plugin sees semiaxes.
        lines.append(f"{number} " + " ".join(map(format_float, (2*a, 2*a, 2*c, *quat))))
    lines.extend(["", "Velocities", ""])
    for number, (position, _, _) in enumerate(particles, 1):
        # Last three values are angular momentum for atom_style ellipsoid.
        lines.append(f"{number} {format_float(shear*position[1])} 0 0 0 0 0")
    path.write_text("\n".join(lines) + "\n")


def verify_packing(particles, a, box, tolerance):
    """Independent Python PW check of every potentially overlapping pair."""
    worst = 0.0
    for first, second in itertools.combinations(particles, 2):
        dr = tuple(v/a for v in minimum_image(first[0], second[0], box))
        if sum(v*v for v in dr) >= 4:
            continue
        contact = contact_function(first[2], second[2], dr)
        worst = max(worst, 1-math.sqrt(max(0, contact)))
    if worst > tolerance*1.05 + 1e-12:
        raise ValueError(f"Packing verification failed: PW overlap {worst:.9g} > {tolerance:.9g}")
    return worst


def read_packing(path, count, a, c, box):
    rows = path.read_text().splitlines()
    header = rows[0].split()
    if len(header) != 5 or int(header[0]) != count:
        raise ValueError("Packer output has an incorrect particle count")
    for actual, expected in zip(map(float, header[1:]), (box, a, c, 1.0)):
        if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=0):
            raise ValueError("Packer output did not reach the requested final geometry")
    if len(rows) != count+1:
        raise ValueError("Packer output has missing/extra particles")
    particles = []
    for row in rows[1:]:
        values = tuple(map(float, row.split()))
        if len(values) != 7 or not all(math.isfinite(v) for v in values):
            raise ValueError("Packer output contains malformed coordinates")
        position, quat = values[:3], values[3:]
        if not all(0 <= x < box for x in position):
            raise ValueError("Packer output lies outside the periodic box")
        if not math.isclose(sum(q*q for q in quat), 1, abs_tol=1e-10, rel_tol=0):
            raise ValueError("Packer output contains a nonunit quaternion")
        particles.append((position, quat, shape_matrix(quat, c/a)))
    return particles


def resolve_lubrication(args, a):
    """Resolve hydro lengths independently of mechanical rough-contact onset."""
    requested = args.lubrication_model
    spherical = math.isclose(args.ar, 1.0, rel_tol=1e-12, abs_tol=0)
    model = ("mari" if spherical else "spheroid") if requested == "auto" else requested
    if model == "mari" and not spherical:
        raise ValueError("lubrication-model mari currently requires AR=1; "
                         "use spheroid for the local-curvature extension or legacy for the old model")
    if model == "spheroid" and (args.lubrication_epsilon_relative != 0.001 or
                                args.lubrication_cutoff_relative != 0.5):
        raise ValueError("spheroid lubrication uses physical lengths; use --lubrication-epsilon-nm "
                         "and --lubrication-cutoff-nm instead of Mari relative-length overrides")
    if model != "spheroid" and args.lubrication_epsilon_nm != 0.15:
        raise ValueError("--lubrication-epsilon-nm applies only to the spheroid model; "
                         "use --lubrication-epsilon-relative for mari or --roughness-nm for legacy")
    if model == "mari":
        regularization = args.lubrication_epsilon_relative*a
    elif model == "spheroid":
        regularization = args.lubrication_epsilon_nm*1e-9
    else:
        regularization = args.roughness_nm*1e-9
    if args.lubrication_cutoff_nm is not None:
        cutoff = args.lubrication_cutoff_nm*1e-9
    else:
        cutoff = (args.lubrication_cutoff_relative*a if model == "mari"
                  else 75e-9 if model == "spheroid" else 7.5e-9)
    if cutoff <= regularization:
        raise ValueError("resolved lubrication cutoff must exceed its regularization length")
    if args.contact_gap_nm*1e-9 >= cutoff:
        raise ValueError("contact-gap-nm must be smaller than the resolved lubrication cutoff")
    return model, regularization, cutoff


def prepare(args):
    for key in ("ar", "phi", "thickness_nm", "density", "roughness_nm", "viscosity",
                "normal_stiffness", "tangential_stiffness", "lubrication_epsilon_relative",
                "lubrication_epsilon_nm", "lubrication_cutoff_relative", "timestep", "initial_phi",
                "growth_fraction", "overlap_tolerance", "solver_ilu_drop"):
        positive(getattr(args, key), key)
    if args.lubrication_cutoff_nm is not None:
        positive(args.lubrication_cutoff_nm, "lubrication_cutoff_nm")
    for key in ("count", "output_interval", "max_relax", "solver_ilu_fill", "solver_ilu_refresh"):
        integer(getattr(args, key), key)
    integer(args.steps, "steps", minimum=0)
    integer(args.prep_steps, "prep-steps", minimum=0)
    integer(args.seed, "seed", minimum=0)
    for key in ("contact_gap_nm", "friction", "average_start_strain"):
        value = getattr(args, key)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    if not math.isfinite(args.shear_rate):
        raise ValueError("shear-rate must be finite")
    if args.phi >= 1:
        raise ValueError("phi must be below 1")
    # AR = D/t = a/c; full thickness is fixed, equatorial axes vary with AR.
    c = args.thickness_nm*0.5e-9
    a = args.ar*c
    volume = 4*math.pi*a*a*c/3
    box = (args.count*volume/args.phi)**(1/3)
    lubrication_model, regularization, cutoff = resolve_lubrication(args, a)
    contact_gap = args.contact_gap_nm*1e-9
    interaction_range = max(cutoff, contact_gap)
    skin = 0.2*c
    required_length = 2*(2*a+interaction_range+skin)
    if box <= required_length:
        required_count = math.floor(args.phi*required_length**3/volume)+1
        raise ValueError(f"For this AR/phi the periodic initializer requires N >= {required_count}; "
                         f"increase --count (currently {args.count}).")
    packer = args.packer.expanduser().resolve()
    if not packer.is_file():
        raise ValueError(f"Missing geometry packer {packer}; rerun build_slurry_cpu.sbatch")
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)

    print(f"Lubrication: requested={args.lubrication_model}, resolved={lubrication_model}; "
          f"regularization={regularization/1e-9:.12g} nm, cutoff={cutoff/1e-9:.12g} nm; "
          f"independent mechanical contact gap={args.contact_gap_nm:g} nm", flush=True)
    if lubrication_model == "spheroid":
        print("Spheroid local-curvature closure: pair cutoff=min(physical cap, 0.5*Rlocal); "
              "Rlocal is the mean of the two local harmonic curvature radii. "
              "Non-spherical quantitative validation remains pending.", flush=True)

    initial_phi = min(args.initial_phi, args.phi)
    scale = (initial_phi/args.phi)**(1/3)
    particles, attempts, _, _ = place_particles(args.count, a*scale, c*scale, box, args.seed, 20000)
    seed_path, packed_path = output/"packing.seed", output/"packing.final"
    with seed_path.open("w") as stream:
        stream.write(f"{args.count} " + " ".join(map(format_float, (box, a, c, scale))) + "\n")
        for position, quat, _ in particles:
            stream.write(" ".join(map(format_float, (*position, *quat))) + "\n")
    command = [str(packer), "--input", str(seed_path), "--output", str(packed_path),
               "--history", str(output/"packing_history.csv"),
               "--growth-fraction", format_float(args.growth_fraction),
               "--overlap-tolerance", format_float(args.overlap_tolerance),
               "--max-relax", str(args.max_relax)]
    print(f"Growing N={args.count}, AR={args.ar:g}, phi={initial_phi:g} -> {args.phi:g}", flush=True)
    # Preserve live progress in Slurm stdout and a separate preparation log.
    with (output/"packing.log").open("w") as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        if process.wait():
            raise ValueError(f"Packing did not reach the target; see {output/'packing_history.csv'}. "
                             "No in.slurry was written and shear will not start.")
    particles = read_packing(packed_path, args.count, a, c, box)
    worst = verify_packing(particles, a, box, args.overlap_tolerance)
    write_data(output/"particles.data", particles, a, c, args.density, box,
               0.0 if args.prep_steps else args.shear_rate)
    fmt = format_float
    lubrication_options = f"lubrication {lubrication_model}"
    if lubrication_model == "mari":
        lubrication_options += f" lubrication_delta {fmt(args.lubrication_epsilon_relative)}"
        if args.lubrication_cutoff_nm is None:
            lubrication_options += f" lubrication_range {fmt(args.lubrication_cutoff_relative)}"
    contact_options = (f"contact_gap {fmt(contact_gap)} "
                       f"friction {fmt(args.friction)} {fmt(args.tangential_stiffness)}")
    solver_options = (f"solver_preconditioner {args.solver_preconditioner} "
                      f"solver_layout {args.solver_layout} "
                      f"solver_forcing {args.solver_forcing} solver_cache {args.solver_cache} "
                      f"solver_ilu {fmt(args.solver_ilu_drop)} {args.solver_ilu_fill} {args.solver_ilu_refresh}")
    preparation = ""
    if args.prep_steps:
        preparation = f'''# Relax the finite rough-contact layer without shear or frictional memory.
fix prepare all slurry/implicit {fmt(args.viscosity)} 0 {fmt(cutoff)} {fmt(regularization)} {fmt(args.normal_stiffness)} {lubrication_options} {solver_options} contact_gap {fmt(contact_gap)} friction 0 {fmt(args.tangential_stiffness)} history preparation_history.csv {args.output_interval} 0
thermo {args.output_interval}
thermo_style custom step time atoms f_prepare[1] f_prepare[2] f_prepare[3] f_prepare[4] f_prepare[5] f_prepare[6] f_prepare[7] f_prepare[8]
thermo_modify flush yes lost error
print "Zero-shear preparation: {args.prep_steps} steps; finite rough-contact relaxation; no production strain"
run {args.prep_steps}
write_data prepared.data nocoeff
unfix prepare
reset_timestep 0
'''
    text = f'''# Generated directly from run_slurry_cpu.sbatch / prepare_case.py CLI.
# Athermal overdamped dynamics; geometry packing finished before this physical run.
# Coupled resistance, rough-surface normal contact and Coulomb sliding; zero inertia.
# Stress decomposition and interval means are recorded in history.csv.
# AR={fmt(args.ar)} phi={fmt(args.phi)} t={fmt(args.thickness_nm)} nm rho={fmt(args.density)} kg/m3
# Lubrication requested={args.lubrication_model}, resolved={lubrication_model}; epsilon/floor={fmt(regularization)} m; cutoff={fmt(cutoff)} m
# Solver preconditioner={args.solver_preconditioner}; MPI layout={args.solver_layout}
# Solver forcing={args.solver_forcing}; cache={args.solver_cache}; ILUT drop/fill/refresh={fmt(args.solver_ilu_drop)}/{args.solver_ilu_fill}/{args.solver_ilu_refresh}
plugin load ${{slurry_plugin}}
units si
dimension 3
boundary p p p
atom_style ellipsoid
atom_modify map array
newton on
read_data particles.data
comm_modify vel yes
pair_style zero {fmt(2*a+interaction_range)}
pair_coeff * *
neighbor {fmt(skin)} bin
neigh_modify every 1 delay 0 check yes
timestep {fmt(args.timestep)}
{preparation}# Production starts at step/time/strain zero, with a fresh tangential contact history.
fix shear all deform 0 xy erate {fmt(args.shear_rate)} remap v units box
fix integrate all slurry/implicit {fmt(args.viscosity)} {fmt(args.shear_rate)} {fmt(cutoff)} {fmt(regularization)} {fmt(args.normal_stiffness)} {lubrication_options} {solver_options} {contact_options} history history.csv {args.output_interval} {fmt(args.average_start_strain)}
variable strain equal {fmt(abs(args.shear_rate))}*time
compute shape all property/atom shapex shapey shapez quatw quati quatj quatk angmomx angmomy angmomz tqx tqy tqz
dump trajectory all custom {args.output_interval} trajectory.lammpstrj id type x y z vx vy vz fx fy fz c_shape[*]
dump_modify trajectory sort id format float %.17g
thermo {args.output_interval}
thermo_style custom step time atoms v_strain f_integrate[1] f_integrate[2] f_integrate[3] f_integrate[4] f_integrate[5] f_integrate[6] f_integrate[7] f_integrate[8]
thermo_modify flush yes lost error
fix diagnostics all ave/time {args.output_interval} 1 {args.output_interval} f_integrate[1] f_integrate[2] f_integrate[3] f_integrate[4] f_integrate[5] f_integrate[6] f_integrate[7] f_integrate[8] file diagnostics.dat ave one title1 "# Instantaneous overdamped solver diagnostics; overlaps/gaps in m" title2 "# step lubrication_pairs overlapping_pairs max_overlap_m scaled_relative_residual linear_iterations newton_iterations predicted_contacts min_candidate_gap_m"
print "Integrator: slurry/implicit; dt={fmt(args.timestep)} s; steps={args.steps}; duration={fmt(args.steps*args.timestep)} s; total strain={fmt(abs(args.shear_rate)*args.steps*args.timestep)}"
print "Lubrication: {lubrication_model}; epsilon/floor={fmt(regularization)} m; cutoff={fmt(cutoff)} m; mechanical contact gap={fmt(contact_gap)} m"
print "Solver preconditioner: {args.solver_preconditioner}; MPI layout: {args.solver_layout}"
print "Solver forcing: {args.solver_forcing}; cache: {args.solver_cache}; ILUT drop/fill/refresh: {fmt(args.solver_ilu_drop)}/{args.solver_ilu_fill}/{args.solver_ilu_refresh}"
run {args.steps}
write_data final.data nocoeff
'''
    (output/"in.slurry").write_text(text)
    # A generated record, not another input/configuration layer.
    metadata = dict(vars(args))
    metadata.update({"a_m": a, "b_m": a, "c_m": c, "box_length_m": box,
                     "particle_volume_m3": volume, "particle_mass_kg": args.density*volume,
                     "volume_fraction_actual": args.count*volume/box**3,
                     "lubrication_model_requested": args.lubrication_model,
                     "lubrication_model_resolved": lubrication_model,
                     "lubrication_reference_radius_m": a if lubrication_model == "mari" else None,
                     "lubrication_regularization_m": regularization,
                     "lubrication_cutoff_m": cutoff,
                     "lubrication_epsilon_scaling": ("epsilon_relative * mean pair radius" if lubrication_model == "mari"
                                                     else "absolute physical length, independent of aspect ratio" if lubrication_model == "spheroid"
                                                     else "absolute legacy floor"),
                     "lubrication_cutoff_scaling": ("cutoff_relative * mean pair radius"
                                                    if lubrication_model == "mari" and args.lubrication_cutoff_nm is None
                                                    else "min(absolute cap, 0.5*Rlocal); Rlocal=(2/trace(Ci)+2/trace(Cj))/2"
                                                    if lubrication_model == "spheroid"
                                                    else "absolute cutoff"),
                     "lubrication_gap_rule": ("max(h,0)+epsilon" if lubrication_model in ("mari", "spheroid")
                                              else "max(h,roughness)"),
                     "lubrication_validation_scope": ("Local-curvature closure; non-spherical pair and bulk validation deferred"
                                                        if lubrication_model == "spheroid" else
                                                        "Mari sphere kernel" if lubrication_model == "mari" else "Legacy spheroid model"),
                     "lubrication_relative_options_use": ("Mari pair-radius scaling" if lubrication_model == "mari" else
                                                           "Not used; spheroid physical epsilon/cap apply" if lubrication_model == "spheroid" else
                                                           "Not used; legacy absolute roughness/cutoff apply"),
                     "packing_initial_phi_actual": initial_phi,
                     "packing_seed_attempts": attempts, "packing_maximum_PW_overlap": worst,
                     "simulated_time_s": args.steps*args.timestep,
                     "accumulated_strain": abs(args.shear_rate)*args.steps*args.timestep,
                     "preparation_time_s": args.prep_steps*args.timestep,
                     "density_source": ("ACS Nano DOI 10.1021/acsnano.6c10201 PDF p.3: 2.4 g/cm3"
                                        if args.density == 2400.0 else "User override; default ACS Nano density is 2400 kg/m3"),
                     "medium": "User-specified Newtonian medium; default 0.01 Pa s is an assumed CMC90k value",
                     "preparation": "Geometric growth, then optional zero-shear frictionless rough-contact relaxation; production time/strain and tangential history reset afterwards",
                     "roughness_use": ("Ignored by Mari lubrication; hydro epsilon is set independently by lubrication_epsilon_relative"
                                       if lubrication_model == "mari" else "Ignored by spheroid lubrication; hydro epsilon is set independently by lubrication_epsilon_nm"
                                       if lubrication_model == "spheroid" else
                                       "Legacy lubrication coefficient floor only; not a minimum surface gap"),
                     "contact_gap_use": "Mechanical contact onset hc; normal compression max(hc-h,0), not a forced minimum gap",
                     "average_window": "Selected production strain window; stationarity is not assumed or automatically established",
                     "integrator": "slurry/implicit: athermal zero-inertia coupled resistance solve with first-order frozen-geometry rough contact and tangential friction",
                     "density_role": "Stored physical property and mass/angular-momentum representation; inertial acceleration is omitted",
                     "scope": "Athermal overdamped model stress decomposition; no adhesion, adhesive rolling, Brownian noise or far-field many-body hydrodynamics"})
    (output/"parameters.txt").write_text("".join(f"{key} = {value}\n" for key, value in metadata.items()))
    print(f"Prepared {output}\nN={args.count}, AR={args.ar:g}, phi={args.phi:g}, L={box:.12g} m", flush=True)
    print(f"dt={args.timestep:g} s, steps={args.steps}, time={args.steps*args.timestep:g} s, "
          f"strain={abs(args.shear_rate)*args.steps*args.timestep:g}; coupled overdamped dynamics.", flush=True)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new run directory")
    parser.add_argument("--packer", type=Path, default=DEFAULT_PACKER)
    parser.add_argument("--ar", type=float, default=2.0)
    parser.add_argument("--phi", type=float, default=0.24)
    parser.add_argument("--shear-rate", type=float, default=100.0)
    parser.add_argument("--thickness-nm", type=float, default=300.0, help="full polar thickness")
    parser.add_argument("--density", type=float, default=2400.0, help="kg/m^3")
    parser.add_argument("--roughness-nm", type=float, default=3.0, help="legacy lubrication gap floor only; ignored by mari and spheroid")
    parser.add_argument("--contact-gap-nm", type=float, default=3.0, help="mechanical roughness-contact onset; independent of lubrication regularization")
    parser.add_argument("--friction", type=float, default=0.1, help="Coulomb sliding coefficient")
    parser.add_argument("--tangential-stiffness", type=float, default=80.0, help="N/m; tangential contact spring")
    parser.add_argument("--count", type=int, default=300)
    parser.add_argument("--viscosity", type=float, default=0.01, help="Newtonian medium, Pa s")
    parser.add_argument("--normal-stiffness", type=float, default=1.0, help="N/m; finite repulsive spring")
    parser.add_argument("--lubrication-model", choices=("auto", "legacy", "mari", "spheroid"), default="auto",
                        help="auto: Mari sphere model at AR=1, local-curvature spheroid extension otherwise")
    parser.add_argument("--lubrication-epsilon-relative", type=float, default=0.001,
                        help="Mari hydro epsilon / sphere radius; independent of contact gap")
    parser.add_argument("--lubrication-epsilon-nm", type=float, default=0.15,
                        help="spheroid physical hydro regularization in nm; independent of AR and contact gap")
    parser.add_argument("--lubrication-cutoff-relative", type=float, default=0.5,
                        help="Mari outer cutoff / sphere radius; overridden by --lubrication-cutoff-nm")
    parser.add_argument("--lubrication-cutoff-nm", type=float, default=None,
                        help="absolute outer range/cap; defaults: legacy 7.5 nm, spheroid 75 nm (also capped by 0.5*Rlocal), mari relative")
    parser.add_argument("--solver-preconditioner", choices=("auto", "sparse", "nested", "lu", "ilu"), default="ilu",
                        help="implicit linear solver backend; lu uses full-Jacobian SparseLU, ilu uses IncompleteLUT, nested retains the previous inner CG")
    parser.add_argument("--solver-layout", choices=("root", "replicated"), default="root",
                        help="root solves once and shares the result; replicated retains the previous MPI layout")
    parser.add_argument("--solver-forcing", choices=("fixed", "adaptive"), default="fixed",
                        help="inner linear-solve forcing; does not change the final force tolerance")
    parser.add_argument("--solver-cache", choices=("off", "on"), default="on",
                        help="reuse eligible solver preconditioner factors between solves")
    parser.add_argument("--solver-ilu-drop", type=float, default=1e-3,
                        help="positive ILUT drop tolerance; applies when the ilu backend is used")
    parser.add_argument("--solver-ilu-fill", type=int, default=4,
                        help="positive ILUT fill factor; applies when the ilu backend is used")
    parser.add_argument("--solver-ilu-refresh", type=int, default=40,
                        help="positive iteration threshold for refreshing the ILUT factor")
    parser.add_argument("--timestep", type=float, default=1e-6, help="s; overdamped step, check temporal convergence")
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--output-interval", type=int, default=100)
    parser.add_argument("--average-start-strain", type=float, default=5.0, help="start of the reported averaging window, not an automatic steady-state test")
    parser.add_argument("--prep-steps", type=int, default=1000, help="optional zero-shear rough-contact relaxation; 0 disables")
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--initial-phi", type=float, default=0.05)
    parser.add_argument("--growth-fraction", type=float, default=0.02, help="relative phi increase per attempt")
    parser.add_argument("--overlap-tolerance", type=float, default=1e-6, help="dimensionless PW contact-scaling deficit")
    parser.add_argument("--max-relax", type=int, default=20000, help="maximum geometric iterations per growth attempt")
    args = parser.parse_args()
    try:
        prepare(args)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(2, f"prepare_case.py: {error}\n")


if __name__ == "__main__":
    main()


