#!/usr/bin/env python3
"""Calculate the nonaffine translational velocity correlation in Singh et al.,
Phys. Rev. Lett. 124, 248005 (2020), Eq. (1), for Test_BS xz shear trajectories.

Example:
  python analyze_velocity_correlation.py RUN --output Fig_2c
  python analyze_velocity_correlation.py RUN_A RUN_B --window-strain 0.5 --output Fig_2c

Inputs per run: trajectory.lammpstrj (or its .gz version) and parameters.json.
If only the dump survives, pass --shear-rate and --timestep-s. The smallest
radius can be read from its first frame's full shape axes or set by --radius-m.
Outputs: radial profile CSV, analysis settings JSON and, unless --csv-only is
given, PDF/PNG. Requires Python 3.8+, NumPy, and Matplotlib for plotting.

Subtract the solver's affine velocity gdot*(z-zlo)*ex before time averaging.
The Singh paper states a mean-flow displacement of roughly one particle for
the velocity-average time. Our explicit default uses a small-species volume-
equivalent diameter 2a and the box-averaged absolute affine speed in the
midplane frame, |gdot|*Lz/4. Thus the window spans strain 8a/Lz. The subtraction
of affine velocity uses the solver's zlo origin, irrespective of that scale
choice. --window-strain overrides the default. Use the same convention for the
runs being compared.

Velocity is averaged by particle ID using trapezoidal integration of sampled
velocities over complete, nonoverlapping windows. Distances use centers and
the xz-tilted periodic cell in the saved frame nearest the window midpoint
(earlier frame on a tie). Each unordered pair in each window counts once.
The CSV retains the signed mean dot product in SI units and scaled by
(|gdot|*a)^2. The block columns are independent calculations on contiguous
groups of windows, for checking temporal stability. They are not confidence
bounds. Shorter/longer time spans or larger bin widths can be set on the CLI.
"""
import argparse
import csv
import gzip
import json
import math
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Frame:
    step: int
    ids: np.ndarray
    pos: np.ndarray
    velocity: np.ndarray
    lengths: np.ndarray
    xz: float
    small_shape_radius: float = math.nan


def read_frames(path, shear_rate):
    """Read the ordinary restricted-triclinic or orthogonal dumps made by Test_BS."""
    opener = gzip.open if path.suffix == ".gz" else open
    required = ("id", "x", "y", "z", "vx", "vy", "vz")
    with opener(path, "rt") as stream:
        last_step = None
        identifiers = None
        while True:
            line = stream.readline()
            if not line:
                return
            if line.strip() != "ITEM: TIMESTEP":
                raise ValueError(f"{path}: expected ITEM: TIMESTEP, got {line.strip()!r}")
            step = int(stream.readline())
            if last_step is not None and step <= last_step:
                raise ValueError(f"{path}: timestep {step} is repeated or out of order")
            last_step = step
            if stream.readline().strip() != "ITEM: NUMBER OF ATOMS":
                raise ValueError(f"{path}, step {step}: missing atom count")
            n = int(stream.readline())
            header = stream.readline().split()
            if header[:3] != ["ITEM:", "BOX", "BOUNDS"]:
                raise ValueError(f"{path}, step {step}: missing BOX BOUNDS")
            style = header[3:]
            triclinic = style[:3] == ["xy", "xz", "yz"]
            if style[3 if triclinic else 0:] != ["pp"] * 3:
                raise ValueError(f"{path}: expected a periodic orthogonal or xz-tilted cell")
            bounds = np.array([[float(v) for v in stream.readline().split()] for _ in range(3)])
            if bounds.shape != (3, 3 if triclinic else 2) or not np.isfinite(bounds).all():
                raise ValueError(f"{path}, step {step}: invalid box")
            lo, hi = bounds[:, 0].copy(), bounds[:, 1].copy()
            xz = 0.
            if triclinic:
                xy, xz, yz = bounds[:, 2]
                if xy != 0 or yz != 0:
                    raise ValueError(f"{path}: expected Test_BS xz-only shear; found xy or yz tilt")
                lo[0] -= min(0., xz)
                hi[0] -= max(0., xz)
            lengths = hi - lo
            if np.any(lengths <= 0):
                raise ValueError(f"{path}, step {step}: invalid cell lengths")
            header = stream.readline().split()
            if header[:2] != ["ITEM:", "ATOMS"]:
                raise ValueError(f"{path}, step {step}: missing atom header")
            columns = header[2:]
            missing = set(required) - set(columns)
            if missing:
                raise ValueError(f"{path}: missing columns {sorted(missing)}")
            iid, *coords = [columns.index(key) for key in required]
            shape_indices = []
            if identifiers is None:
                for names in (("c_shape[1]", "c_shape[2]", "c_shape[3]"), ("shapex", "shapey", "shapez")):
                    if all(name in columns for name in names):
                        shape_indices = [columns.index(name) for name in names]
                        break
            axes = np.empty((n, 3)) if shape_indices else None
            ids = np.empty(n, dtype=np.int64)
            values = np.empty((n, 6))
            for i in range(n):
                row = stream.readline().split()
                if len(row) != len(columns):
                    raise ValueError(f"{path}, step {step}: incomplete atom row {i + 1}")
                ids[i] = int(row[iid])
                values[i] = [row[k] for k in coords]
                if axes is not None:
                    axes[i] = [row[k] for k in shape_indices]
            if not np.isfinite(values).all():
                raise ValueError(f"{path}, step {step}: nonfinite position or velocity")
            if not np.all(ids[1:] > ids[:-1]):
                order = np.argsort(ids)
                ids, values = ids[order], values[order]
            if identifiers is None:
                if n < 2 or np.any(np.diff(ids) <= 0):
                    raise ValueError(f"{path}: need at least two atoms with unique IDs")
                identifiers = ids.copy()
            elif not np.array_equal(identifiers, ids):
                raise ValueError(f"{path}, step {step}: particle IDs changed")
            values[:, 3] -= shear_rate * (values[:, 2] - lo[2])
            small_radius = math.nan
            if axes is not None and np.isfinite(axes).all() and (axes > 0).all():
                # compute property/atom shapex, shapey, shapez outputs full axes.
                small_radius = float(np.min(np.cbrt(np.prod(axes, axis=1)))) / 2
            yield Frame(step, ids, values[:, :3], values[:, 3:], lengths, xz, small_radius)


def minimum_distances(displacements, lengths, xz):
    """Exact minimum image for an xz-sheared rectangular prism.

    For any z image the best x and y images are independent roundings.
    After trying the nearest z image the best squared xz distance is at
    most (Lx^2+Lz^2)/4. This bounds how many additional z images can win
    even for non-cubic cells. A component-wise fractional wrap alone does
    not in general minimize the Cartesian distance.
    """
    lx, ly, lz = lengths
    dx, dy, dz = displacements.T
    z_image = np.rint(dz / lz)
    dx = dx - z_image * xz
    dz = dz - z_image * lz
    dx = dx - lx * np.rint(dx / lx)
    dy = dy - ly * np.rint(dy / ly)
    best = dx * dx + dz * dz
    if xz != 0.:
        span = int(math.floor((math.hypot(lx, lz) / lz + 1.) / 2.))
        for offset in range(1, span + 1):
            for k in (-offset, offset):
                candidate_z = dz - k * lz
                keep = candidate_z * candidate_z < best
                if np.any(keep):
                    candidate_x = dx[keep] - k * xz
                    candidate_x -= lx * np.rint(candidate_x / lx)
                    best[keep] = np.minimum(best[keep], candidate_x ** 2 + candidate_z[keep] ** 2)
    return np.sqrt(best + dy * dy)


def shell_sums(frame, velocities, a, edges, pair_ids):
    i, j = pair_ids
    r = minimum_distances(frame.pos[j] - frame.pos[i], frame.lengths, frame.xz) / a
    bins = np.searchsorted(edges, r, side="right") - 1
    keep = (bins >= 0) & (bins < len(edges) - 1)
    bins = bins[keep]
    dots = np.einsum("ij,ij->i", velocities[i[keep]], velocities[j[keep]])
    return (np.bincount(bins, weights=dots, minlength=len(edges) - 1),
            np.bincount(bins, minlength=len(edges) - 1))


def average_windows(frames, first, strain_per_step, start, end, width):
    """Integrate at exact window edges and keep one physical midpoint frame."""
    prev = first
    g0 = first.step * strain_per_step
    left = max(start, g0)
    origin = left
    k = 0
    right = origin + width
    integral = np.zeros_like(first.velocity)
    best_frame = None
    best_dist = np.inf
    max_spacing = 0.
    for frame in frames:
        g1 = frame.step * strain_per_step
        sample = min(g1, end)
        if sample > left:
            begin = max(g0, left)
            eps = 32 * np.finfo(float).eps * max(1., g1, right)
            while begin < sample - eps:
                stop = min(sample, right)
                # Integral of the piecewise-linear interpolant in stored velocity.
                fraction = (0.5 * (begin + stop) - g0) / (g1 - g0)
                integral += (stop - begin) * (prev.velocity + fraction * (frame.velocity - prev.velocity))
                mid = left + 0.5 * width
                for candidate, g in ((prev, g0), (frame, g1)):
                    dist = abs(g - mid)
                    if dist + eps < best_dist:
                        best_frame, best_dist = candidate, dist
                max_spacing = max(max_spacing, g1 - g0)
                if right - stop <= eps:
                    yield (mid, best_frame, integral / width, max_spacing)
                    k += 1
                    left = origin + k * width
                    right = origin + (k + 1) * width
                    integral = np.zeros_like(first.velocity)
                    best_frame, best_dist, max_spacing = None, np.inf, 0.
                begin = stop
        if g1 >= end:
            break
        prev, g0 = frame, g1


def safe_ratio(total, count):
    return np.divide(total, count, out=np.full_like(total, np.nan, dtype=float), where=count > 0)


def check_substeps(run, steps):
    """A subdivided macrostep can have a stage velocity offset from the dump geometry."""
    history = run / "history.csv"
    if not history.is_file():
        return None
    with history.open() as stream:
        records = csv.DictReader(stream)
        if not {"step", "substeps"}.issubset(records.fieldnames or ()):
            return None
        return sum(int(row["step"]) in steps and float(row["substeps"]) > 1 for row in records)


def analyze(run, args):
    parameter_file = run / "parameters.json"
    params = json.loads(parameter_file.read_text()) if parameter_file.is_file() else {}
    rate = args.shear_rate if args.shear_rate is not None else params.get("shear_rate")
    dt = args.timestep_s if args.timestep_s is not None else params.get("timestep")
    if rate is None or dt is None:
        raise ValueError(f"{run}: supply parameters.json or --shear-rate and --timestep-s")
    rate, dt = float(rate), float(dt)
    if not math.isfinite(rate) or rate == 0 or not math.isfinite(dt) or dt <= 0.:
        raise ValueError(f"{run}: shear_rate must be nonzero and timestep positive")
    path = run / "trajectory.lammpstrj"
    if not path.is_file():
        path = run / "trajectory.lammpstrj.gz"
    frames = read_frames(path, rate)
    first = next(frames, None)
    if first is None:
        raise ValueError(f"empty trajectory in {run}")
    if args.radius_m:
        a, radius_source = args.radius_m, "--radius-m"
    elif params.get("size_populations"):
        a = min(float(species["volume_equivalent_diameter_m"])
                for species in params["size_populations"]) / 2
        radius_source = "parameters.json size_populations"
    elif math.isfinite(first.small_shape_radius):
        a, radius_source = first.small_shape_radius, "smallest volume-equivalent radius from first dump frame full axes"
    elif params.get("size_distribution") == "monodisperse":
        a = float(params["nominal_diameter_m"]) / 2
        radius_source = "monodisperse nominal_diameter_m"
    else:
        raise ValueError(f"{run}: small-species radius missing; supply --radius-m in metres")
    if not math.isfinite(a) or a <= 0:
        raise ValueError(f"{run}: invalid particle radius")
    start = max(args.start_strain, first.step * abs(rate) * dt)
    stop = args.end_strain if args.end_strain is not None else math.inf
    if stop <= start:
        raise ValueError(f"{run}: no requested strain interval within the trajectory")
    width = args.window_strain or (8 * a / first.lengths[2])
    r_max = args.r_max or (0.5 * min(first.lengths) / a)
    edges = np.arange(0., r_max, args.bin_width)
    # Discard any near-zero final bin created by roundoff.
    if r_max - edges[-1] < args.bin_width * 1e-12 and len(edges) > 1:
        edges = edges[:-1]
    edges = np.append(edges, r_max)
    i, j = np.triu_indices(len(first.ids), k=1)
    mids, sums, counts, steps = [], [], [], set()
    max_spacing = 0.
    first_used_step = None
    last_used_step = None

    def tracked_frames():
        nonlocal first_used_step, last_used_step
        last = first
        for frame in frames:
            if (frame.step * abs(rate) * dt >= start and last.step * abs(rate) * dt <= stop):
                if first_used_step is None:
                    first_used_step = last.step
                steps.update((last.step, frame.step))
                last_used_step = frame.step
            yield frame
            last = frame

    for mid, frame, ubar, spacing in average_windows(
            tracked_frames(), first, abs(rate) * dt, start, stop, width):
        total, count = shell_sums(frame, ubar, a, edges, (i, j))
        mids.append(mid)
        sums.append(total)
        counts.append(count)
        max_spacing = max(max_spacing, spacing)
    if not mids:
        raise ValueError(f"{run}: no complete velocity window (requested strain width {width:.6g})")
    if max_spacing > width / 2 * (1 + 1e-8):
        warnings.warn(f"{run}: largest dump spacing {max_spacing:g} spans more than half the "
                      f"averaging window {width:g}; check window sensitivity")
    subdivided = check_substeps(run, steps)
    if subdivided:
        warnings.warn(f"{run}: {subdivided} saved steps have substeps > 1; their velocity uses "
                      "the last internal stage while the dump position uses the macrostep geometry")
    sums, counts = np.array(sums), np.array(counts)
    nblocks = min(args.blocks, len(mids))
    blocks = np.array_split(np.arange(len(mids)), nblocks)
    block_means = np.array([safe_ratio(sums[idx].sum(axis=0), counts[idx].sum(axis=0)) for idx in blocks])
    scale = (abs(rate) * a) ** 2
    profile = safe_ratio(sums.sum(axis=0), counts.sum(axis=0))
    meta = dict(run=str(run), AR=params.get("ar"), phi=params.get("phi"), shear_rate_s_inv=rate,
                timestep_s=dt, radius_m=a, radius_source=radius_source, velocity_scale_m_s=abs(rate) * a,
                c_m2_s2_definition="mean pair dot product of window-averaged nonaffine translation in m^2/s^2",
                c_scaled_definition="c_m2_s2 / (abs(shear_rate_s_inv) * radius_m)^2",
                bin_width_over_a=args.bin_width, r_max_over_a=r_max,
                window_strain=width, window_method=("user" if args.window_strain else
                                                    "2a / (mean midplane-frame affine speed Lz*|gdot|/4)"),
                covered_strain=[start, start + len(mids) * width], complete_windows=len(mids),
                max_dump_spacing_strain=max_spacing, checked_frame_step_range=[first_used_step, last_used_step],
                checked_frames_with_substeps_gt_1=subdivided, block_windows=[len(idx) for idx in blocks],
                mu_s=params.get("friction"), mu_r=params.get("rolling_friction"))
    centers = 0.5 * (edges[:-1] + edges[1:])
    rows = []
    for k, radius in enumerate(centers):
        row = dict(run=str(run), AR=meta["AR"], phi=meta["phi"], r_lo_over_a=edges[k],
                   r_hi_over_a=edges[k+1], r_over_a=radius, pair_samples=int(counts[:, k].sum()),
                   n_windows=int(np.count_nonzero(counts[:, k])), c_m2_s2=profile[k],
                   c_scaled=profile[k] / scale)
        row.update({f"block_{idx+1}_c_scaled": block_means[idx, k] / scale for idx in range(nblocks)})
        rows.append(row)
    return meta, rows


def write_plot(path, results, labels):
    import matplotlib as mpl
    mpl.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import AutoMinorLocator

    mpl.rcParams.update({"font.family": ["Liberation Sans", "DejaVu Sans"], "font.size": 8.5,
                         "mathtext.fontset": "stix", "axes.labelsize": 13, "axes.linewidth": .8,
                         "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True,
                         "ytick.right": True, "xtick.major.size": 3.2, "ytick.major.size": 3.2,
                         "xtick.minor.size": 1.8, "ytick.minor.size": 1.8,
                         "xtick.major.width": .7, "ytick.major.width": .7,
                         "xtick.minor.width": .5, "ytick.minor.width": .5,
                         "legend.frameon": False, "pdf.fonttype": 42})
    colors = ["#171717", "#2060A3", "#BA342D", "#00836A", "#79509B"]
    symbols = ["D", "o", "s", "^", "v"]
    fig = plt.figure(figsize=(3.54, 2.78))
    ax = fig.add_axes([.144, .155, .828, .77])
    fig.text(.017, .939, "(c)", fontsize=10, weight="bold")
    signed = []
    used_styles = {}
    for index, (meta, rows) in enumerate(results):
        ar = meta["AR"]
        style = int(ar) - 1 if ar in (1, 2, 3, 4, 5) else index % len(colors)
        variant = used_styles.get(style, 0)
        used_styles[style] = variant + 1
        linestyle = ("-", "--", "-.", ":")[variant % 4]
        x = [r["r_over_a"] for r in rows]
        y = np.array([r["c_scaled"] for r in rows])
        signed.extend(y[np.isfinite(y)])
        ax.plot(x, y, lw=.94, ls=linestyle, color=colors[style], marker=symbols[style], ms=3.75,
                mfc="white", mew=.82, label=labels[index])
    if signed and min(signed) > 0:
        ax.set_yscale("log")
    else:
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.set_xlabel(r"$r/a$", labelpad=0)
    # A compact ordinate fits the Fig. 1 panel; normalization is given in the
    # figure metadata and CSV. Use the conventional star for dimensionless c.
    ax.set_ylabel(r"$c^*(r)$", labelpad=0)
    ax.tick_params(axis="both", which="major", pad=2.2)
    ax.legend(loc="best", fontsize=7.6, handlelength=1.6, labelspacing=.14,
              handletextpad=.38, borderpad=.05, borderaxespad=.35)
    fig.savefig(str(path) + ".pdf", metadata={"Title": "Nonaffine velocity correlation",
                "Subject": "c*(r)=c(r)/(|shear rate| a)^2. a = small-species volume-equivalent radius "
                           "unless overridden. Lines connect binned sample means."})
    fig.savefig(str(path) + ".png", dpi=600, facecolor="white")
    plt.close(fig)


def legend_labels(metadata):
    """Prefer the parameters which distinguish the cases, including friction."""
    labels = [[] for _ in metadata]
    all_friction = {(m["mu_s"], m["mu_r"]) for m in metadata}
    for key, symbol in (("AR", "AR"), ("phi", r"$\phi$")):
        varying = len({m[key] for m in metadata}) > 1
        for i, m in enumerate(metadata):
            if m[key] is not None and (varying or len(metadata) == 1):
                labels[i].append(f"{symbol} = {m[key]:g}")
    if len(all_friction) > 1:
        for i, m in enumerate(metadata):
            sliding, rolling = m["mu_s"], m["mu_r"]
            if sliding is not None and rolling is not None:
                if sliding == rolling:
                    labels[i].append(r"$\mu_s=\mu_r=" + f"{sliding:g}$")
                else:
                    labels[i].append(r"$(\mu_s,\mu_r)=" + f"({sliding:g},{rolling:g})$")
    plain = [", ".join(parts) for parts in labels]
    return [label if label and plain.count(label) == 1 else
            (f"{label}, run {i + 1}" if label else Path(metadata[i]["run"]).resolve().name)
            for i, label in enumerate(plain)]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", type=Path, nargs="*", default=[Path(".")], metavar="RUN")
    parser.add_argument("--output", type=Path, help="output filename prefix (default RUN/Fig_2c for a single run)")
    parser.add_argument("--start-strain", type=float, default=10.)
    parser.add_argument("--end-strain", type=float, help="default: last available frame")
    parser.add_argument("--window-strain", type=float, help="velocity average length in strain (default 8a/Lz)")
    parser.add_argument("--shear-rate", type=float, help="override the signed shear rate, in s^-1")
    parser.add_argument("--timestep-s", type=float, help="override the integration timestep, in seconds")
    parser.add_argument("--radius-m", type=float, help="override small-species volume-equivalent radius in metres")
    parser.add_argument("--bin-width", type=float, default=.25, help="radial bin width in units of a")
    parser.add_argument("--r-max", type=float, help="outer edge in units of a (default half shortest box side/a)")
    parser.add_argument("--blocks", type=int, default=4, help="number of contiguous blocks for stability checks")
    parser.add_argument("--label", action="append", default=[], help="legend entry, once per run in input order")
    parser.add_argument("--csv-only", action="store_true", help="write CSV and settings JSON without Matplotlib")
    args = parser.parse_args()
    for key in ("window_strain", "radius_m", "bin_width", "r_max"):
        val = getattr(args, key)
        if val is not None and (not math.isfinite(val) or val <= 0):
            parser.error(f"--{key.replace('_', '-')} must be positive")
    if not math.isfinite(args.start_strain) or args.start_strain < 0 or args.blocks < 1:
        parser.error("--start-strain must be finite and nonnegative; --blocks must be positive")
    if args.end_strain is not None and (not math.isfinite(args.end_strain) or args.end_strain <= args.start_strain):
        parser.error("--end-strain must exceed --start-strain")
    if args.label and len(args.label) != len(args.runs):
        parser.error("supply one --label for each run")
    path = args.output or ((args.runs[0] if len(args.runs) == 1 else Path(".")) / "Fig_2c")
    results = [analyze(run, args) for run in args.runs]
    labels = args.label or legend_labels([meta for meta, _ in results])
    path.parent.mkdir(parents=True, exist_ok=True)
    all_rows = [row for _, rows in results for row in rows]
    columns = list(dict.fromkeys(key for row in all_rows for key in row))
    with Path(str(path) + ".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(all_rows)
    with Path(str(path) + ".json").open("w") as stream:
        json.dump([meta for meta, _ in results], stream, indent=2, allow_nan=False)
        stream.write("\n")
    if not args.csv_only:
        write_plot(path, results, labels)
    for meta, _ in results:
        print(f"{meta['run']}: {meta['complete_windows']} complete velocity windows, "
              f"window strain {meta['window_strain']:.5g}, radius {meta['radius_m']:.5g} m")
    print(str(path) + ".csv")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError) as exc:
        sys.exit(f"{Path(sys.argv[0]).name}: {exc}")
