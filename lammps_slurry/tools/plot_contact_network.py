#!/usr/bin/env python3
"""Render Fig. 2(d) from contact_network.csv and a matching production trajectory.

    python plot_contact_network.py RUN
    python plot_contact_network.py RUN --step 200000 --force-max-n 2e-6

The default frame has contact number closest to the strain-averaged sampled
contact number at strain >= 10. At fixed N this also minimizes the deviation in
Z=2C/N. Distances, shapes and forces are SI. Each glyph's three semiaxes are
scaled identically for display. Color and thickness increase linearly with
mechanical force magnitude. The force scale defaults to the largest recorded
force in the eligible frames. Use the same --force-max-n to compare panels.
"""
import argparse
from collections import namedtuple
import csv
import gzip
from itertools import groupby, product
from pathlib import Path

import numpy as np


Snapshot = namedtuple("Snapshot", "step time strain count peak")


def contact_groups(path):
    fields = ("step", "time_s", "strain", "id_i", "id_j", "dx_m", "dy_m", "dz_m", "F_contact_N")
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if not set(fields).issubset(reader.fieldnames or []):
            raise ValueError(f"{path}: missing required contact CSV columns")
        last_step = -1
        for step, rows in groupby(reader, key=lambda r: int(r["step"])):
            if step <= last_step:
                raise ValueError(f"{path}: frame steps are not strictly increasing")
            pairs = []
            time = strain = None
            blank = False
            for row in rows:
                stamps = (float(row["time_s"]), float(row["strain"]))
                if not np.isfinite(stamps).all() or (time is not None and (time, strain) != stamps):
                    raise ValueError(f"Invalid frame timestamp at step {step}")
                time, strain = stamps
                if row["id_i"] == "" and row["id_j"] == "":
                    if pairs or blank or any(row[k] for k in fields[5:]):
                        raise ValueError(f"Invalid empty-frame marker at step {step}")
                    blank = True
                    continue
                i, j = int(row["id_i"]), int(row["id_j"])
                values = np.array([float(row[k]) for k in fields[5:]])
                if blank or i == j or not np.isfinite(values).all() or values[-1] < 0:
                    raise ValueError(f"Invalid contact at step {step}: {(i, j)}")
                pairs.append((i, j, values[:3], values[-1]))
            peak = max((p[3] for p in pairs), default=0.)
            yield Snapshot(step, time, strain, len(pairs), peak), pairs
            last_step = step


def choose_frame(path, start, step):
    eligible = []
    selected = None
    for info, pairs in contact_groups(path):
        if info.strain >= start - 1e-10:
            eligible.append(info)
        if info.step == step:
            selected = (info, pairs)
    if step is not None:
        if selected is None:
            raise ValueError(f"No recorded contact frame at step {step}")
        return selected[0], selected[1], max([selected[0].peak] + [s.peak for s in eligible])
    if not eligible:
        raise ValueError(f"No recorded frame at strain >= {start:g}; use --step or --start-strain")
    strain = np.array([s.strain for s in eligible])
    counts = np.array([s.count for s in eligible])
    if len(strain) > 1 and np.any(np.diff(strain) <= 0):
        raise ValueError("Recorded frame strains do not increase")
    target = (np.sum((counts[1:] + counts[:-1]) * .5 * np.diff(strain)) /
              (strain[-1] - strain[0])) if len(strain) > 1 else counts[0]
    middle = .5 * (strain[0] + strain[-1])
    best = min(eligible, key=lambda s: (abs(s.count - target), abs(s.strain - middle), s.step))
    return best, next(pairs for s, pairs in contact_groups(path) if s.step == best.step), max(s.peak for s in eligible)


def read_geometry(path, selected):
    """Read only the selected frame's atoms, converting bound extents to real box vectors."""
    opener = gzip.open if path.suffix == ".gz" else open
    shape_fields = [f"c_shape[{k}]" for k in range(1, 8)]
    with opener(path, "rt") as stream:
        for line in stream:
            if line.strip() != "ITEM: TIMESTEP":
                raise ValueError(f"{path}: unexpected dump section {line.strip()!r}")
            step = int(next(stream))
            if next(stream).strip() != "ITEM: NUMBER OF ATOMS":
                raise ValueError("Missing atom count")
            n = int(next(stream))
            bounds_header = next(stream).split()
            if bounds_header[:3] != ["ITEM:", "BOX", "BOUNDS"]:
                raise ValueError("Missing LAMMPS box bounds")
            bounds = np.array([next(stream).split() for _ in range(3)], dtype=float)
            columns = next(stream).split()
            if columns[:2] != ["ITEM:", "ATOMS"]:
                raise ValueError("Missing LAMMPS atom columns")
            if step > selected:
                break
            if step < selected:
                for _ in range(n):
                    next(stream)
                continue
            lo, hi = bounds[:, 0].copy(), bounds[:, 1].copy()
            xy = xz = yz = 0.
            if bounds.shape[1] == 3 and bounds_header[3:6] == ["xy", "xz", "yz"]:
                xy, xz, yz = bounds[:, 2]
                lo[0] -= min(0., xy, xz, xy + xz)
                hi[0] -= max(0., xy, xz, xy + xz)
                lo[1] -= min(0., yz)
                hi[1] -= max(0., yz)
            elif bounds.shape[1] != 2:
                raise ValueError("Expected an orthogonal or restricted triclinic LAMMPS dump")
            cell = np.array([[hi[0]-lo[0], xy, xz], [0., hi[1]-lo[1], yz], [0., 0., hi[2]-lo[2]]])
            if not np.isfinite(cell).all() or np.any(np.diag(cell) <= 0):
                raise ValueError("Invalid simulation cell dimensions")
            names = columns[2:]
            indices = [names.index(k) for k in ["id", "x", "y", "z", *shape_fields]]
            ids = []
            data = []
            for _ in range(n):
                tokens = next(stream).split()
                ids.append(int(tokens[indices[0]]))
                data.append([float(tokens[index]) for index in indices[1:]])
            data = np.asarray(data)
            if len(set(ids)) != n or not np.isfinite(data).all() or np.any(data[:, 3:6] <= 0):
                raise ValueError(f"Invalid ellipsoid records at step {step}")
            frac = np.linalg.solve(cell, (data[:, :3] - lo).T).T % 1.
            # LAMMPS compute property/atom shapex/y/z returns full axis lengths.
            return cell, ids, frac, data[:, 3:6]*.5, data[:, 6:10]
    raise ValueError(f"No particle trajectory frame at contact step {selected} in {path}")


def periodic_segments(start, delta):
    """Wrap the entire segment by splitting it where it crosses any cell face."""
    breaks = [0., 1.]
    for x, dx in zip(start, delta):
        if abs(dx) < 1e-14:
            continue
        low, high = sorted((x, x+dx))
        for face in range(int(np.floor(low))+1, int(np.ceil(high))):
            t = (face-x)/dx
            if 1e-13 < t < 1.-1e-13:
                breaks.append(t)
    breaks = np.unique(breaks)
    for a, b in zip(breaks[:-1], breaks[1:]):
        if b-a > 1e-12:
            shift = np.floor(start + .5*(a + b)*delta)
            yield np.stack((start + a*delta - shift, start + b*delta - shift))


def ellipse_shape(q, axes, view):
    # Project the orthographic silhouette of the rotated ellipsoid exactly.
    norm = np.linalg.norm(q)
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("Invalid ellipsoid quaternion")
    w, x, y, z = q / norm
    rotation = np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                         [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                         [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])
    transform = (view @ rotation) * axes
    vals, vecs = np.linalg.eigh(transform @ transform.T)
    if vals[0] <= 0:
        raise ValueError("Degenerate projected particle shape")
    return 2*np.sqrt(vals[1]), 2*np.sqrt(vals[0]), np.degrees(np.arctan2(vecs[1,1], vecs[0,1]))


def draw(geometry, pairs, force_max, shrink, stem, info):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    from matplotlib.lines import Line2D

    cell, ids, frac, semi, quat = geometry
    by_id = {tag: k for k, tag in enumerate(ids)}
    az, el = np.deg2rad([24., 14.])
    view = np.array([[np.cos(az), np.sin(az), 0.],
                     [-np.sin(az)*np.sin(el), np.cos(az)*np.sin(el), np.cos(el)]])
    normal = np.cross(view[0], view[1])
    coordinates = frac @ cell.T
    corners = np.array(list(product([0., 1.], repeat=3)))
    projected_corners = corners @ cell.T @ view.T
    specs = {"font.family":"Liberation Sans", "mathtext.fontset":"stix", "font.size":8.5,
             "pdf.fonttype":42, "ps.fonttype":42}
    with plt.rc_context(specs):
        fig = plt.figure(figsize=(3.54, 2.78))
        ax = fig.add_axes([.015, .115, .97, .815], aspect="equal")
        ax.set_axis_off()
        for corner in corners:
            for d in range(3):
                if corner[d] == 0:
                    edge = np.stack([corner, corner + np.eye(3)[d]]) @ cell.T @ view.T
                    ax.plot(*edge.T, color=".73", lw=.42, zorder=0)
        objects = []
        for xyz, a, q in zip(coordinates, semi, quat):
            wid, hei, ang = ellipse_shape(q, a * shrink, view)
            glyph = Ellipse(view@xyz, wid, hei, angle=ang, facecolor=".91",
                            edgecolor=".66", linewidth=.22)
            objects.append((xyz@normal, glyph))
        inv = np.linalg.inv(cell)
        for i, j, branch, force in pairs:
            if i not in by_id or j not in by_id:
                raise ValueError(f"Unknown trajectory ID in contact {i}, {j}")
            start, other = frac[by_id[i]], frac[by_id[j]]
            delta = inv @ branch
            residual = start + delta - other
            if np.max(np.abs(residual - np.rint(residual))) > 2e-7:
                raise ValueError(f"Contact {i}, {j}: displacement disagrees with this trajectory/box at step {info.step}")
            ratio = min(force/force_max, 1.) if force_max else 0.
            for segment in periodic_segments(start, delta):
                xyz = segment @ cell.T
                line = Line2D(*(xyz @ view.T).T, color=str(.71-.67*ratio),
                              lw=.32+2.25*ratio, solid_capstyle="round")
                objects.append((xyz.mean(axis=0) @ normal, line))
        for _, obj in sorted(objects, key=lambda item: item[0]):
            obj.set_zorder(2)
            ax.add_artist(obj)
        # Include complete displayed ellipsoids near periodic faces.
        pad = shrink * semi.max()
        ax.set_xlim(projected_corners[:,0].min()-pad, projected_corners[:,0].max()+pad)
        ax.set_ylim(projected_corners[:,1].min()-pad, projected_corners[:,1].max()+pad)
        fig.text(.017, .939, "(d)", fontsize=10, weight="bold")
        triad = fig.add_axes([.073, .026, .167, .16], aspect="equal")
        triad.set(xlim=(-.07, 1.18), ylim=(-.23, 1.17))
        triad.set_axis_off()
        for proj, label in zip(view.T, ("x", "y", "z")):
            triad.annotate("", xy=proj, xytext=(0., 0.),
                           arrowprops=dict(arrowstyle="-|>", lw=.62, color=".28",
                                           mutation_scale=5., shrinkA=0., shrinkB=0.))
            triad.text(*(proj*1.09), label, fontsize=7, va="center", ha="left")
        if force_max:
            # A legend in physical units uses exactly the same line rendering as a contact.
            scales = [(1., "N"), (1e-3, "mN"), (1e-6, "µN"), (1e-9, "nN"),
                      (1e-12, "pN"), (1e-15, "fN")]
            scale, unit = next(((s,u) for s,u in scales if force_max>=s), scales[-1])
            fig.text(.37,.066,rf"$|F_{{\rm C}}|$ ({unit})", va="center", ha="right", fontsize=8)
            for ratio, xloc in ((0.,.45),(.5,.64),(1.,.83)):
                line = Line2D([xloc-.063,xloc+.063],[.082,.082], transform=fig.transFigure,
                              color=str(.71-.67*ratio), lw=.32+2.25*ratio, solid_capstyle="round")
                fig.add_artist(line)
                val = f"{force_max*ratio/scale:.2g}"
                fig.text(xloc,.03,val,va="center",ha="center",fontsize=7.6)
        stem.parent.mkdir(parents=True, exist_ok=True)
        description = (f"Step {info.step}, strain {info.strain:.9g}, {len(ids)} particles, "
                       f"{len(pairs)} active pair contacts. Displayed ellipsoid semiaxes scaled by {shrink:g}. "
                       f"Force scale 0 to {force_max:.9g} N.")
        fig.savefig(str(stem)+".pdf", metadata={"Title":"Fig. 2(d) — Contact force network",
                                                 "Subject":description})
        fig.savefig(str(stem)+".png", dpi=600, facecolor="white")
        plt.close(fig)
    return description


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path, help="run directory containing contact_network.csv and trajectory.lammpstrj[.gz]")
    parser.add_argument("--step", type=int, help="select an exact recorded step, overriding the representative-frame rule")
    parser.add_argument("--start-strain", type=float, default=10., help="minimum strain for representative-frame selection (10)")
    parser.add_argument("--particle-scale", type=float, default=.52, help="fraction of physical axis lengths shown (0.52)")
    parser.add_argument("--force-max-n", type=float, help="shared upper scale in newtons; higher forces are saturated")
    parser.add_argument("--output", type=Path, help="output filename stem (default RUN/Fig_2d)")
    args = parser.parse_args()
    if not np.isfinite(args.start_strain) or not 0 < args.particle_scale <= 1:
        parser.error("require finite --start-strain and 0 < --particle-scale <= 1")
    if args.force_max_n is not None and not (np.isfinite(args.force_max_n) and args.force_max_n > 0):
        parser.error("--force-max-n must be finite and positive")
    dump = args.run / "trajectory.lammpstrj"
    if not dump.is_file():
        dump = dump.with_suffix(".lammpstrj.gz")
    try:
        info, pairs, peak = choose_frame(args.run/"contact_network.csv", args.start_strain, args.step)
        geometry = read_geometry(dump, info.step)
        stem = args.output or args.run/"Fig_2d"
        print(draw(geometry, pairs, args.force_max_n or peak, args.particle_scale, stem, info))
        print(f"{stem}.pdf\n{stem}.png")
    except (OSError, ValueError, StopIteration) as error:
        parser.exit(1, f"Contact network: {error or 'truncated trajectory'}\n")


if __name__ == "__main__":
    main()
