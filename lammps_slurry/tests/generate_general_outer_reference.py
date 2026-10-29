#!/usr/bin/env python3
"""Freeze selected Python/C++ port comparisons; never used by the runtime.

Pass the directory containing the previously evaluated general_pair_eval
Python package. This generates test fixtures, not a model response table.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference-root", required=True, type=Path)
    p.add_argument("--output", type=Path,
                   default=Path(__file__).parent / "data/general_outer_reference.dat")
    args = p.parse_args()
    sys.path.insert(0, str(args.reference_root.resolve()))
    import numpy as np
    from general_pair_eval.vendor.geometry import make_geometry
    from general_pair_eval.outer import evaluate_outer

    cases = [
        ("sphere_disk", 1., .2, .31, 1.12, .77, 4, 16, False),
        ("ar2_generic_disk", 2., .3, .31, 1.12, .77, 8, 32, False),
        ("ar5_generic_disk", 5., 2., .17, 1.34, .41, 8, 32, False),
        ("ar10_far", 10., 100., .63, 1.21, .52, 4, 16, True),
        ("ar20_face_disk", 20., .02, 0., 0., 0., 16, 64, False),
        ("ar3p7_rotated_disk", 3.7, 1., .31, 1.12, .77, 4, 16, False),
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = ["SLURRY_GENERAL_OUTER_REFERENCE_V1", str(len(cases))]
    info = []
    for name, ar, h, ti, tj, phi, nr, np_, far in cases:
        g = make_geometry(ar, h, c_nm=1., theta_i=ti, theta_j=tj, phi=phi)
        if "rotated" in name:
            from types import SimpleNamespace
            axis = np.array([.3, -.4, .5]); axis /= np.linalg.norm(axis)
            x, y, z = axis
            k = np.array([[0.,-z,y], [z,0.,-x], [-y,x,0.]])
            w = np.eye(3) + np.sin(.73)*k + (1-np.cos(.73))*(k@k)
            g = SimpleNamespace(a=g.a, centers=g.centers@w.T,
                                rotations=np.einsum('ij,bjk->bik', w, g.rotations),
                                shape_tensors=np.einsum('ij,bjk,lk->bil', w, g.shape_tensors, w))
        result = evaluate_outer(g, nr, np_, method="far" if far else "disk")
        rows.append(name)
        rows.append(" ".join(map(str, [ar, h, nr, np_, int(far)])))
        for value in [g.centers, g.rotations, result["grand17"],
                      result["isolated17"], result["mobility22"]]:
            rows.append(" ".join("%.17g" % x for x in np.asarray(value).ravel()))
        info.append({"name": name, "ar": ar, "gap_over_c": h,
                     "method": "far" if far else "disk", "nrad": nr, "nphi": np_,
                     "minimum_grand_eigenvalue": result["minimum_grand_eigenvalue"]})
    args.output.write_text("\n".join(rows) + "\n")
    package = args.reference_root.resolve() / "general_pair_eval"
    files = ["outer.py", "vendor/kim_far.py", "vendor/kim_disk.py", "vendor/geometry.py"]
    metadata = {"scope": "C++ port regression, not independent Stokes accuracy certification",
                "runtime_reads_fixture": False, "cases": info,
                "source_sha256": {s: hashlib.sha256((package/s).read_bytes()).hexdigest() for s in files},
                "fixture_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest()}
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2)+"\n")
    print(args.output)


if __name__ == "__main__":
    main()
