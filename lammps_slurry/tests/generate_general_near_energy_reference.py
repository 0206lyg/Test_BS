#!/usr/bin/env python3
"""One failed integration geometry, independently completed in NumPy.

Use the directory containing the previously evaluated general_pair_eval
package. The prior Python source and released archive are left unchanged.
This fixture is a port regression and is never read by the runtime model.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", required=True, type=Path)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).parent / "data/general_near_energy_python_v1.txt")
    args = parser.parse_args()
    sys.path.insert(0, str(args.reference_root.resolve()))
    import numpy as np
    from general_pair_eval.vendor.geometry import make_geometry
    from general_pair_eval.vendor.claeys_near import singular_expansion
    from general_pair_eval.near import compact_factors

    ar, h = 20., .007800524472922343
    pi = np.array([.0459959091219549, 0., .9989416280964796])
    pj = np.array([.18424127321483846, .029663839054549584, .9824333106611978])
    ti, tj, phi = np.arccos(pi[2]), np.arccos(pj[2]), np.arctan2(pj[1], pj[0])
    g = make_geometry(ar, h, c_nm=1., theta_i=ti, theta_j=tj, phi=phi)
    expansion = singular_expansion(g)
    fa, fb = compact_factors(h, min(expansion.reference_length, 1.))
    a, b, t = expansion.inverse_gap6, expansion.logarithm6, expansion.contact_map
    k = fa*a[2, 2] + fb*b[2, 2]
    c = fb*b[:, 2].copy(); c[2] = 0.
    other = fb*b.copy(); other[2, :] = 0.; other[:, 2] = 0.
    square = c/k; square[2] = 1.
    # Positive-square form, independently of the C++ raw-plus-completion sum.
    full = t.T @ (k*np.outer(square, square) + other) @ t
    completion = t.T @ (np.outer(c, c)/k) @ t
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        f.write("GENERAL_NEAR_ENERGY_PYTHON_V1\n")
        f.write(" ".join("%.17g" % x for x in (ar, h, ti, tj, phi)) + "\n")
        for matrix in (full, completion):
            for row in matrix:
                f.write(" ".join("%.17g" % x for x in row) + "\n")
    package = args.reference_root.resolve() / "general_pair_eval"
    files = ("near.py", "vendor/claeys_near.py", "vendor/geometry.py")
    meta = {
        "scope": "NumPy positive-square construction at actual AR20 failed pair; not independent Stokes accuracy",
        "runtime_reads_fixture": False,
        "source_sha256": {name: hashlib.sha256((package/name).read_bytes()).hexdigest() for name in files},
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "fixture_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "case": {"ar": ar, "h_over_c": h, "particle_tags": [1, 64], "origin": "general_smoke_AR20 N64 phi0.04 initial geometry"},
    }
    args.output.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
