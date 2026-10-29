#!/usr/bin/env python3
"""Check the ABI and policy in separate processes to avoid symbol interposition."""
import argparse
import ctypes
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys


def probe(library):
    lib = ctypes.CDLL(str(library.resolve()), mode=ctypes.RTLD_LOCAL)
    routine = lib.get_quadparams_adap_
    real = ctypes.c_double
    integer = ctypes.c_int64
    routine.argtypes = [ctypes.POINTER(real), ctypes.POINTER(integer), ctypes.POINTER(integer),
                       ctypes.POINTER(real), ctypes.POINTER(integer), ctypes.POINTER(integer)]
    routine.restype = None
    eps_values = [.01, .001, .0001, 1e-7, 1e-9, 1e-11]
    for threshold in [.5e-2, .5e-3, .5e-6, .5e-9]:
        eps_values.extend([threshold, math.nextafter(threshold, 0.)])
    expected = [(12, 1e-2, 1, 8), (12, .2e-2, 1, 10), (16, .1e-4, 1, 16),
                (22, 7e-8, 1, 22), (30, 3e-10, 1, 30)]
    records = []
    for patch_type in [1, 11]:
        for eps in eps_values:
            iprec = sum(eps < threshold for threshold in [.5e-2, .5e-3, .5e-6, .5e-9])
            samples = []
            for _ in range(3):
                e, ip = real(eps), integer(patch_type)
                order, adaptive_eps, level, far_order = integer(-1), real(-1), integer(-1), integer(-1)
                routine(ctypes.byref(e), ctypes.byref(ip), ctypes.byref(order), ctypes.byref(adaptive_eps),
                        ctypes.byref(level), ctypes.byref(far_order))
                samples.append([order.value, adaptive_eps.value, level.value, far_order.value])
            records.append({"eps": eps, "patch_type": patch_type, "iprec": iprec,
                            "values": samples[0], "repeatable": all(v == samples[0] for v in samples),
                            "matches_high_order_policy": samples[0] == list(expected[iprec])})
    return {"library": str(library.resolve()), "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
            "output_columns": ["nqorder", "eps_adap", "nlev", "nqorder_f"], "records": records,
            "all_repeatable": all(r["repeatable"] for r in records),
            "all_match_high_order_policy": all(r["matches_high_order_policy"] for r in records)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--old-library", type=Path)
    ap.add_argument("--new-library", type=Path)
    ap.add_argument("--output", type=Path)
    ap.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.worker:
        print(json.dumps(probe(args.worker)))
        return
    if not all((args.old_library, args.new_library, args.output)):
        ap.error("--old-library, --new-library and --output are required")
    result = {"schema": "fmm3dbie_quadparams_policy_probe_v1", "physical_accuracy_qualified": False,
              "scope": "Parameter selection and ABI only; no operator or physical accuracy qualification"}
    for name, library in [("old", args.old_library), ("new", args.new_library)]:
        result[name] = json.loads(subprocess.check_output([sys.executable, str(Path(__file__).resolve()),
                                                          "--worker", str(library.resolve())], text=True))
    result["passed"] = result["new"]["all_repeatable"] and result["new"]["all_match_high_order_policy"]
    result["changed_case_count"] = sum(a["values"] != b["values"] for a, b in
                                      zip(result["old"]["records"], result["new"]["records"]))
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ["passed", "changed_case_count", "physical_accuracy_qualified"]}))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
