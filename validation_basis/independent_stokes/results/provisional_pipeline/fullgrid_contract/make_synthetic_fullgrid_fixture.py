#!/usr/bin/env python3
"""Create an explicitly synthetic AR20 full-grid serialization fixture.

No Stokes calculation or physical reference is represented by these matrices.
The anchor is 1e6 times the identity in the work-conjugate full17 coordinates;
it is SPD and invariant under all proper global rotations. All 64 numeric
theta_i/theta_j/phi rows are supplied, including both poles and equators.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    folder = args.output_dir
    folder.mkdir(parents=True, exist_ok=True)
    records = folder / "synthetic_records"
    records.mkdir(exist_ok=True)
    source = folder / "synthetic_ar20_fullgrid.source17"
    if source.exists():
        raise SystemExit("Fresh source output is required")
    theta = [0.0, math.pi / 6, math.pi / 3, math.pi / 2]
    phi = [-math.pi + 2 * math.pi * k / 4 for k in range(4)]
    rows = []
    text = [
        "SLURRY_REFERENCE_SOURCE17_V1",
        "MODEL JANOSCHEK_MATCHED_HARDFLOOR_FULL17_V4",
        "CONVENTION slurry_full17_c_eta_1_v1",
        "STATUS PROVISIONAL 0",
        "UNITS 1.5e-7 1e-9",
        "AR 20",
        "POSES 64",
    ]
    raw = " ".join("1000000" if i == j else "0" for i in range(17) for j in range(17))
    for ti in theta:
        for tj in theta:
            for azimuth in phi:
                index = len(rows)
                record = {
                    "schema": "synthetic_reference17_contract_row_v1",
                    "synthetic": True,
                    "physical_accuracy_qualified": False,
                    "contains_stokes_solve": False,
                    "matrix_definition": "1000000 * I17; arbitrary covariant SPD anchor; not physical resistance",
                    "aspect_ratio": 20,
                    "theta_i": ti,
                    "theta_j": tj,
                    "phi": azimuth,
                    "anchor_h_over_c": 1,
                }
                encoded = (json.dumps(record, indent=2, sort_keys=True) + "\n").encode()
                name = f"pose_{index:03d}.synthetic.json"
                (records / name).write_bytes(encoded)
                rows.append({"index": index, "synthetic_json": name, "sha256": digest(encoded)})
                text.extend([f"POSE {ti:.17g} {tj:.17g} {azimuth:.17g} 1 {digest(encoded)} -", "RAW17", raw, "END_POSE"])
    text.append("END")
    payload = ("\n".join(text) + "\n").encode()
    source.write_bytes(payload)
    manifest = {
        "schema": "synthetic_reference17_fullgrid_fixture_v1",
        "synthetic": True,
        "physical_accuracy_qualified": False,
        "contains_stokes_solve": False,
        "scope": "64-row AR20 full-grid generator, pole/equator, rotation, floor and runtime transport contract only",
        "source": source.name,
        "source_sha256": digest(payload),
        "matrix_definition": "1000000 * I17 for each anchor, not a physical Stokes result",
        "theta_count": 4,
        "phi_count": 4,
        "pose_count": len(rows),
        "source_json_hashes_are_synthetic_records": True,
        "rows": rows,
    }
    (folder / "synthetic_fixture.manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(source)


if __name__ == "__main__":
    main()
