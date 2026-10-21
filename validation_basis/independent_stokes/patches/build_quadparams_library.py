#!/usr/bin/env python3
"""Replace one object in a COPY of the pinned archive; never touch its build."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess

REVISION = "be6b2b3700b77e9cbf20c512ff4f0922aaee45d9"
SOURCE_SHA256 = "298c2580f8129f7a998394c4323ef6c19f33899661b7082e2d13c63b898e2241"
POLICY = "get_quadparams_adap_high_order_p_ge_4_v1"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--upstream", type=Path, required=True)
    ap.add_argument("--output-prefix", type=Path, required=True)
    ap.add_argument("--fc", required=True)
    args = ap.parse_args()
    upstream = args.upstream.resolve()
    output = args.output_prefix.resolve()
    compiler = str(Path(args.fc).resolve())
    blas_link_flags = os.environ.get('REFERENCE_BLAS', '-lblas -llapack')
    patch = Path(__file__).with_name("fmm3dbie-get-quadparams-high-order-v1.patch")
    source = upstream / "src/quadratures/near_field_routs.f"
    archive = upstream / "lib-static/libfmm3dbie.a"
    library = upstream / "lib/libfmm3dbie.so"
    revision = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    if revision != REVISION or sha(source) != SOURCE_SHA256:
        raise SystemExit("Pinned upstream revision or source hash does not match; refusing patch")
    subprocess.run(["git", "-C", str(upstream), "diff", "--exit-code", "HEAD", "--"], check=True)
    members = subprocess.check_output(["ar", "t", str(archive)], text=True).splitlines()
    if members.count("near_field_routs.o") != 1:
        raise SystemExit("Expected exactly one near_field_routs.o archive member")
    original_hashes = {str(p): sha(p) for p in (source, archive, library)}
    # No existing prefix is reused: an interrupted build cannot silently replace a library.
    output.mkdir(parents=True, exist_ok=False)
    (output / "src/quadratures").mkdir(parents=True)
    (output / "lib-static").mkdir()
    (output / "lib").mkdir()
    (output / ".mod").mkdir()
    patched_source = output / "src/quadratures/near_field_routs.f"
    patched_archive = output / "lib-static/libfmm3dbie.a"
    patched_library = output / "lib/libfmm3dbie.so"
    shutil.copy2(source, patched_source)
    shutil.copy2(archive, patched_archive)
    shutil.copy2(patch, output / patch.name)
    commands = []

    def run(command):
        commands.append(command)
        with (output / "build.log").open("a") as log:
            log.write("COMMAND " + repr(command) + "\n")
            log.flush()
            subprocess.run(command, cwd=output, stdout=log, stderr=subprocess.STDOUT, check=True)

    run(["patch", "--batch", "--fuzz=0", "-p1", "-i", str(patch)])
    flags = ["-fPIC", "-O3", "-std=legacy", "-w", "-fopenmp", "-J", str(output / ".mod")]
    object_path = output / "near_field_routs.o"
    run([compiler, *flags, "-c", str(patched_source), "-o", str(object_path)])
    run(["ar", "r", str(patched_archive), str(object_path)])
    run(["ranlib", str(patched_archive)])
    run([compiler, "-shared", "-fPIC", "-fopenmp", "-Wl,--whole-archive", str(patched_archive),
         "-Wl,--no-whole-archive", "-o", str(patched_library), "-lm", "-lstdc++", "-lgomp", *shlex.split(blas_link_flags)])
    # Confirm the archive operation changed only the one intended object.
    new_members = subprocess.check_output(["ar", "t", str(patched_archive)], text=True).splitlines()
    if members != new_members:
        raise RuntimeError("Archive member list changed")
    changed_members = []
    for member in members:
        before = subprocess.check_output(["ar", "p", str(archive), member])
        after = subprocess.check_output(["ar", "p", str(patched_archive), member])
        if before != after:
            changed_members.append(member)
    if changed_members != ["near_field_routs.o"]:
        raise RuntimeError(f"Unexpected changed archive members: {changed_members}")
    if any(sha(p) != expected for p, expected in original_hashes.items()):
        raise RuntimeError("Original upstream artifacts changed during the build")
    manifest = {
        "schema": "fmm3dbie_single_object_patch_build_v1", "upstream_revision": revision,
        "patch_policy": POLICY, "supported_panel_order_minimum": 4,
        "physical_accuracy_qualified": False, "compiler": compiler,
        "compiler_version": subprocess.check_output([compiler, "--version"], text=True).splitlines()[0],
        "compile_flags": flags, "blas_link_flags": blas_link_flags, "commands": commands, "upstream_files_sha256": original_hashes,
        "patch_sha256": sha(patch), "patched_source_sha256": sha(patched_source),
        "patched_object_sha256": sha(object_path), "patched_archive_sha256": sha(patched_archive),
        "library": str(patched_library), "library_sha256": sha(patched_library),
        "changed_archive_members": changed_members, "unchanged_archive_member_count": len(members) - 1,
        "original_artifacts_unchanged": True,
    }
    manifest_path = output / "patch_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"library": str(patched_library), "sha256": sha(patched_library), "manifest": str(manifest_path)}))


if __name__ == "__main__":
    main()
