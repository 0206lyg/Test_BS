#!/usr/bin/env python3
"""Expose the IMEX nonlinear ceiling without replacing existing run settings."""
import argparse
from pathlib import Path
import re
import shutil


def update(original):
    """Return patched bytes; preserve all existing bytes and custom values."""
    text = original.decode("utf-8", errors="surrogateescape")
    newline = "\r\n" if "\r\n" in text else "\n"
    anchors = list(re.finditer(r"(?m)^# End of run settings\.[^\r\n]*", text))
    if not anchors:
        anchors = list(re.finditer(r"(?m)^set -Eeuo pipefail[ \t]*(?=\r?$)", text))
    if len(anchors) != 1:
        raise ValueError("Expected one '# End of run settings.' or 'set -Eeuo pipefail' anchor; file unchanged")
    position = anchors[0].start()
    assignments = list(re.finditer(r"(?m)^[ \t]*(?:export[ \t]+)?IMEX_MAX_ITERATIONS=[^\r\n]*", text))
    if len(assignments) > 1:
        raise ValueError("Repeated IMEX_MAX_ITERATIONS assignment; file unchanged")
    if assignments and assignments[0].start() >= position:
        raise ValueError("IMEX_MAX_ITERATIONS must appear in the settings section; file unchanged")
    exports = list(re.finditer(r"(?m)^[ \t]*export[ \t]+SLURRY_IMEX_MAX_ITERATIONS=\"\$IMEX_MAX_ITERATIONS\"[ \t]*(?:#[^\r\n]*)?(?=\r?$)", text))
    environment_assignments = list(re.finditer(r"(?m)^[ \t]*(?:export[ \t]+)?SLURRY_IMEX_MAX_ITERATIONS=[^\r\n]*", text))
    if len(environment_assignments) != len(exports) or len(exports) > 1:
        raise ValueError("Existing SLURRY_IMEX_MAX_ITERATIONS assignment is not the supported export; file unchanged")
    if exports and (not assignments or exports[0].start() < assignments[0].start()):
        raise ValueError("Existing IMEX export must follow its settings assignment; file unchanged")
    insertion = []
    if not assignments:
        insertion.append("IMEX_MAX_ITERATIONS=1000  # total nonlinear updates per inner IMEX solve")
    if not exports:
        insertion.append('export SLURRY_IMEX_MAX_ITERATIONS="$IMEX_MAX_ITERATIONS"')
    if not insertion:
        return original
    result = text[:position] + newline.join(insertion) + newline + text[position:]
    return result.encode("utf-8", errors="surrogateescape")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", nargs="?", type=Path,
                        default=Path(__file__).resolve().parent / "run_slurry_cpu.sbatch")
    args = parser.parse_args()
    path = args.file.resolve()
    try:
        original = path.read_bytes()
        revised = update(original)
        if revised != original:
            backup = path.with_name(path.name + ".before_solver_limits_update")
            try:
                with backup.open("xb") as output:
                    output.write(original)
            except FileExistsError:
                pass
            else:
                shutil.copystat(path, backup)
            path.write_bytes(revised)  # Writing the existing file retains its mode.
    except (OSError, ValueError) as error:
        parser.exit(1, "Solver limit update failed: {}\n".format(error))
    print("IMEX_MAX_ITERATIONS is exported to the solver; existing values and other run settings are preserved.")


if __name__ == "__main__":
    main()
