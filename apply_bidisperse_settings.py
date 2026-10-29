#!/usr/bin/env python3
"""Enable the bidisperse mixture in an existing run script without changing its run settings."""
import argparse
from pathlib import Path
import re
import shutil


def update(text):
    text = text.replace("# s; explicit Heun predictor-corrector",
                        "# s; nearfield frozen-cap IMEX")
    settings = {"SIZE_DISTRIBUTION": "bidisperse", "SIZE_RATIO": "1.4"}
    for key, value in settings.items():
        expression = r"(?m)^"+key+r"=[^\n]*$"
        matches = list(re.finditer(expression, text))
        if len(matches) > 1:
            raise ValueError("Repeated assignment: "+key)
        if matches:
            text = re.sub(expression, key+"="+value, text)
        else:
            anchor = r"(?m)^NOMINAL_DIAMETER_UM=[^\n]*$"
            if len(re.findall(anchor, text)) != 1:
                raise ValueError("Expected one NOMINAL_DIAMETER_UM assignment")
            text = re.sub(anchor, lambda match: match.group(0)+"\n"+key+"="+value, text)
    for option in ("--size-distribution", "--size-ratio"):
        count = text.count(option)
        if count > 1:
            raise ValueError("Repeated option: "+option)
    if "--size-distribution" not in text and "--size-ratio" not in text:
        anchor = '    --shape "$PARTICLE_SHAPE"'
        if text.count(anchor) != 1:
            raise ValueError("Expected one --shape argument in SLURRY_ARGUMENTS")
        text = text.replace(anchor, anchor+'\n    --size-distribution "$SIZE_DISTRIBUTION" --size-ratio "$SIZE_RATIO"')
    elif not all(option in text for option in ("--size-distribution", "--size-ratio")):
        raise ValueError("Existing size options are incomplete")
    elif ('--size-distribution "$SIZE_DISTRIBUTION"' not in text or
          '--size-ratio "$SIZE_RATIO"' not in text):
        raise ValueError("Existing size options must refer to SIZE_DISTRIBUTION and SIZE_RATIO")
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=Path,
                        default=Path(__file__).resolve().parent/"run_slurry_cpu.sbatch")
    args = parser.parse_args()
    path = args.file.resolve()
    original = path.read_text()
    revised = update(original)
    if revised != original:
        backup = path.with_name(path.name+".before_bidisperse_update")
        if not backup.exists():
            shutil.copy2(path, backup)
        path.write_text(revised)
    print("Enabled size-distribution=bidisperse, size-ratio=1.4.")
    print("Particle count, AR, phi, stiffnesses, timestep, duration, output, paths and resources preserved.")


if __name__ == "__main__":
    main()
