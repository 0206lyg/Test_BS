#!/usr/bin/env python3
"""90k CMC mass balance and Fig. 2(c) medium viscosity; no force model.

Concentrations are g/L of liquid. Adsorption is the project's nominal
Langmuir closure constrained by qmax=2.8 and adsorbed(4)=2.7 g/L. Viscosity
nodes are digitized ACS Nano 10.1021/acsnano.6c10201 Fig. 2(c) values, with
equal-weight monotonic pooling at 4/5 and 8/9 g/L. See the root README.
"""
import argparse
import json
import math


MODEL_ID = "cmc90k_fig2c_medium_v1"
SOURCE_DOI = "10.1021/acsnano.6c10201"
QMAX_G_L = 2.8
KD_G_L = 1.3 * (QMAX_G_L / 2.7 - 1.)
MAX_TOTAL_G_L = 16.
# (free CMC g/L, absolute medium viscosity mPa s), including water anchor.
MEDIUM_MPA_S = (
    (0., 0.89),
    (1.5, 2.2200536928048247),
    (3., 2.6320341855240157),
    (4., 3.1726500559930906),
    (5., 3.1726500559930906),
    (6., 3.7900857542108404),
    (7., 4.467310796917457),
    (8., 4.924922244091771),
    (9., 4.924922244091771),
    (10., 5.864917005417395),
    (12., 6.498789916280678),
    (16., 8.927466453848531),
)


def resolve(total_g_l):
    """Return composition and viscosity in SI; never compute attraction."""
    try:
        total = float(total_g_l)
    except (TypeError, ValueError):
        raise ValueError("CMC_TOTAL must be a finite number in [0, 16] g/L")
    if not math.isfinite(total) or not 0. <= total <= MAX_TOTAL_G_L:
        raise ValueError("CMC_TOTAL must be a finite number in [0, 16] g/L")
    # Positive root of cf^2 + (qmax + Kd - ct)*cf - Kd*ct = 0.
    # Rationalize when b >= 0 to avoid cancellation at small total CMC.
    b = QMAX_G_L + KD_G_L - total
    root = math.sqrt(b * b + 4. * KD_G_L * total)
    free = (2. * KD_G_L * total / (root + b)
            if b >= 0. else .5 * (root - b))
    for (c0, eta0), (c1, eta1) in zip(MEDIUM_MPA_S, MEDIUM_MPA_S[1:]):
        if free <= c1:
            viscosity = (eta0 + (eta1 - eta0) * (free - c0) / (c1 - c0)) * 1e-3
            break
    else:
        raise ValueError("Free CMC lies outside the viscosity table")
    return {
        "model_id": MODEL_ID,
        "source_doi": SOURCE_DOI,
        "total_g_l": total,
        "adsorbed_g_l": total - free,
        "free_g_l": free,
        "viscosity_Pa_s": viscosity,
        "concentration_basis": "g/L of liquid",
        "adsorption": {"type": "nominal Langmuir mass balance",
                       "qmax_g_l": QMAX_G_L, "Kd_g_l": KD_G_L},
        "viscosity_source": "Fig. 2(c), 90k, eta_water*(1+eta_specific)",
        "viscosity_interpolation": "piecewise linear in free CMC and absolute viscosity",
        "force_mapping": "none; contact work and background amplitude are manual inputs",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total", type=float, required=True, help="total CMC, g/L of liquid (0--16)")
    parser.add_argument("--shell-values", action="store_true",
                        help="emit adsorbed_g_l free_g_l viscosity_Pa_s for the sbatch launcher")
    args = parser.parse_args()
    try:
        result = resolve(args.total)
    except ValueError as error:
        parser.error(str(error))
    if args.shell_values:
        print("{adsorbed_g_l:.17g} {free_g_l:.17g} {viscosity_Pa_s:.17g}".format(**result))
    else:
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
