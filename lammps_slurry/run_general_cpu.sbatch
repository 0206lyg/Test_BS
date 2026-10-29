#!/bin/bash
#SBATCH --job-name=slurry_general
#SBATCH --account=ardekani
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=14-00:00:00
#SBATCH --output=slurry_general_%j.out

# Submit from the repository root. Defaults are a short integration run;
# override steps, preparation and averaging explicitly for a production run.
# Pair geometry and general hydrodynamic quadrature are distributed over MPI
# ranks. The implicit solve runs once on rank 0; no OpenMP threads are needed.
set -Eeuo pipefail
SLURRY_GENERAL_BASE="${SLURRY_ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}}"
if [[ "$(basename -- "$SLURRY_GENERAL_BASE")" == lammps_slurry ]]; then
    SLURRY_GENERAL_BASE="$(cd -- "$SLURRY_GENERAL_BASE/.." && pwd)"
fi
if [[ ! -f "$SLURRY_GENERAL_BASE/lammps_slurry/run_slurry_cpu.sbatch" ]]; then
    printf 'Submit from the repository root or export SLURRY_ROOT. Missing launcher in %s\n' "$SLURRY_GENERAL_BASE" >&2
    exit 2
fi
export SLURRY_ROOT="$SLURRY_GENERAL_BASE"
exec bash "$SLURRY_GENERAL_BASE/lammps_slurry/run_slurry_cpu.sbatch" \
    --lubrication-model general --ar 2 --count 300 \
    --steps 20 --prep-steps 0 --output-interval 1 --average-start-strain 0 \
    "$@"
