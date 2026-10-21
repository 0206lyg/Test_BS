#!/bin/bash
#SBATCH --job-name=slurry_bridge
#SBATCH --account=ardekani
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=14-00:00:00
#SBATCH --output=slurry_bridge_%j.out

# New compact-bridge path: no basis or resistance-table generation.
# sbatch lammps_slurry/run_bridge_cpu.sbatch --output runs/ar20_bridge --ar 20 --count 108 --phi 0.24 --steps 100
set -Eeuo pipefail
SLURRY_BASE="${SLURRY_ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}}"
if [[ "$(basename -- "$SLURRY_BASE")" == lammps_slurry ]]; then
  SLURRY_BASE="$(cd -- "$SLURRY_BASE/.." && pwd)"
fi
if [[ ! -f "$SLURRY_BASE/lammps_slurry/tools/run_bridge.py" ]]; then
  printf 'Cannot find Test_BS compact-bridge sources in %s; set SLURRY_ROOT.\n' "$SLURRY_BASE" >&2
  exit 2
fi
# Cluster modules are optional for direct local execution. The normal project
# build on the user's cluster uses these same module defaults.
if type module >/dev/null 2>&1; then
  module purge
  module load "${SLURRY_PYTHON_MODULE:-anaconda/2024.02-py311}"
  module load "${SLURRY_GCC_MODULE:-gcc/12.2.0}"
  module load "${SLURRY_MPI_MODULE:-openmpi/4.1.4}"
fi
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
exec python3 "$SLURRY_BASE/lammps_slurry/tools/run_bridge.py" "$@"
