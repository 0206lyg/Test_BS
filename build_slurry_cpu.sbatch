#!/bin/bash
#SBATCH --job-name=build_slurry
#SBATCH --account=ardekani
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=build_slurry_%j.out

# Submit from the source root: sbatch build_slurry_cpu.sbatch
set -Eeuo pipefail
trap 'printf "LAMMPS build failed at line %s (exit %s).\n" "$LINENO" "$?" >&2' ERR
readonly SLURRY_LAUNCHER_RELEASE=20261016-path-fix
# Slurm executes a spool copy; an explicit root or submission directory takes
# precedence. Direct bash execution without Slurm uses this script's directory.
SLURRY_BASE="${SLURRY_ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}}"
if [[ "$(basename -- "$SLURRY_BASE")" == lammps_slurry && -f "$SLURRY_BASE/tools/bundle_guard.py" ]]; then
    SLURRY_BASE="$(cd -- "$SLURRY_BASE/.." && pwd)"
fi
if [[ ! -f "$SLURRY_BASE/lammps_slurry/tools/bundle_guard.py" ]]; then
    printf 'Wrong or outdated source root: %s\nThis launcher requires release %s. Change to the directory containing the updated build_slurry_cpu.sbatch and lammps_slurry/tools/bundle_guard.py, then submit again; or set SLURRY_ROOT to that directory.\n' "$SLURRY_BASE" "$SLURRY_LAUNCHER_RELEASE" >&2
    exit 1
fi
SLURRY_BASE="$(cd -- "$SLURRY_BASE" && pwd)"
printf 'Source root: %s\nLauncher release: %s\n' "$SLURRY_BASE" "$SLURRY_LAUNCHER_RELEASE"
module purge
module load "${SLURRY_PYTHON_MODULE:-anaconda/2024.02-py311}"
module load "${SLURRY_GCC_MODULE:-gcc/12.2.0}"
module load "${SLURRY_MPI_MODULE:-openmpi/4.1.4}"
module load "${SLURRY_CMAKE_MODULE:-cmake/3.23.1}"
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
python3 "$SLURRY_BASE/lammps_slurry/tools/bundle_guard.py" check-source \
    --root "$SLURRY_BASE" --expected-release "$SLURRY_LAUNCHER_RELEASE"
exec python3 "$SLURRY_BASE/lammps_slurry/tools/build.py" \
    --jobs "${SLURM_CPUS_PER_TASK:-4}" "$@"
