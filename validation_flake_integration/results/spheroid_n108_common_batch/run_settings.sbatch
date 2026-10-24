#!/bin/bash
#SBATCH --job-name=slurry
#SBATCH --account=ardekani
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --mem=96G
#SBATCH --time=14-00:00:00
#SBATCH --output=slurry_%j.out

# Edit this section, then submit from the slurry repository: sbatch run_slurry_cpu.sbatch
# Exactly one simulation is launched. No command-line parameters are needed.
PARTICLE_SHAPE=spheroid
AR=20
PHI=.24
COUNT=108
STEPS=3
OUTPUT_INTERVAL=1
TIMESTEP=1e-6
AVERAGE_START_STRAIN=0

THICKNESS_NM=300
DENSITY=2400
VISCOSITY=0.01
SHEAR_RATE=100
GAP_FLOOR_NM=1
CONTACT_GAP_NM=3
FRICTION=0.1
NORMAL_STIFFNESS=100
TANGENTIAL_STIFFNESS=80

INITIALIZATION=random                 # random or aligned; no automatic switch
SEED=1729
INITIAL_PHI=0.05
GROWTH_FRACTION=0.02
OVERLAP_TOLERANCE=1e-6
MAX_RELAX=20000

MOBILITY_BACKEND=mesh                  # FFTW mesh; direct is a reference Fourier sum
SPECTRAL=2
SPECTRAL_TOLERANCE=1e-8
SOLVER_TOLERANCE=1e-6
KRYLOV_ITERATIONS=1000

OUTPUT_DIR="/workspace/scratch/0520ed9b6708/flake-common-batch-validation/spheroid"
BUILD_ENV="/workspace/scratch/0520ed9b6708/flake-build/env.sh"
PREPARE_ONLY=0
TRACE_SOLVER=0
# End of run settings. Resource requests are the #SBATCH lines above.

set -Eeuo pipefail
if (( $# != 0 )); then
    printf 'Edit the settings in run_slurry_cpu.sbatch, then submit it without arguments.\n' >&2
    exit 2
fi
case "$PARTICLE_SHAPE" in
    spheroid|flake) ;;
    *) printf 'PARTICLE_SHAPE must be spheroid or flake.\n' >&2; exit 2 ;;
esac
SLURRY_BASE="${SLURRY_ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}}"
if [[ "$(basename -- "$SLURRY_BASE")" == lammps_slurry ]]; then
    SLURRY_BASE="$(cd -- "$SLURRY_BASE/.." && pwd)"
fi
SLURRY_BASE="$(cd -- "$SLURRY_BASE" && pwd)"
if [[ ! -f "$SLURRY_BASE/lammps_slurry/tools/run_bridge.py" ]]; then
    printf 'Cannot find slurry sources in %s; submit from the repository root.\n' "$SLURRY_BASE" >&2
    exit 2
fi
if type module >/dev/null 2>&1; then
    module purge
    module load "${SLURRY_PYTHON_MODULE:-anaconda/2024.02-py311}"
    module load "${SLURRY_GCC_MODULE:-gcc/12.2.0}"
    module load "${SLURRY_MPI_MODULE:-openmpi/4.1.4}"
fi
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OMP_DYNAMIC=FALSE OMP_PLACES=cores OMP_PROC_BIND=close OMP_MAX_ACTIVE_LEVELS=1
export MKL_NUM_THREADS="$OMP_NUM_THREADS" MKL_DYNAMIC=FALSE OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1 SLURRY_BRIDGE_TRACE="$TRACE_SOLVER"
if [[ -z "$OUTPUT_DIR" ]]; then
    SLURRY_RUN_ID="${SLURM_JOB_ID:-local_$(date -u +%Y%m%dT%H%M%SZ)_$$}"
    OUTPUT_DIR="${SLURRY_RESULTS_ROOT:-$SLURRY_BASE/runs}/${PARTICLE_SHAPE}_ar${AR}_phi${PHI}_n${COUNT}_${SLURRY_RUN_ID}"
elif [[ "$OUTPUT_DIR" != /* ]]; then
    OUTPUT_DIR="$SLURRY_BASE/$OUTPUT_DIR"
fi
if [[ -z "$BUILD_ENV" ]]; then
    BUILD_ENV="$SLURRY_BASE/build/lammps/env.sh"
elif [[ "$BUILD_ENV" != /* ]]; then
    BUILD_ENV="$SLURRY_BASE/$BUILD_ENV"
fi
SLURRY_ARGUMENTS=(
    --output "$OUTPUT_DIR" --build-env "$BUILD_ENV"
    --shape "$PARTICLE_SHAPE"
    --ar "$AR" --phi "$PHI" --count "$COUNT" --steps "$STEPS"
    --output-interval "$OUTPUT_INTERVAL" --timestep "$TIMESTEP"
    --average-start-strain "$AVERAGE_START_STRAIN"
    --thickness-nm "$THICKNESS_NM" --density "$DENSITY"
    --viscosity "$VISCOSITY" --shear-rate "$SHEAR_RATE"
    --gap-floor-nm "$GAP_FLOOR_NM" --contact-gap-nm "$CONTACT_GAP_NM"
    --friction "$FRICTION" --normal-stiffness "$NORMAL_STIFFNESS"
    --tangential-stiffness "$TANGENTIAL_STIFFNESS"
    --initialization "$INITIALIZATION" --seed "$SEED" --initial-phi "$INITIAL_PHI"
    --growth-fraction "$GROWTH_FRACTION" --overlap-tolerance "$OVERLAP_TOLERANCE"
    --max-relax "$MAX_RELAX" --mobility-backend "$MOBILITY_BACKEND"
    --spectral "$SPECTRAL" --spectral-tolerance "$SPECTRAL_TOLERANCE"
    --tolerance "$SOLVER_TOLERANCE" --iterations "$KRYLOV_ITERATIONS"
)
if [[ "$PREPARE_ONLY" == 1 ]]; then
    SLURRY_ARGUMENTS+=(--prepare-only)
elif [[ "$PREPARE_ONLY" != 0 ]]; then
    printf 'PREPARE_ONLY must be 0 or 1.\n' >&2
    exit 2
fi
printf 'SHAPE=%s AR=%s PHI=%s COUNT=%s STEPS=%s THREADS=%s\nRun directory: %s\n' \
    "$PARTICLE_SHAPE" "$AR" "$PHI" "$COUNT" "$STEPS" "$OMP_NUM_THREADS" "$OUTPUT_DIR"
exec python3 "$SLURRY_BASE/lammps_slurry/tools/run_bridge.py" "${SLURRY_ARGUMENTS[@]}"
