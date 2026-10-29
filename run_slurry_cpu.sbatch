#!/bin/bash
#SBATCH --job-name=slurry
#SBATCH --account=ardekani
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --hint=nomultithread
#SBATCH --mem=32G
#SBATCH --time=7-00:00:00
#SBATCH --output=slurry_%j.out

# Edit this section, then submit from the slurry repository: sbatch run_slurry_cpu.sbatch
# Exactly one simulation is launched. No command-line parameters are needed.
PARTICLE_SHAPE=spheroid               # this contact-overlap model supports spheroids
AR=5                                 # supported range: 1 through 5
PHI=0.34
COUNT=1000
STEPS=200000
OUTPUT_INTERVAL=400
TIMESTEP=2.5e-6  # s; nearfield frozen-cap IMEX
AVERAGE_START_STRAIN=5
# If reducing TIMESTEP tenfold, increase STEPS tenfold for the same total strain.

NOMINAL_DIAMETER_UM=8                 # volume-equivalent D; fixed while AR changes
SIZE_RATIO=1.4
SIZE_DISTRIBUTION=bidisperse
LUBRICATION_EPSILON=0.001             # additive gap epsilon*(D/2) = 4 nm at D=8 um
DENSITY=2400
VISCOSITY=0.01
SHEAR_RATE=100
FRICTION=10                          # strong-friction trial; saturation must be checked
NORMAL_STIFFNESS=100  # N/m
TANGENTIAL_STIFFNESS=100  # N/m
ROLLING_FRICTION=10                  # strong-friction trial; not a measured graphite value
ROLLING_STIFFNESS=100  # N/m

INITIALIZATION=random                 # random or aligned; no automatic switch
SEED=1729
INITIAL_PHI=0.05
GROWTH_FRACTION=0.02
OVERLAP_TOLERANCE=1e-6
MAX_RELAX=20000

MOBILITY_BACKEND=nearfield  # isolated spheroid drag + local lubrication; no FFT
SPECTRAL=2
SPECTRAL_TOLERANCE=1e-8
SOLVER_TOLERANCE=1e-6
KRYLOV_ITERATIONS=1000

OUTPUT_DIR=""                         # empty: runs/<shape>_ar<AR>_phi<PHI>_n<COUNT>_<job ID>
BUILD_ENV=""                          # empty: build/lammps/env.sh
PREPARE_ONLY=0
TRACE_SOLVER=0
IMEX_MAX_ITERATIONS=1000  # total nonlinear updates per inner IMEX solve
export SLURRY_IMEX_MAX_ITERATIONS="$IMEX_MAX_ITERATIONS"
# End of run settings. Resource requests are the #SBATCH lines above.

set -Eeuo pipefail
if (( $# != 0 )); then
    printf 'Edit the settings in run_slurry_cpu.sbatch, then submit it without arguments.\n' >&2
    exit 2
fi
case "$PARTICLE_SHAPE" in
    spheroid) ;;
    *) printf 'PARTICLE_SHAPE must be spheroid for the contact-overlap model.\n' >&2; exit 2 ;;
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
    --size-distribution "$SIZE_DISTRIBUTION" --size-ratio "$SIZE_RATIO"
    --ar "$AR" --phi "$PHI" --count "$COUNT" --steps "$STEPS"
    --output-interval "$OUTPUT_INTERVAL" --timestep "$TIMESTEP"
    --average-start-strain "$AVERAGE_START_STRAIN"
    --nominal-diameter-um "$NOMINAL_DIAMETER_UM" --density "$DENSITY"
    --viscosity "$VISCOSITY" --shear-rate "$SHEAR_RATE"
    --lubrication-epsilon "$LUBRICATION_EPSILON"
    --friction "$FRICTION" --normal-stiffness "$NORMAL_STIFFNESS"
    --tangential-stiffness "$TANGENTIAL_STIFFNESS"
    --rolling-friction "$ROLLING_FRICTION" --rolling-stiffness "$ROLLING_STIFFNESS"
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
