#!/bin/bash
#SBATCH --job-name=cmc
#SBATCH --account=ardekani
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --hint=nomultithread
#SBATCH --mem=4G
#SBATCH --time=7-00:00:00
#SBATCH --output=/scratch/negishi/lee6456/cmc_%j.out

# Edit this section, then submit from the slurry repository: sbatch run_cmc.sbatch
# Exactly one simulation is launched. No command-line parameters are needed.
CMC_TOTAL=4  # g/L of liquid, 90k CMC; supported range 0--16
# Only medium viscosity is computed from CMC. Both attraction inputs stay manual.
PARTICLE_SHAPE=spheroid               # this contact-overlap model supports spheroids
AR=4
PHI=0.24
COUNT=1000
STEPS=500000000  # strain 50 at 0.1/s and dt=1e-6; 500 s of physical time
OUTPUT_INTERVAL=400
TIMESTEP=1e-6  # s; nominal IMEX timestep
AVERAGE_START_STRAIN=5
# Recover Newton failures and excessive overlap at the same dt first.
# If those recovery attempts fail, the existing limited dt subdivision remains.

NOMINAL_DIAMETER_UM=8                 # volume-equivalent D; fixed while AR changes
SIZE_RATIO=1.4
SIZE_DISTRIBUTION=bidisperse
LUBRICATION_EPSILON=0.001             # additive gap epsilon*(D/2) = 4 nm at D=8 um
DENSITY=2400
# VISCOSITY is calculated from free CMC using ACS Nano Fig. 2(c).
SHEAR_RATE=0.1
FRICTION=10
NORMAL_STIFFNESS=900  # N/m; initial and minimum normal stiffness
TANGENTIAL_STIFFNESS=300  # N/m
ROLLING_FRICTION=10
ROLLING_STIFFNESS=300  # N/m

# Manual force amplitudes; inherited defaults are NOT a fit for CMC_TOTAL.
# Physical interactions match the current N1000 attraction benchmark settings.
# Packing matches its bidisperse 1.4-ratio, random-seed-1729 fixture (733/267).
# For contact + lubrication baseline, set COHESION_WORK=0 and BACKGROUND_HAMAKER=0.
COHESION_WORK=0.0212434877286055  # J/m^2
COHESION_RANGE=6.7e-10  # m; 0.67 nm retained
BACKGROUND_HAMAKER=9.9e-20  # J
BACKGROUND_REFERENCE_GAP=2e-9  # m; potential reference, not contact onset
BACKGROUND_SIGMA=4.197e-10  # m
SURFACE_ADAPTIVE=0  # no microscopic pair-change timestep gate in practical mode
SURFACE_ROUNDOFF=yes
SURFACE_INTEGRATOR=practical
SURFACE_LINEAR=1e-4

# Normal-stiffness control is part of this launcher; it is always enabled.
# All overlap fractions below use the pair's smaller minimum semiaxis.
# Tangential/rolling stiffness and friction coefficients are not rescaled.
KN_TARGET=0.02          # target overlap fraction after increasing kn
KN_RAISE=0.03           # increase kn after a step exceeds this fraction
KN_LOWER=0.01           # relax kn after staying below this fraction
KN_RELAX_STRAIN=0.02    # strain window for relaxation toward NORMAL_STIFFNESS
KN_MAX_FACTOR=1000      # maximum kn / NORMAL_STIFFNESS

INITIALIZATION=random                 # random or aligned; no automatic switch
SEED=1729
INITIAL_PHI=0.05
GROWTH_FRACTION=0.02
OVERLAP_TOLERANCE=1e-6
MAX_RELAX=20000

MOBILITY_BACKEND=nearfield  # isolated spheroid drag + local lubrication; no FFT
SPECTRAL=2
SPECTRAL_TOLERANCE=1e-8
SOLVER_TOLERANCE=1e-4
KRYLOV_ITERATIONS=1000

RESULTS_ROOT="/scratch/negishi/lee6456"  # default output parent; never falls back to the source tree
OUTPUT_DIR=""  # empty: cmc_c<TOTAL>_cw<WORK:.4f>_h<A/1e-20>_n<COUNT>_sr<RATE>_<JOB_ID>; relative names are under RESULTS_ROOT
BUILD_ENV=""                          # empty: build/lammps/env.sh
PREPARE_ONLY=0
TRACE_SOLVER=0
IMEX_MAX_ITERATIONS=100  # Newton updates per solve; same-dt ADMM recovery is additional
export SLURRY_IMEX_MAX_ITERATIONS="$IMEX_MAX_ITERATIONS"
# End of run settings. Resource requests are the #SBATCH lines above.

set -Eeuo pipefail
if (( $# != 0 )); then
    printf 'Edit the settings in run_cmc.sbatch, then submit it without arguments.\n' >&2
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
# Resolve composition before creating any output or reading a build.
# Command substitution propagates a failed/out-of-range CMC lookup under set -e.
CMC_VALUES="$(python3 "$SLURRY_BASE/lammps_slurry/tools/cmc_medium.py" --total "$CMC_TOTAL" --shell-values)"
read -r CMC_ADSORBED CMC_FREE VISCOSITY <<< "$CMC_VALUES"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OMP_DYNAMIC=FALSE OMP_PLACES=cores OMP_PROC_BIND=close OMP_MAX_ACTIVE_LEVELS=1
export MKL_NUM_THREADS="$OMP_NUM_THREADS" MKL_DYNAMIC=FALSE OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1 SLURRY_BRIDGE_TRACE="$TRACE_SOLVER"
if [[ "$RESULTS_ROOT" != /* ]]; then
    printf 'RESULTS_ROOT must be an absolute path.\n' >&2
    exit 2
fi
if [[ -z "$OUTPUT_DIR" ]]; then
    SLURRY_RUN_ID="${SLURM_JOB_ID:-local_$(date -u +%Y%m%dT%H%M%SZ)_$$}"
    SLURRY_SR_TAG="$(LC_ALL=C awk -v value="$SHEAR_RATE" 'BEGIN { printf "%.12g", value }')"
    # Round only the directory label; pass the full COHESION_WORK to the solver.
    SLURRY_CW_TAG="$(LC_ALL=C printf '%.4f' "$COHESION_WORK")"
    SLURRY_CMC_TAG="$(LC_ALL=C awk -v value="$CMC_TOTAL" 'BEGIN { printf "%.12g", value }')"
    # h is Hamaker amplitude in units of 1e-20 J, not its scientific-notation mantissa.
    SLURRY_H_TAG="$(LC_ALL=C awk -v value="$BACKGROUND_HAMAKER" 'BEGIN { printf "%.12g", value/1e-20 }')"
    OUTPUT_DIR="$RESULTS_ROOT/cmc_c${SLURRY_CMC_TAG}_cw${SLURRY_CW_TAG}_h${SLURRY_H_TAG}_n${COUNT}_sr${SLURRY_SR_TAG}_${SLURRY_RUN_ID}"
elif [[ "$OUTPUT_DIR" != /* ]]; then
    OUTPUT_DIR="$RESULTS_ROOT/$OUTPUT_DIR"
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
    --kn-adapt yes --kn-target "$KN_TARGET" --kn-raise "$KN_RAISE"
    --kn-lower "$KN_LOWER" --kn-relax-strain "$KN_RELAX_STRAIN" --kn-max-factor "$KN_MAX_FACTOR"
    --cohesion-work "$COHESION_WORK" --cohesion-range "$COHESION_RANGE"
    --background-hamaker "$BACKGROUND_HAMAKER"
    --background-reference-gap "$BACKGROUND_REFERENCE_GAP" --background-sigma "$BACKGROUND_SIGMA"
    --surface-adaptive "$SURFACE_ADAPTIVE" --surface-roundoff "$SURFACE_ROUNDOFF"
    --surface-integrator "$SURFACE_INTEGRATOR" --surface-linear "$SURFACE_LINEAR"
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
printf 'CMC_TOTAL=%s g/L; ADSORBED_CMC=%s g/L; FREE_CMC=%s g/L; MEDIUM_VISCOSITY=%s Pa s\n' \
    "$CMC_TOTAL" "$CMC_ADSORBED" "$CMC_FREE" "$VISCOSITY"
printf 'Medium source: ACS Nano 10.1021/acsnano.6c10201 Fig. 2(c); nominal Langmuir adsorption\n'
printf 'Manual COHESION_WORK=%s J/m^2; BACKGROUND_HAMAKER=%s J\n' \
    "$COHESION_WORK" "$BACKGROUND_HAMAKER"
printf 'SHAPE=%s AR=%s PHI=%s COUNT=%s STEPS=%s THREADS=%s\nRun directory: %s\n' \
    "$PARTICLE_SHAPE" "$AR" "$PHI" "$COUNT" "$STEPS" "$OMP_NUM_THREADS" "$OUTPUT_DIR"
exec python3 "$SLURRY_BASE/lammps_slurry/tools/run_bridge.py" "${SLURRY_ARGUMENTS[@]}"


