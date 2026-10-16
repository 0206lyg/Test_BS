#!/bin/bash
#SBATCH --job-name=slurry_md
#SBATCH --account=ardekani
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=7-00:00:00
#SBATCH --output=slurry_md_%j.out

set -Eeuo pipefail

# Edit the physical conditions here, or override them after the sbatch filename:
# sbatch lammps_slurry/run_slurry_cpu.sbatch --ar 2 --phi 0.24 --shear-rate 100
# Oblate semiaxes are (AR*t/2, AR*t/2, t/2); t is the full thickness.
AR=10
PHI=0.24
SHEAR_RATE=100                 # 1/s
THICKNESS_NM=300
DENSITY=2400                   # kg/m^3 (2.4 g/cm^3)
ROUGHNESS_NM=3                 # legacy hydro gap floor; ignored by the Mari model
CONTACT_GAP_NM=3               # rough-surface mechanical contact starts at h < hc
FRICTION=0.1                   # Coulomb sliding coefficient
TANGENTIAL_STIFFNESS=80        # N/m; tangential contact spring
COUNT=300
VISCOSITY=0.01                 # Pa s; assumed Newtonian CMC 90k medium
NORMAL_STIFFNESS=1             # N/m; numerical normal repulsion stiffness
LUBRICATION_MODEL=auto         # mari at AR=1; legacy for nonspherical particles
LUBRICATION_EPSILON_RELATIVE=0.001 # Mari epsilon / mean sphere radius
LUBRICATION_CUTOFF_RELATIVE=0.5    # Mari outer gap cutoff / mean sphere radius
LUBRICATION_CUTOFF_NM=""       # optional absolute override; blank preserves model default
TIMESTEP=1e-6                  # s; coupled overdamped resistance/contact integrator
STEPS=100000                   # 0.1 s, strain 10 at SHEAR_RATE=100
OUTPUT_INTERVAL=100
AVERAGE_START_STRAIN=5         # averaging window start; not a steady-state claim
PREP_STEPS=1000                # zero-shear relaxation; 0 disables it
SEED=1729

# Initial particle growth and overlap relaxation prepare the geometry only.
# They are not physical dynamics and do not contribute to shear strain or time.
INITIAL_PHI=0.05
GROWTH_FRACTION=0.02
OVERLAP_TOLERANCE=1e-6
MAX_RELAX=20000

SLURRY_RUN=""
SLURRY_PREPARE_OVERRIDES=()
while (($#)); do
    case "$1" in
        --output)
            [[ $# -ge 2 ]] || { printf 'Missing value for %s\n' "$1" >&2; exit 2; }
            SLURRY_RUN="$2"
            shift 2 ;;
        --average-start-strain)
            [[ $# -ge 2 ]] || { printf 'Missing value for %s\n' "$1" >&2; exit 2; }
            AVERAGE_START_STRAIN="$2"
            shift 2 ;;
        --ar|--phi|--shear-rate|--thickness-nm|--density|--roughness-nm|--contact-gap-nm|--friction|--tangential-stiffness|--count|--viscosity|--normal-stiffness|--lubrication-model|--lubrication-epsilon-relative|--lubrication-cutoff-relative|--lubrication-cutoff-nm|--timestep|--steps|--output-interval|--prep-steps|--seed|--initial-phi|--growth-fraction|--overlap-tolerance|--max-relax)
            [[ $# -ge 2 ]] || { printf 'Missing value for %s\n' "$1" >&2; exit 2; }
            SLURRY_PREPARE_OVERRIDES+=("$1" "$2")
            shift 2 ;;
        --help|-h)
            cat <<'USAGE'
Usage: sbatch lammps_slurry/run_slurry_cpu.sbatch [OPTIONS]

Edit the variables at the top of this script or override them with:
  --ar AR --phi PHI --shear-rate RATE --steps N --seed N --output NEW_DIR
  --thickness-nm T --density RHO --roughness-nm H --count N --viscosity ETA
  --normal-stiffness K --lubrication-cutoff-nm H --timestep DT
  --lubrication-model auto|legacy|mari --lubrication-epsilon-relative DELTA
  --lubrication-cutoff-relative RANGE
  --contact-gap-nm HC --friction MU --tangential-stiffness KT
  --prep-steps N --average-start-strain STRAIN
  --output-interval N --initial-phi PHI --growth-fraction FRACTION
  --overlap-tolerance TOL --max-relax N

The preparation helper writes particles.data and in.slurry directly.
No JSON configuration file is needed.
Lubrication auto selects mari at AR=1 and legacy otherwise, and logs its choice.
Mari defaults: epsilon=0.001*a, outer gap cutoff=0.5*a; a is sphere radius.
Legacy defaults: gap floor=3 nm, outer gap cutoff=7.5 nm.
Mechanical contact defaults to 3 nm independently; use --contact-gap-nm 0
for a sphere-surface contact benchmark. An absolute cutoff overrides the relative one.
USAGE
            exit 0 ;;
        *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
    esac
done

module purge
module load "${SLURRY_PYTHON_MODULE:-anaconda/2024.02-py311}"
module load "${SLURRY_GCC_MODULE:-gcc/12.2.0}"
module load "${SLURRY_MPI_MODULE:-openmpi/4.1.4}"
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1

# sbatch copies this script into a spool directory: locate the repository from
# the submission directory (or explicit SLURRY_ROOT), not the copied filename.
SLURRY_BASE="${SLURRY_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
if [[ -f "$SLURRY_BASE/lammps_slurry/tools/prepare_case.py" ]]; then
    :
elif [[ -f "$SLURRY_BASE/tools/prepare_case.py" && "$(basename "$SLURRY_BASE")" == lammps_slurry ]]; then
    SLURRY_BASE="$(cd "$SLURRY_BASE/.." && pwd)"
else
    printf 'Submit from the repository root or lammps_slurry directory, or set SLURRY_ROOT.\n' >&2
    exit 1
fi
SLURRY_BASE="$(cd "$SLURRY_BASE" && pwd)"
if [[ ! -f "$SLURRY_BASE/build/lammps/env.sh" ]]; then
    printf 'Missing build/lammps/env.sh. Run sbatch build_slurry_cpu.sbatch from the repository root first.\n' >&2
    exit 1
fi
source "$SLURRY_BASE/build/lammps/env.sh"
: "${SLURRY_LAMMPS_EXECUTABLE:?Build environment did not define SLURRY_LAMMPS_EXECUTABLE}"
: "${SLURRY_LAMMPS_PLUGIN:?Build environment did not define SLURRY_LAMMPS_PLUGIN}"
SLURRY_PACKER_EXECUTABLE="${SLURRY_PACKER_EXECUTABLE:-$SLURRY_BASE/build/lammps-slurry/slurry_pack}"
[[ -x "$SLURRY_LAMMPS_EXECUTABLE" ]] || { printf 'Missing LAMMPS executable: %s\n' "$SLURRY_LAMMPS_EXECUTABLE" >&2; exit 1; }
[[ -f "$SLURRY_LAMMPS_PLUGIN" ]] || { printf 'Missing slurry plugin: %s\n' "$SLURRY_LAMMPS_PLUGIN" >&2; exit 1; }
[[ -x "$SLURRY_PACKER_EXECUTABLE" ]] || { printf 'Missing particle packer: %s. Rebuild with build_slurry_cpu.sbatch.\n' "$SLURRY_PACKER_EXECUTABLE" >&2; exit 1; }
case "$SLURRY_LAMMPS_PLUGIN" in
    *\"*|*$'\n'*|*$'\r'*) printf 'The LAMMPS plugin filename cannot contain double quotes or newlines.\n' >&2; exit 1 ;;
esac

if [[ -z "$SLURRY_RUN" ]]; then
    SLURRY_RESULTS_BASE="${SLURRY_RESULTS_ROOT:-/scratch/negishi/${USER:?USER is not set}}"
    SLURRY_RUN="$SLURRY_RESULTS_BASE/slurry_md_${SLURM_JOB_ID:-local}_$(date -u +%Y%m%dT%H%M%SZ)"
fi
SLURRY_LUBRICATION_OVERRIDES=()
if [[ -n "$LUBRICATION_CUTOFF_NM" ]]; then
    SLURRY_LUBRICATION_OVERRIDES+=(--lubrication-cutoff-nm "$LUBRICATION_CUTOFF_NM")
fi
python3 "$SLURRY_BASE/lammps_slurry/tools/prepare_case.py" \
    --ar "$AR" --phi "$PHI" --shear-rate "$SHEAR_RATE" \
    --thickness-nm "$THICKNESS_NM" --density "$DENSITY" \
    --roughness-nm "$ROUGHNESS_NM" --count "$COUNT" --viscosity "$VISCOSITY" \
    --contact-gap-nm "$CONTACT_GAP_NM" --friction "$FRICTION" \
    --tangential-stiffness "$TANGENTIAL_STIFFNESS" \
    --normal-stiffness "$NORMAL_STIFFNESS" --lubrication-model "$LUBRICATION_MODEL" \
    --lubrication-epsilon-relative "$LUBRICATION_EPSILON_RELATIVE" \
    --lubrication-cutoff-relative "$LUBRICATION_CUTOFF_RELATIVE" \
    --timestep "$TIMESTEP" --steps "$STEPS" --output-interval "$OUTPUT_INTERVAL" \
    --prep-steps "$PREP_STEPS" --average-start-strain "$AVERAGE_START_STRAIN" \
    --seed "$SEED" --initial-phi "$INITIAL_PHI" --growth-fraction "$GROWTH_FRACTION" \
    --overlap-tolerance "$OVERLAP_TOLERANCE" --max-relax "$MAX_RELAX" \
    --packer "$SLURRY_PACKER_EXECUTABLE" --output "$SLURRY_RUN" \
    "${SLURRY_LUBRICATION_OVERRIDES[@]}" \
    "${SLURRY_PREPARE_OVERRIDES[@]}"
cd "$SLURRY_RUN"
printf 'LAMMPS executable: %s\nPlugin: %s\nRun directory: %s\n' "$SLURRY_LAMMPS_EXECUTABLE" "$SLURRY_LAMMPS_PLUGIN" "$PWD"
# Slurm stdout remains in the submission directory; only run data goes to scratch.
mpirun -np "${SLURM_NTASKS:-1}" "$SLURRY_LAMMPS_EXECUTABLE" \
    -var slurry_plugin "\"$SLURRY_LAMMPS_PLUGIN\"" -in in.slurry -log log.lammps
python3 "$SLURRY_BASE/lammps_slurry/tools/summarize_rheology.py" history.csv \
    --start-strain "$AVERAGE_START_STRAIN" --output rheology_summary.txt
