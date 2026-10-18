#!/bin/bash
#SBATCH --job-name=slurry_md
#SBATCH --account=ardekani
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=14-00:00:00
#SBATCH --output=slurry_md_%j.out

# Root solves are serial: use one rank by default to avoid reserving idle CPUs.
# MPI geometry partitioning remains available; multi-rank speedup is not established.
set -Eeuo pipefail

# Edit the physical conditions here, or override them after the sbatch filename:
# sbatch lammps_slurry/run_slurry_cpu.sbatch --ar 2 --phi 0.24 --shear-rate 100
# Oblate semiaxes are (AR*t/2, AR*t/2, t/2); t is the full thickness.
AR=25
PHI=0.24
SHEAR_RATE=100                 # 1/s
THICKNESS_NM=300
DENSITY=2400                   # kg/m^3 (2.4 g/cm^3)
ROUGHNESS_NM=3                 # legacy hydro gap floor; ignored by mari and spheroid
CONTACT_GAP_NM=3               # rough-surface mechanical contact starts at h < hc
FRICTION=0.1                   # Coulomb sliding coefficient
TANGENTIAL_STIFFNESS=80        # N/m; tangential contact spring
COUNT=300
VISCOSITY=0.01                 # Pa s; assumed Newtonian CMC 90k medium
NORMAL_STIFFNESS=1             # N/m; numerical normal repulsion stiffness
LUBRICATION_MODEL="${LUBRICATION_MODEL:-auto}" # preserve an explicit exported model selection
LUBRICATION_EPSILON_RELATIVE=0.001 # Mari epsilon / mean sphere radius
LUBRICATION_EPSILON_NM=0.15     # spheroid physical epsilon; fixed when AR changes
LUBRICATION_CUTOFF_RELATIVE=0.5    # Mari outer gap cutoff / mean sphere radius
FINITE_GAP_BASIS="${FINITE_GAP_BASIS:-}" # explicit experimental .ssb; mutually exclusive with table
FINITE_GAP_TABLE="${FINITE_GAP_TABLE:-}" # fixed-geometry .sfg; no extrapolation
FINITE_GAP_FAST_TABLE="${FINITE_GAP_FAST_TABLE:-}" # explicit trial .sft; missing cache needs a source
FINITE_GAP_FAST_SOURCE="${FINITE_GAP_FAST_SOURCE:-}" # offline generator basis; reused cache by default
FINITE_GAP_FAST_GENERATOR="${FINITE_GAP_FAST_GENERATOR:-}" # optional generator executable override
FINITE_GAP_FAST_FILM_GRID="${FINITE_GAP_FAST_FILM_GRID:-161}" # offline source resolution only
FINITE_GAP_FAST_NPHI="${FINITE_GAP_FAST_NPHI:-48}" # selected azimuth count; must match the table
FINITE_GAP_FAST_THREADS="${FINITE_GAP_FAST_THREADS:-1}" # offline generation; reserve matching cpus-per-task
FINITE_GAP_FAST_SOURCE_GATE="${FINITE_GAP_FAST_SOURCE_GATE:-}" # optional current physical gate report
FINITE_GAP_FAST_GATE_EXECUTABLE="${FINITE_GAP_FAST_GATE_EXECUTABLE:-}" # optional physical-check executable
FINITE_GAP_FAST_INTERPOLATION_EXECUTABLE="${FINITE_GAP_FAST_INTERPOLATION_EXECUTABLE:-}" # optional table-check executable
FINITE_GAP_FAST_BASIS_GENERATOR="${FINITE_GAP_FAST_BASIS_GENERATOR:-}" # optional actual AR-basis generator
FINITE_GAP_FAST_BASIS_DEGREE="${FINITE_GAP_FAST_BASIS_DEGREE:-12}" # candidate near order; physical gate still required
FINITE_GAP_FAST_FAR_DEGREE="${FINITE_GAP_FAST_FAR_DEGREE:-8}"
FINITE_GAP_FILM_GRID="${FINITE_GAP_FILM_GRID:-161}" # hybrid source grid; use 321 for a convergence check
FINITE_GAP_RANGE=0.5           # experimental outer gap / equatorial semiaxis
LUBRICATION_CUTOFF_NM=""       # optional absolute override; blank preserves model default
SOLVER_PRECONDITIONER=ilu      # verified ILUT backend; auto|sparse|nested|lu|ilu available
SOLVER_LAYOUT=root             # root|replicated; solve once on rank 0 by default
SOLVER_FORCING=fixed           # fixed|adaptive; final force tolerance is unchanged
SOLVER_CACHE=on                # off|on; reuse eligible preconditioner factors
SOLVER_ILU_DROP=1e-3           # positive ILUT drop tolerance
SOLVER_ILU_FILL=4              # positive ILUT fill factor
SOLVER_ILU_REFRESH=40          # iteration threshold for refreshing the ILUT factor
TIMESTEP=1e-6                  # s; coupled overdamped resistance/contact integrator
STEPS=100000                   # 0.1 s, strain 10 at SHEAR_RATE=100
OUTPUT_INTERVAL=1000
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
        --finite-gap-fast-smoke)
            SLURRY_PREPARE_OVERRIDES+=("$1")
            shift ;;
        --ar|--phi|--shear-rate|--thickness-nm|--density|--roughness-nm|--contact-gap-nm|--friction|--tangential-stiffness|--count|--viscosity|--normal-stiffness|--lubrication-model|--lubrication-epsilon-relative|--lubrication-epsilon-nm|--lubrication-cutoff-relative|--lubrication-cutoff-nm|--finite-gap-basis|--finite-gap-table|--finite-gap-fast-table|--finite-gap-fast-source|--finite-gap-fast-generator|--finite-gap-fast-film-grid|--finite-gap-fast-nphi|--finite-gap-fast-threads|--finite-gap-fast-source-gate|--finite-gap-fast-gate-executable|--finite-gap-fast-interpolation-executable|--finite-gap-fast-basis-generator|--finite-gap-fast-basis-degree|--finite-gap-fast-far-degree|--finite-gap-fast-basis-source-rings|--finite-gap-fast-basis-source-phi|--finite-gap-range|--finite-gap-film-grid|--solver-preconditioner|--solver-layout|--solver-forcing|--solver-cache|--solver-ilu-drop|--solver-ilu-fill|--solver-ilu-refresh|--timestep|--steps|--output-interval|--prep-steps|--seed|--initial-phi|--growth-fraction|--overlap-tolerance|--max-relax)
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
  --lubrication-model auto|legacy|mari|spheroid|finite_gap|finite_gap_fast
  --lubrication-epsilon-relative DELTA --lubrication-cutoff-relative RANGE
  --lubrication-epsilon-nm H
  --finite-gap-basis FILE.ssb | --finite-gap-table FILE.sfg
  --finite-gap-fast-table FILE.sft
  --finite-gap-fast-source FILE.ssb --finite-gap-fast-generator EXECUTABLE
  --finite-gap-fast-basis-generator EXECUTABLE --finite-gap-fast-basis-degree 12
  --finite-gap-fast-far-degree 8 --finite-gap-fast-basis-source-rings 96
  --finite-gap-fast-basis-source-phi 128
  --finite-gap-fast-film-grid 161 --finite-gap-fast-nphi 48
  --finite-gap-fast-threads N [--finite-gap-fast-smoke]
  --finite-gap-fast-source-gate REPORT.json --finite-gap-fast-gate-executable EXECUTABLE
  --finite-gap-fast-interpolation-executable EXECUTABLE
  --finite-gap-range LAMBDA --finite-gap-film-grid 161
  --solver-preconditioner auto|sparse|nested|lu|ilu --solver-layout root|replicated
  --solver-forcing fixed|adaptive --solver-cache off|on
  --solver-ilu-drop TOL --solver-ilu-fill FACTOR --solver-ilu-refresh ITERATIONS
  --contact-gap-nm HC --friction MU --tangential-stiffness KT
  --prep-steps N --average-start-strain STRAIN
  --output-interval N --initial-phi PHI --growth-fraction FRACTION
  --overlap-tolerance TOL --max-relax N

The preparation helper writes particles.data and in.slurry directly.
No JSON configuration file is needed.
Lubrication auto selects mari at AR=1 and spheroid otherwise, and logs its choice.
Mari defaults: epsilon=0.001*a, outer gap cutoff=0.5*a; a is sphere radius.
Spheroid defaults: physical epsilon=0.15 nm; cutoff=min(75 nm, 0.5*Rlocal),
where Rlocal is the mean of the two local harmonic curvature radii.
--lubrication-cutoff-nm sets the physical cap; relative length options are Mari-only.
Non-spherical pair and bulk validation of this extension remain pending.
finite_gap is explicit: its basis must match the selected oblate AR; AR1 uses Mari.
Exactly one basis/table is required; it is copied and hashed into the run directory.
Unsupported geometry/gaps and failed global passivity checks stop without fallback.
finite_gap_fast is a separately selected precomputed matrix-log/Fourier trial, not an auto default.
It requires a matching complete fast table. If absent, --finite-gap-fast-source
invokes the offline generator and caches its output before particle preparation.
At AR10, full generation first checks the direct near basis and matched provider at 5%.
After generation it runs a separate complete-table interpolation check at 1%.
Independent physical pair references currently cover AR10 only. Other oblate ARs
run analytic isolated and available boundary checks, then actual table interpolation
checks; their independent pair accuracy is recorded as not checked.
Source and interpolation results remain separate in finite_gap_manifest.json.
With a source but no table path, numerical-header/basis/generator/AR/grid/length
hashes select the cache. Offline film-grid resolution defaults to 161.
--finite-gap-fast-threads controls offline workers; reserve matching --cpus-per-task.
--finite-gap-fast-smoke uses a coarse connection test and is never physical validation.
The selected azimuth count defaults to 48 and must match an explicit table.
Other even resolutions require --finite-gap-fast-nphi; this does not certify them.
No runtime Stokes or film-grid solve and no old-model fallback are requested by this mode.
Successful preparation does not certify accuracy; use only with a passed trial gate.
Legacy defaults: gap floor=3 nm, outer gap cutoff=7.5 nm.
Mechanical contact defaults to 3 nm independently; use --contact-gap-nm 0
for a sphere-surface contact benchmark. An absolute cutoff overrides the relative one.
Solver defaults: ilu with root layout, fixed forcing and cache on.
ILUT drop/fill/refresh defaults are 1e-3/4/40. Optional lu uses full-Jacobian
SparseLU; nested retains the previous inner-CG preconditioner.
USAGE
            exit 0 ;;
        *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
    esac
done

readonly SLURRY_LAUNCHER_RELEASE=20261018-ar-basis-pipeline-trial
# Slurm executes a spool copy; an explicit root or submission directory takes
# precedence. Direct bash execution without Slurm uses this script's parent.
SLURRY_BASE="${SLURRY_ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}}"
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
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
python3 "$SLURRY_BASE/lammps_slurry/tools/bundle_guard.py" check-source \
    --root "$SLURRY_BASE" --expected-release "$SLURRY_LAUNCHER_RELEASE"
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
python3 "$SLURRY_BASE/lammps_slurry/tools/bundle_guard.py" check-build \
    --root "$SLURRY_BASE" --expected-release "$SLURRY_LAUNCHER_RELEASE" \
    --executable "$SLURRY_LAMMPS_EXECUTABLE" --plugin "$SLURRY_LAMMPS_PLUGIN" \
    --packer "$SLURRY_PACKER_EXECUTABLE" \
    --manifest "${SLURRY_BUILD_MANIFEST:-$SLURRY_BASE/build/lammps/build-manifest.json}"
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
if [[ -n "$FINITE_GAP_BASIS" ]]; then
    SLURRY_LUBRICATION_OVERRIDES+=(--finite-gap-basis "$FINITE_GAP_BASIS")
fi
if [[ -n "$FINITE_GAP_TABLE" ]]; then
    SLURRY_LUBRICATION_OVERRIDES+=(--finite-gap-table "$FINITE_GAP_TABLE")
fi
if [[ -n "$FINITE_GAP_FAST_TABLE" ]]; then
    SLURRY_LUBRICATION_OVERRIDES+=(--finite-gap-fast-table "$FINITE_GAP_FAST_TABLE")
fi
if [[ -n "$FINITE_GAP_FAST_SOURCE" ]]; then
    SLURRY_LUBRICATION_OVERRIDES+=(--finite-gap-fast-source "$FINITE_GAP_FAST_SOURCE")
fi
if [[ -n "$FINITE_GAP_FAST_GENERATOR" ]]; then
    SLURRY_LUBRICATION_OVERRIDES+=(--finite-gap-fast-generator "$FINITE_GAP_FAST_GENERATOR")
fi
if [[ -n "$FINITE_GAP_FAST_SOURCE_GATE" ]]; then
    SLURRY_LUBRICATION_OVERRIDES+=(--finite-gap-fast-source-gate "$FINITE_GAP_FAST_SOURCE_GATE")
fi
if [[ -n "$FINITE_GAP_FAST_GATE_EXECUTABLE" ]]; then
    SLURRY_LUBRICATION_OVERRIDES+=(--finite-gap-fast-gate-executable "$FINITE_GAP_FAST_GATE_EXECUTABLE")
fi
if [[ -n "$FINITE_GAP_FAST_INTERPOLATION_EXECUTABLE" ]]; then
    SLURRY_LUBRICATION_OVERRIDES+=(--finite-gap-fast-interpolation-executable "$FINITE_GAP_FAST_INTERPOLATION_EXECUTABLE")
fi
if [[ -n "$FINITE_GAP_FAST_BASIS_GENERATOR" ]]; then
    SLURRY_LUBRICATION_OVERRIDES+=(--finite-gap-fast-basis-generator "$FINITE_GAP_FAST_BASIS_GENERATOR")
fi
python3 "$SLURRY_BASE/lammps_slurry/tools/prepare_case.py" \
    --ar "$AR" --phi "$PHI" --shear-rate "$SHEAR_RATE" \
    --thickness-nm "$THICKNESS_NM" --density "$DENSITY" \
    --roughness-nm "$ROUGHNESS_NM" --count "$COUNT" --viscosity "$VISCOSITY" \
    --contact-gap-nm "$CONTACT_GAP_NM" --friction "$FRICTION" \
    --tangential-stiffness "$TANGENTIAL_STIFFNESS" \
    --normal-stiffness "$NORMAL_STIFFNESS" --lubrication-model "$LUBRICATION_MODEL" \
    --lubrication-epsilon-relative "$LUBRICATION_EPSILON_RELATIVE" \
    --lubrication-epsilon-nm "$LUBRICATION_EPSILON_NM" \
    --lubrication-cutoff-relative "$LUBRICATION_CUTOFF_RELATIVE" \
    --finite-gap-range "$FINITE_GAP_RANGE" \
    --finite-gap-film-grid "$FINITE_GAP_FILM_GRID" \
    --finite-gap-fast-film-grid "$FINITE_GAP_FAST_FILM_GRID" \
    --finite-gap-fast-nphi "$FINITE_GAP_FAST_NPHI" \
    --finite-gap-fast-threads "$FINITE_GAP_FAST_THREADS" \
    --finite-gap-fast-basis-degree "$FINITE_GAP_FAST_BASIS_DEGREE" \
    --finite-gap-fast-far-degree "$FINITE_GAP_FAST_FAR_DEGREE" \
    --solver-preconditioner "$SOLVER_PRECONDITIONER" --solver-layout "$SOLVER_LAYOUT" \
    --solver-forcing "$SOLVER_FORCING" --solver-cache "$SOLVER_CACHE" \
    --solver-ilu-drop "$SOLVER_ILU_DROP" --solver-ilu-fill "$SOLVER_ILU_FILL" \
    --solver-ilu-refresh "$SOLVER_ILU_REFRESH" \
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
