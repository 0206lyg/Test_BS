#!/usr/bin/env bash
# run_slurry_practical_cpu.sbatch와 같은 디렉토리에 두고 실행:
#   bash sweep_practical_slurry.sh
# 제출 없이 복사본만 생성: bash sweep_practical_slurry.sh --dry-run
# 원본은 수정하지 않습니다. 각 복사본의 COHESION_WORK 값만 변경합니다.
set -Eeuo pipefail

# COHESION_WORK [J/m^2]: 원하는 값으로 수정하세요.
# 예시: 기존 0.0212434877286055의 100%, 50%, 25%, 10%, 0%.
# COHESION_RANGE, BACKGROUND_HAMAKER와 나머지 설정은 원본 그대로입니다.
# 주의: 0으로 해도 BACKGROUND_HAMAKER에 의한 인력은 남습니다.
COHESION_WORK_VALUES=(
    0.0212434877286055
    0.01062174386430275
    0.005310871932151375
    0.00212434877286055
    0
)

DRY_RUN=0
case "${1:-}" in
    "") ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help)
        printf 'Edit COHESION_WORK_VALUES in this file, then run: bash sweep_practical_slurry.sh [--dry-run]\n'
        exit 0 ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; exit 1 ;;
esac
(( $# == 0 )) || { printf 'Unexpected arguments.\n' >&2; exit 1; }

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE="$ROOT/run_slurry_practical_cpu.sbatch"
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
[[ -f "$TEMPLATE" ]] || die "Missing template: $TEMPLATE"
(( ${#COHESION_WORK_VALUES[@]} > 0 )) || die 'COHESION_WORK_VALUES is empty.'

# 제출 전에 확인합니다. 템플릿을 실행하거나 source하지 않습니다.
awk '
    /^[[:space:]]*COHESION_WORK=/ { count++ }
    END { exit !(count == 1) }
' "$TEMPLATE" || die 'Template must contain exactly one COHESION_WORK= assignment.'

# 빈 OUTPUT_DIR이면 기존 runner가 job ID별로 별도 결과 폴더를 만듭니다.
awk '
    /^OUTPUT_DIR=/ {
        count++
        value=$0
        sub(/^[[:space:]]*OUTPUT_DIR=/, "", value)
        sub(/[[:space:]]*#.*/, "", value)
        gsub(/[[:space:]]/, "", value)
        empty=(value == "\"\"" || value == "\047\047" || value == "")
    }
    END { exit !(count == 1 && empty) }
' "$TEMPLATE" || die 'Set OUTPUT_DIR="" in the template so each job gets its own result folder.'

for work in "${COHESION_WORK_VALUES[@]}"; do
    awk -v work="$work" 'BEGIN {
        number="^([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$"
        exit !(work ~ number && work+0 >= 0)
    }' || die "Invalid COHESION_WORK: $work (requires a nonnegative number in J/m^2)."
done
bash -n "$TEMPLATE"
if (( ! DRY_RUN )); then
    command -v sbatch >/dev/null || die 'sbatch not found; run on the cluster login node.'
fi

mkdir -p -- "$ROOT/sweeps"
SWEEP_DIR="$(mktemp -d "$ROOT/sweeps/cohesion_$(date +%Y%m%d_%H%M%S)_XXXXXX")"
cp -- "$TEMPLATE" "$SWEEP_DIR/template.sbatch"
SCRIPTS=()
for i in "${!COHESION_WORK_VALUES[@]}"; do
    work="${COHESION_WORK_VALUES[$i]}"
    printf -v case_name '%03d_cw%s' "$((i+1))" "$work"
    mkdir -- "$SWEEP_DIR/$case_name"
    script="$SWEEP_DIR/$case_name/run_slurry_practical_cpu.sbatch"
    awk -v work="$work" '
        /^[[:space:]]*COHESION_WORK=/ { sub(/=[^[:space:]#]*/, "=" work) }
        { print }
    ' "$SWEEP_DIR/template.sbatch" > "$script"
    bash -n "$script"
    SCRIPTS+=("$script")
done

printf 'Sweep directory: %s\n' "$SWEEP_DIR"
printf 'index\tCOHESION_WORK\tjob_id\tscript\n' > "$SWEEP_DIR/jobs.tsv"
# 복사본 디렉토리가 아니라 repository에서 제출합니다.
cd -- "$ROOT"
for i in "${!SCRIPTS[@]}"; do
    work="${COHESION_WORK_VALUES[$i]}"
    if (( DRY_RUN )); then
        job_id=DRY_RUN
    elif job_id="$(SLURRY_ROOT="$ROOT" sbatch --parsable --chdir="$ROOT" "${SCRIPTS[$i]}")"; then
        :
    else
        printf '%s\t%s\tFAILED\t%s\n' "$((i+1))" "$work" "${SCRIPTS[$i]}" >> "$SWEEP_DIR/jobs.tsv"
        die "Submission failed. Previously accepted jobs remain queued; see $SWEEP_DIR/jobs.tsv"
    fi
    printf '%s\t%s\t%s\t%s\n' "$((i+1))" "$work" "$job_id" "${SCRIPTS[$i]}" >> "$SWEEP_DIR/jobs.tsv"
    printf 'COHESION_WORK=%s J/m^2 -> %s\n' "$work" "$job_id"
done
printf 'Recorded %s case(s): %s/jobs.tsv\n' "${#SCRIPTS[@]}" "$SWEEP_DIR"
