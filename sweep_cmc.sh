#!/usr/bin/env bash
# run_cmc.sbatch와 같은 디렉토리에 두고: bash sweep_cmc.sh
# 제출 없이 복사본만 생성하려면: bash sweep_cmc.sh --dry-run
# 각 복사본의 COHESION_WORK와 BACKGROUND_HAMAKER만 변경합니다.
# CMC_TOTAL, 매질 점도 계산, solver 및 나머지 설정은 원본 그대로입니다.
set -Eeuo pipefail

# 각 줄은 "COHESION_WORK [J/m^2] BACKGROUND_HAMAKER [J]" 순서쌍입니다.
# 아래는 편집용 예시입니다. 원하는 순서쌍으로 수정하세요.
# 전체 조합(cartesian product)이 아니라 아래에 적은 쌍만 제출합니다.
# BACKGROUND_HAMAKER는 폴더 표시값 9.9가 아니라 실제 SI 값 9.9e-20을 적습니다.
PAIRS=(
    "0.00212434877286055 9.9e-20"
    "0 9.9e-20"
    "0 9.9e-21"
    "0 0"
)

DRY_RUN=0
case "${1:-}" in
    "") ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help)
        printf 'Edit PAIRS in this file, then run: bash sweep_cmc.sh [--dry-run]\n'
        exit 0 ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; exit 1 ;;
esac
(( $# == 0 )) || { printf 'Unexpected arguments.\n' >&2; exit 1; }

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE="$ROOT/run_cmc.sbatch"
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
[[ -f "$TEMPLATE" ]] || die "Missing template: $TEMPLATE"
(( ${#PAIRS[@]} > 0 )) || die 'PAIRS is empty.'

# 제출 전에 전체 입력을 확인합니다. 템플릿을 실행하거나 source하지 않습니다.
awk '
    /^[[:space:]]*COHESION_WORK=/ { work++ }
    /^[[:space:]]*BACKGROUND_HAMAKER=/ { hamaker++ }
    END { exit !(work == 1 && hamaker == 1) }
' "$TEMPLATE" || die 'Template must contain exactly one COHESION_WORK= and one BACKGROUND_HAMAKER= assignment.'

# 빈 OUTPUT_DIR이면 runner가 job ID별로 별도 결과 폴더를 만듭니다.
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

WORKS=()
HAMAKERS=()
for pair in "${PAIRS[@]}"; do
    read -r work hamaker extra <<< "$pair"
    [[ -n "$work" && -n "$hamaker" && -z "$extra" ]] || die "Expected one COHESION_WORK BACKGROUND_HAMAKER pair: $pair"
    awk -v work="$work" -v hamaker="$hamaker" 'BEGIN {
        number="^([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$"
        exit !(work ~ number && hamaker ~ number && work+0 >= 0 && hamaker+0 >= 0 \
               && work+0 <= 1.7976931348623157e308 && hamaker+0 <= 1.7976931348623157e308)
    }' || die "Invalid pair: $pair (requires finite nonnegative values in J/m^2 and J)."
    WORKS+=("$work")
    HAMAKERS+=("$hamaker")
done
bash -n "$TEMPLATE"
if (( ! DRY_RUN )); then
    command -v sbatch >/dev/null || die 'sbatch not found; run on the cluster login node.'
fi

SWEEP_ROOT="/scratch/negishi/lee6456/sweeps"
mkdir -p -- "$SWEEP_ROOT"
SWEEP_DIR="$(mktemp -d "$SWEEP_ROOT/cmc_$(date +%Y%m%d_%H%M%S)_XXXXXX")"
cp -- "$TEMPLATE" "$SWEEP_DIR/template.sbatch"
SCRIPTS=()
for i in "${!WORKS[@]}"; do
    work="${WORKS[$i]}"
    hamaker="${HAMAKERS[$i]}"
    printf -v case_name '%03d_cw%s_A%s' "$((i+1))" "$work" "$hamaker"
    mkdir -- "$SWEEP_DIR/$case_name"
    script="$SWEEP_DIR/$case_name/run_cmc.sbatch"
    awk -v work="$work" -v hamaker="$hamaker" '
        /^[[:space:]]*COHESION_WORK=/ { sub(/=[^[:space:]#]*/, "=" work) }
        /^[[:space:]]*BACKGROUND_HAMAKER=/ { sub(/=[^[:space:]#]*/, "=" hamaker) }
        { print }
    ' "$SWEEP_DIR/template.sbatch" > "$script"
    bash -n "$script"
    SCRIPTS+=("$script")
done

printf 'Sweep directory: %s\n' "$SWEEP_DIR"
printf 'index\tCOHESION_WORK\tBACKGROUND_HAMAKER\tjob_id\tscript\n' > "$SWEEP_DIR/jobs.tsv"
# 복사본 디렉토리가 아니라 repository에서 제출합니다.
cd -- "$ROOT"
for i in "${!SCRIPTS[@]}"; do
    work="${WORKS[$i]}"
    hamaker="${HAMAKERS[$i]}"
    if (( DRY_RUN )); then
        job_id=DRY_RUN
    elif job_id="$(SLURRY_ROOT="$ROOT" sbatch --parsable --chdir="$ROOT" "${SCRIPTS[$i]}")"; then
        :
    else
        printf '%s\t%s\t%s\tFAILED\t%s\n' "$((i+1))" "$work" "$hamaker" "${SCRIPTS[$i]}" >> "$SWEEP_DIR/jobs.tsv"
        die "Submission failed. Previously accepted jobs remain queued; see $SWEEP_DIR/jobs.tsv"
    fi
    printf '%s\t%s\t%s\t%s\t%s\n' "$((i+1))" "$work" "$hamaker" "$job_id" "${SCRIPTS[$i]}" >> "$SWEEP_DIR/jobs.tsv"
    printf 'COHESION_WORK=%s J/m^2 BACKGROUND_HAMAKER=%s J -> %s\n' "$work" "$hamaker" "$job_id"
done
printf 'Recorded %s case(s): %s/jobs.tsv\n' "${#SCRIPTS[@]}" "$SWEEP_DIR"
