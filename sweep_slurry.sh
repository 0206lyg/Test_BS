#!/usr/bin/env bash
# run_slurry_cpu.sbatch와 같은 디렉토리에 두고: bash sweep_slurry.sh
# 제출 없이 복사본만 생성하려면: bash sweep_slurry.sh --dry-run
set -Eeuo pipefail

# 각 줄은 "AR PHI" 한 쌍입니다. 원하는 조합을 추가/수정하세요.
# 전체 조합(cartesian product)이 아니라 아래에 적은 쌍만 제출합니다.
PAIRS=(
    "2 0.20"
    "2 0.30"
)

DRY_RUN=0
case "${1:-}" in
    "") ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help)
        printf 'Edit PAIRS in this file, then run: bash sweep_slurry.sh [--dry-run]\n'
        exit 0 ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; exit 1 ;;
esac
(( $# == 0 )) || { printf 'Unexpected arguments.\n' >&2; exit 1; }

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE="$ROOT/run_slurry_cpu.sbatch"
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
[[ -f "$TEMPLATE" ]] || die "Missing template: $TEMPLATE"
(( ${#PAIRS[@]} > 0 )) || die 'PAIRS is empty.'

# Validate everything before submitting any jobs; never execute/source the template.
awk '
    /^[[:space:]]*AR=/ { ar++ }
    /^[[:space:]]*PHI=/ { phi++ }
    END { exit !(ar == 1 && phi == 1) }
' "$TEMPLATE" || die 'Template must contain exactly one AR= and one PHI= assignment.'

# The existing runner creates a unique output folder using AR, PHI and job ID.
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

ARS=()
PHIS=()
for pair in "${PAIRS[@]}"; do
    read -r ar phi extra <<< "$pair"
    [[ -n "$ar" && -n "$phi" && -z "$extra" ]] || die "Expected one AR PHI pair: $pair"
    awk -v ar="$ar" -v phi="$phi" 'BEGIN {
        number="^([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$"
        exit !(ar ~ number && phi ~ number && ar+0 >= 1 && ar+0 <= 5 && phi+0 > 0 && phi+0 < 1)
    }' || die "Invalid pair: $pair (requires 1 <= AR <= 5 and 0 < PHI < 1)."
    ARS+=("$ar")
    PHIS+=("$phi")
done
bash -n "$TEMPLATE"
if (( ! DRY_RUN )); then
    command -v sbatch >/dev/null || die 'sbatch not found; run on the cluster login node.'
fi

mkdir -p -- "$ROOT/sweeps"
SWEEP_DIR="$(mktemp -d "$ROOT/sweeps/sweep_$(date +%Y%m%d_%H%M%S)_XXXXXX")"
cp -- "$TEMPLATE" "$SWEEP_DIR/template.sbatch"
SCRIPTS=()
for i in "${!ARS[@]}"; do
    ar="${ARS[$i]}"
    phi="${PHIS[$i]}"
    printf -v case_name '%03d_ar%s_phi%s' "$((i+1))" "$ar" "$phi"
    mkdir -- "$SWEEP_DIR/$case_name"
    script="$SWEEP_DIR/$case_name/run_slurry_cpu.sbatch"
    awk -v ar="$ar" -v phi="$phi" '
        /^[[:space:]]*AR=/ { sub(/=[^[:space:]#]*/, "=" ar) }
        /^[[:space:]]*PHI=/ { sub(/=[^[:space:]#]*/, "=" phi) }
        { print }
    ' "$SWEEP_DIR/template.sbatch" > "$script"
    bash -n "$script"
    SCRIPTS+=("$script")
done

printf 'Sweep directory: %s\n' "$SWEEP_DIR"
printf 'index\tAR\tPHI\tjob_id\tscript\n' > "$SWEEP_DIR/jobs.tsv"
# Submit from the repository, not the directory holding generated copies.
cd -- "$ROOT"
for i in "${!SCRIPTS[@]}"; do
    if (( DRY_RUN )); then
        job_id=DRY_RUN
    elif job_id="$(SLURRY_ROOT="$ROOT" sbatch --parsable --chdir="$ROOT" "${SCRIPTS[$i]}")"; then
        :
    else
        printf '%s\t%s\t%s\tFAILED\t%s\n' "$((i+1))" "${ARS[$i]}" "${PHIS[$i]}" "${SCRIPTS[$i]}" >> "$SWEEP_DIR/jobs.tsv"
        die "Submission failed. Previously accepted jobs remain queued; see $SWEEP_DIR/jobs.tsv"
    fi
    printf '%s\t%s\t%s\t%s\t%s\n' "$((i+1))" "${ARS[$i]}" "${PHIS[$i]}" "$job_id" "${SCRIPTS[$i]}" >> "$SWEEP_DIR/jobs.tsv"
    printf 'AR=%s PHI=%s -> %s\n' "${ARS[$i]}" "${PHIS[$i]}" "$job_id"
done
printf 'Recorded %s case(s): %s/jobs.tsv\n' "${#SCRIPTS[@]}" "$SWEEP_DIR"
