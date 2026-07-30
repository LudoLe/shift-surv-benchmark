#!/usr/bin/env bash
# Run every real-data transfer (public registries + I3LUNG + I3LUNG LOCO) in
# parallel, each with the per-seed patient-level bootstrap.
#
# The transfers are completely independent -- separate cohorts, separate output
# prefixes, no shared state -- so they are run as concurrent jobs with a slot
# cap rather than the sequential loops in run_all_public.sh / run_all_realdata.sh.
#
# Usage:
#   bash experiments/run_all_parallel.sh [I3LUNG_DATA_DIR] [JOBS] [N_SEEDS]
#
#   I3LUNG_DATA_DIR  directory with cb.csv, outcomes.csv, features.json.
#                    Omit (or pass "") to run the public registries only.
#   JOBS             max concurrent jobs (default: #cores/2, min 1, max 10)
#   N_SEEDS          seeds per transfer (default: 10)
#
# Each job is pinned to a single BLAS/OMP thread. Without this, every job's
# numpy/torch would try to fan out over all cores and the jobs would fight each
# other for them -- with this many short jobs, oversubscription costs more than
# the intra-job parallelism gains.
#
# Logs:    outputs/logs/<prefix>.log   (one per transfer)
# Tables:  outputs/tables/<prefix>_{raw,summary,bootstrap,bootstrap_byseed,
#                                   bootstrap_ensemble,bootreps,shiftdiag}.csv
set -uo pipefail

I3DIR="${1:-}"
NCORES="$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)"
JOBS="${2:-$(( NCORES / 2 ))}"
N_SEEDS="${3:-10}"
(( JOBS < 1 )) && JOBS=1
(( JOBS > 10 )) && JOBS=10

# Resolve the I3LUNG dir against the *caller's* cwd before cd'ing to the repo
# root, so a relative path like "I3LUNG_DATA" works from wherever it was typed.
if [[ -n "$I3DIR" ]]; then
  I3DIR="$(cd "$I3DIR" 2>/dev/null && pwd)" || {
    echo "ERROR: I3LUNG dir '${1}' does not exist (from $(pwd))." >&2; exit 1; }
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p outputs/logs outputs/tables

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1 TORCH_NUM_THREADS=1 PYTHONUNBUFFERED=1

# --- job list: "prefix|args..." -------------------------------------------
jobs_list=()

# Public registries, both directions.
for t in "Rotterdam GBSG2" "GBSG2 Rotterdam"; do
  s="${t% *}"; d="${t#* }"
  jobs_list+=("public_rotterdam_gbsg_${s}_${d}_def|--dataset rotterdam_gbsg --source $s --target $d")
done

# I3LUNG center-to-center transfers + leave-one-center-out.
if [[ -n "$I3DIR" ]]; then
  if [[ ! -f "$I3DIR/cb.csv" ]]; then
    echo "ERROR: $I3DIR/cb.csv not found — is the I3LUNG path correct?" >&2
    exit 1
  fi
  for t in "GHD INT" "INT GHD" "INT MH" "INT SZMC" "INT UOC" "INT VHIO" "VHIO INT"; do
    s="${t% *}"; d="${t#* }"
    jobs_list+=("i3lung_${s}_${d}|--data_dir $I3DIR --source $s --target $d")
  done
  jobs_list+=("i3lung_loco|--data_dir $I3DIR --loco")
else
  echo "NOTE: no I3LUNG data dir given — running public registries only."
fi

echo "=== $(date +%F\ %T) — ${#jobs_list[@]} transfers, $JOBS concurrent, $N_SEEDS seeds each"
echo "=== logs: $ROOT/outputs/logs/"
echo

# --- dispatcher ------------------------------------------------------------
# Indexed arrays rather than an associative one: `declare -A` needs bash 4, and
# macOS still ships 3.2. Same for `wait -n` below, which falls back to polling.
pids=()
pref=()
run_one() {
  local prefix="$1"; shift
  python -m experiments.run_real_data "$@" \
      --n_seeds "$N_SEEDS" --out_prefix "$prefix" \
      > "outputs/logs/${prefix}.log" 2>&1
}

for entry in "${jobs_list[@]}"; do
  prefix="${entry%%|*}"
  args="${entry#*|}"
  # Block until a slot frees up.
  while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do wait -n 2>/dev/null || sleep 2; done
  echo "  [$(date +%T)] start  $prefix"
  run_one "$prefix" $args &
  pids+=("$!")
  pref+=("$prefix")
done

# --- collect ---------------------------------------------------------------
failed=()
i=0
while [ $i -lt ${#pids[@]} ]; do
  if ! wait "${pids[$i]}"; then failed+=("${pref[$i]}"); fi
  i=$((i + 1))
done

echo
echo "=== $(date +%F\ %T) — done"
for entry in "${jobs_list[@]}"; do
  prefix="${entry%%|*}"
  # LOCO writes one table per held-out centre (<prefix>_<CENTRE>_bootstrap.csv),
  # never a bare <prefix>_bootstrap.csv, so match the glob rather than the exact
  # name -- otherwise a completed LOCO run is reported as FAIL.
  n_out=$(ls -1 "outputs/tables/${prefix}"*_bootstrap.csv 2>/dev/null | wc -l | tr -d ' ')
  if [ "$n_out" -gt 0 ]; then
    echo "  OK    $prefix   ($n_out table(s))"
  else
    echo "  FAIL  $prefix   (see outputs/logs/${prefix}.log)"
  fi
done

if (( ${#failed[@]} )); then
  echo
  echo "=== ${#failed[@]} job(s) exited non-zero: ${failed[*]}"
  exit 1
fi
echo
echo "=== next: python -m experiments.max_statistic_bootstrap --prefix i3lung"
echo "===       python -m experiments.max_statistic_bootstrap --prefix public"
