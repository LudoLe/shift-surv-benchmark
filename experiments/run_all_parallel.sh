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
# Was capped at 10 because unpinned TensorFlow workers oversubscribed the
# machine; taskset below makes each worker occupy one core.
(( JOBS > NCORES )) && JOBS=$NCORES

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
       NUMEXPR_NUM_THREADS=1 TORCH_NUM_THREADS=1 PYTHONUNBUFFERED=1 \
       TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 TF_CPP_MIN_LOG_LEVEL=3

# Pin each job to one CPU. TensorFlow does not read OMP_NUM_THREADS -- it has its
# own pools, and they size themselves to the machine -- so on a 192-core server
# each DANN/CDAN worker opened about 175 native threads. Sixty-four workers came
# to 11181 threads over 192 hardware ones, load average 381 at 46% CPU: slower
# than running serially. taskset makes the kernel enforce what the libraries
# would not, and because the pools read sched_getaffinity they also stop creating
# the threads in the first place: 451 threads instead of 11181, seven per worker.
if command -v taskset >/dev/null 2>&1; then
  PIN=1
else
  PIN=0
  echo "  nota: taskset non disponibile; riduci JOBS per non sovrascrivere la macchina"
fi
NICE="${NICE:-10}"
_pin_prefix() {   # $1 = job index
  if [ "$PIN" = 1 ]; then printf 'taskset -c %d nice -n %d' "$(( $1 % NCORES ))" "$NICE"
  else printf 'nice -n %d' "$NICE"; fi
}


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
  # All 30 ordered pairs, not a chosen seven. The previous list had INT as the
  # source in five of its seven entries, which meant the pooled effect was
  # largely one center's model carried to the others, and it left the paper
  # having to justify the selection. It also disagreed with the survival check of
  # Section V-D, which already refits all thirty ordered pairs; running the same
  # thirty here makes the two analyses answer the same question.
  #
  # Centers are those surviving the min_n=50 filter in load_i3lung_all.
  I3_CENTERS=(GHD INT VHIO UOC MH SZMC)
  for s_c in "${I3_CENTERS[@]}"; do
    for d_c in "${I3_CENTERS[@]}"; do
      [ "$s_c" = "$d_c" ] && continue
      jobs_list+=("i3lung_${s_c}_${d_c}|--data_dir $I3DIR --source $s_c --target $d_c")
    done
  done
  for t in; do
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
  $(_pin_prefix "${JOB_IDX:-0}") python -m experiments.run_real_data "$@" \
      --n_seeds "$N_SEEDS" --out_prefix "$prefix" \
      > "outputs/logs/${prefix}.log" 2>&1
}

for entry in "${jobs_list[@]}"; do
  prefix="${entry%%|*}"
  args="${entry#*|}"
  # Block until a slot frees up.
  while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do wait -n 2>/dev/null || sleep 2; done
  echo "  [$(date +%T)] start  $prefix"
  JOB_IDX=${#pids[@]} run_one "$prefix" $args &
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
# --exclude loco: the prefix is a substring, so i3lung also matches the
# leave-one-centre-out folds, which are a different design and must not be
# pooled with the pairwise transfers.
echo "=== next: python -m experiments.max_statistic_bootstrap --prefix i3lung --exclude loco"
echo "===       python -m experiments.max_statistic_bootstrap --prefix i3lung_loco"
echo "===       python -m experiments.max_statistic_bootstrap --prefix public"
