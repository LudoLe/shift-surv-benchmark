#!/usr/bin/env bash
# Synthetic twins of the real transfers, in parallel.
#
# Independent of run_all_synthetic.sh: the twins read the generator and the real
# cohorts, not the benchmark tables, so the two can run concurrently -- but they
# compete for the same cores, so do not launch both at full width on one box.
#
# They DO need re-running after any generator change: build_twin constructs the
# twin by calling generate_scenario("covariate_shift", ...), so a change to that
# regime changes every twin, and the MMD bisection re-tunes to a different scale.
#
# Usage:
#   bash experiments/run_all_twins.sh [I3LUNG_DATA_DIR] [JOBS] [N_SEEDS]
#
#   N_SEEDS must match the real transfers' --n_seeds, or the twin and real
#   intervals stop being comparable, which is the whole point of the twin.
set -uo pipefail

I3DIR="${1:-}"
if [[ -n "$I3DIR" ]]; then
  I3DIR="$(cd "$I3DIR" 2>/dev/null && pwd)" || {
    echo "ERROR: I3LUNG dir '${1}' inesistente (da $(pwd))." >&2; exit 1; }
fi

NCORES="$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)"
JOBS="${2:-$(( NCORES / 4 ))}"
N_SEEDS="${3:-10}"
(( JOBS < 1 )) && JOBS=1
(( JOBS > 6 )) && JOBS=6

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p outputs/logs outputs/tables

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1 TORCH_NUM_THREADS=1 PYTHONUNBUFFERED=1

# "tag|args" — tag is used only for the log filename; synthetic_twin.py names
# its own output table twin_<source>_<target>[_tau<NN>].csv.
jobs_list=(
  "twin_GBSG2_Rotterdam|--dataset rotterdam_gbsg --tau 36 --source GBSG2 --target Rotterdam"
  "twin_Rotterdam_GBSG2|--dataset rotterdam_gbsg --tau 36 --source Rotterdam --target GBSG2"
)
if [[ -n "$I3DIR" ]]; then
  jobs_list+=(
    "twin_INT_GHD|--data_dir $I3DIR --source INT --target GHD"
    "twin_GHD_INT|--data_dir $I3DIR --source GHD --target INT"
  )
else
  echo "NOTA: nessuna dir I3LUNG — solo i gemelli del registry pubblico."
fi

echo "=== $(date +%F\ %T) — ${#jobs_list[@]} gemelli, $JOBS concorrenti, $N_SEEDS seed"
echo

pids=(); pref=()
for entry in "${jobs_list[@]}"; do
  tag="${entry%%|*}"; args="${entry#*|}"
  while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do wait -n 2>/dev/null || sleep 2; done
  echo "  [$(date +%T)] start  $tag"
  python -m experiments.synthetic_twin $args --n_seeds "$N_SEEDS" \
      > "outputs/logs/${tag}.log" 2>&1 &
  pids+=("$!"); pref+=("$tag")
done

failed=(); i=0
while [ $i -lt ${#pids[@]} ]; do
  if ! wait "${pids[$i]}"; then failed+=("${pref[$i]}"); fi
  i=$((i + 1))
done

echo
echo "=== $(date +%F\ %T) — done"
for entry in "${jobs_list[@]}"; do
  tag="${entry%%|*}"
  # synthetic_twin.py writes twin_<source>_<target>*.csv; match on the pair.
  pair="${tag#twin_}"
  n=$(ls -1 outputs/tables/twin_${pair}*.csv 2>/dev/null | wc -l | tr -d ' ')
  if [ "$n" -gt 0 ]; then echo "  OK    $tag"
  else echo "  FAIL  $tag   (vedi outputs/logs/${tag}.log)"; fi
done

if (( ${#failed[@]} )); then
  echo; echo "=== ${#failed[@]} job usciti con errore: ${failed[*]}"; exit 1
fi
echo
echo "=== confronto gemello vs reale:  python -m experiments.twin_vs_real"
