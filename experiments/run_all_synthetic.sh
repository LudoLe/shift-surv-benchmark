#!/usr/bin/env bash
# Re-run the whole synthetic benchmark in parallel, after a generator change.
#
# The five runs are independent -- separate output prefixes, no shared state --
# so wall-clock time is the slowest job (the n-sweep), not the sum.
#
# Usage:
#   bash experiments/run_all_synthetic.sh [JOBS] [N_SEEDS]
#     JOBS     max concurrent jobs (default: #cores/2, min 1, max 5)
#     N_SEEDS  seeds per run (default: 10)
#
# Existing synthetic tables are moved to outputs/_archive_synth_<timestamp>/
# rather than deleted, and the script diffs the untouched scenarios against
# that archive at the end: only covariate_shift should differ. If anything else
# does, the benchmark is not reproducible from its seeds and that is worth
# knowing before the numbers go in the paper.
#
# Each job is pinned to one BLAS/OMP thread; without it the jobs fight over the
# same cores and finish later than they would in sequence.
set -uo pipefail

NCORES="$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)"
JOBS="${1:-$(( NCORES / 2 ))}"
N_SEEDS="${2:-10}"
(( JOBS < 1 )) && JOBS=1
(( JOBS > 6 )) && JOBS=6

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p outputs/logs outputs/tables

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1 TORCH_NUM_THREADS=1 PYTHONUNBUFFERED=1

ARCHIVE="outputs/_archive_synth_$(date +%F_%H%M)"
mkdir -p "$ARCHIVE"
moved=0
for f in outputs/tables/benchmark_* outputs/tables/n_sweep_* outputs/tables/ablation_*; do
  [ -e "$f" ] || continue
  mv "$f" "$ARCHIVE"/ && moved=$((moved + 1))
done
echo "=== archiviate $moved tabelle in $ARCHIVE"

# --- job list: "prefix|args..." -------------------------------------------
# The sweep prefixes are dictated by paper_assets/_gen_fig_nsweep.py, which
# reads n_sweep_iwcv_n<N>_raw.csv for five regimes and n_sweep_iwcv_prior_n<N>_raw.csv
# for label-prior -- the latter was run separately and its CSVs lived outside
# the repo, under ~/Downloads/shift-surv-outputs. Regenerating it here puts it
# in outputs/tables with everything else, so the figure no longer depends on a
# directory that is not version-controlled and not on every machine.
# N grid is _gen_fig_nsweep.py's N_GRID.
# Every prefix below is taken from the CONSUMER, not from the README: the paper
# figures read benchmark_1s1t_iwcv_*, not benchmark_1s1t_* (make_figures.py,
# _gen_fig1_dotwhisker.py, _gen_fig1_transfer.py, _gen_fig2_bars.py,
# _gen_fig3_ece_bars.py, _gen_table_survival.py). Running under the default
# prefix produces correct numbers in files no figure generator opens, and the
# mismatch only surfaces when the figures are regenerated.
jobs_list=(
  "benchmark_1s1t_iwcv|"
  "benchmark_3s1t|--n_source_domains 3"
  "benchmark_1s1t_full|--include_appendix"
  "n_sweep_iwcv|--n_sweep 250,500,1000,2000"
  "n_sweep_iwcv_prior|--n_sweep 250,500,1000,2000 --scenarios prior_shift"
  "ablation_no_iwcv|--scenarios covariate_shift --no_iwcv"
)

echo "=== $(date +%F\ %T) — ${#jobs_list[@]} run, $JOBS concorrenti, $N_SEEDS seed"
echo "=== log: $ROOT/outputs/logs/"
echo

pids=()
pref=()
for entry in "${jobs_list[@]}"; do
  prefix="${entry%%|*}"
  args="${entry#*|}"
  while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do wait -n 2>/dev/null || sleep 2; done
  echo "  [$(date +%T)] start  $prefix"
  python -m experiments.run_adaptation $args \
      --n_seeds "$N_SEEDS" --out_prefix "$prefix" \
      > "outputs/logs/${prefix}.log" 2>&1 &
  pids+=("$!")
  pref+=("$prefix")
done

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
  # A plain run writes <prefix>_raw.csv; a sweep writes one per n,
  # <prefix>_n<N>_raw.csv. Match those two shapes exactly rather than
  # globbing <prefix>*: that would let benchmark_1s1t be satisfied by
  # benchmark_1s1t_full's output, and n_sweep_iwcv by n_sweep_iwcv_prior's.
  if [ -s "outputs/tables/${prefix}_raw.csv" ]; then
    n_out=1
  else
    n_out=$(ls -1 outputs/tables/${prefix}_n[0-9]*_raw.csv 2>/dev/null | wc -l | tr -d ' ')
  fi
  if [ "$n_out" -gt 0 ]; then
    echo "  OK    $prefix   ($n_out raw table(s))"
  else
    echo "  FAIL  $prefix   (vedi outputs/logs/${prefix}.log)"
  fi
done

# --- reproducibility check -------------------------------------------------
# Only covariate_shift was changed. Every other scenario is generated from the
# same seeds and must come out byte-identical to the archived run.
echo
echo "=== controllo riproducibilita' (scenari NON toccati vs $ARCHIVE)"
for prefix in benchmark_1s1t_iwcv benchmark_3s1t benchmark_1s1t_full; do
  old="$ARCHIVE/${prefix}_raw.csv"
  new="outputs/tables/${prefix}_raw.csv"
  if [ ! -f "$old" ] || [ ! -f "$new" ]; then
    echo "  --    $prefix   (manca una delle due tabelle, salto)"
    continue
  fi
  if diff -q <(grep -v covariate_shift "$old") <(grep -v covariate_shift "$new") >/dev/null; then
    echo "  OK    $prefix   scenari non toccati identici"
  else
    echo "  DIVERSI  $prefix   <-- il benchmark non e' riproducibile dai seed, da investigare:"
    echo "      diff <(grep -v covariate_shift $old) <(grep -v covariate_shift $new) | head"
  fi
done

if (( ${#failed[@]} )); then
  echo
  echo "=== ${#failed[@]} job usciti con errore: ${failed[*]}"
  exit 1
fi
