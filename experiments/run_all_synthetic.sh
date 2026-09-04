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
# The cap was 8 because unpinned TensorFlow workers oversubscribed the
# machine; with taskset below each worker occupies one core, so the limit is
# now just how many cores you want to take.
(( JOBS > NCORES )) && JOBS=$NCORES

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


ARCHIVE="outputs/_archive_synth_$(date +%F_%H%M)"
mkdir -p "$ARCHIVE"
moved=0
for f in outputs/tables/benchmark_* outputs/tables/n_sweep_* outputs/tables/ablation_*; do
  [ -e "$f" ] || continue
  mv "$f" "$ARCHIVE"/ && moved=$((moved + 1))
done
echo "=== archiviate $moved tabelle in $ARCHIVE"

# --- job list: "prefix|args..." -------------------------------------------
# The sweep prefixes feed the paper's sample-size analysis, which
# reads n_sweep_iwcv_n<N>_raw.csv for five regimes and n_sweep_iwcv_prior_n<N>_raw.csv
# for label-prior -- the latter was run separately and its CSVs lived outside
# the repo, under ~/Downloads/shift-surv-outputs. Regenerating it here puts it
# in outputs/tables with everything else, so the figure no longer depends on a
# directory that is not version-controlled and not on every machine.
# Every prefix below matches what the figure generators and the paper's
# Sample-size paragraph read (paper_assets/, outputs/tables/n_sweep_iwcv_*).
# prior_shift is now in CORE_SCENARIOS, so the main benchmark emits it and the
# dedicated *_prior jobs below are redundant for any table produced from here on.
# They stay until the next full re-run: the tables currently on disk came from the
# old split, and _gen_fig_disc.py / _gen_fig_cal.py still read label-prior from
# those prefixes. Drop the four *_prior jobs, and the PRIOR_PREFIX map in the two
# figure generators, once the main prefixes carry prior_shift rows.
jobs_list=(
  "benchmark_1s1t_iwcv|"
  "benchmark_1s1t_prior|--scenarios prior_shift"
  "benchmark_3s1t|--n_source_domains 3"
  "benchmark_3s1t_prior|--scenarios prior_shift --n_source_domains 3"
  "n_sweep_iwcv|--n_sweep 250,500,1000,2000"
  "n_sweep_iwcv_prior|--n_sweep 250,500,1000,2000 --scenarios prior_shift"
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
  # REUSE=1 takes the DANN and CDAN rows from the previous run of the same
  # prefix instead of recomputing them. Their code and inputs are unchanged by
  # the symmetric-tuning switch -- their fits do not depend on any other
  # method's hyperparameter -- and they are the expensive part, ten trainings
  # per cell against one for everything else. Off by default: it is only valid
  # while that assumption holds, so it has to be asked for.
  REUSE_ARG=()
  # --reuse_dir points at the archive: the block above has already moved the
  # previous tables out of outputs/tables, so without it every job dies on a
  # missing raw CSV.
  [ "${REUSE:-0}" = 1 ] && REUSE_ARG=(--reuse_methods dann cdan --reuse_dir "$ARCHIVE")
  $(_pin_prefix "${#pids[@]}") python -m experiments.run_adaptation $args \
      "${REUSE_ARG[@]}" \
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
for prefix in benchmark_1s1t_iwcv benchmark_3s1t; do
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

# run_survival.py is a different entry point: it turns on exponential censoring
# and fits a Cox model, and produces survival_cox_summary.csv, which
# _gen_table_survival.py reads. It was missing from every runner.
echo
echo "=== validita' survival (Cox su dati censurati)"
python -m experiments.run_survival --out_prefix survival_cox \
    > outputs/logs/survival_cox.log 2>&1 \
  && echo "  OK    survival_cox" \
  || { echo "  FAIL  survival_cox   (vedi outputs/logs/survival_cox.log)"; failed+=("survival_cox"); }

if (( ${#failed[@]} )); then
  echo
  echo "=== ${#failed[@]} job usciti con errore: ${failed[*]}"
  exit 1
fi
