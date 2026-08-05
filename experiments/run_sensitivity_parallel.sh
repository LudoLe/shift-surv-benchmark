#!/usr/bin/env bash
# Run the sensitivity battery in parallel, split by regime.
#
# The natural unit is (mode, n_source_domains), but that gives four jobs of very
# unequal size -- worlds at one source is 600 configurations on its own, about
# 21 hours, so four jobs would finish in 21 hours no matter how many cores are
# free. Splitting each of those four by regime gives 24 independent jobs of
# roughly equal size, which is what actually uses a machine.
#
# Each job writes its own CSV; the shards are concatenated at the end. Nothing is
# shared between them: the derived config files each process writes go to its own
# mkdtemp directory, so concurrent jobs cannot collide over them.
#
# Usage:
#   ./experiments/run_sensitivity_parallel.sh [JOBS] [N_SEEDS]
#
#   JOBS     concurrent processes (default: half the cores, capped at 12)
#   N_SEEDS  seeds per configuration (default: 10)
#
# Run experiments/calibrate.py first: without configs/parameter_space_calibrated.yaml
# the sweep exits immediately.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

NCORES="$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)"
JOBS="${1:-$(( NCORES / 2 ))}"
N_SEEDS="${2:-10}"
(( JOBS < 1 )) && JOBS=1
(( JOBS > 12 )) && JOBS=12

# One thread per process: every job is already a whole core, and letting BLAS
# fan out inside each of them makes twenty-four jobs fight over the same cores.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1

if [ ! -f configs/parameter_space_calibrated.yaml ]; then
  echo "ERRORE: manca configs/parameter_space_calibrated.yaml"
  echo "        lancia prima:  python -m experiments.calibrate"
  exit 1
fi

mkdir -p outputs/logs outputs/tables
REGIMES=(covariate_shift observational_shift spurious_shift
         baseline_hazard_shift prior_shift concept_shift)

# (mode, n_source_domains, extra args)
UNITS=(
  "oat|1|"
  "oat|3|"
  "worlds|1|--n_worlds 100"
  "worlds|3|--n_worlds 50"
)

jobs_list=()
for unit in "${UNITS[@]}"; do
  IFS='|' read -r mode nsrc extra <<< "$unit"
  for reg in "${REGIMES[@]}"; do
    jobs_list+=("${mode}|${nsrc}|${reg}|${extra}")
  done
done

echo "=== $(date +%F\ %T) — ${#jobs_list[@]} job, $JOBS concorrenti, $N_SEEDS seed"
echo "=== log: $ROOT/outputs/logs/"
echo

pids=(); names=()
for job in "${jobs_list[@]}"; do
  IFS='|' read -r mode nsrc reg extra <<< "$job"
  name="sens_${mode}_${nsrc}src_${reg}"
  while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do wait -n 2>/dev/null || sleep 5; done
  echo "  [$(date +%T)] start  $name"
  # shellcheck disable=SC2086
  python -m experiments.sensitivity_sweep \
      --mode "$mode" --n_source_domains "$nsrc" --regimes "$reg" \
      --n_seeds "$N_SEEDS" --out_prefix "$name" $extra \
      > "outputs/logs/${name}.log" 2>&1 &
  pids+=("$!"); names+=("$name")
done

failed=()
for i in "${!pids[@]}"; do
  wait "${pids[$i]}" || failed+=("${names[$i]}")
done

echo
echo "=== $(date +%F\ %T) — screening finito"
[ ${#failed[@]} -gt 0 ] && { echo "  FALLITI:"; printf '    %s\n' "${failed[@]}"; }

# --- merge the per-regime shards -------------------------------------------
# Header from the first shard, rows from all of them. A missing shard is
# reported rather than silently dropped: a merged file that is quietly short is
# worse than one that is missing.
for unit in "${UNITS[@]}"; do
  IFS='|' read -r mode nsrc extra <<< "$unit"
  out="outputs/tables/sensitivity_${mode}_${nsrc}src.csv"
  shards=()
  for reg in "${REGIMES[@]}"; do
    f="outputs/tables/sens_${mode}_${nsrc}src_${reg}.csv"
    if [ -s "$f" ]; then shards+=("$f"); else echo "  MANCA  $f"; fi
  done
  if [ ${#shards[@]} -eq 0 ]; then echo "  nessuno shard per $mode/$nsrc"; continue; fi
  head -1 "${shards[0]}" > "$out"
  for f in "${shards[@]}"; do tail -n +2 "$f" >> "$out"; done
  echo "  unito  $out  ($(( $(wc -l < "$out") - 1 )) righe da ${#shards[@]} shard)"
done

# --- confirmation, after everything else ------------------------------------
# It reads the screening CSVs, so it cannot start until they exist. Both
# settings are confirmed on the same disjoint seed block declared in the frozen
# protocol.
echo
echo "=== $(date +%F\ %T) — conferma dei candidati"
for nsrc in 1 3; do
  srcs=()
  for mode in oat worlds; do
    f="outputs/tables/sensitivity_${mode}_${nsrc}src.csv"
    [ -s "$f" ] && srcs+=("$f")
  done
  if [ ${#srcs[@]} -eq 0 ]; then echo "  nessun CSV di screening per ${nsrc} sorgente/i"; continue; fi
  name="sens_confirm_${nsrc}src"
  echo "  [$(date +%T)] start  $name  (da ${srcs[*]})"
  python -m experiments.sensitivity_sweep --mode confirm \
      --n_source_domains "$nsrc" --confirm_from "${srcs[@]}" \
      --n_seeds "$N_SEEDS" --out_prefix "$name" \
      > "outputs/logs/${name}.log" 2>&1 \
      && echo "    ok" || echo "    FALLITO (vedi outputs/logs/${name}.log)"
done

echo
echo "=== $(date +%F\ %T) — fatto"
ls -la outputs/tables/sensitivity_*.csv outputs/tables/sens_confirm_*.csv 2>/dev/null
