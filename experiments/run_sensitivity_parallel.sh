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
#   JOBS     concurrent processes (default: a quarter of the cores)
#   N_SEEDS  seeds per configuration (default: 10)
#
# On a shared machine, read the load before choosing JOBS. Every job is one
# thread after the TensorFlow settings below, so JOBS is very nearly the number
# of cores this run will occupy. The jobs are also niced, so they yield to
# anyone else's work rather than competing with it: on an idle machine that
# costs nothing, and on a busy one it is the difference between finishing
# slightly later and making everyone else's day worse.
#
# Run experiments/calibrate.py first: without configs/parameter_space_calibrated.yaml
# the sweep exits immediately.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

NCORES="$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)"
JOBS="${1:-$(( NCORES / 4 ))}"
N_SEEDS="${2:-10}"
(( JOBS < 1 )) && JOBS=1

# Leave room for whoever else is on the machine. The one-minute load average is
# a fair estimate of how many cores are already busy; asking for more than what
# is left slows everybody down, this run included.
NICE="${NICE:-10}"
if [ -r /proc/loadavg ]; then
  LOAD1="$(awk '{printf "%d", $1}' /proc/loadavg)"
  FREE_CORES=$(( NCORES - LOAD1 ))
  (( FREE_CORES < 1 )) && FREE_CORES=1
  # Two thirds of what is free, so the machine does not sit at exactly 100%.
  POLITE=$(( FREE_CORES * 2 / 3 ))
  (( POLITE < 1 )) && POLITE=1
  if (( JOBS > POLITE )); then
    echo "  load attuale ${LOAD1} su ${NCORES} thread -> ~${FREE_CORES} liberi"
    echo "  $JOBS job richiesti, limito a $POLITE per lasciare spazio agli altri"
    echo "  (per forzare: JOBS_FORCE=1 ./$(basename "$0") $JOBS)"
    [ "${JOBS_FORCE:-0}" = 1 ] || JOBS=$POLITE
  fi
fi
echo "  $JOBS job concorrenti, nice $NICE"

# No hard cap on a big machine, but memory is the binding constraint rather than
# cores: DANN and CDAN run on TensorFlow and every process reserves its own
# block, so a hundred concurrent jobs is a hundred TF runtimes. If free memory is
# readable, cap the job count at what fits with 2 GB each and say so, because a
# run that dies at hour four from an OOM costs more than one that used half the
# cores.
if command -v free >/dev/null 2>&1; then
  FREE_GB="$(free -g | awk '/^Mem:/ {print $7}')"
  if [ -n "${FREE_GB:-}" ] && [ "$FREE_GB" -gt 0 ]; then
    MEM_CAP=$(( FREE_GB / 2 ))
    (( MEM_CAP < 1 )) && MEM_CAP=1
    if (( JOBS > MEM_CAP )); then
      echo "  nota: $JOBS job richiesti, ${FREE_GB} GB liberi -> limito a $MEM_CAP"
      echo "        (stima 2 GB per processo TensorFlow; --no_nets ne usa molta meno)"
      JOBS=$MEM_CAP
    fi
  fi
fi

# One thread per process: every job is already a whole core, and letting BLAS
# fan out inside each of them makes twenty-four jobs fight over the same cores.
# TensorFlow does not read OMP_NUM_THREADS, so the BLAS variables alone left
# every DANN/CDAN process opening about sixty-five threads: ninety-six jobs came
# to six thousand threads over a hundred and ninety-two hardware ones, load
# average 1006 at 46% CPU, and the machine spent its time context-switching. The
# TF variables are set in experiments/adaptation_methods.py before the import
# that initialises the runtime, and repeated here so a job launched by hand
# behaves the same way.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1 \
       TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 \
       TF_CPP_MIN_LOG_LEVEL=3

# Pin each job to one CPU. The library environment variables help but do not
# finish the job: measured, a worker holds 14 native threads with them set and 36
# without, and those pools size themselves to the machine -- on a 192-core server
# that came to 175 threads per process, 11181 in total across 64 workers, load
# average 381. Python's threading.active_count() reports 1 either way, because
# the pools are native and invisible to it, which is why this went unnoticed.
#
# taskset makes the kernel enforce what the libraries would not: the threads
# still exist, but they share one core, so sixty-four workers occupy sixty-four
# cores no matter how many pools TensorFlow decides to open.
if command -v taskset >/dev/null 2>&1; then
  PIN=1
else
  PIN=0
  echo "  nota: taskset non disponibile, i job non saranno vincolati a una CPU"
  echo "        (su questa macchina il rischio e' sovrascrizione: riduci JOBS)"
fi

# A sanity check worth the two seconds: if a job still opens more than a couple
# of threads, the whole parallel plan is wrong and it is better to find out now
# than four hours in.
_THR="$(python - <<'PYEOF' 2>/dev/null
import threading, warnings; warnings.filterwarnings("ignore")
import experiments.adaptation_methods  # noqa: F401
print(threading.active_count())
PYEOF
)"
# Informative only: threading.active_count() cannot see the native pools that
# actually caused the oversubscription, so this check never had the power to
# catch it. taskset is what makes the job safe; this just reports.
if [ -n "${_THR:-}" ]; then
  echo "  thread Python per processo: $_THR (i pool nativi non sono visibili qui)"
fi

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

# 24 jobs saturates about 24 cores. On a larger machine the limit stops being
# the job count and becomes the size of the largest job, so WORLD_SHARDS splits
# each worlds unit into contiguous blocks of the SAME hypercube (--world_slice).
# Not a different seed per block: that would be a different design, and the two
# blocks would explore two different spaces while reporting as one run.
WORLD_SHARDS="${WORLD_SHARDS:-1}"

# The one-at-a-time units are the long pole once worlds is sharded: 36
# configurations per regime, about 108 minutes at three sources. Splitting them
# by axis needs no care -- each axis perturbs the same centre independently, so a
# subset produces exactly the rows the full run would have -- and brings the
# longest job down to a handful of configurations.
AXES_ALL=(lambda0 w_norm sigma n_per_domain p nu k_inv k_spu w_shape amplitude)
OAT_BY_AXIS="${OAT_BY_AXIS:-0}"

jobs_list=()
for unit in "${UNITS[@]}"; do
  IFS='|' read -r mode nsrc extra <<< "$unit"
  for reg in "${REGIMES[@]}"; do
    if [ "$mode" = oat ] && [ "$OAT_BY_AXIS" = 1 ]; then
      for ax in "${AXES_ALL[@]}"; do
        jobs_list+=("${mode}|${nsrc}|${reg}|${extra} --axes ${ax}|ax${ax}")
      done
    elif [ "$mode" = worlds ] && [ "$WORLD_SHARDS" -gt 1 ]; then
      for k in $(seq 0 $(( WORLD_SHARDS - 1 ))); do
        jobs_list+=("${mode}|${nsrc}|${reg}|${extra} --world_slice ${k}/${WORLD_SHARDS}|w${k}")
      done
    else
      jobs_list+=("${mode}|${nsrc}|${reg}|${extra}|")
    fi
  done
done

echo "=== $(date +%F\ %T) — ${#jobs_list[@]} job, $JOBS concorrenti, $N_SEEDS seed"
echo "=== log: $ROOT/outputs/logs/"
echo

pids=(); names=()
for job in "${jobs_list[@]}"; do
  IFS='|' read -r mode nsrc reg extra sfx <<< "$job"
  name="sens_${mode}_${nsrc}src_${reg}${sfx:+_$sfx}"
  while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do wait -n 2>/dev/null || sleep 5; done
  echo "  [$(date +%T)] start  $name"
  # shellcheck disable=SC2086
  if [ "$PIN" = 1 ]; then
    CPU=$(( ${#pids[@]} % NCORES ))
    PREFIX=(taskset -c "$CPU" nice -n "$NICE")
  else
    PREFIX=(nice -n "$NICE")
  fi
  "${PREFIX[@]}" python -m experiments.sensitivity_sweep \
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
    for f in outputs/tables/sens_${mode}_${nsrc}src_${reg}.csv \
             outputs/tables/sens_${mode}_${nsrc}src_${reg}_w*.csv \
             outputs/tables/sens_${mode}_${nsrc}src_${reg}_ax*.csv; do
      [ -s "$f" ] && shards+=("$f")
    done
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
  taskset_pfx=(); command -v taskset >/dev/null 2>&1 && taskset_pfx=(taskset -c 0-$(( NCORES/4 )))
  "${taskset_pfx[@]}" nice -n "$NICE" python -m experiments.sensitivity_sweep --mode confirm \
      --n_source_domains "$nsrc" --confirm_from "${srcs[@]}" \
      --n_seeds "$N_SEEDS" --out_prefix "$name" \
      > "outputs/logs/${name}.log" 2>&1 \
      && echo "    ok" || echo "    FALLITO (vedi outputs/logs/${name}.log)"
done

echo
echo "=== $(date +%F\ %T) — fatto"
ls -la outputs/tables/sensitivity_*.csv outputs/tables/sens_confirm_*.csv 2>/dev/null
