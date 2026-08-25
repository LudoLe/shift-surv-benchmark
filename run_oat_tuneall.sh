#!/bin/bash
# One-at-a-time sensitivity sweep, every method selected per run.
#
# The worlds mode already runs under --tune_all; the OAT mode did not, which left
# the paper's configuration count split across two selection protocols. This
# closes that.
#
# Each shard is one (n_source, regime, axis) triple and writes its own CSV, so an
# interruption costs at most one shard. Rerunning this script resumes from what is
# missing.
#
#   tmux new -s oat
#   conda activate shift310 && ./run_oat_tuneall.sh
cd ~/shift-surv-benchmark
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 TF_CPP_MIN_LOG_LEVEL=2
mkdir -p outputs/logs/oat

REGS="covariate_shift observational_shift spurious_shift baseline_hazard_shift prior_shift concept_shift"
AXES="lambda0 w_norm sigma n_per_domain p nu k_inv k_spu w_shape amplitude"
MAXJOBS=48

while true; do
  missing=0
  for ns in 1 3; do
    for reg in $REGS; do
      for ax in $AXES; do
        out=outputs/tables/oat_tuneall_${ns}src_${reg}_${ax}.csv
        if [ -s "$out" ]; then
          continue
        fi
        missing=$((missing + 1))
        if pgrep -f "oat_tuneall_${ns}src_${reg}_${ax}$" > /dev/null; then
          continue
        fi
        while [ "$(pgrep -fc sensitivity_sweep)" -ge "$MAXJOBS" ]; do
          sleep 30
        done
        setsid nohup nice -n 10 python -m experiments.sensitivity_sweep \
            --mode oat --n_source_domains "$ns" --regimes "$reg" --axes "$ax" \
            --n_seeds 10 --with_nets --tune_all \
            --out_prefix "oat_tuneall_${ns}src_${reg}_${ax}" \
            > "outputs/logs/oat/${ns}src_${reg}_${ax}.log" 2>&1 &
        sleep 2
      done
    done
  done
  if [ "$missing" -eq 0 ]; then
    date +"%H:%M tutte le 120 fette OAT complete"
    break
  fi
  date +"%H:%M in attesa"
  echo "  mancano $missing fette, ricontrollo fra 5 minuti"
  sleep 300
done
