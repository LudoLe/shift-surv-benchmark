#!/bin/bash
cd ~/shift-surv-benchmark
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 TF_CPP_MIN_LOG_LEVEL=2
mkdir -p outputs/logs/tuneall
REGS="covariate_shift observational_shift spurious_shift baseline_hazard_shift prior_shift concept_shift"
while true; do
  missing=0
  for ns in 1 3; do
    for reg in $REGS; do
      for k in $(seq 0 31); do
        f=outputs/tables/sens_tuneall_${ns}src_${reg}_s${k}.csv
        [ -s "$f" ] && continue
        missing=$((missing+1))
        pgrep -f "sens_tuneall_${ns}src_${reg}_s${k}$" >/dev/null && continue
        while [ "$(pgrep -fc sensitivity_sweep)" -ge 48 ]; do sleep 30; done
        setsid nohup nice -n 10 python -m experiments.sensitivity_sweep \
            --mode worlds --n_source_domains $ns --regimes "$reg" \
            --n_seeds 10 --n_worlds 100 --world_seed 12345 \
            --world_slice ${k}/32 --with_nets --tune_all \
            --out_prefix sens_tuneall_${ns}src_${reg}_s${k} \
            > outputs/logs/tuneall/${ns}src_${reg}_s${k}.log 2>&1 &
        sleep 2
      done
    done
  done
  [ "$missing" -eq 0 ] && { echo "$(date +%H:%M) tutte le 384 fette complete"; break; }
  echo "$(date +%H:%M) mancano $missing fette, ricontrollo fra 5 minuti"
  sleep 300
done
