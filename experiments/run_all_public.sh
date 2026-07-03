#!/usr/bin/env bash
# Run all public-registry multicentric transfers (each with the patient-level
# bootstrap + covariate-shift diagnostic), mirroring run_all_realdata.sh.
#
# Usage:
#   bash experiments/run_all_public.sh [DATASET] [TAU]
#     DATASET  public registry name (default: rotterdam_gbsg)
#     TAU      fixed horizon in months (default: the registry's TAU)
#
# Results: <repo>/outputs/tables/public_<DATASET>_<S>_<T>_{raw,summary,bootstrap,shiftdiag}.csv
set -euo pipefail

DATASET="${1:-rotterdam_gbsg}"
TAU="${2:-}"
N_SEEDS="${3:-10}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Source/target center pairs per dataset (both directions).
case "$DATASET" in
  rotterdam_gbsg) transfers=("Rotterdam GBSG2" "GBSG2 Rotterdam") ;;
  *) echo "ERROR: no transfer list defined for dataset '$DATASET'" >&2; exit 1 ;;
esac

TAU_ARG=(); TAULAB="def"
if [[ -n "$TAU" ]]; then TAU_ARG=(--tau "$TAU"); TAULAB="tau${TAU%.*}"; fi

for t in "${transfers[@]}"; do
  s="${t% *}"; d="${t#* }"
  echo "=== $(date +%H:%M:%S)  [$DATASET] $s -> $d ($TAULAB) ==="
  python -m experiments.run_real_data --dataset "$DATASET" \
      --source "$s" --target "$d" "${TAU_ARG[@]}" --n_seeds "$N_SEEDS" \
      --out_prefix "public_${DATASET}_${s}_${d}_${TAULAB}"
done
echo "=== ALL DONE ($(date +%H:%M:%S)) — tables in outputs/tables/ ==="
