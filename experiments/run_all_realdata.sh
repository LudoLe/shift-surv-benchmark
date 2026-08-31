#!/usr/bin/env bash
# Run all 30 ordered I3LUNG center-to-center transfers (each with the
# patient-level bootstrap).
#
# Usage:
#   bash experiments/run_all_realdata.sh /real/path/to/I3LUNG_DATA
#
# The data dir must contain: cb.csv, outcomes.csv, features.json.
# Results are written to <repo>/outputs/tables/i3lung_<S>_<T>_{raw,summary,bootstrap}.csv
#
# The transfer list is generated, not hardcoded: an earlier version enumerated
# seven pairs by hand, and every later "full" run silently covered only those,
# which is how a rerun meant to refresh thirty transfers refreshed eight.
set -euo pipefail
DIR="${1:?Usage: bash experiments/run_all_realdata.sh /path/to/I3LUNG_DATA}"
if [[ ! -f "$DIR/cb.csv" ]]; then
  echo "ERROR: $DIR/cb.csv not found — is the data path correct?" >&2
  exit 1
fi
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
CENTERS=(GHD INT MH SZMC UOC VHIO)
for s in "${CENTERS[@]}"; do
  for d in "${CENTERS[@]}"; do
    [[ "$s" == "$d" ]] && continue
    echo "=== $(date +%H:%M:%S)  $s -> $d ==="
    python -m experiments.run_real_data --data_dir "$DIR" \
        --source "$s" --target "$d" --out_prefix "i3lung_${s}_${d}"
  done
done
echo "=== ALL DONE ($(date +%H:%M:%S)) — 30 transfers in outputs/tables/ ==="
