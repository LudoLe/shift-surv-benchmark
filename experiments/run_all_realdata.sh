#!/usr/bin/env bash
# Run all 7 I3LUNG center-to-center transfers (each with the patient-level bootstrap).
#
# Usage:
#   bash experiments/run_all_realdata.sh /real/path/to/I3LUNG_DATA
#
# The data dir must contain: cb.csv, outcomes.csv, features.json.
# Results are written to <repo>/outputs/tables/i3lung_<S>_<T>_{raw,summary,bootstrap}.csv
set -euo pipefail

DIR="${1:?Usage: bash experiments/run_all_realdata.sh /path/to/I3LUNG_DATA}"
if [[ ! -f "$DIR/cb.csv" ]]; then
  echo "ERROR: $DIR/cb.csv not found — is the data path correct?" >&2
  exit 1
fi

# Run from the repo root so `python -m experiments...` resolves regardless of cwd.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

transfers=("GHD INT" "INT GHD" "INT MH" "INT SZMC" "INT UOC" "INT VHIO" "VHIO INT")
for t in "${transfers[@]}"; do
  s="${t% *}"; d="${t#* }"
  echo "=== $(date +%H:%M:%S)  $s -> $d ==="
  python -m experiments.run_real_data --data_dir "$DIR" \
      --source "$s" --target "$d" --out_prefix "i3lung_${s}_${d}"
done
echo "=== ALL DONE ($(date +%H:%M:%S)) — bootstrap CSVs in outputs/tables/ ==="
