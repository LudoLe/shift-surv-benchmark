#!/usr/bin/env bash
# Regenerate the three figures the paper includes, from the tracked result
# tables in outputs/tables/. The fourth figure in the paper (the generator
# diagram) is TikZ and lives in the LaTeX sources.
#
# Usage:  bash paper_assets/make_paper_figures.sh
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p outputs/logs outputs/figures/paper

gens=(
  "_gen_fig_disc.py"      # fig_disc_a  — synthetic discrimination (panel b is written too, unused by the paper)
  "_gen_fig_cal.py"       # fig_cal_a   — synthetic calibration
  "_gen_fig_realmain.py"  # fig_realmain — real-cohort forest plot
)

failed=()
for g in "${gens[@]}"; do
  printf "  %-24s" "$g"
  if python "paper_assets/$g" > "outputs/logs/${g%.py}.log" 2>&1; then
    echo "OK"
  else
    echo "FAIL  (outputs/logs/${g%.py}.log)"
    failed+=("$g")
  fi
done

echo
echo "Paper figures in outputs/figures/paper/: fig_disc_a, fig_cal_a, fig_realmain"
if (( ${#failed[@]} )); then
  echo "${#failed[@]} generator(s) failed: ${failed[*]}"
  exit 1
fi
