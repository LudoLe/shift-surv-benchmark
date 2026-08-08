#!/usr/bin/env bash
# Regenerate every figure the paper includes, into one directory in the repo.
#
# Before this existed the chain had two untracked manual steps:
#
#   CSV -> generator -> ~/Desktop/shift-surv-paper-images/   (copy by hand)
#       -> ~/Desktop/shift-surv-figures-FINAL/               (the .tex reads here)
#
# and it had already drifted: on 2026-07-31 the fig_disc and fig_cal PDFs the
# paper was compiling against dated from 2026-07-23, i.e. before the per-seed
# bootstrap, the shift-regime purity fixes and the observed-data oracle. Nothing
# reported the mismatch, because a stale figure looks exactly like a fresh one.
#
# Everything now writes to outputs/figures/paper/ and the .tex points there with
#   \graphicspath{{../SHIFT-SURV/outputs/figures/paper/}}
# so there is no step left to forget.
#
# Usage:  bash paper_assets/make_paper_figures.sh
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
OUT="outputs/figures/paper"
mkdir -p "$OUT"

gens=(
  "_gen_fig_disc.py"             # fig_disc_a/b        discriminazione
  "_gen_fig_cal.py"              # fig_cal_a/b         calibrazione
  "_gen_fig_nsweep.py"           # fig_nsweep_a/b      sweep numerosita'
  "_gen_fig_whyfails.py"         # fig_whyfails_a/b    geometria decisionale
  "_gen_fig_realdata_panels.py"  # fig_i3lung_*, fig_registry_*
  "_gen_fig_predictability.py"   # fig_predictability
  "_gen_fig3_dissociation.py"    # fig3_dissociation
)
tables=(
  "_gen_table_scenarios.py"      # table_scenarios.tex
  "_gen_table_survival.py"       # table_survival.tex
)

failed=()
for g in "${gens[@]}" "${tables[@]}"; do
  printf "  %-30s" "$g"
  if python paper_assets/"$g" > "outputs/logs/${g%.py}.log" 2>&1; then
    echo "OK"
  else
    echo "FAIL  (outputs/logs/${g%.py}.log)"
    failed+=("$g")
  fi
done

# --- wiring check ----------------------------------------------------------
# Every figure the paper \includegraphics must exist here afterwards. This is
# the check that would have caught all three producer/consumer prefix
# mismatches found on 2026-07-31, each of which left a figure silently
# rendering a previous run.
TEX="${TEX_PATH:-$HOME/Desktop/shift-surv-paper-PATCHED.tex}"
echo
if [ -f "$TEX" ]; then
  echo "=== figure richieste dal .tex e non prodotte"
  missing=0
  for f in $(grep -oE '\\includegraphics(\[[^]]*\])?\{[^}]*\}' "$TEX" \
             | sed -E 's/.*\{([^}]*)\}/\1/' | sed 's/\.[a-z]*$//' | sort -u); do
    ls "$OUT/$f".* >/dev/null 2>&1 || { echo "  MANCA  $f"; missing=$((missing+1)); }
  done
  [ "$missing" -eq 0 ] && echo "  nessuna: tutte presenti"
else
  echo "NOTA: .tex non trovato in $TEX — salto il controllo di cablaggio."
  echo "      Passa TEX_PATH=/percorso/al/paper.tex per attivarlo."
fi

echo
echo "=== $(ls -1 "$OUT" 2>/dev/null | wc -l | tr -d ' ') file in $OUT"
if (( ${#failed[@]} )); then
  echo "=== ${#failed[@]} generatori falliti: ${failed[*]}"
  exit 1
fi
