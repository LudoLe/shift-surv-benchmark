# Superseded figure generators

These produced an earlier generation of figures. The paper no longer includes
any of them, so they are kept out of `paper_assets/` to make it unambiguous
which generators are live — but not deleted, because two of them were the only
surviving record of how the current figures were built.

| file | produced | superseded by |
|---|---|---|
| `make_figures.py` | `fig1_divergence`, `fig2_headroom` | `_gen_fig_disc.py` |
| `make_assets.py` | `table3_auroc`, `table4_ece`, `table5_real`, fig1, fig2 | `_gen_table_*.py` |
| `_gen_fig1_dotwhisker.py` | `fig1_divergence` | `_gen_fig_disc.py` |
| `_gen_fig1_transfer.py` | `fig1_divergence_transfer` | `_gen_fig_disc.py` |
| `_gen_fig2_bars.py` | `fig2_headroom` | `_gen_fig_disc.py` |
| `_gen_fig3_ece_bars.py` | `fig3_ece_headroom` | `_gen_fig_cal.py` |
| `_gen_fig4_realdata.py` | `fig4_realdata` | `_gen_fig_realdata_panels.py` |
| `_gen_fig_realdata_forest.py` | forest variant of the real-data panels | `_gen_fig_realdata_panels.py` |

`fig_disc` and `fig_cal` had no generator at all: they existed only as PDFs, in
`~/Desktop/shift-surv-figures-FINAL/`. They were rebuilt on 2026-07-31 from
`_gen_fig2_bars.py` and `_gen_fig3_ece_bars.py`, which read the right tables and
carried the layout conventions. That is why these files stay in the repository.

## Still without a generator

`fig_i3lung_shift`, `fig3_dissociation` and `fig1_generator` are included by the
paper and produced by nothing here. `fig1_generator` is a schematic and is
presumably drawn by hand, which is fine. The other two plot data and should be
rebuilt, or the paper should stop claiming they are reproducible.
