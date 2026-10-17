# Paper

> This work has been submitted to the IEEE for possible publication. Copyright may be transferred without
> notice, after which this version may no longer be accessible.

Preprint of **When to Replan and How Much to Correct: Residual Reinforcement Learning for Disturbance Recovery
in Vision-Language-Action Policies** (Chang-Hoon Ji and Tae-Eui Kam, Korea University), an ICEIC 2027 submission
(IEEE Publication Track).

**[ji2027replan.pdf](ji2027replan.pdf)** is the manuscript as of 2026-10-17 (6 pages) with the notice above added in
the top margin of page 1; the text is otherwise identical to `main_iceic_residual.tex`.

| File | Content |
|---|---|
| `main_iceic_residual.tex` | the paper |
| `fig_overview_res.tex` | Fig. 1 (TikZ) |
| `figures/res_forest.pdf`, `figures/res_general.pdf` | Figs. 2 and 3, written by `scripts/fig_paper.py` |
| `figures/res_qual.pdf` | Fig. 4, written by `scripts/qual_frames.py` and `scripts/fig_qual.py` |
| `refs_res.bib` | references |
| `main_preprint.tex` | adds the notice and builds `main_preprint.pdf` (= `ji2027replan.pdf`) |
| `IEEEtran.cls` | IEEE conference class, V1.8b, unmodified |

## Build

```bash
cd paper
latexmk -pdf main_iceic_residual.tex   # the paper without the notice
latexmk -pdf main_preprint.tex         # preprint with the notice
```

Tested with TeX Live 2026 (pdfLaTeX, BibTeX). `main_preprint.tex` needs LaTeX 2020-10 or later.

Figs. 2 and 3 can be regenerated without a GPU from the result files in this repository (see
[Reproduce the paper's numbers](../README.md#reproduce-the-papers-numbers-no-gpu-or-simulator-needed));
Fig. 4 needs the simulator and a policy server. The main README says where every other number in the paper
comes from.

## Copyright

The paper's text and figures are © 2026 the authors and are **not** covered by the MIT License of the code in
this repository. If the paper is accepted, the copyright may be transferred to the IEEE; this preprint will then
be replaced by the accepted version with the IEEE copyright notice and, once assigned, its DOI. `IEEEtran.cls` is
© Michael Shell and contributors and is distributed under the LaTeX Project Public License; the complete package
is on [CTAN](https://ctan.org/pkg/ieeetran).
