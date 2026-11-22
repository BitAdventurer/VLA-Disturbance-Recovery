# Paper

> © 2027 IEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for all other
> uses, in any current or future media, including reprinting/republishing this material for advertising or
> promotional purposes, creating new collective works, for resale or redistribution to servers or lists, or reuse
> of any copyrighted component of this work in other works.

Accepted version of **When to Replan and How Much to Correct: Residual Reinforcement Learning for Disturbance
Recovery in Vision-Language-Action Policies** (Chang-Hoon Ji and Tae-Eui Kam, Korea University), accepted at
ICEIC 2027 (IEEE Publication Track).

**[ji2027replan.pdf](ji2027replan.pdf)** is the accepted version (6 pages; text as of 2026-11-23) with an acceptance line
and the notice above added in the top margin of page 1; the text is otherwise identical to `main_iceic_residual.tex`.
The DOI will be added here once the paper is in IEEE Xplore.

| File | Content |
|---|---|
| `main_iceic_residual.tex` | the paper |
| `fig_overview_res.tex` | Fig. 1 (TikZ) |
| `figures/res_overview.pdf` | Fig. 1 on its own page, for slides and posters (`scripts/fig1_standalone.py`, then `pdflatex fig1_standalone.tex`); not used by the paper build |
| `figures/res_forest.pdf`, `figures/res_general.pdf` | Figs. 2 and 3, written by `scripts/fig_paper.py` |
| `figures/res_qual.pdf` | Fig. 4, written by `scripts/qual_frames.py` and `scripts/fig_qual.py` |
| `refs_res.bib` | references |
| `main_preprint.tex` | adds the notice and builds `main_preprint.pdf` (= `ji2027replan.pdf`) |
| `IEEEtran.cls` | IEEE conference class, V1.8b, unmodified |

## Build

```bash
cd paper
latexmk -pdf main_iceic_residual.tex   # the paper without the notice
latexmk -pdf main_preprint.tex         # accepted version with the notice
```

Tested with TeX Live 2026 (pdfLaTeX, BibTeX). `main_preprint.tex` needs LaTeX 2020-10 or later.

Figs. 2 and 3 can be regenerated without a GPU from the result files in this repository (see
[Reproduce the paper's numbers](../README.md#reproduce-the-papers-numbers-no-gpu-or-simulator-needed));
Fig. 4 needs the simulator and a policy server. The main README says where every other number in the paper
comes from.

## Copyright

The paper's text and figures are © 2027 IEEE (see the notice above) and are **not** covered by the MIT License of
the code in this repository. This folder holds the accepted version, not the version published by the IEEE; the
DOI will be added once the paper is in IEEE Xplore. `IEEEtran.cls` is © Michael Shell and contributors and is
distributed under the LaTeX Project Public License; the complete package is on [CTAN](https://ctan.org/pkg/ieeetran).
