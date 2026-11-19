#!/usr/bin/env python3
"""fig1_standalone.py [PAPER_DIR]: writes PAPER_DIR/fig1_standalone.tex (default: ../paper), Fig. 1 of the paper on
its own page: the TikZ picture of fig_overview_res.tex with the paper's class, fonts and scale (0.97 textwidth), no
caption, the page cropped to the figure. Then `pdflatex fig1_standalone.tex` in PAPER_DIR gives the single figure
(released as paper/figures/res_overview.pdf; the paper itself inputs fig_overview_res.tex directly)."""
import os
import sys

D = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "paper")
src = open(os.path.join(D, "fig_overview_res.tex")).read()
body = src[src.index("\\definecolor{cexec}"):src.index("\\caption{Overview.")].rstrip()

doc = r"""\documentclass[conference,a4paper]{IEEEtran}
% Written by scripts/fig1_standalone.py from fig_overview_res.tex: Fig. 1 alone, page cropped to the figure.
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{graphicx}
\usepackage{xcolor}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,decorations.pathreplacing,calc,positioning}
\newsavebox{\figbox}
\begin{document}
\sbox{\figbox}{%
""" + body + r"""}
\pdfpagewidth=\wd\figbox
\pdfpageheight=\dimexpr\ht\figbox+\dp\figbox\relax
\hoffset=-1in \voffset=-1in
\shipout\hbox{\usebox{\figbox}}
\end{document}
"""
out = os.path.join(D, "fig1_standalone.tex")
open(out, "w").write(doc)
print("wrote", out)
