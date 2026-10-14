#!/usr/bin/env python3
"""fig_qual.py OUT.pdf [DPI]: qualitative frame strip of one disturbed episode (task 6, init 37, mug displaced 5.1 cm)
for the base policy and the residual student, from results_res/qual/{base,student}_t6_e37/frame_%04d.png.
Without DPI the frames are rasterized at matplotlib's default 100 dpi (the paper figure); the poster copy uses
DPI=600 so that the frames are not downsampled to 56 px when printed on A0."""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

Q = os.path.expanduser(os.environ.get("VLA_EXPERIMENTS", "~/disk2/vla-inference-scheduling/experiments") + "/results_res/qual")
plt.rcParams.update({"font.size": 6.5, "pdf.fonttype": 42, "font.family": "sans-serif",
                     "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"]})
ORANGE, GREEN, VERM, GREY = "#E69F00", "#009E73", "#D55E00", "#7F7F7F"

rows = [
    ("$\\pi_{0.5}$ alone\n(timeout)", "base_t6_e37", [40, 50, 70, 130, 430],
     ["approach\n$t{=}40$", "displaced, no\nreaction $t{=}50$", "closes at old\nposition $t{=}70$", "pushes the\nmug $t{=}130$", "knocked\nover $t{=}430$"]),
    ("+ residual\nstudent\n(success)", "student_t6_e37", [40, 48, 68, 97, 257],
     ["approach\n$t{=}40$", "detected, replan\n+ $\\Delta$, $t{=}48$", "shifted toward\nmug $t{=}68$", "grasp\n$t{=}97$", "task done\n$t{=}257$"]),
]
crop = (36, 28, 204, 200)      # x0, y0, x1, y1 of the 224x224 agent view (table region incl. the plate)
ncol = len(rows[0][2])
fig = plt.figure(figsize=(3.45, 1.62))
gs = fig.add_gridspec(2, ncol, left=0.165, right=0.99, top=0.885, bottom=0.015, wspace=0.03, hspace=0.5)
for r, (label, d, ts, caps) in enumerate(rows):
    for c, (t, cap) in enumerate(zip(ts, caps)):
        ax = fig.add_subplot(gs[r, c])
        im = Image.open(os.path.join(Q, d, "frame_%04d.png" % t)).crop(crop)
        ax.imshow(np.asarray(im))
        ax.set_xticks([]); ax.set_yticks([])
        col = GREY if r == 0 else GREEN
        for s in ax.spines.values():
            s.set_edgecolor(col); s.set_linewidth(0.8)
        ax.set_title(cap, fontsize=5.8, pad=1.5, color=(VERM if (c == 1) else "black"), linespacing=0.95)
        if c == 0:
            ax.text(-0.1, 0.5, label, transform=ax.transAxes, ha="right", va="center", fontsize=6.5, color=col,
                    fontweight="bold", linespacing=1.1)
if len(sys.argv) > 2:
    fig.savefig(sys.argv[1], dpi=int(sys.argv[2]))
else:
    fig.savefig(sys.argv[1])
print("saved", sys.argv[1])
