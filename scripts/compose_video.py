#!/usr/bin/env python3
"""compose_video.py DIR OUT.mp4 [FPS]: side-by-side video of the Fig. 4 episode (task 6, init 37) from the per-step
agent-view frames written by video_frames.py: left the base policy alone, right with the detector and the residual
student. t is the control step (20 Hz) as in Fig. 4; the captions are the figure's. FPS 20 is real time."""
import json
import os
import subprocess
import sys

from PIL import Image, ImageDraw, ImageFont

DIR, OUT = sys.argv[1], sys.argv[2]
FPS = int(sys.argv[3]) if len(sys.argv) > 3 else 20
FONT = "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
BOLD = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
INK, GREY, ORANGE, GREEN = (33, 33, 33), (110, 110, 110), (213, 94, 0), (0, 140, 100)
S, G, TOP, BOT = 512, 24, 96, 76            # panel size, gap, header and footer heights
W, H = 2 * S + 3 * G, TOP + S + BOT

f_title, f_head, f_cap, f_small = (ImageFont.truetype(BOLD, 22), ImageFont.truetype(BOLD, 24),
                                   ImageFont.truetype(FONT, 24), ImageFont.truetype(FONT, 18))
meta = json.load(open(os.path.join(DIR, "student_t6_e37", "meta.json")))
d = meta["disturbance"]
moved = "mug moved %.1f cm" % (100 * d["mag"])

# captions of Fig. 4 (start t, end t, text, colour); the last one stays to the end
CAP = {
    "base": [(40, 47, "approach", INK), (47, 70, moved + ", no reaction", ORANGE), (70, 130, "closes at the old position", INK),
             (130, 200, "pushes the mug", INK), (430, 480, "knocks it over", INK), (520, 10**6, "timeout (t = 530)", GREY)],
    "student": [(40, 47, "approach", INK), (47, 68, moved + ": detected, replan + Δ", ORANGE),
                (68, 97, "shifted toward the mug", INK), (97, 130, "grasp", INK), (257, 10**6, "task done", GREEN)],
}
HEAD = {"base": ("alone", GREY), "student": ("+ detector and residual student", GREEN)}
f_sub = ImageFont.truetype(BOLD, 16)


def head(dr, x, y, arm):
    """Panel title; the base one starts with pi_0.5, its subscript drawn smaller (the font has no subscript digits)."""
    txt, col = HEAD[arm]
    if arm == "base":
        dr.text((x, y), "π", font=f_head, fill=col)
        x += dr.textlength("π", font=f_head)
        dr.text((x, y + 11), "0.5", font=f_sub, fill=col)
        x += dr.textlength("0.5 ", font=f_sub)
    dr.text((x, y), txt, font=f_head, fill=col)


def frames(arm):
    p = os.path.join(DIR, "%s_t6_e37" % arm, "steps")
    n = len([f for f in os.listdir(p) if f.startswith("agent_")])
    return p, n


def caption(arm, t):
    for a, b, txt, col in CAP[arm]:
        if a <= t < b:
            return txt, col
    return None, None


pb, nb = frames("base")
ps, ns = frames("student")
T0, T1 = 10, max(nb, ns)                    # start after the 10 settling steps
tmp = os.path.join(DIR, "composed")
os.makedirs(tmp, exist_ok=True)
for k, t in enumerate(range(T0, T1 + 1)):
    im = Image.new("RGB", (W, H), "white")
    dr = ImageDraw.Draw(im)
    dr.text((G, 14), "LIBERO-10: " + meta["task"], font=f_title, fill=INK)
    for i, (arm, p, n) in enumerate((("base", pb, nb), ("student", ps, ns))):
        x = G + i * (S + G)
        fr = Image.open(os.path.join(p, "agent_%04d.png" % min(t, n))).resize((S, S), Image.LANCZOS)
        im.paste(fr, (x, TOP))
        head(dr, x, TOP - 34, arm)
        done = t >= n
        if arm == "student" and t >= 257:
            dr.rectangle([x - 3, TOP - 3, x + S + 2, TOP + S + 2], outline=GREEN, width=5)
        txt, c = caption(arm, t)
        if txt:
            dr.text((x, TOP + S + 12), txt, font=f_cap, fill=c)
        if done:
            dr.text((x + S - 4, TOP + S + 14), "episode ended", font=f_small, fill=GREY, anchor="ra")
    dr.text((W // 2, H - 22), "t = %d   (control step, 20 Hz; same start state and sampling noise in both)" % t,
            font=f_small, fill=GREY, anchor="ms")
    im.save(os.path.join(tmp, "c_%04d.png" % k))
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", os.path.join(tmp, "c_%04d.png"),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-movflags", "+faststart", OUT], check=True)
print("wrote", OUT, "frames", T1 - T0 + 1, "fps", FPS)
