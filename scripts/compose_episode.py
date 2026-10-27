#!/usr/bin/env python3
"""compose_episode.py OUT.mp4 --dirs base:DIR,replan:DIR,student:DIR --label TEXT [--fps 20] [--thumb OUT.jpg]
Side-by-side video of one episode replayed by video_frames.py under each arm (same start state, disturbance and
sampling noise). Captions come from each arm's meta.json: when the object is moved, each detector firing (a firing
before the move, or in an undisturbed episode, is a false alarm), and the end (task done or timeout). t is the control
step at 20 Hz; --fps 20 is real time. --thumb writes a still from shortly after the move for the README."""
import argparse
import json
import os
import shutil
import subprocess

from PIL import Image, ImageDraw, ImageFont

FONT = "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
BOLD = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
INK, GREY, ORANGE, GREEN = (33, 33, 33), (110, 110, 110), (213, 94, 0), (0, 140, 100)
S, G, TOP, BOT = 512, 24, 96, 76            # panel size, gap, header and footer heights
TITLE = {"base": ("alone", GREY), "replan": ("+ replan at the detected event", INK),
         "student": ("+ replan + residual student", GREEN)}
f_title, f_head, f_sub, f_small = (ImageFont.truetype(BOLD, 22), ImageFont.truetype(BOLD, 24),
                                   ImageFont.truetype(BOLD, 16), ImageFont.truetype(FONT, 18))


def font_fit(dr, text, width, size=24):
    while size > 14:
        f = ImageFont.truetype(FONT, size)
        if dr.textlength(text, font=f) <= width:
            return f
        size -= 1
    return ImageFont.truetype(FONT, size)


def pretty(obj):
    parts = obj.split("_")
    return " ".join(parts[:-1] if parts[-1].isdigit() else parts)


def events(meta, arm, t_end):
    """(start, end, text, colour); the caption shown at t is the one with the latest start among those active."""
    ev = []
    d = meta.get("disturbance") or None
    ta = d.get("applied_t") if d else None
    moved = "%s moved %.1f cm" % (pretty(d["object"]), 100 * d["mag"]) if ta is not None else None
    dets = meta.get("detections") or []
    caught = ta is not None and any(ta <= td <= ta + 10 for td in dets)
    if moved and not caught:                  # a detection right at the move gets one merged caption below
        ev.append((ta, ta + 30, moved + (", no reaction" if arm == "base" else ""), ORANGE))
    what = "replan" if arm == "replan" else "replan + Δ"
    for td in dets:
        if ta is not None and ta <= td <= ta + 10:
            ev.append((ta, td + 30, "%s: detected, %s" % (moved, what), ORANGE))
        elif ta is None or td < ta:
            ev.append((td, td + 30, "false alarm: " + what, GREY))
        else:                                 # a later firing, not right at the move
            ev.append((td, td + 30, "detector fires: " + what, ORANGE))
    done = float(meta.get("success") or 0) > 0.5
    ev.append((t_end, 10 ** 6, ("task done (t = %d)" if done else "timeout (t = %d)") % t_end, GREEN if done else GREY))
    return ev, done


def caption(ev, t):
    act = [e for e in ev if e[0] <= t < e[1]]
    return max(act, key=lambda e: e[0])[2:] if act else (None, None)


def head(dr, x, y, arm):
    txt, col = TITLE[arm]
    if arm == "base":                        # pi_0.5; the font has no subscript digits
        dr.text((x, y), "π", font=f_head, fill=col)
        x += dr.textlength("π", font=f_head)
        dr.text((x, y + 11), "0.5", font=f_sub, fill=col)
        x += dr.textlength("0.5 ", font=f_sub)
    dr.text((x, y), txt, font=f_head, fill=col)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("--dirs", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--thumb")
    args = p.parse_args()
    panels = []
    for item in args.dirs.split(","):
        arm, d = item.split(":", 1)
        meta = json.load(open(os.path.join(d, "meta.json")))
        steps = os.path.join(d, "steps")
        n = len([f for f in os.listdir(steps) if f.startswith("agent_")])
        ev, done = events(meta, arm, n)
        panels.append(dict(arm=arm, steps=steps, n=n, ev=ev, done=done, meta=meta))
    m0 = panels[0]["meta"]
    W, H = len(panels) * S + (len(panels) + 1) * G, TOP + S + BOT
    T0, T1 = 10, max(pn["n"] for pn in panels)            # start after the 10 settling steps
    tmp = os.path.splitext(args.out)[0] + "_frames"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    d = m0.get("disturbance") or None
    ta = d.get("applied_t") if d else None
    det = [t for t in (panels[-1]["meta"].get("detections") or []) if ta is None or t >= ta]
    t_thumb = min(T1, (det[0] + 20) if det else ((ta + 20) if ta is not None else T0 + 120))   # last panel's response
    for k, t in enumerate(range(T0, T1 + 1)):
        im = Image.new("RGB", (W, H), "white")
        dr = ImageDraw.Draw(im)
        dr.text((G, 14), "LIBERO-10 task %d: %s" % (m0["task_id"], m0["task"]), font=f_title, fill=INK)
        for i, pn in enumerate(panels):
            x = G + i * (S + G)
            fr = Image.open(os.path.join(pn["steps"], "agent_%04d.png" % min(t, pn["n"]))).resize((S, S), Image.LANCZOS)
            im.paste(fr, (x, TOP))
            head(dr, x, TOP - 34, pn["arm"])
            if pn["done"] and t >= pn["n"]:
                dr.rectangle([x - 3, TOP - 3, x + S + 2, TOP + S + 2], outline=GREEN, width=5)
            txt, col = caption(pn["ev"], t)
            if txt:
                dr.text((x, TOP + S + 12), txt, font=font_fit(dr, txt, S - 4), fill=col)
        dr.text((W // 2, H - 22), "t = %d  (control step, 20 Hz)  ·  %s  ·  same start state and sampling noise in all panels"
                % (t, args.label), font=f_small, fill=GREY, anchor="ms")
        im.save(os.path.join(tmp, "c_%04d.png" % k))
        if args.thumb and t == t_thumb:
            im.resize((W * 720 // W, H * 720 // W), Image.LANCZOS).save(args.thumb, quality=88)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(args.fps), "-i", os.path.join(tmp, "c_%04d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", "-movflags", "+faststart", args.out], check=True)
    shutil.rmtree(tmp)
    print("wrote", args.out, "frames", T1 - T0 + 1)


if __name__ == "__main__":
    main()
