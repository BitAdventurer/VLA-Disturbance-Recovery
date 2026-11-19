#!/usr/bin/env python3
"""compose_traj_panels.py DIR OUT.mp4 [--task T --init I] [--object NAME] [--slow 40,70,4] [--stride 2] [--fps 20]
[--topview]: the rollouts of compose_traj.py as three larger panels without plots. Besides the end effector's path,
each panel shows the wrist camera top right (--topview: compose_traj.py's top-view inset instead), marks every
gripper closure on the path (the first one larger) with a running count, leaves a ghost of the object at its old
position after a displacement, and pulses at every policy query (orange for the query a detection triggers). Steps
inside --slow A,B,F are shown F frames each (slow motion), outside it every --stride-th step; at --fps 20 a stride of
2 is twice real time."""
import argparse
import os
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compose_traj as CT     # noqa: E402

S, G, TOP, BOT = 448, 16, 78, 34
PIP = 156                     # wrist-camera inset
CT.S = S                      # the shared title and inset helpers lay out against this panel size


def closures(st, t0, t1):
    """Steps at which the gripper starts to close: the finger gap falls below 90% of its opening after being open."""
    gap = {t: st[t]["gripper"][0] - st[t]["gripper"][1] for t in sorted(st) if t0 <= t <= t1}
    top, out, armed = max(gap.values()), [], True
    for t, g in gap.items():
        if armed and g < 0.9 * top:
            out.append(t)
            armed = False
        elif not armed and g > 0.97 * top:
            armed = True
    return out


def ghost_layer(frame, proj, old, size):
    """The object at its old position, cut from a frame before the displacement, with a soft elliptic mask."""
    bx, by = proj(old)
    _, ty = proj(np.add(old, [0, 0, 0.10]))                  # about the top of a mug
    h = max(20.0, by - ty) + 14
    r = 0.62 * h
    box = [int(bx - r), int(by - h), int(bx + r), int(by + 12)]
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).ellipse(box, fill=120)
    return frame, mask.filter(ImageFilter.GaussianBlur(5))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dir")
    p.add_argument("out")
    p.add_argument("--task", type=int, default=6)
    p.add_argument("--init", type=int, default=37)
    p.add_argument("--object", default="porcelain_mug_1")
    p.add_argument("--slow", default="40,70,4", help="A,B,F: steps A..B shown F frames each")
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--topview", action="store_true")
    a = p.parse_args()
    sa, sb, sf = (int(v) for v in a.slow.split(","))
    runs = {arm: CT.load(os.path.join(a.dir, "%s_t%d_e%d" % (arm, a.task, a.init))) for arm, _, _ in CT.ARMS}
    meta0 = runs["student"][0]
    dist = meta0.get("disturbance")
    obj = dist["object"] if dist else a.object
    ev_t = dist["applied_t"] if dist else None
    noun = obj.rsplit("_", 1)[0].split("_")[-1]
    W, H = 3 * S + 4 * G, TOP + S + BOT
    T0 = 10
    ends = {arm: (r[0]["episode_length"] + T0, bool(r[0]["success"])) for arm, r in runs.items()}
    T1 = max(t for t, _ in ends.values())
    old = new = None
    st_s = runs["student"][1]
    if ev_t is not None:
        old = st_s[ev_t - 1]["objects"][obj]
        new = next(st_s[t]["objects"][obj] for t in sorted(st_s) if t >= ev_t
                   and np.linalg.norm(np.subtract(st_s[t]["objects"][obj][:2], old[:2])) > 0.01)
    proj, closes, queries, dets, ghosts = {}, {}, {}, {}, {}
    for arm, (meta, st, cam) in runs.items():
        proj[arm] = CT.projector(cam, S)
        closes[arm] = closures(st, T0, ends[arm][0])
        queries[arm] = [c["t"] for c in meta.get("calls", []) if c.get("t") is not None]
        dets[arm] = list(meta.get("detections") or [])
        if old is not None:
            src = Image.open(os.path.join(a.dir, "%s_t%d_e%d" % (arm, a.task, a.init), "steps", "agent_%04d.png" % T0))
            ghosts[arm] = ghost_layer(src.convert("RGB").resize((S, S), Image.LANCZOS), proj[arm], old, S)
    rlog = meta0.get("residual_log") or []
    est, win = None, None
    det_s = [t for t in dets["student"] if ev_t is not None and t >= ev_t]
    if det_s and rlog:
        after = [q for q in rlog if q["t"] >= det_s[0]]
        if after and after[0].get("d_hat") is not None:
            est = np.array([old[0] + after[0]["d_hat"][0], old[1] + after[0]["d_hat"][1], old[2]])
        active = [q["t"] for q in after if q.get("delta") is not None and np.linalg.norm(q["delta"]) > 1e-6]
        win = (det_s[0], (max(active) if active else det_s[0]) + 10)
    steps = []
    for t in range(T0, T1 + 1):
        if sa <= t <= sb:
            steps += [t] * sf
        elif (t - T0) % a.stride == 0 or t == T1:
            steps.append(t)
    tmp = os.path.splitext(a.out)[0] + "_frames"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    for k, t in enumerate(steps):
        im = Image.new("RGB", (W, H), "white")
        dr = ImageDraw.Draw(im)
        dr.text((W // 2, 12), "LIBERO-10 task %d, held-out start state %d: %s" % (a.task, a.init, meta0["task"]),
                font=CT.f_small, fill=CT.INK, anchor="ma")
        slow = sa <= t <= sb
        dr.text((W - G, 12), "slow motion ×1/%d" % sf if slow else "×%d speed" % a.stride, font=CT.f_lab,
                fill=CT.ORANGE if slow else CT.GREY, anchor="ra")
        for i, (arm, lab, col) in enumerate(CT.ARMS):
            meta, st, cam = runs[arm]
            te, ok = ends[arm]
            tt = min(t, te)
            x = G + i * (S + G)
            fr = Image.open(os.path.join(a.dir, "%s_t%d_e%d" % (arm, a.task, a.init), "steps", "agent_%04d.png" % tt))
            fr = fr.convert("RGB").resize((S, S), Image.LANCZOS)
            if old is not None and t > ev_t:
                g, m = ghosts[arm]
                fr = Image.composite(g, fr, m)
            d = ImageDraw.Draw(fr)
            if old is not None and t >= ev_t:
                ox, oy = proj[arm](old)
                d.line([(ox - 6, oy - 6), (ox + 6, oy + 6)], fill=CT.ORANGE, width=3)
                d.line([(ox - 6, oy + 6), (ox + 6, oy - 6)], fill=CT.ORANGE, width=3)
                if t < ev_t + 40:
                    d.text((ox, oy + 12), "old position", font=CT.f_tiny, fill=CT.ORANGE, anchor="ma",
                           stroke_width=2, stroke_fill="white")
                show_est = arm == "student" and win is not None and win[0] <= t <= win[1]
                if a.topview:
                    CT.inset(d, proj[arm], old, new, st[tt]["eef"], col, est if show_est else None)
            pts = [proj[arm](st[s]["eef"]) for s in range(T0, tt + 1) if s in st]
            if len(pts) > 1:
                d.line(pts, fill=col, width=3, joint="curve")
            done = [c for c in closes[arm] if c <= tt]
            for j, c in enumerate(done):                       # gripper closures on the path
                cx, cy = proj[arm](st[c]["eef"])
                r = 8 if j == 0 else 6
                d.polygon([(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)], fill=(20, 20, 20), outline="white")
                if j == 0 and tt < c + 40:
                    d.text((cx + 10, cy - 8), "1st close", font=CT.f_tiny, fill=(20, 20, 20), stroke_width=2,
                           stroke_fill="white")
            if pts:
                cx, cy = pts[-1]
                for q in queries[arm]:                          # a ring that widens after every policy query
                    if 0 <= tt - q <= 3:
                        rr = 11 + 5 * (tt - q)
                        qc = CT.ORANGE if q in dets[arm] else col
                        d.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], outline=qc, width=max(1, 3 - (tt - q) // 2))
                d.ellipse([cx - 7, cy - 7, cx + 7, cy + 7], fill="white", outline=col, width=3)
            d.text((8, S - 26), "gripper closes: %d" % len(done), font=CT.f_lab, fill="white", stroke_width=2,
                   stroke_fill=(30, 30, 30))
            if any(td <= t < td + 20 for td in dets[arm]):
                d.text((8, 8), "detected → replan" + (" + Δ" if arm == "student" else ""), font=CT.f_lab,
                       fill=CT.ORANGE, stroke_width=2, stroke_fill="white")
            if t >= te:
                d.text((8, 8), "task done (t = %d)" % te if ok else "timeout (t = %d)" % te, font=CT.f_lab,
                       fill="white", stroke_width=2, stroke_fill=(40, 40, 40))
            if not a.topview:                                  # wrist camera, top right
                wr = Image.open(os.path.join(a.dir, "%s_t%d_e%d" % (arm, a.task, a.init), "steps", "wrist_%04d.png" % tt))
                wr = wr.convert("RGB").resize((PIP, PIP), Image.LANCZOS)
                wx, wy = S - PIP - 8, 8
                d.rectangle([wx - 2, wy - 2, wx + PIP + 1, wy + PIP + 1], outline="white", width=2)
                fr.paste(wr, (wx, wy))
                d.text((wx + 5, wy + 4), "wrist camera", font=CT.f_tiny, fill="white", stroke_width=2,
                       stroke_fill=(30, 30, 30))
            im.paste(fr, (x, TOP))
            CT.title(dr, x, 42, arm, lab, col)
            if t >= te and ok:
                dr.rectangle([x - 3, TOP - 3, x + S + 2, TOP + S + 2], outline=col, width=4)
        tail = "× and ghost: the %s's old position" % noun if old is not None else "no displacement"
        dr.text((W // 2, H - 12), "t = %d (control step, 20 Hz)  ·  black diamonds: gripper closes  ·  rings: policy "
                "queries (orange: at a detection)  ·  %s" % (t, tail), font=CT.f_small, fill=CT.GREY, anchor="ms")
        im.save(os.path.join(tmp, "c_%04d.png" % k))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(a.fps), "-i", os.path.join(tmp, "c_%04d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart", a.out], check=True)
    shutil.rmtree(tmp)
    print("wrote", a.out, "frames", len(steps), "size %dx%d" % (W, H))


if __name__ == "__main__":
    main()
