#!/usr/bin/env python3
"""compose_traj.py DIR OUT.mp4 [--task T --init I] [--object NAME] [--fps 20] [--stride 1]: side-by-side video of
one start state under the base policy, the detector replan and the residual student, from rollouts written by
`video_frames.py --log_state`.

Panels: the end effector's path projected into the agent view, a marker at its current position and, after a
displacement, an x at the object's old position and a top-view inset (cm, oriented as the camera view) with the true
move, the gripper's horizontal position and, on the student panel, its displacement estimate while its residual is
active. Plots below, with one cursor at the current step: the policy queries of
each arm; the learned detector's per-step score against its threshold; the student's residual at each query; the
horizontal distance from the end effector to the object (solid) and to its old position (dashed) until the object is
first lifted. Without a displacement (--disturb none rollouts) --object names the object to track. t is the control
step at 20 Hz; --fps 20 --stride 1 is real time, --stride 2 twice as fast."""
import argparse
import json
import os
import shutil
import subprocess

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

FONT = "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
BOLD = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
INK, GREY, ORANGE = (33, 33, 33), (110, 110, 110), (213, 94, 0)
ARMS = [("base", "alone", (110, 110, 110)), ("replan", "+ replan at the event", (0, 114, 178)),
        ("student", "+ replan + residual student", (0, 158, 115))]
S, G, TOP, PH, BOT = 384, 16, 78, 452, 34            # panel size, gap, task and title rows, plots, footer
TAU = 1.628                                          # detector threshold (step_tau of the paper's runs)
f_head, f_sub, f_small, f_lab = (ImageFont.truetype(BOLD, 21), ImageFont.truetype(BOLD, 14),
                                 ImageFont.truetype(FONT, 15), ImageFont.truetype(FONT, 14))
f_tiny = ImageFont.truetype(FONT, 12)


def rgb(c):
    return tuple(v / 255 for v in c)


def load(d):
    meta = json.load(open(os.path.join(d, "meta.json")))
    st = [json.loads(l) for l in open(os.path.join(d, "steps", "states.jsonl"))]
    cam = json.load(open(os.path.join(d, "steps", "camera.json")))
    return meta, {s["t"]: s for s in st}, cam


def projector(cam, size):
    """World point -> pixel in the saved (180-degree rotated) frame, scaled to the panel size."""
    m, h, w = np.asarray(cam["world_to_pixel"]), cam["height"], cam["width"]

    def proj(p):
        q = m @ np.append(np.asarray(p, float), 1.0)
        return (w - 1 - q[0] / q[2]) * size / w, q[1] / q[2] * size / h     # checked against the frames: x mirrored
    return proj


def title(dr, x, y, arm, txt, col):
    """Panel title; the base one starts with pi_0.5, its subscript drawn smaller (the font has no subscript digits)."""
    w = dr.textlength(txt, font=f_head) + (dr.textlength("π0.5 ", font=f_sub) + 4 if arm == "base" else 0)
    x += (S - w) / 2
    if arm == "base":
        dr.text((x, y), "π", font=f_head, fill=col)
        x += dr.textlength("π", font=f_head)
        dr.text((x, y + 9), "0.5", font=f_sub, fill=col)
        x += dr.textlength("0.5 ", font=f_sub)
    dr.text((x, y), txt, font=f_head, fill=col)


def arrow(d, p0, p1, col, dashed=False, width=3):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    v = p1 - p0
    n = np.linalg.norm(v)
    if n < 1:
        return
    u = v / n
    tip = p1 - 9 * u                                   # shaft stops at the head
    if dashed:
        for a in np.arange(0, max(0.0, n - 9), 8):
            b = min(a + 4.5, n - 9)
            d.line([tuple(p0 + a * u), tuple(p0 + b * u)], fill=col, width=width)
    else:
        d.line([tuple(p0), tuple(tip)], fill=col, width=width)
    perp = np.array([-u[1], u[0]])
    d.polygon([tuple(p1), tuple(tip + 5 * perp), tuple(tip - 5 * perp)], fill=col)


def inset(d, proj, old, new, eef, col, est=None, size=132, span=8.0):
    """Top view around the old position (x), oriented like the camera view: the true move (orange), the
    displacement estimate (dashed) and the gripper's horizontal position (dot); +-span cm are shown."""
    o = np.array(proj(old))
    ex = np.array(proj(np.add(old, [0.01, 0, 0]))) - o
    ey = np.array(proj(np.add(old, [0, 0.01, 0]))) - o
    ex, ey = ex / np.linalg.norm(ex), ey / np.linalg.norm(ey)
    c = np.array([S - size / 2 - 8, S - size / 2 - 8])
    k = (size / 2 - 6) / span                          # pixels per cm

    def at(p):
        v = c + k * (100 * (p[0] - old[0]) * ex + 100 * (p[1] - old[1]) * ey)
        return tuple(np.clip(v, c - size / 2 + 4, c + size / 2 - 4))
    d.rectangle([c[0] - size / 2, c[1] - size / 2, c[0] + size / 2, c[1] + size / 2], fill=(250, 250, 250),
                outline=(170, 170, 170), width=1)
    d.text((c[0] - size / 2 + 5, c[1] - size / 2 + 3), "top view", font=f_tiny, fill=GREY)
    d.text((c[0] - size / 2 + 5, c[1] + size / 2 - 30), "\u2192 true move", font=f_tiny, fill=ORANGE)
    if est is not None:
        d.text((c[0] - size / 2 + 5, c[1] + size / 2 - 16), "- - estimate", font=f_tiny, fill=INK)
    d.line([(c[0] + size / 2 - 5 - 2 * k, c[1] - size / 2 + 10), (c[0] + size / 2 - 5, c[1] - size / 2 + 10)],
           fill=GREY, width=2)
    d.text((c[0] + size / 2 - 9 - 2 * k, c[1] - size / 2 + 3), "2 cm", font=f_tiny, fill=GREY, anchor="ra")
    arrow(d, c, at(new), ORANGE)
    if est is not None:
        arrow(d, c, at(est), INK, dashed=True)
    d.line([(c[0] - 6, c[1] - 6), (c[0] + 6, c[1] + 6)], fill=ORANGE, width=3)
    d.line([(c[0] - 6, c[1] + 6), (c[0] + 6, c[1] - 6)], fill=ORANGE, width=3)
    gx, gy = at(eef)
    d.ellipse([gx - 6, gy - 6, gx + 6, gy + 6], fill="white", outline=col, width=3)


def plots(data, t0, t1, width, noun):
    """Four stacked plots as one image; returns it with the pixel x of t0 and t1 and the rows of the top and bottom."""
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
                         "font.size": 9.5, "axes.linewidth": 0.6})
    fig = plt.figure(figsize=(width / 100, PH / 100), dpi=100)
    gs = fig.add_gridspec(4, 1, height_ratios=[0.42, 1.0, 0.72, 1.3], hspace=0.16, left=0.095, right=0.99,
                          top=0.985, bottom=0.10)
    ax = [fig.add_subplot(gs[i]) for i in range(4)]
    ev_t = data["ev_t"]
    for a in ax:
        a.set_xlim(t0, t1)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        if ev_t is not None:
            a.axvline(ev_t, color=rgb(ORANGE), lw=1.0, ls=(0, (3, 2)))
        if a is not ax[-1]:
            a.tick_params(labelbottom=False)
    # 1. policy queries, one lane per arm
    for i, (arm, lab, col) in enumerate(ARMS):
        ax[0].vlines(data["queries"][arm], 2 - i + 0.18, 2 - i + 0.82, color=rgb(col), lw=1.3)
    ax[0].set_ylim(0, 3)
    ax[0].set_yticks([])
    ax[0].spines["left"].set_visible(False)
    ax[0].set_ylabel("policy\nqueries", rotation=0, ha="right", va="center", labelpad=6)
    # 2. detector score per step
    hi = TAU + 1.0
    for arm, lab, col in ARMS:
        sc = data["scores"].get(arm)
        if not sc:
            continue
        t, v = zip(*sc)
        ax[1].plot(t, v, color=rgb(col), lw=1.1)
        for td in data["dets"].get(arm, []):
            ax[1].plot([td], [np.interp(td, t, v)], marker="v", color=rgb(col), ms=6, mec="white", mew=0.6,
                       zorder=5)
        hi = max(hi, max(v) + 0.8)
    ax[1].axhline(TAU, color=rgb(GREY), lw=0.9, ls=(0, (4, 3)))
    ax[1].text(t1, TAU, " threshold", color=rgb(GREY), fontsize=8.5, va="bottom", ha="right")
    lo = min([min(v for _, v in sc) for sc in data["scores"].values() if sc] + [0.0]) - 0.5
    ax[1].set_ylim(lo, hi)
    if ev_t is not None:
        ax[1].text(ev_t + 9, hi - 0.06 * (hi - lo), "%s moved %.1f cm" % (noun, data["mag"]), color=rgb(ORANGE),
                   fontsize=9, va="top")
    ax[1].set_ylabel("detector\nscore")
    # 3. the student's residual at each query
    rq = data["resid"]
    if rq:
        t, n = zip(*rq)
        ax[2].bar(t, n, width=5, color=rgb(ARMS[2][2]), align="center")
    ax[2].set_ylim(0, max([0.1] + [v for _, v in rq]) * 1.25)
    ax[2].set_ylabel("student\n|Δ|")
    # 4. distance to the object (solid) and to its old position (dashed), until the first lift
    top = 1.0
    for arm, lab, col in ARMS:
        t, d = data["dist"][arm]
        ax[3].plot(t, d, color=rgb(col), lw=1.6)
        top = max(top, max(d))
        if data["dist_old"].get(arm):
            to, do = data["dist_old"][arm]
            ax[3].plot(to, do, color=rgb(col), lw=1.1, ls=(0, (3, 2)))
            top = max(top, max(do))
        if data["lifts"].get(arm) is not None:
            ax[3].plot([t[-1]], [d[-1]], marker="^", color=rgb(col), ms=7, mec="white", mew=0.6, zorder=5)
        te, ok = data["ends"][arm]
        ax[3].plot([te], [0], marker="o" if ok else "s", color=rgb(col), ms=7, mec="white", mew=0.6, zorder=6,
                   clip_on=False)
    ax[3].set_ylim(0, top * 1.12)
    ax[3].set_ylabel("gripper to %s\n(horizontal, cm)" % noun)
    ax[3].plot([], [], color="k", lw=1.4, label="to the %s" % noun)
    if ev_t is not None:
        ax[3].plot([], [], color="k", lw=1.0, ls=(0, (3, 2)), label="to its old position")
    ax[3].legend(loc="upper right", frameon=False, ncol=2, fontsize=9, handlelength=2.2)
    ax[3].set_xlabel("control step (20 Hz)    ▼ detection    ▲ %s lifted    ● task done    ■ timeout" % noun, labelpad=2)
    for a in ax[1:]:
        a.grid(axis="y", alpha=0.25, lw=0.5)
    fig.canvas.draw()
    img = Image.frombuffer("RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba()).convert("RGB")
    x0 = ax[3].transData.transform((t0, 0))[0]
    x1 = ax[3].transData.transform((t1, 0))[0]
    y_top = img.height - ax[0].transAxes.transform((0, 1))[1]
    y_bot = img.height - ax[3].transAxes.transform((0, 0))[1]
    plt.close(fig)
    return img, x0, x1, y_top, y_bot


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dir")
    p.add_argument("out")
    p.add_argument("--task", type=int, default=6)
    p.add_argument("--init", type=int, default=37)
    p.add_argument("--object", default="porcelain_mug_1", help="object to track when there is no displacement")
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--stride", type=int, default=1)
    a = p.parse_args()
    runs = {arm: load(os.path.join(a.dir, "%s_t%d_e%d" % (arm, a.task, a.init))) for arm, _, _ in ARMS}
    meta0 = runs["student"][0]
    dist = meta0.get("disturbance")
    obj = dist["object"] if dist else a.object
    ev_t = dist["applied_t"] if dist else None
    noun = obj.rsplit("_", 1)[0].split("_")[-1]                # porcelain_mug_1 -> mug
    W = 3 * S + 4 * G
    H = TOP + S + 10 + PH + BOT
    H += H % 2
    T0 = 10                                                   # the first control step after settling
    ends = {arm: (r[0]["episode_length"] + T0, bool(r[0]["success"])) for arm, r in runs.items()}
    T1 = max(t for t, _ in ends.values())
    old = new = None
    if ev_t is not None:
        st_s = runs["student"][1]
        old = st_s[ev_t - 1]["objects"][obj]                  # the object just before the displacement
        new = next(st_s[t]["objects"][obj] for t in sorted(st_s) if t >= ev_t       # first step it has moved
                   and np.linalg.norm(np.subtract(st_s[t]["objects"][obj][:2], old[:2])) > 0.01)
    data = {"ev_t": ev_t, "mag": 100 * dist["mag"] if dist else 0.0, "queries": {}, "scores": {}, "dets": {},
            "resid": [], "dist": {}, "dist_old": {}, "lifts": {}, "ends": ends}
    proj = {}
    for arm, (meta, st, cam) in runs.items():
        proj[arm] = projector(cam, S)
        data["queries"][arm] = [c["t"] for c in meta.get("calls", []) if c.get("t") is not None]
        data["scores"][arm] = [tuple(x) for x in meta.get("detector_scores") or []]
        data["dets"][arm] = list(meta.get("detections") or [])
        z0 = st[T0]["objects"][obj][2]
        up = [t for t in sorted(st) if t > T0 and st[t]["objects"][obj][2] - z0 > 0.03]
        data["lifts"][arm] = up[0] if up else None            # first lift of the tracked object (3 cm)
        ts = sorted(t for t in st if T0 <= t <= (data["lifts"][arm] or ends[arm][0]))
        data["dist"][arm] = (ts, [100 * np.linalg.norm(np.subtract(st[t]["objects"][obj][:2], st[t]["eef"][:2]))
                                  for t in ts])
        if old is not None:
            to = [t for t in ts if t >= ev_t]
            data["dist_old"][arm] = (to, [100 * np.linalg.norm(np.subtract(old[:2], st[t]["eef"][:2])) for t in to])
    rlog = runs["student"][0].get("residual_log") or []
    data["resid"] = [(q["t"], float(np.linalg.norm(q["delta"]))) for q in rlog if q.get("delta") is not None]
    # the student's displacement estimate and the steps its residual is active after the first detection
    est, win = None, None
    det_s = [t for t in data["dets"]["student"] if ev_t is not None and t >= ev_t]
    if det_s and rlog:
        after = [q for q in rlog if q["t"] >= det_s[0]]
        if after and after[0].get("d_hat") is not None:
            est = np.array([old[0] + after[0]["d_hat"][0], old[1] + after[0]["d_hat"][1], old[2]])
        active = [q["t"] for q in after if q.get("delta") is not None and np.linalg.norm(q["delta"]) > 1e-6]
        win = (det_s[0], (max(active) if active else det_s[0]) + 10)
    pimg, px0, px1, ptop, pbot = plots(data, T0, T1, W - 2 * G, noun)
    tmp = os.path.splitext(a.out)[0] + "_frames"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    k = -1
    for k, t in enumerate(range(T0, T1 + 1, a.stride)):
        im = Image.new("RGB", (W, H), "white")
        dr = ImageDraw.Draw(im)
        dr.text((W // 2, 12), "LIBERO-10 task %d, held-out start state %d: %s" % (a.task, a.init, meta0["task"]),
                font=f_small, fill=INK, anchor="ma")
        for i, (arm, lab, col) in enumerate(ARMS):
            meta, st, cam = runs[arm]
            te, ok = ends[arm]
            tt = min(t, te)
            x = G + i * (S + G)
            fr = Image.open(os.path.join(a.dir, "%s_t%d_e%d" % (arm, a.task, a.init), "steps", "agent_%04d.png" % tt))
            fr = fr.convert("RGB").resize((S, S), Image.LANCZOS)
            d = ImageDraw.Draw(fr)
            if old is not None and t >= ev_t:
                ox, oy = proj[arm](old)
                d.line([(ox - 7, oy - 7), (ox + 7, oy + 7)], fill=ORANGE, width=3)
                d.line([(ox - 7, oy + 7), (ox + 7, oy - 7)], fill=ORANGE, width=3)
            if old is not None and t >= ev_t:
                show_est = arm == "student" and win is not None and win[0] <= t <= win[1]
                inset(d, proj[arm], old, new, st[tt]["eef"], col, est if show_est else None)
            pts = [proj[arm](st[s]["eef"]) for s in range(T0, tt + 1) if s in st]
            if len(pts) > 1:
                d.line(pts, fill=col, width=3, joint="curve")
            if pts:
                cx, cy = pts[-1]
                d.ellipse([cx - 7, cy - 7, cx + 7, cy + 7], fill="white", outline=col, width=3)
            if any(td <= t < td + 20 for td in data["dets"].get(arm, [])):
                d.text((8, 8), "detected → replan" + (" + Δ" if arm == "student" else ""), font=f_lab,
                       fill=ORANGE, stroke_width=2, stroke_fill="white")
            im.paste(fr, (x, TOP))
            title(dr, x, 42, arm, lab, col)
            if t >= te:
                if ok:
                    dr.rectangle([x - 3, TOP - 3, x + S + 2, TOP + S + 2], outline=col, width=4)
                dr.text((x + S / 2, TOP + S - 26), "task done (t = %d)" % te if ok else "timeout (t = %d)" % te,
                        font=f_lab, fill="white", anchor="ma", stroke_width=2, stroke_fill=(40, 40, 40))
        im.paste(pimg, (G, TOP + S + 10))
        cx = G + px0 + (px1 - px0) * (t - T0) / max(1, T1 - T0)
        dr.line([(cx, TOP + S + 10 + ptop), (cx, TOP + S + 10 + pbot)], fill=(60, 60, 60), width=2)
        tail = "× the %s's old position" % noun if old is not None else "no displacement"
        dr.text((W // 2, H - 12), "t = %d (control step, 20 Hz)  ·  same start state and sampling noise in all panels  ·  %s"
                % (t, tail), font=f_small, fill=GREY, anchor="ms")
        im.save(os.path.join(tmp, "c_%04d.png" % k))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(a.fps), "-i", os.path.join(tmp, "c_%04d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart", a.out], check=True)
    shutil.rmtree(tmp)
    print("wrote", a.out, "frames", k + 1, "size %dx%d" % (W, H))


if __name__ == "__main__":
    main()
