#!/usr/bin/env python3
"""Where does the gripper close after the displacement: near the object's OLD or NEW position?
Reads --log_exec runs (exec rows: t, action(7), eef(3), displaced object pos(3)). Old position = new - (mag*dir, 0)."""
import glob
import json
import os
import sys

import numpy as np

E = os.path.expanduser(os.environ.get("VLA_EXPERIMENTS", "~/disk2/vla-inference-scheduling/experiments") + "/results_res")
rows = []
for f in sorted(glob.glob(os.path.join(E, sys.argv[1] if len(sys.argv) > 1 else "diag_*.jsonl"))):
    for line in open(f):
        r = json.loads(line)
        d = r.get("disturbance") or {}
        if not r.get("exec") or d.get("object") is None:
            continue
        X = np.array(r["exec"], dtype=float)
        t, a, eef, po = X[:, 0], X[:, 1:8], X[:, 8:11], X[:, 11:14]
        at = d["applied_t"]
        post = t > at
        if not post.any() or np.isnan(po[post][0]).any():
            continue
        p_new0 = po[post][0]
        p_old = p_new0.copy(); p_old[:2] -= d["mag"] * np.array(d["dir"])
        z0 = p_new0[2]
        g = a[:, 6]
        closes = [i for i in range(1, len(t)) if post[i] and g[i] > 0 and g[i - 1] <= 0]
        info = []
        for i in closes:
            if po[i][2] - z0 > 0.03:          # object already lifted: later sub-task
                break
            dn = np.linalg.norm(eef[i, :2] - po[i, :2]); do = np.linalg.norm(eef[i, :2] - p_old[:2])
            info.append((dn, do))
        lifted = bool((po[post][:, 2] - z0 > 0.03).any())
        # first 15 post-displacement steps: eef displacement direction vs (new - eef) and (old - eef)
        i0 = int(np.argmax(post)); i1 = min(len(t) - 1, i0 + 15)
        mv = eef[i1, :2] - eef[i0, :2]
        cn = np.dot(mv, po[i0, :2] - eef[i0, :2]) / (np.linalg.norm(mv) * np.linalg.norm(po[i0, :2] - eef[i0, :2]) + 1e-9)
        co = np.dot(mv, p_old[:2] - eef[i0, :2]) / (np.linalg.norm(mv) * np.linalg.norm(p_old[:2] - eef[i0, :2]) + 1e-9)
        rows.append({"task": r["task_id"], "success": r["success"], "n_close_before_lift": len(info),
                     "first_close": info[0] if info else None, "lifted": lifted, "cos_new": cn, "cos_old": co,
                     "mag": d["mag"]})
S = np.array([r["success"] for r in rows])
print("episodes %d, success %.1f%%" % (len(rows), 100 * S.mean()))
for lab, m in (("success", S == 1), ("failure", S == 0)):
    rr = [r for r, k in zip(rows, m) if k]
    fc = [r["first_close"] for r in rr if r["first_close"]]
    nearer_old = np.mean([do < dn for dn, do in fc]) if fc else float("nan")
    print("\n%s: n=%d  lifted the displaced object %.0f%%  closes before lift (mean) %.1f" % (
        lab, len(rr), 100 * np.mean([r["lifted"] for r in rr]), np.mean([r["n_close_before_lift"] for r in rr])))
    if fc:
        print("  first close: xy distance to NEW pos median %.3f m, to OLD pos median %.3f m; closer to OLD in %.0f%%" % (
            np.median([x[0] for x in fc]), np.median([x[1] for x in fc]), 100 * nearer_old))
    print("  first 15 steps after the displacement: cos(motion, to-new) %.2f, cos(motion, to-old) %.2f" % (
        np.nanmean([r["cos_new"] for r in rr]), np.nanmean([r["cos_old"] for r in rr])))
print("\nper task success:", {t: "%.0f%%" % (100 * np.mean([r["success"] for r in rows if r["task"] == t])) for t in sorted({r["task"] for r in rows})})
