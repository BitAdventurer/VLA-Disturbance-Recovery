#!/usr/bin/env python3
"""Learned-detector event-triggered replanning (det_*) vs the (2,10) noise-draw pool and the oracle react_now arms.
Primary: inits 25-49 (never seen by the detector, trained on inits 0-24 frames); secondary: all 500 episodes.
Also detection quality in closed loop (hit = a detection within 0..2 steps after the displacement frame) and cost."""
import json
import os

import numpy as np

E = os.path.expanduser(os.environ.get("VLA_EXPERIMENTS", "~/disk2/vla-inference-scheduling/experiments") + "/results_dist")


def L(n):
    p = os.path.join(E, n + ".jsonl")
    p = p if os.path.exists(p) else p + ".part"
    if not os.path.exists(p):
        return None
    return {(r["task_id"], r["init_state_id"]): r for r in map(json.loads, open(p)) if "success" in r and not r.get("error")}


rng = np.random.default_rng(0)
base = [d for d in (L(n) for n in ["obj_static_n2_h10_s0", "obj_static_n2_h10_ns1_s0", "obj_static_n2_h10_ns2_s0",
                                    "obj_static_n2_h10_ns3_s0", "obj_static_n2_h10_ns4_s0"])
        if d is not None and len(d) == 500]          # complete draws only
print("baseline pool: %d draws" % len(base))
bm = {k: np.mean([b[k]["success"] for b in base]) for k in base[0]}
bcost = {k: np.mean([b[k]["cost_ms"] for b in base]) for k in base[0]}
ARMS = {"oracle replan (2,10)": ["obj_react_now_n2h10_s0", "obj_react_now_n2h10_ns1_s0", "obj_react_now_n2h10_ns2_s0"],
        "oracle replan (10,10)": ["obj_react_now_n10h10_s0", "obj_react_now_n10h10_ns1_s0", "obj_react_now_n10h10_ns2_s0"],
        "detector replan (2,10)": ["det_n2h10_s0", "det_n2h10_ns1_s0", "det_n2h10_ns2_s0", "det_n2h10_ns3_s0",
                                   "det_n2h10_ns4_s0"],
        "detector replan (10,10)": ["det_n10h10_s0", "det_n10h10_ns1_s0", "det_n10h10_ns2_s0"]}
for split, f in (("held-out inits 25-49", lambda k: k[1] >= 25), ("all 500", lambda k: True)):
    print("\n== %s" % split)
    for arm, names in ARMS.items():
        D = [d for d in (L(n) for n in names) if d]
        if not D:
            continue
        # per episode: mean over the draws that contain it (unfinished draws contribute where available)
        ks = sorted(k for k in set().union(*[set(d) for d in D]) & set(bm) if f(k))
        if not ks:
            continue
        ys = {k: [d[k]["success"] for d in D if k in d] for k in ks}
        x = np.array([np.mean(ys[k]) - bm[k] for k in ks])
        tasks = np.array([k[0] for k in ks])
        grp = [np.flatnonzero(tasks == t) for t in sorted(set(tasks))]
        bs = [x[np.concatenate([g[rng.integers(0, len(g), len(g))] for g in grp])].mean() for _ in range(4000)]
        cost = np.mean([d[k]["cost_ms"] for d in D for k in ks if k in d]) / np.mean([bcost[k] for k in ks])
        ndraw = np.mean([len(ys[k]) for k in ks])
        print("  %-24s draws %d (mean %.1f/episode) n=%3d  success %.1f vs pool %.1f  delta %+.1f [%+.1f, %+.1f]  cost x%.3f" % (
            arm, len(D), ndraw, len(ks), 100 * np.mean([np.mean(ys[k]) for k in ks]),
            100 * np.mean([bm[k] for k in ks]), 100 * x.mean(), *(100 * np.percentile(bs, [2.5, 97.5])), cost))
print("\n== closed-loop detection")
for n in ARMS["detector replan (2,10)"] + ARMS["detector replan (10,10)"]:
    d = L(n)
    if not d:
        continue
    hit = fa = ev = 0
    for k, r in d.items():
        at = (r.get("disturbance") or {}).get("applied_t")
        dets = r.get("detections", [])
        if at is not None and r["disturbance"].get("object") is not None:
            ev += 1
            hit += any(at + 1 <= t <= at + 3 for t in dets)
            fa += sum(1 for t in dets if not (at + 1 <= t <= at + 3))
        else:
            fa += len(dets)
    print("  %-22s n=%d events %d hit %.1f%%  false detections/episode %.3f" % (n, len(d), ev, 100 * hit / max(ev, 1), fa / len(d)))
nd = L("det_none_n2h10_s0")
for ref in ("cal_none_s0",):
    rr = L(ref)
    if nd and rr:
        ks = sorted(set(nd) & set(rr))
        print("\n== undisturbed: detector run %.1f%% vs (2,10) %s %.1f%% (n=%d); false detections/episode %.3f; episodes with any %.1f%%" % (
            100 * np.mean([nd[k]["success"] for k in ks]), ref, 100 * np.mean([rr[k]["success"] for k in ks]), len(ks),
            np.mean([len(nd[k].get("detections", [])) for k in ks]), 100 * np.mean([len(nd[k].get("detections", [])) > 0 for k in ks])))
