#!/usr/bin/env python3
"""Residual evaluations (results_res/<arm>.jsonl, aligned resets = same population as the sequential results_dist runs)
against (a) the same-noise reference run (exactly paired) and (b) the mean of several noise draws of the reference.
Usage: analyze_res.py ARM [ARM ...] [--ref static|det] [--split heldout|all]"""
import argparse
import json
import os

import numpy as np

E = os.path.expanduser(os.environ.get("VLA_EXPERIMENTS", "~/disk2/vla-inference-scheduling/experiments"))
REFS = {"static": ["obj_static_n2_h10_s0", "obj_static_n2_h10_ns1_s0", "obj_static_n2_h10_ns2_s0",
                   "obj_static_n2_h10_ns3_s0", "obj_static_n2_h10_ns4_s0"],
        "det": ["det_n2h10_s0", "det_n2h10_ns1_s0", "det_n2h10_ns2_s0", "det_n2h10_ns3_s0", "det_n2h10_ns4_s0"]}


def L(p):
    p = p if os.path.exists(p) else p + ".part"
    if not os.path.exists(p):
        return None
    return {(r["task_id"], r["init_state_id"]): r for r in map(json.loads, open(p)) if "success" in r}


def boot(x, tasks, rng, B=4000):
    grp = [np.flatnonzero(tasks == t) for t in sorted(set(tasks))]
    bs = [x[np.concatenate([g[rng.integers(0, len(g), len(g))] for g in grp])].mean() for _ in range(B)]
    return np.percentile(bs, [2.5, 97.5])


ap = argparse.ArgumentParser()
ap.add_argument("arms", nargs="+")
ap.add_argument("--ref", default="static")
ap.add_argument("--split", default="all")
a = ap.parse_args()
rng = np.random.default_rng(0)
draws = [d for d in (L("%s/results_dist/%s.jsonl" % (E, n)) for n in REFS[a.ref]) if d]
s0 = draws[0]
pool = {k: np.mean([d[k]["success"] for d in draws if k in d]) for k in s0}
for arm in a.arms:
    d = L("%s/results_res/%s.jsonl" % (E, arm))
    if not d:
        print("%-18s missing" % arm); continue
    ks = sorted(k for k in d if k in s0 and (a.split == "all" or k[1] >= 25))
    y = np.array([d[k]["success"] for k in ks], float)
    x0 = y - np.array([s0[k]["success"] for k in ks]); xp = y - np.array([pool[k] for k in ks])
    tasks = np.array([k[0] for k in ks])
    same = np.mean([d[k]["episode_length"] == s0[k]["episode_length"] for k in ks])
    act = np.mean([len(d[k].get("D") or []) > 0 for k in ks])
    print("%-18s n=%3d success %.1f%% | vs same-noise %s ref %+.1f [%+.1f,%+.1f] (wins %d, losses %d) | vs %d-draw pool %+.1f [%+.1f,%+.1f] | residual active in %.0f%% of eps, identical-to-ref length %.0f%%" % (
        arm, len(ks), 100 * y.mean(), a.ref, 100 * x0.mean(), *(100 * boot(x0, tasks, rng)),
        int((x0 > 0).sum()), int((x0 < 0).sum()), len(draws), 100 * xp.mean(), *(100 * boot(xp, tasks, rng)),
        100 * act, 100 * same))
    print("%18s per task vs pool: %s" % ("", {t: "%+d" % int(round(100 * xp[tasks == t].mean())) for t in sorted(set(tasks))}))
