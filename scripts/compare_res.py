#!/usr/bin/env python3
"""compare_res.py ARM[,ARM2...] --ref REF[,REF2...] [--split all|heldout] [--ref_dir results_dist] [--arm_dir results_res]
Paired comparison of an arm (mean over its listed draws per episode) against the mean of the reference draws on the
same (task, init) episodes; task-stratified bootstrap 95% CI. Also reports the arm's undisturbed-style stats."""
import argparse
import json
import os

import numpy as np

E = os.path.expanduser(os.environ.get("VLA_EXPERIMENTS", "~/disk2/vla-inference-scheduling/experiments"))


def L(path):
    p = path if os.path.exists(path) else path + ".part"
    if not os.path.exists(p):
        return None
    return {(r["task_id"], r["init_state_id"]): r for r in map(json.loads, open(p)) if "success" in r and not r.get("error")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arms")
    ap.add_argument("--ref", required=True)
    ap.add_argument("--split", default="all")
    ap.add_argument("--ref_dir", default="results_dist")
    ap.add_argument("--arm_dir", default="results_res")
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    refs = [d for d in (L("%s/%s/%s.jsonl" % (E, a.ref_dir, n)) for n in a.ref.split(",")) if d]
    arms = [d for d in (L("%s/%s/%s.jsonl" % (E, a.arm_dir, n)) for n in a.arms.split(",")) if d]
    if not refs or not arms:
        print("missing files: refs %d arms %d" % (len(refs), len(arms))); return
    ks = sorted(k for k in set.intersection(*[set(d) for d in refs]) & set().union(*[set(d) for d in arms])
                if a.split == "all" or k[1] >= 25)
    y = np.array([np.mean([d[k]["success"] for d in arms if k in d]) for k in ks])
    r = np.array([np.mean([d[k]["success"] for d in refs]) for k in ks])
    x = y - r
    tasks = np.array([k[0] for k in ks]); grp = [np.flatnonzero(tasks == t) for t in sorted(set(tasks))]
    bs = [x[np.concatenate([g[rng.integers(0, len(g), len(g))] for g in grp])].mean() for _ in range(6000)]
    act = np.mean([any(any(abs(v) > 1e-6 for v in dd) for dd in (d[k].get("D") or [])) for d in arms for k in ks if k in d])
    print("%s (%d draw%s) vs %s (%d draw%s), %s, n=%d:" % (a.arms, len(arms), "s" if len(arms) > 1 else "", a.ref, len(refs),
                                                      "s" if len(refs) > 1 else "", a.split, len(ks)))
    print("  success %.1f%% vs %.1f%%  delta %+.1f [%+.1f, %+.1f]  residual active in %.0f%% of episodes" % (
        100 * y.mean(), 100 * r.mean(), 100 * x.mean(), *(100 * np.percentile(bs, [2.5, 97.5])), 100 * act))
    print("  per task delta: %s" % {t: "%+d" % int(round(100 * x[tasks == t].mean())) for t in sorted(set(tasks))})


if __name__ == "__main__":
    main()
