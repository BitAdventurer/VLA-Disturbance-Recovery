#!/usr/bin/env python3
"""Offline single-decision gain of heuristic selectors on branch-run labels (replay candidates + outcomes).
For each decision and continuation c with labels for candidates ks, a selector picks k* among ks; gain = Y(k*,c) -
mean_k Y(k,c). Weighted (strat HT weights for br0 episodes, 1 for d1-only), bootstrap CI. Uses train_critic's loaders."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_critic as T                              # noqa: E402
import selectors_bok as S                             # noqa: E402

E = T.E
rep = T.load_replay()
d1 = T.load_runs(["d1_k%d_c1" % k for k in range(8)] + ["d2_k%d_c0" % k for k in range(8)])
br0 = T.load_runs(["br0_k%d_c%d" % (k, c) for k in range(4) for c in (0, 1)])
w = json.load(open(E + "/bok/strat_weights.json"))
G = np.array(json.load(open(E + "/bok/aim_gain.json")))


def collect(runs):
    lab = {}
    for name, d in runs.items():
        k = int(name.split("_k")[1].split("_")[0]); c = int(name.rsplit("_c", 1)[1])
        for key, r in d.items():
            if key not in rep or T.branch_hash(r) != str(rep[key]["hashes"][k]):
                continue
            lab.setdefault(key, {}).setdefault(c, {}).setdefault(k, float(r["success"]))
    return lab


lab = collect({**d1, **br0})
sels = ["medoid", "antimedoid", "continuity", "discontinuity", "priv_aim"]
rng = np.random.default_rng(0)
for split, filt in (("train inits 0-24", lambda key: key[1] < 25), ("test inits 25-49", lambda key: key[1] >= 25)):
    rows = {s: [] for s in sels + ["oracle_split"]}
    for key, by_c in lab.items():
        if not filt(key):
            continue
        e = rep[key]
        ctx = {"rng": np.random.default_rng(0), "last_action": None if np.isnan(e["last_action"]).any() else e["last_action"],
               "eef": e["eef"], "p_obj": None, "gripper_open": True, "aim_gain": G}
        z = np.load("%s/results_bok/cands/replay_k8/t%d_e%d.npz" % (E, *key))
        p = z["0_p_obj"] if "0_p_obj" in z else None
        ctx["p_obj"] = None if p is None or np.isnan(p).any() else p
        wt = w.get("%d_%d" % key, 1.0) if key in br0.get("br0_k0_c0", {}) else 1.0
        for c, labs in by_c.items():
            ks = sorted(labs)
            if len(ks) < 2:
                continue
            Y = np.array([labs[k] for k in ks])
            cands = np.asarray(e["cands"], float)[ks]
            for s in sels:
                _, sc = S.select(s, cands, ctx)
                best = np.flatnonzero(sc == sc.max())
                rows[s].append((wt, Y[best].mean() - Y.mean(), len(ks)))
        if 0 in by_c and 1 in by_c and sorted(by_c[0]) == sorted(by_c[1]):      # split oracle where both c exist
            ks = sorted(by_c[0]); Y = np.array([[by_c[c][k] for c in (0, 1)] for k in ks])
            g = np.mean([Y[np.flatnonzero(Y[:, c] == Y[:, c].max()), 1 - c].mean() - Y[:, 1 - c].mean() for c in (0, 1)])
            rows["oracle_split"].append((wt, g, len(ks)))
    print("\n== %s" % split)
    for s, rr in rows.items():
        if not rr:
            continue
        W = np.array([r[0] for r in rr]); g = np.array([r[1] for r in rr])
        bs = [np.sum(W[i] * g[i]) / np.sum(W[i]) for i in [rng.integers(0, len(rr), len(rr)) for _ in range(4000)]]
        print("  %-14s groups %4d  gain %+5.1f [%+5.1f, %+5.1f] pts" % (s, len(rr), 100 * np.sum(W * g) / W.sum(),
                                                                    *(100 * np.percentile(bs, [2.5, 97.5]))))
