#!/usr/bin/env python3
"""Surprise signals from cal2_* runs (--log_chunks, --log_embed): which signal flags the first query after the
displacement? Signals at query q (all computed from information available at q before acting on it):
  img        camera change since the previous query (grey thumbnails)
  emb_all/base/wrist/prompt   1 - cosine between the pooled prefix embedding at q and at q-1
  embv       1 - cosine of the concatenated per-layer V means
  heading    angle (deg) between the xy translation of the n=2 chunk at q and that of q-1 (needs the q chunk)
  jump       scaled distance between the first action of chunk q and the last action of chunk q-1
Prints AUROC vs pre-displacement queries and vs undisturbed queries, and the hit rate at thresholds giving
10% / 5% false alarms per undisturbed episode."""
import glob
import json
import os

import numpy as np

E = os.path.expanduser(os.environ.get("VLA_EXPERIMENTS", "~/disk2/vla-inference-scheduling/experiments"))
SCALE = np.array([0.842, 0.828, 0.937, 0.128, 0.175, 0.253])
NAMES = ["img", "emb_all", "emb_base", "emb_wrist", "emb_prompt", "embv", "heading", "jump"]


def load(name):
    p = "%s/results_dist/%s.jsonl" % (E, name)
    p = p if os.path.exists(p) else p + ".part"
    return [json.loads(l) for l in open(p) if l.strip()]


def shard(name, r):
    p = "%s/results_bok/cands/%s/t%d_e%d.npz" % (E, name, r["task_id"], r["init_state_id"])
    if not os.path.exists(p):
        return None
    z = np.load(p)
    n = int(z["n"])
    return {int(z["%d_q" % i]): (z["%d_emb" % i].astype(np.float32), z["%d_emb_v" % i].astype(np.float32).ravel())
            for i in range(n) if "%d_emb" % i in z}


def cosd(a, b):
    return float(1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def signals(r, emb):
    calls = r["calls"]
    out = []
    for q in range(1, len(calls)):
        c, p = calls[q], calls[q - 1]
        s = {"t": c["t"], "img": (c.get("surprise") or [np.nan])[0]}
        if emb is not None and q in emb and q - 1 in emb:
            e1, v1 = emb[q]
            e0, v0 = emb[q - 1]
            for j, nm in enumerate(["emb_all", "emb_base", "emb_wrist", "emb_prompt"]):
                s[nm] = cosd(e1[j], e0[j])
            s["embv"] = cosd(v1, v0)
        if "chunk" in c and "chunk" in p:
            a1, a0 = np.asarray(c["chunk"]), np.asarray(p["chunk"])
            u, v = a1[:, :2].sum(0), a0[:, :2].sum(0)
            s["heading"] = float(np.degrees(np.arccos(np.clip(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9), -1, 1))))
            s["jump"] = float(np.sqrt((((a1[0, :6] - a0[-1, :6]) / SCALE) ** 2).sum()))
        out.append(s)
    return out


def auroc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    pos, neg = pos[~np.isnan(pos)], neg[~np.isnan(neg)]
    if not len(pos) or not len(neg):
        return np.nan
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort() + 1
    return (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def main():
    obj = load("cal2_obj_s0")
    try:
        none = load("cal2_none_s0")
    except FileNotFoundError:
        none = []
    S_obj = [(r, signals(r, shard("cal2_obj", r))) for r in obj]
    S_none = [signals(r, shard("cal2_none", r)) for r in none]
    print("%d displaced, %d undisturbed episodes" % (len(S_obj), len(S_none)))
    print("%-11s %8s %8s %9s %9s %9s" % ("signal", "AUC_pre", "AUC_none", "hit@FA10", "hit@FA5", "med hit/none"))
    for nm in NAMES:
        hit, pre = [], []
        for r, ss in S_obj:
            at = (r.get("disturbance") or {}).get("applied_t")
            if at is None:
                continue
            first = next((s for s in ss if s["t"] > at), None)
            for s in ss:
                if s["t"] <= at:
                    pre.append(s.get(nm, np.nan))
            if first is not None:
                hit.append(first.get(nm, np.nan))
        qn = [s.get(nm, np.nan) for ss in S_none for s in ss]
        epmax = np.array([np.nanmax([s.get(nm, np.nan) for s in ss] + [-1]) for ss in S_none] or [np.inf])
        hit = np.asarray(hit, float)
        res = []
        for fa in (0.10, 0.05):
            tau = np.quantile(epmax, 1 - fa)
            res.append(100 * np.nanmean(hit > tau))
        print("%-11s %8.3f %8.3f %8.1f%% %8.1f%%   %.4f/%.4f" % (nm, auroc(hit, pre), auroc(hit, qn), res[0], res[1],
                                                            np.nanmedian(hit), np.nanmedian(qn)))


if __name__ == "__main__":
    main()
