#!/usr/bin/env python3
"""Learned disturbance detector for event-triggered replanning (react_step with --detector).

Data: --log_frames runs (results_bok/frames/<run>/t{task}_e{init}.npz: per executed step t, 64x64 grey frames of the
agentview and wrist cameras after the step, eef position, commanded action) and their jsonl (disturbance applied_t).
A sample is the consecutive pair (frame t-1, frame t) with the action that produced it; it is positive iff
t == applied_t + 1 (the first frame after the displacement). Hard negatives: every pair within 15 steps of the event;
plus 40 random pairs per episode and all pairs of undisturbed runs (subsampled).
Model: CNN on [prev, cur, cur-prev] x 2 cameras (6 x 64 x 64) + MLP on (action, eef delta) -> logit.
Split: inits 0-24 train, 25-49 test (all runs). Reports AUROC on test pairs and, for thresholds set on undisturbed
test episodes (any step above tau = a false alarm), the hit rate at the event step (and within +2 steps).
Exports a TorchScript model + threshold table for the client.
Run with the openpi venv (CUDA torch) or the LIBERO venv (CPU torch)."""
import argparse
import glob
import json
import os

import numpy as np
import torch

E = os.path.expanduser(os.environ.get("VLA_EXPERIMENTS", "~/disk2/vla-inference-scheduling/experiments"))


def load_run(name, frames_dir):
    recs = {}
    p = "%s/results_bok/%s.jsonl" % (E, name)
    p = p if os.path.exists(p) else p + ".part"
    for l in open(p):
        try:
            r = json.loads(l)
        except ValueError:
            continue
        if "success" in r:
            recs[(r["task_id"], r["init_state_id"])] = r
    eps = []
    for f in sorted(glob.glob("%s/results_bok/frames/%s/t*_e*.npz" % (E, frames_dir))):
        if ".tmp" in f:
            continue
        t, e = os.path.basename(f)[1:-4].split("_e")
        key = (int(t), int(e))
        if key not in recs:
            continue
        z = np.load(f)
        d = recs[key].get("disturbance") or {}
        ev = d.get("applied_t") if d.get("object") is not None else None
        eps.append({"key": key, "t": z["t"], "img": z["img"], "wrist": z["wrist"], "eef": z["eef"], "act": z["act"],
                    "event": None if ev is None else int(ev) + 1})
    return eps


def pair_tensor(ep, idx):
    """idx: indices i >= 1 into the episode arrays -> (images [N,6,64,64] float, vec [N,10] float)."""
    i = np.asarray(idx)
    im0, im1 = ep["img"][i - 1].astype(np.float32) / 255.0, ep["img"][i].astype(np.float32) / 255.0
    w0, w1 = ep["wrist"][i - 1].astype(np.float32) / 255.0, ep["wrist"][i].astype(np.float32) / 255.0
    x = np.stack([im0, im1, im1 - im0, w0, w1, w1 - w0], 1)
    v = np.concatenate([ep["act"][i], ep["eef"][i] - ep["eef"][i - 1]], 1) * np.array([1] * 7 + [50] * 3, np.float32)
    return x, v.astype(np.float32)


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        c = torch.nn
        self.cnn = c.Sequential(c.Conv2d(6, 32, 5, 2, 2), c.GELU(), c.Conv2d(32, 64, 3, 2, 1), c.GELU(),
                                c.Conv2d(64, 64, 3, 2, 1), c.GELU(), c.Conv2d(64, 64, 3, 1, 1), c.GELU())
        self.vec = c.Sequential(c.Linear(10, 32), c.GELU())
        self.head = c.Sequential(c.Linear(64 * 2 + 32, 64), c.GELU(), c.Linear(64, 1))

    def forward(self, x, v):
        h = self.cnn(x)
        h = torch.cat([h.amax((2, 3)), h.mean((2, 3)), self.vec(v)], 1)
        return self.head(h).squeeze(-1)


def auroc(pos, neg):
    allv = np.concatenate([pos, neg]); r = allv.argsort().argsort() + 1
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="frames_obj_s0:obj_s0,frames_obj_ns1:obj_ns1,frames_none_s0:none_s0")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--out", default=E + "/bok/detector_v1.pt")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    eps = []
    for spec in a.runs.split(","):
        name, fd = spec.split(":")
        try:
            got = load_run(name, fd)
        except FileNotFoundError:
            print("missing", name); continue
        for e in got:
            e["run"] = name
        eps += got
        print("%s: %d episodes (%d with an event)" % (name, len(got), sum(e["event"] is not None for e in got)))
    # displaced runs: inits 0-24 train / 25-49 test; undisturbed runs (inits 0-24 only): even / odd init
    is_test = lambda e: (e["key"][1] % 2 == 1) if e["run"].startswith("frames_none") else (e["key"][1] >= 25)
    train = [e for e in eps if not is_test(e)]
    test = [e for e in eps if is_test(e)]
    print("train episodes %d, test episodes %d (undisturbed test %d)" % (
        len(train), len(test), sum(e["event"] is None for e in test)))

    def samples(ep_list, n_rand=40):
        X, V, Y = [], [], []
        for ep in ep_list:
            T = len(ep["t"])
            pos = [] if ep["event"] is None else list(np.flatnonzero(ep["t"] == ep["event"]))
            pos = [p for p in pos if p >= 1]
            hard = [] if not pos else [i for i in range(max(1, pos[0] - 15), min(T, pos[0] + 16)) if i != pos[0]]
            rnd = list(rng.choice(np.arange(1, T), size=min(n_rand, T - 1), replace=False)) if T > 1 else []
            idx = sorted(set(pos + hard + rnd) - set())
            if not idx:
                continue
            x, v = pair_tensor(ep, idx)
            X.append(x); V.append(v); Y.append(np.array([1.0 if i in pos else 0.0 for i in idx], np.float32))
        return np.concatenate(X), np.concatenate(V), np.concatenate(Y)

    Xtr, Vtr, Ytr = samples(train)
    print("train pairs %d (positives %d)" % (len(Ytr), int(Ytr.sum())))
    net = Net().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    pw = torch.tensor((len(Ytr) - Ytr.sum()) / max(Ytr.sum(), 1), device=dev)
    for ep_i in range(a.epochs):
        net.train()
        perm = rng.permutation(len(Ytr)); tot = 0.0
        for b in range(0, len(perm), 256):
            i = perm[b:b + 256]
            x = torch.tensor(Xtr[i], device=dev); v = torch.tensor(Vtr[i], device=dev); y = torch.tensor(Ytr[i], device=dev)
            # augmentation: random brightness shift of both frames of a pair (keeps the difference channels)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(net(x, v), y, pos_weight=pw)
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss) * len(i)
        print("epoch %d loss %.4f" % (ep_i, tot / len(perm)), flush=True)
    net.eval()

    def score_episode(ep):
        T = len(ep["t"])
        if T < 2:
            return np.zeros(0)
        out = []
        for b in range(1, T, 512):
            idx = list(range(b, min(T, b + 512)))
            x, v = pair_tensor(ep, idx)
            with torch.no_grad():
                out.append(net(torch.tensor(x, device=dev), torch.tensor(v, device=dev)).cpu().numpy())
        return np.concatenate(out)            # score for pair (i-1, i), i = 1..T-1

    pos, neg, ep_max_none, hit_scores, hit2_scores = [], [], [], [], []
    for ep in test:
        s = score_episode(ep)
        if ep["event"] is None:
            neg += list(s); ep_max_none.append(s.max() if len(s) else -1e9)
            continue
        j = np.flatnonzero(ep["t"][1:] == ep["event"])
        if not len(j):
            neg += list(s); continue
        j = int(j[0])
        pos.append(s[j]); hit_scores.append(s[j]); hit2_scores.append(s[j:j + 3].max())
        neg += list(np.delete(s, j))
    pos, neg, ep_max_none = np.array(pos), np.array(neg), np.array(ep_max_none)
    print("test: %d events, %d other pairs, %d undisturbed episodes; AUROC %.4f" % (
        len(pos), len(neg), len(ep_max_none), auroc(pos, neg)))
    table = []
    for fq in (1e-2, 3e-3, 1e-3, 3e-4):                 # per-step false-alarm rate on all non-event pairs
        tau = float(np.quantile(neg, 1 - fq))
        print("  FA/step %.4f: tau %+.3f  hit %.1f%%  within +2 %.1f%%  FA/undisturbed episode %.1f%%" % (
            fq, tau, 100 * np.mean(np.array(hit_scores) > tau), 100 * np.mean(np.array(hit2_scores) > tau),
            100 * np.mean(ep_max_none > tau) if len(ep_max_none) else float("nan")))
    for fa in (0.2, 0.1, 0.05, 0.02):
        tau = float(np.quantile(ep_max_none, 1 - fa)) if len(ep_max_none) else float(np.quantile(neg, 0.999))
        table.append({"fa_per_episode": fa, "tau": tau, "hit": float(np.mean(np.array(hit_scores) > tau)),
                      "hit_within_2": float(np.mean(np.array(hit2_scores) > tau))})
        print("  FA/episode %4.0f%%: tau %+.3f  hit %.1f%%  hit within +2 steps %.1f%%" % (
            100 * fa, tau, 100 * table[-1]["hit"], 100 * table[-1]["hit_within_2"]))
    net = net.cpu()
    torch.save({k: v.clone() for k, v in net.state_dict().items()}, a.out + ".state.pt")   # portable weights
    ts = torch.jit.trace(net, (torch.zeros(1, 6, 64, 64), torch.zeros(1, 10)))
    ts.save(a.out)
    json.dump({"thresholds": table, "auroc": float(auroc(pos, neg)) if len(pos) else None}, open(a.out + ".json", "w"), indent=1)
    print("saved", a.out)


def export(state, out):
    """Rebuild the net from portable weights and save TorchScript with THIS torch (e.g. the client's older torch)."""
    net = Net()
    net.load_state_dict(torch.load(state, map_location="cpu"))
    net.eval()
    torch.jit.trace(net, (torch.zeros(1, 6, 64, 64), torch.zeros(1, 10))).save(out)
    print("exported", out, "with torch", torch.__version__)


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 4 and sys.argv[1] == "export":
        export(sys.argv[2], sys.argv[3])
    else:
        main()
