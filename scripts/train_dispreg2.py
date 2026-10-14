#!/usr/bin/env python3
"""Displacement regressor v2: 128x128 RGB frames right before/after the disturbance (both cameras) + action/eef delta
-> horizontal object displacement d (metres). Data: run_libero.py --log_event_frames DIR (one npz per episode with
before_img, after_img, before_wrist, after_wrist, action, eef_before, eef_after, d, moved). Split by init state:
inits 0-24 train, 25-49 test. Exports portable weights; export() writes TorchScript with the running torch."""
import argparse
import glob
import json
import os

import numpy as np
import torch
import torch.nn as nn

E = os.path.expanduser(os.environ.get("VLA_EXPERIMENTS", "~/disk2/vla-inference-scheduling/experiments"))


def load(dirs):
    X, V, Y, K = [], [], [], []
    for d in dirs.split(","):
        for f in sorted(glob.glob("%s/results_bok/event_frames/%s/t*_e*.npz" % (E, d))):
            if ".tmp" in f:
                continue
            z = np.load(f)
            if not bool(z["moved"]):
                continue
            t, e = os.path.basename(f)[1:-4].split("_e")
            X.append(np.concatenate([z["before_img"], z["after_img"], z["before_wrist"], z["after_wrist"]], -1))
            V.append(np.concatenate([z["action"], (z["eef_after"] - z["eef_before"]) * 50.0]))
            Y.append(z["d"]); K.append((d, int(t), int(e)))
    return np.stack(X), np.stack(V).astype(np.float32), np.stack(Y).astype(np.float32), K


def to_input(x):
    """uint8 [N,128,128,12] -> float [N,18,128,128]: the 4 frames and the 2 after-before differences."""
    x = torch.as_tensor(x).float().permute(0, 3, 1, 2) / 255.0
    return torch.cat([x, x[:, 3:6] - x[:, 0:3], x[:, 9:12] - x[:, 6:9]], 1)


class Block(nn.Module):
    def __init__(self, c_in, c_out, stride):
        super().__init__()
        self.a = nn.Sequential(nn.Conv2d(c_in, c_out, 3, stride, 1), nn.BatchNorm2d(c_out), nn.GELU(),
                               nn.Conv2d(c_out, c_out, 3, 1, 1), nn.BatchNorm2d(c_out))
        self.s = nn.Sequential(nn.Conv2d(c_in, c_out, 1, stride), nn.BatchNorm2d(c_out))

    def forward(self, x):
        return torch.nn.functional.gelu(self.a(x) + self.s(x))


class Reg2(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(Block(18, 32, 2), Block(32, 64, 2), Block(64, 96, 2), Block(96, 128, 2))
        self.vec = nn.Sequential(nn.Linear(10, 32), nn.GELU())
        self.head = nn.Sequential(nn.Linear(128 * 2 + 32, 128), nn.GELU(), nn.Linear(128, 2))

    def forward(self, x, v):
        h = self.cnn(x)
        h = torch.cat([h.amax((2, 3)), h.mean((2, 3)), self.vec(v)], 1)
        return self.head(h) / 10.0


def evaluate(net, X, V, Y, dev, bs=256):
    net.eval()
    out = []
    with torch.no_grad():
        for b in range(0, len(X), bs):
            out.append(net(to_input(X[b:b + bs]).to(dev), torch.as_tensor(V[b:b + bs]).to(dev)).cpu().numpy())
    p = np.concatenate(out)
    err = np.linalg.norm(p - Y, axis=1)
    cos = (p * Y).sum(1) / (np.linalg.norm(p, axis=1) * np.linalg.norm(Y, axis=1) + 1e-9)
    return p, {"n": int(len(Y)), "err_median_cm": float(100 * np.median(err)), "err_p90_cm": float(100 * np.quantile(err, .9)),
               "angle_median_deg": float(np.degrees(np.median(np.arccos(np.clip(cos, -1, 1))))),
               "mag_ratio_median": float(np.median(np.linalg.norm(p, axis=1) / np.linalg.norm(Y, axis=1)))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", default="s1,s2,s3,s4,s5,s6")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--out", default=E + "/bok/dispreg_v2.pt")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    X, V, Y, K = load(a.dirs)
    te = np.array([k[2] >= 25 for k in K])
    print("events: train %d, test %d; |d| median %.1f cm" % ((~te).sum(), te.sum(), 100 * np.median(np.linalg.norm(Y, axis=1))))
    net = Reg2().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    steps = a.epochs * int(np.ceil((~te).sum() / 64))
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps)
    Xtr, Vtr, Ytr = X[~te], V[~te], Y[~te]
    rng = np.random.default_rng(0)
    for ep in range(a.epochs):
        net.train()
        perm = rng.permutation(len(Ytr))
        for b in range(0, len(perm), 64):
            i = perm[b:b + 64]
            x = to_input(Xtr[i]).to(dev)
            x[:, :12] = x[:, :12] * (1 + (torch.rand(len(i), 1, 1, 1, device=dev) - 0.5) * 0.2)   # brightness jitter
            loss = ((net(x, torch.as_tensor(Vtr[i]).to(dev)) - torch.as_tensor(Ytr[i]).to(dev)) ** 2).sum(-1).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        if ep % 5 == 0 or ep == a.epochs - 1:
            _, m = evaluate(net, X[te], V[te], Y[te], dev)
            print("epoch %d loss %.6f test %s" % (ep, float(loss), json.dumps(m)), flush=True)
    p, m = evaluate(net, X[te], V[te], Y[te], dev)
    print("FINAL", json.dumps(m))
    net = net.cpu()
    torch.save({k: v.clone() for k, v in net.state_dict().items()}, a.out + ".state.pt")
    json.dump(m, open(a.out + ".json", "w"), indent=1)


def export(state, out):
    net = Reg2()
    net.load_state_dict(torch.load(state, map_location="cpu"))
    net.eval()
    torch.jit.trace(net, (torch.zeros(1, 18, 128, 128), torch.zeros(1, 10))).save(out)
    print("exported", out, "with torch", torch.__version__)


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 4 and sys.argv[1] == "export":
        export(sys.argv[2], sys.argv[3])
    else:
        main()
