#!/usr/bin/env python3
"""Decision-time chunk critic for best-of-K selection (first query after the displacement).

Data
  replay shards  results_bok/cands/replay_k8/t{task}_e{init}.npz: at jp=0, the 8 legacy candidates (seeds k=0..7),
                 the prefix embedding emb [4,2048], emb_v [18,256], proprio state (8), eef, last executed action.
  labels         branch runs (forced candidate k at jp=0, then continuation salt c): d1_k{k}_c1 (train inits 0-24,
                 k=0..7) and br0_k{k}_c{c} (strat subset of all inits, k=0..3, c=0/1).
  replay check   a label is used only if the branch run's executed chunk hash at the branch query equals the hash of
                 replay candidate k (same observation, same seed).
Model  per candidate: [PCA(emb) | state | scaled chunk | chunk - set mean | distance to the set medoid] -> MLP -> logit
       of success; loss = BCE + listwise softmax CE within an episode's candidate set (groups with mixed outcomes);
       ensemble of 5 (bootstrap over episodes). Exported as numpy weights for the client (selectors_bok.critic).
Eval   held-out inits 25-49 on the br0 labels: the critic picks k in 0..3 and is scored on both continuations
       (offline single-decision gain, Hajek-weighted, vs mean over k), next to the split oracle on the same subset.
"""
import argparse
import hashlib
import json
import os

import numpy as np

E = os.path.expanduser(os.environ.get("VLA_EXPERIMENTS", "~/disk2/vla-inference-scheduling/experiments"))
SCALE = np.array([0.842, 0.828, 0.937, 0.128, 0.175, 0.253, 1.0])


def load_runs(pattern_ks, allow_partial=False):
    out = {}
    for name in pattern_ks:
        p = "%s/results_bok/%s.jsonl" % (E, name)
        if not os.path.exists(p):
            if allow_partial and os.path.exists(p + ".part"):
                print("WARNING: using unfinished %s.part" % name)
                p += ".part"
            else:
                print("missing (or unfinished) run %s: skipped" % name)
                continue
        d = {}
        for l in open(p):
            try:
                r = json.loads(l)
            except ValueError:
                continue
            if "success" in r and not r.get("error"):
                d[(r["task_id"], r["init_state_id"])] = r
        out[name] = d
    return out


def chash(chunk):
    return hashlib.sha1(np.ascontiguousarray(chunk, dtype=np.float64).tobytes()).hexdigest()[:12]


def branch_hash(r):
    for c in r["calls"]:
        if c.get("how") == "branch":
            return c.get("ch")
    return None


def load_replay():
    rep = {}
    d = "%s/results_bok/cands/replay_k8" % E
    for f in os.listdir(d) if os.path.isdir(d) else []:
        if not f.endswith(".npz") or ".tmp" in f:
            continue
        t, e = f[1:-4].split("_e")
        z = np.load(os.path.join(d, f))
        if int(z["n"]) < 1 or "0_emb" not in z:
            continue
        rep[(int(t), int(e))] = {k: z["0_" + k] for k in ("cands", "emb", "emb_v", "state", "eef", "last_action", "seeds",
                                                          "hashes") if "0_" + k in z}
    return rep


def features(rep_e, pca):
    """[K, F] features of one replay entry (candidates of one decision)."""
    c = np.asarray(rep_e["cands"], dtype=np.float64) / SCALE            # [K, H, 7]
    K = len(c)
    flat = c.reshape(K, -1)
    rel = flat - flat.mean(0, keepdims=True)
    d = np.sqrt(((flat[:, None] - flat[None]) ** 2).sum(-1))
    med = d.sum(1)
    medf = (d[:, int(med.argmin())])[:, None]
    obs = np.concatenate([np.asarray(rep_e["emb"], np.float64).ravel()])
    zobs = pca(obs[None])[0]
    state = np.asarray(rep_e["state"], np.float64)
    return np.concatenate([np.repeat(zobs[None], K, 0), np.repeat(state[None], K, 0), flat, rel, medf], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pca", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--ens", type=int, default=5)
    ap.add_argument("--wd", type=float, default=1e-3)
    ap.add_argument("--rank_w", type=float, default=1.0)
    ap.add_argument("--out", default=E + "/bok/critic_v1.npz")
    ap.add_argument("--allow_partial", action="store_true", help="also use unfinished (.part) runs")
    a = ap.parse_args()
    import torch
    torch.manual_seed(0)
    rep = load_replay()
    d1 = load_runs(["d1_k%d_c1" % k for k in range(8)] + ["d2_k%d_c0" % k for k in range(8)],
                   allow_partial=a.allow_partial)
    br0 = load_runs(["br0_k%d_c%d" % (k, c) for k in range(4) for c in (0, 1)], allow_partial=a.allow_partial)
    print("replay entries %d; d1 runs %d; br0 runs %d" % (len(rep), len(d1), len(br0)))

    # ---- labels: (key) -> {k: [outcomes]} with replay-hash check
    def collect(runs, keyfilter):
        """lab[key][c][k] = success. d1_k{k}_c1 and br0_k{k}_c1 are the same experiment (fresh env, same seeds) on
        the episodes both cover, so one (key, c, k) is stored once (first source wins; outcomes are identical)."""
        lab, bad, dup = {}, 0, 0
        for name, d in runs.items():
            k = int(name.split("_k")[1].split("_")[0])
            c = int(name.rsplit("_c", 1)[1])
            for key, r in d.items():
                if not keyfilter(key) or key not in rep or k >= len(rep[key]["cands"]):
                    continue
                if branch_hash(r) != str(rep[key]["hashes"][k]):
                    bad += 1
                    continue
                slot = lab.setdefault(key, {}).setdefault(c, {})
                if k in slot:
                    dup += 1
                    continue
                slot[k] = float(r["success"])
        return lab, bad, dup
    train_lab, bad_tr, dup_tr = collect({**d1, **br0}, lambda key: key[1] < 25)
    print("train decisions %d (hash mismatches dropped: %d, duplicate experiments skipped: %d)" % (
        len(train_lab), bad_tr, dup_tr))

    # ---- PCA of the observation embedding, fit on train decisions
    X_obs = np.stack([np.asarray(rep[k]["emb"], np.float64).ravel() for k in train_lab])
    mu = X_obs.mean(0)
    U, S, Vt = np.linalg.svd(X_obs - mu, full_matrices=False)
    P = Vt[:a.pca]
    sd = (S[:a.pca] / np.sqrt(len(X_obs))) + 1e-6
    pca = lambda x: ((x - mu) @ P.T) / sd

    groups = []                    # one listwise group per (decision, continuation): same continuation noise
    for key, by_c in train_lab.items():
        F = features(rep[key], pca)
        for c, labs in sorted(by_c.items()):
            ks = sorted(labs)
            if len(ks) < 2:
                continue
            groups.append((key, F[ks], np.array([labs[k] for k in ks]), np.ones(len(ks))))
    Fall = np.concatenate([g[1] for g in groups])
    fmu, fsd = Fall.mean(0), Fall.std(0) + 1e-6
    print("feature dim %d; groups with mixed outcomes: %d / %d" % (
        Fall.shape[1], sum(1 for g in groups if g[2].min() != g[2].max()), len(groups)))

    def make_net():
        return torch.nn.Sequential(torch.nn.Linear(Fall.shape[1], a.hidden), torch.nn.GELU(),
                                   torch.nn.Dropout(0.2), torch.nn.Linear(a.hidden, a.hidden), torch.nn.GELU(),
                                   torch.nn.Linear(a.hidden, 1))

    rng = np.random.default_rng(0)
    nets = []
    for m in range(a.ens):
        idx = rng.integers(0, len(groups), len(groups))
        G = [groups[i] for i in idx]
        net = make_net()
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=a.wd)
        for ep in range(a.epochs):
            net.train()
            perm = rng.permutation(len(G))
            for b in range(0, len(G), 32):
                loss = 0.0
                for i in perm[b:b + 32]:
                    _, F, y, w = G[i]
                    x = torch.tensor((F - fmu) / fsd, dtype=torch.float32)
                    yt = torch.tensor(y, dtype=torch.float32)
                    logit = net(x).squeeze(-1)
                    loss = loss + torch.nn.functional.binary_cross_entropy_with_logits(
                        logit, yt, weight=torch.tensor(w, dtype=torch.float32)) / len(perm[b:b + 32])
                    if y.max() != y.min():
                        tgt = yt / yt.sum()
                        loss = loss - a.rank_w * (tgt * torch.log_softmax(logit, 0)).sum() / len(perm[b:b + 32])
                opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        nets.append(net)
        print("member %d trained" % m, flush=True)

    def score(F):
        x = torch.tensor((F - fmu) / fsd, dtype=torch.float32)
        with torch.no_grad():
            return np.mean([n(x).squeeze(-1).numpy() for n in nets], 0)

    # ---- offline evaluation on held-out inits (br0 labels, k=0..3, c=0/1)
    w = json.load(open(E + "/bok/strat_weights.json"))
    cnt = json.load(open(E + "/bok/succ_counts.json"))
    test_lab, bad_te, _ = collect(br0, lambda key: key[1] >= 25)
    rows = []
    for key, by_c in test_lab.items():
        if sorted(by_c) != [0, 1] or any(sorted(by_c[c]) != [0, 1, 2, 3] for c in (0, 1)):
            continue
        Y = np.array([[by_c[c][k] for c in (0, 1)] for k in range(4)])  # [4, 2]
        F = features(rep[key], pca)[:4]
        s = score(F)
        kstar = np.flatnonzero(s == s.max())
        crit = Y[kstar].mean()
        rand = Y.mean()
        orac = np.mean([Y[np.flatnonzero(Y[:, c] == Y[:, c].max()), 1 - c].mean() for c in (0, 1)])
        rows.append((key, w.get("%d_%d" % key, 1.0), crit - rand, orac - rand,
                     0 if cnt["%d_%d" % key] == 0 else (2 if cnt["%d_%d" % key] == 4 else 1)))
    if rows:
        W = np.array([r[1] for r in rows]); gc = np.array([r[2] for r in rows]); go = np.array([r[3] for r in rows])
        st = np.array([r[4] for r in rows])
        grp = [np.flatnonzero(st == s) for s in range(3) if (st == s).any()]
        bs = []
        for _ in range(10000):                       # bootstrap within strata, as in the prereg
            i = np.concatenate([g[rng.integers(0, len(g), len(g))] for g in grp])
            bs.append(np.sum(W[i] * gc[i]) / np.sum(W[i]))
        print("held-out decisions %d (hash mismatches %d): critic gain %+.1f [%+.1f, %+.1f] pts; split oracle %+.1f" % (
            len(rows), bad_te, 100 * np.sum(W * gc) / W.sum(), *(100 * np.percentile(bs, [2.5, 97.5])),
            100 * np.sum(W * go) / W.sum()))
    # ---- export numpy weights
    arrs = {"mu": mu, "P": P, "sd": sd, "fmu": fmu, "fsd": fsd, "n_members": np.asarray(len(nets))}
    for m, n in enumerate(nets):
        lin = [l for l in n if isinstance(l, torch.nn.Linear)]
        for j, l in enumerate(lin):
            arrs["m%d_W%d" % (m, j)] = l.weight.detach().numpy()
            arrs["m%d_b%d" % (m, j)] = l.bias.detach().numpy()
    np.savez(a.out, **arrs)
    print("saved", a.out)


if __name__ == "__main__":
    main()
