#!/usr/bin/env python3
"""DAgger distillation of the privileged residual teacher (residual_rl.py) into a vision-only student.

Student (residual_rl.StudentAgent): gate = learned per-step detector (react_step), d_hat = displacement regressor on
the event frames, features = proprio + base chunk + d_hat + timing (no object state). Rounds: beta = 1.0 (teacher acts,
student features recorded), then 0.5, then 0.0 (student acts); every round labels the visited states with the
teacher's greedy residual and retrains the student on all data so far (MSE in action space).

  python student_distill.py --teacher ckpt/res_v3.pt --dispreg bok/dispreg_v2_t24.pt --schedule react_step \\
      --detector bok/detector_v1_t24.pt --step_tau 1.628 --ports 8000,8001 --workers_per_port 6 --out ckpt/student_v1.pt
  then: residual_rl.py --eval student:ckpt/student_v1.pt --dispreg ... --schedule react_step --detector ... --step_tau ...
"""
import argparse
import json
import os
import time

import numpy as np
import torch

import residual_rl as RR
import run_libero as R


def collect(pool, args, teacher_sd, student_sd, beta, rnd, rng):
    combos = [(t, i) for t in range(RR.N_TASKS) for i in range(args.train_inits)]
    pairs = [c for c in combos for _ in range(args.reps)]
    by = {}
    for j, (tid, init) in enumerate(pairs):
        disturb = "none" if rng.random() < args.p_none else "object"
        by.setdefault(tid, []).append({"key": (RR.TRAIN_STREAM + 2, rnd * 1000 + j, tid, init), "init": init, "disturb": disturb})
    batches = []
    for tid, eps in by.items():
        for s in range(0, len(eps), args.batch_episodes):
            batches.append({"tid": tid, "episodes": eps[s:s + args.batch_episodes],
                            "student": {"sd": student_sd, "teacher_sd": teacher_sd, "beta": beta, "window": args.window,
                                        "no_dhat": bool(args.no_dhat)}})
    return RR.run_batches(pool, batches)


def fit(student, X, Y, args):
    opt = torch.optim.Adam(student.parameters(), lr=args.lr, weight_decay=1e-5)
    Xt, Yt = torch.from_numpy(X), torch.from_numpy(Y)
    scale = torch.tensor(RR.ACT_SCALE, dtype=torch.float32)
    n = len(Xt)
    for ep in range(args.epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for b in range(0, n, 256):
            i = perm[b:b + 256]
            pred = scale * torch.tanh(student(Xt[i]))
            loss = ((pred - Yt[i]) ** 2).sum(-1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(i)
    with torch.no_grad():
        pred = scale * torch.tanh(student(Xt))
        base = (Yt ** 2).sum(-1).mean()
        resid = ((pred - Yt) ** 2).sum(-1).mean()
    return {"fit_loss": tot / n, "label_var": float(base), "resid_var": float(resid), "r2": float(1 - resid / (base + 1e-9))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", required=True)
    p.add_argument("--out", default="ckpt/student_v1.pt")
    p.add_argument("--betas", default="1.0,0.5,0.0")
    p.add_argument("--reps", type=int, default=2, help="episodes per (task, init) per round")
    p.add_argument("--train_inits", type=int, default=25)
    p.add_argument("--p_none", type=float, default=0.1)
    p.add_argument("--window", type=int, default=6)
    p.add_argument("--no_dhat", action="store_true", help="ablation: zero the displacement estimate in the student input")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_episodes", type=int, default=6)
    # rollout settings shared with residual_rl.py
    p.add_argument("--ports", default="8000,8001")
    p.add_argument("--workers_per_port", type=int, default=6)
    p.add_argument("--suite", default="libero_10")
    p.add_argument("--n", type=int, default=2)
    p.add_argument("--h", type=int, default=10)
    p.add_argument("--schedule", default="react_step", choices=["static", "react_step"])
    p.add_argument("--react_setting", default="2,10")
    p.add_argument("--detector", default=None)
    p.add_argument("--step_tau", type=float, default=None)
    p.add_argument("--dispreg", required=True)
    p.add_argument("--gate", default="cum", choices=["cum", "jump"], help="privileged teacher gate (see residual_rl.py)")
    p.add_argument("--disturb_trigger", default="approach")
    p.add_argument("--disturb_mag", default="0.04,0.06")
    p.add_argument("--disturb_window", default="40,160")
    p.add_argument("--drift_steps", type=int, default=60)
    p.add_argument("--push_steps", type=int, default=5)
    p.add_argument("--mock", action="store_true")
    a = p.parse_args()
    base = R.build_parser().parse_args([])
    for k, v in vars(a).items():
        setattr(base, k, v)
    base.disturb = "object"
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    teacher_sd = torch.load(a.teacher, map_location="cpu")["model"]
    student = RR.StudentNet()
    pool = RR.make_pool(base)
    X_all, Y_all = [], []
    log = open(a.out + ".log.jsonl", "w")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    for rnd, beta in enumerate(float(b) for b in a.betas.split(",")):
        t0 = time.time()
        sd = {k: v.clone() for k, v in student.state_dict().items()}
        eps = collect(pool, base, teacher_sd, sd, beta, rnd, rng)
        act = [e for e in eps if e.get("X") is not None]
        X_all += [e["X"] for e in act]; Y_all += [e["Y"] for e in act]
        X, Y = np.concatenate(X_all), np.concatenate(Y_all)
        stats = fit(student, X, Y, a)
        dist = [e for e in eps if e["disturb"] == "object"]
        rec = {"round": rnd, "beta": beta, "episodes": len(eps), "success": float(np.mean([e["success"] for e in eps])),
               "success_disturbed": float(np.mean([e["success"] for e in dist])) if dist else None,
               "active_episodes": len(act), "samples_total": int(len(X)),
               "label_mean_abs": float(np.abs(Y).mean()), "sec": time.time() - t0, **stats}
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n"); log.flush()
        RR.atomic_save({"model": student.state_dict(), "window": a.window, "round": rnd, "args": vars(a),
                        "no_dhat": bool(a.no_dhat)}, a.out)
    pool.close()


if __name__ == "__main__":
    main()
