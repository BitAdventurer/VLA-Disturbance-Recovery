#!/usr/bin/env python3
"""Residual RL on top of the pi0.5 policy, whose weights are not updated (LIBERO-10, object displacement).

At every query of the base policy ((n,h) = (2,10) by default) a small Gaussian policy outputs a residual
delta = ACT_SCALE * tanh(u) for the translation actions (dx, dy, dz) that is added to every action of the window that
follows (run_libero.run_episode(..., residual=agent)). Version 1 is a privileged teacher: its observation contains the
true positions of the objects of interest relative to the gripper and their displacement since the start of the
episode (a vision student is distilled later). Initial policy mean = 0 (identical to the base policy).

Training: PPO with GAE over the ~37 decisions of an episode, reward = success at the end - res_cost * |delta/scale|^2
per decision; episodes from train initial states (inits 0..24, keys (TRAIN_STREAM, iteration, task, init)), a fraction
undisturbed. Rollouts in a spawn pool of LIBERO workers spread over the policy-server ports.
Evaluation (--eval): greedy residual on inits 25..49 with the evaluation keys (0, seed, task, init), comparable with
the (2,10) noise-draw pool in results_dist.

  python residual_rl.py --out ckpt/res_v1.pt --ports 8000,8001 --workers_per_port 6 --iters 80
  python residual_rl.py --eval ckpt/res_v1.pt --ports 8000,8001 --workers_per_port 6 --eval_out results_res/res_v1_eval.jsonl
"""
import argparse
import copy
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn

import run_libero as R

TRAIN_STREAM = 9
ACT_SCALE = np.array([0.3, 0.3, 0.15])       # max |residual| per step on x, y, z (action units; ~0.012 m per unit)
N_OBJ = 3
N_TASKS = 10
FEAT_DIM = 8 + 8 + N_OBJ * 7 + 6 + N_TASKS
MIN_DISP, LIFT_DZ = 0.02, 0.03               # trigger: object moved > 2 cm without lift; gate off once lifted 3 cm


def find_displaced(env, ref):
    """(name, joint, displacement) of an object of interest that moved > MIN_DISP horizontally without being lifted."""
    for name, j in R.movable_objects(env):
        p = R.object_pos(env, j)
        if p is None or name not in ref:
            continue
        disp = p - ref[name]
        if np.linalg.norm(disp[:2]) > MIN_DISP and abs(disp[2]) < 0.02:
            return name, j, disp
    return None


def find_jump(env, prev_pos, eef, ref, grip_prev, min_jump=0.025, min_dist=0.05, max_dz=0.015):
    """(name, joint, jump) of an object of interest that moved > min_jump horizontally since the previous query while
    resting at its initial height, with the gripper open at the previous query and > min_dist away now: a sudden
    external displacement, not a nudge or a carried object (checked on undisturbed episodes: 0 false triggers / 453)."""
    if grip_prev > 0:
        return None
    for name, j in R.movable_objects(env):
        p = R.object_pos(env, j)
        if p is None or name not in prev_pos or name not in ref:
            continue
        jump = p - prev_pos[name]
        if (np.linalg.norm(jump[:2]) > min_jump and abs(p[2] - ref[name][2]) < max_dz
                and np.linalg.norm(p - np.asarray(eef, float)) > min_dist):
            return name, j, jump
    return None


def snapshot(env):
    return {name: R.object_pos(env, j).copy() for name, j in R.movable_objects(env) if R.object_pos(env, j) is not None}


def features(env, obs, chunk, info, ref, prev_delta, task_id, first=None, since=0):
    """Privileged low-dimensional state at a query. The triggering (displaced) object, if any, takes the first slot;
    other objects of interest follow by distance to the gripper."""
    q2aa = R._quat2axisangle
    eef = np.asarray(obs["robot0_eef_pos"], float)
    proprio = np.concatenate([eef, q2aa(obs["robot0_eef_quat"]), np.asarray(obs["robot0_gripper_qpos"], float)])
    c = np.asarray(chunk, float)
    base = np.concatenate([c[:, :3].mean(0), c[:, :3].sum(0) / 10.0, [c[0, 6], c[-1, 6]]])
    objs = []
    for name, j in R.movable_objects(env):
        p = R.object_pos(env, j)
        if p is None:
            continue
        rel = p - eef
        disp = p - ref.get(name, p)
        key = -1.0 if name == first else float(np.linalg.norm(rel))
        objs.append((key, np.concatenate([rel * 10.0, disp * 20.0, [1.0]])))
    objs.sort(key=lambda o: o[0])
    of = [o[1] for o in objs[:N_OBJ]] + [np.zeros(7)] * max(0, N_OBJ - len(objs))
    tt = np.array([info["t"] / info["limit"], float(info["last_gripper"]), min(since, 20) / 10.0])
    onehot = np.zeros(N_TASKS)
    onehot[task_id % N_TASKS] = 1.0
    x = np.concatenate([proprio, base, np.concatenate(of), tt, np.asarray(prev_delta, float) / ACT_SCALE, onehot])
    return x.astype(np.float32)


class ResPolicy(nn.Module):
    def __init__(self, d=FEAT_DIM, hidden=256, act_dim=3, log_std_init=-1.2):
        super().__init__()
        self.pi = nn.Sequential(nn.Linear(d, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(),
                                nn.Linear(hidden, act_dim))
        nn.init.zeros_(self.pi[-1].weight)
        nn.init.zeros_(self.pi[-1].bias)                     # initial residual mean 0 = the base policy
        self.log_std = nn.Parameter(torch.full((act_dim,), float(log_std_init)))
        self.v = nn.Sequential(nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Linear(hidden, hidden),
                               nn.LayerNorm(hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.LayerNorm(hidden),
                               nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.pi(x), self.log_std.clamp(np.log(0.05), 0.5), self.v(x).squeeze(-1)

    def log_prob(self, x, u):
        mu, ls, v = self(x)
        lp = (-0.5 * ((u - mu) / ls.exp()) ** 2 - ls - 0.5 * np.log(2 * np.pi)).sum(-1)
        ent = (ls + 0.5 * np.log(2 * np.pi * np.e)).sum() * torch.ones_like(lp)
        return lp, v, ent


class ResidualAgent:
    """run_episode residual hook, gated: inactive (zero residual, nothing recorded) until an object of interest has been
    displaced (privileged trigger; a student uses the detector), and again once that object has been lifted.
    Records (state, u, logp, value, delta) at every active query. script=(alpha, chunks): execute the scripted shift
    instead of the policy but record the same states (behaviour-cloning warm start)."""

    def __init__(self, model, task_id, greedy=False, seed=None, disabled=False, script=None, gate="cum"):
        self.model, self.task_id, self.greedy, self.disabled, self.script = model, task_id, greedy, disabled, script
        self.gate = gate                                     # "cum": moved > 2 cm since reset; "jump": sudden per-query jump
        self.gen = torch.Generator()
        self.gen.manual_seed(int(seed or 0))

    def reset(self, env):
        self.ref = {}
        for name, j in R.movable_objects(env):
            p = R.object_pos(env, j)
            if p is not None:
                self.ref[name] = p.copy()
        self.prev = np.zeros(3)
        self.prev_pos, self.prev_grip = dict(self.ref), -1.0
        self.trig, self.off, self.since = None, False, 0
        self.S, self.U, self.LP, self.V, self.D = [], [], [], [], []

    def at_query(self, env, obs, chunk, info):
        delta = np.zeros(7)
        if self.disabled or self.off:
            return delta
        if self.trig is None:
            hit = (find_jump(env, self.prev_pos, obs["robot0_eef_pos"], self.ref, self.prev_grip) if self.gate == "jump"
                   else find_displaced(env, self.ref))
            self.prev_pos, self.prev_grip = snapshot(env), float(info["last_gripper"])
            if hit is None:
                return delta
            self.trig = hit                                  # (name, joint, displacement at the trigger)
            self.ref = {**self.ref, self.trig[0]: R.object_pos(env, self.trig[1]) - self.trig[2]}   # pre-jump position
        p = R.object_pos(env, self.trig[1])
        if p is not None and p[2] - self.ref[self.trig[0]][2] > LIFT_DZ:
            self.off = True                                  # displaced object picked up: back to the base policy
            return delta
        x = features(env, obs, chunk, info, self.ref, self.prev, self.task_id, first=self.trig[0], since=self.since)
        with torch.no_grad():
            xt = torch.from_numpy(x)[None]
            mu, ls, v = self.model(xt)
            if self.script is not None:
                alpha, k = self.script
                dxy = alpha * self.trig[2][:2] / (ScriptedShift.GAIN * 10 * k) if self.since < k else np.zeros(2)
                tgt = np.clip(np.concatenate([dxy, [0.0]]) / ACT_SCALE, -0.995, 0.995)
                u = torch.from_numpy(np.arctanh(tgt).astype(np.float32))[None]
            else:
                u = mu if self.greedy else mu + ls.exp() * torch.randn(mu.shape, generator=self.gen)
            lp = (-0.5 * ((u - mu) / ls.exp()) ** 2 - ls - 0.5 * np.log(2 * np.pi)).sum(-1)
        u = u[0].numpy()
        delta[:3] = ACT_SCALE * np.tanh(u)
        self.S.append(x)
        self.U.append(u.astype(np.float32))
        self.LP.append(float(lp[0]))
        self.V.append(float(v[0]))
        self.D.append(delta[:3].copy())
        self.prev = delta[:3].copy()
        self.since += 1
        return delta


class ScriptedShift:
    """Privileged scripted residual (probe): once an object of interest has moved > min_disp metres without being
    lifted, add alpha * d (its horizontal displacement) spread over the next `chunks` queries:
    delta_xy = alpha * d / (GAIN * 10 * chunks) per step (GAIN = measured eef metres per action unit per step)."""
    GAIN = 0.012

    def __init__(self, alpha=1.0, chunks=2, min_disp=0.02, max_delta=0.5, gate="cum"):
        self.alpha, self.chunks, self.min_disp, self.max_delta, self.gate = alpha, chunks, min_disp, max_delta, gate
        self.D = []

    def reset(self, env):
        self.ref = snapshot(env)
        self.prev_pos, self.prev_grip = dict(self.ref), -1.0
        self.left, self.d, self.D = None, None, []

    def at_query(self, env, obs, chunk, info):
        delta = np.zeros(7)
        if self.left is None:                            # not triggered yet: look for a displaced object
            if self.gate == "jump":
                hit = find_jump(env, self.prev_pos, obs["robot0_eef_pos"], self.ref, self.prev_grip)
                self.prev_pos, self.prev_grip = snapshot(env), float(info["last_gripper"])
                if hit is not None:
                    self.d, self.left = hit[2][:2].copy(), self.chunks
            else:
                for name, j in R.movable_objects(env):
                    p = R.object_pos(env, j)
                    if p is None or name not in self.ref:
                        continue
                    disp = p - self.ref[name]
                    if np.linalg.norm(disp[:2]) > self.min_disp and abs(disp[2]) < 0.02:
                        self.d, self.left = disp[:2].copy(), self.chunks
                        break
        if self.left:
            delta[:2] = np.clip(self.alpha * self.d / (self.GAIN * 10 * self.chunks), -self.max_delta, self.max_delta)
            self.left -= 1
        self.D.append(delta[:3].copy())
        return delta


class VisionShift:
    """Vision-only residual (deployable version of ScriptedShift): the learned per-step detector (react_step) marks a
    disturbance and hands over the 128x128 RGB frames right before/after it (info["event"]); a displacement regressor
    (train_dispreg2.py, TorchScript) predicts the object's horizontal displacement d_hat from them, and alpha * d_hat is
    spread over the next `chunks` queries exactly like the scripted shift."""

    wants_event = True

    def __init__(self, reg_path, alpha=0.5, chunks=2, max_delta=0.5):
        import torch
        torch.set_num_threads(2)
        self.reg = torch.jit.load(reg_path).eval()
        self.alpha, self.chunks, self.max_delta = alpha, chunks, max_delta
        self.D, self.preds = [], []

    def reset(self, env):
        self.left, self.d, self.D, self.preds = None, None, [], []

    def predict(self, ev):
        import torch
        x = np.concatenate([ev["before_img"], ev["after_img"], ev["before_wrist"], ev["after_wrist"]], -1)
        x = torch.as_tensor(x[None]).float().permute(0, 3, 1, 2) / 255.0
        x = torch.cat([x, x[:, 3:6] - x[:, 0:3], x[:, 9:12] - x[:, 6:9]], 1)
        v = torch.as_tensor(np.concatenate([ev["action"], ev["deef"] * 50.0])[None].astype(np.float32))
        with torch.no_grad():
            return self.reg(x, v)[0].numpy().astype(float)

    def at_query(self, env, obs, chunk, info):
        delta = np.zeros(7)
        ev = info.get("event")
        if ev is not None:                               # a detection since the previous query: (re)estimate d
            self.d, self.left = self.predict(ev), self.chunks
            self.preds.append([int(info["t"]), float(self.d[0]), float(self.d[1])])
        if self.left:
            delta[:2] = np.clip(self.alpha * self.d / (ScriptedShift.GAIN * 10 * self.chunks), -self.max_delta, self.max_delta)
            self.left -= 1
        self.D.append(delta[:3].copy())
        return delta


STUDENT_DIM = 8 + 8 + 3 + 1 + 1 + 1 + 1 + 3 + N_TASKS


class StudentNet(nn.Module):
    """Vision-only residual student: MLP on non-privileged features -> residual (ACT_SCALE * tanh)."""

    def __init__(self, d=STUDENT_DIM, hidden=256):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(d, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 3))
        nn.init.zeros_(self.f[-1].weight)
        nn.init.zeros_(self.f[-1].bias)

    def forward(self, x):
        return self.f(x)


def student_features(obs, chunk, info, d_hat, active, since, prev_delta, task_id, no_dhat=False):
    """Non-privileged features at a query: proprio, base-chunk summary, estimated displacement d_hat (from the
    regressor at the detection), activity flag, queries since the detection, time, last gripper, previous residual,
    task one-hot."""
    q2aa = R._quat2axisangle
    eef = np.asarray(obs["robot0_eef_pos"], float)
    proprio = np.concatenate([eef, q2aa(obs["robot0_eef_quat"]), np.asarray(obs["robot0_gripper_qpos"], float)])
    c = np.asarray(chunk, float)
    base = np.concatenate([c[:, :3].mean(0), c[:, :3].sum(0) / 10.0, [c[0, 6], c[-1, 6]]])
    dh = np.zeros(2) if (d_hat is None or no_dhat) else np.asarray(d_hat, float)
    onehot = np.zeros(N_TASKS)
    onehot[task_id % N_TASKS] = 1.0
    x = np.concatenate([proprio, base, dh * 20.0, [np.linalg.norm(dh) * 20.0], [float(active)], [min(since, 20) / 10.0],
                        [info["t"] / info["limit"]], [float(info["last_gripper"])], np.asarray(prev_delta, float) / ACT_SCALE,
                        onehot])
    return x.astype(np.float32)


class StudentAgent:
    """Vision-only residual for distillation (DAgger). Gate = the learned per-step detector (react_step hands over the
    event frames via info["event"]); d_hat from the displacement regressor; active for `window` queries after each
    detection. Acts with the student (greedy) or, with probability beta, with the privileged teacher (a greedy
    ResidualAgent run in parallel); records (student features, teacher residual) for every active query."""
    wants_event = True

    def __init__(self, student, teacher, task_id, reg_path, beta=0.0, window=6, seed=0, gate="cum", no_dhat=False):
        import torch
        torch.set_num_threads(2)
        self.student, self.task_id, self.beta, self.window, self.no_dhat = student, task_id, beta, window, no_dhat
        self.teacher = ResidualAgent(teacher, task_id, greedy=True, gate=gate) if teacher is not None else None
        self.reg = torch.jit.load(reg_path).eval()
        self.rng = np.random.default_rng(seed)

    def reset(self, env):
        if self.teacher is not None:
            self.teacher.reset(env)
        self.d_hat, self.left, self.since, self.prev = None, 0, 0, np.zeros(3)
        self.X, self.Y, self.D = [], [], []

    def at_query(self, env, obs, chunk, info):
        import torch
        label = self.teacher.at_query(env, obs, chunk, info)[:3] if self.teacher is not None else np.zeros(3)
        ev = info.get("event")
        if ev is not None:
            self.d_hat, self.left, self.since = VisionShift.predict(self, ev), self.window, 0
        delta = np.zeros(7)
        if self.left <= 0:
            self.D.append(delta[:3].copy())
            return delta
        x = student_features(obs, chunk, info, self.d_hat, True, self.since, self.prev, self.task_id, self.no_dhat)
        with torch.no_grad():
            mu = ACT_SCALE * np.tanh(self.student(torch.from_numpy(x)[None])[0].numpy())
        act = np.asarray(label, float) if self.rng.random() < self.beta else mu
        delta[:3] = act
        self.X.append(x)
        self.Y.append(np.asarray(label, np.float32))
        self.D.append(delta[:3].copy())
        self.prev = delta[:3].copy()
        self.left -= 1
        self.since += 1
        return delta


# ----------------------------------------------------------------------------- rollout workers
# The LIBERO simulator state depends only on how many times an env was reset since it was built (not on what happened
# in earlier episodes): the k-th episode of a sequential run equals a fresh env with k throwaway resets. So a worker runs
# a batch of episodes of ONE task on ONE env, and evaluation batches align every episode's reset count with its init
# index, which reproduces the sequential runs in results_dist exactly (same population, paired comparisons).
_W = {}


def _worker_init(args, ports):
    torch.set_num_threads(1)
    wargs = copy.copy(args)
    wargs.port = ports[os.getpid() % len(ports)]     # also works for replacement workers after a crash
    policy, tools, get_env, _ = R.make_world(wargs)
    _W.update(args=wargs, policy=policy, tools=tools, get_env=get_env, model=ResPolicy())


def _run_one(slot, ep, b):
    a = copy.copy(_W["args"])
    a.disturb = ep["disturb"]
    if "noise_salt" in b:
        a.noise_salt = b["noise_salt"]
    key, tid, init_id = tuple(ep["key"]), b["tid"], ep["init"]
    if b.get("probe") is not None:
        agent = ScriptedShift(*b["probe"], gate=getattr(_W["args"], "gate", "cum"))
    elif b.get("vision") is not None:
        agent = VisionShift(_W["args"].dispreg, *b["vision"])
    elif b.get("student") is not None:
        st = b["student"]
        if "student_net" not in _W:
            _W["student_net"], _W["teacher_net"] = StudentNet(), ResPolicy()
        _W["student_net"].load_state_dict(st["sd"])
        teacher = None
        if st.get("teacher_sd") is not None:
            _W["teacher_net"].load_state_dict(st["teacher_sd"])
            teacher = _W["teacher_net"]
        agent = StudentAgent(_W["student_net"], teacher, tid, _W["args"].dispreg, beta=st.get("beta", 0.0),
                             window=st.get("window", 6), seed=R.stream_seed(10, *key), gate=getattr(_W["args"], "gate", "cum"),
                             no_dhat=bool(st.get("no_dhat", False)))
    else:
        agent = ResidualAgent(_W["model"], tid, greedy=b.get("greedy", False), seed=R.stream_seed(8, *key),
                              disabled=b.get("disabled", False), script=b.get("bc"), gate=getattr(_W["args"], "gate", "cum"))
    sched = R.Schedule(a.schedule, a.n, a.h, react_setting=tuple(int(x) for x in a.react_setting.split(",")))

    if not 0 <= init_id < len(slot.init_states):        # was a silent modulo wrap (50 -> 0); identical for valid ids
        raise IndexError("init %d outside the %d loaded initial states (states 50-99 need --init_file full)"
                         % (init_id, len(slot.init_states)))

    def run():
        slot.env._init_state = slot.init_states[init_id]
        slot.env.seed(R.sim_seed(key))
        slot.resets += 1
        return R.run_episode(slot.env, _W["policy"], slot.description, sched, None, a, _W["tools"], None,
                             episode_key=key, residual=agent)
    res = R.run_safely(run, slot.rebuild, _W["policy"], "res %s" % (key,))
    if res.get("error"):
        return None
    d = res.get("disturbance") or {}
    return {"key": list(key), "task_id": tid, "init_state_id": init_id, "disturb": ep["disturb"],
            "success": float(res["success"]), "episode_length": res["episode_length"], "n_calls": res["n_calls"],
            "applied_t": d.get("applied_t"), "object": d.get("object"),
            "S": np.stack(agent.S) if getattr(agent, "S", None) else np.zeros((0, FEAT_DIM), np.float32),
            "U": np.stack(agent.U) if getattr(agent, "U", None) else np.zeros((0, 3), np.float32),
            "LP": np.array(getattr(agent, "LP", []), np.float32), "V": np.array(getattr(agent, "V", []), np.float32),
            "D": np.stack(agent.D) if agent.D else np.zeros((0, 3)), "detections": res.get("detections"),
            "preds": getattr(agent, "preds", None), "true_d": ([float(x) for x in np.asarray(d.get("dir", [0, 0])) * d.get("mag", 0.0)]
                                                              if d.get("object") is not None else None),
            "X": np.stack(agent.X) if getattr(agent, "X", None) else None,
            "Y": np.stack(agent.Y) if getattr(agent, "Y", None) else None}


def _worker_batch(b):
    """b: {"tid", "episodes": [{"key", "init", "disturb"}], "sd", "align", greedy/disabled/probe/bc/noise_salt}."""
    if b.get("sd") is not None:
        _W["model"].load_state_dict(b["sd"])
    slot = R.EnvSlot(_W["get_env"], b["tid"])
    slot.resets = 0
    rebuild = slot.rebuild

    def rebuild_and_count():                         # a rebuilt env starts counting resets again
        rebuild()
        slot.resets = 0
    slot.rebuild = rebuild_and_count
    out = []
    try:
        for ep in b["episodes"]:
            if b.get("align"):
                while slot.resets < ep["init"]:          # throwaway resets: same state as a sequential run
                    slot.env.reset()
                    slot.resets += 1
            out.append(_run_one(slot, ep, b))
    finally:
        slot.close()
    return out


def make_pool(args):
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    plist = [int(p) for p in args.ports.split(",")]
    n = len(plist) * args.workers_per_port
    print("%d rollout workers on ports %s" % (n, plist), flush=True)
    return ctx.Pool(n, initializer=_worker_init, initargs=(args, plist))


def run_batches(pool, batches, per_episode_s=60.0, retries=2):
    """Run batches with at most one batch in flight per worker (so a batch's clock starts when it starts); a batch
    whose result does not arrive within its time limit (worker died) is resubmitted. Returns the flattened results."""
    n_slots = max(1, getattr(pool, "_processes", 1))
    queue = list(range(len(batches)))
    pending, tries, results = {}, {i: 0 for i in queue}, {}

    def submit(i):
        pending[i] = (pool.apply_async(_worker_batch, (batches[i],)), time.time())
    while queue or pending:
        while queue and len(pending) < n_slots:
            submit(queue.pop(0))
        for i, (ar, t0) in list(pending.items()):
            limit = 300.0 + per_episode_s * len(batches[i]["episodes"])
            if ar.ready():
                del pending[i]
                try:
                    results[i] = ar.get()
                except Exception as e:                   # noqa: BLE001
                    print("batch %d failed: %r" % (i, e), flush=True)
                    results[i] = None
                if results[i] is None and tries[i] < retries:
                    tries[i] += 1
                    queue.insert(0, i)
            elif time.time() - t0 > limit:
                del pending[i]
                print("batch %d timed out after %.0f s (lost worker?)%s" % (
                    i, time.time() - t0, "; resubmitting" if tries[i] < retries else "; dropped"), flush=True)
                if tries[i] < retries:
                    tries[i] += 1
                    queue.insert(0, i)
                else:
                    results[i] = None
        time.sleep(0.5)
    return [e for i in sorted(results) for e in (results[i] or []) if e is not None]


# ----------------------------------------------------------------------------- PPO
def gae(ep, args):
    """Per-decision rewards: -res_cost*|delta/scale|^2, + success at the last decision. Returns (adv, ret)."""
    T = len(ep["LP"])
    r = -args.res_cost * ((ep["D"] / ACT_SCALE) ** 2).sum(-1)
    if T:
        r[-1] += ep["success"]
    v = ep["V"]
    adv = np.zeros(T, np.float32)
    last = 0.0
    for t in reversed(range(T)):
        nv = v[t + 1] if t + 1 < T else 0.0
        delta = r[t] + args.gamma * nv - v[t]
        last = delta + args.gamma * args.lam * last
        adv[t] = last
    return adv, adv + v


def ppo_update(model, opts, eps, args, actor=True):
    eps = [e for e in eps if len(e["LP"])]
    if not eps:
        return {"kl": 0.0, "clipfrac": 0.0, "vloss": 0.0, "n": 0, "epochs": 0}
    S = torch.from_numpy(np.concatenate([e["S"] for e in eps]))
    U = torch.from_numpy(np.concatenate([e["U"] for e in eps]))
    LP0 = torch.from_numpy(np.concatenate([e["LP"] for e in eps]))
    A, RET = zip(*[gae(e, args) for e in eps])
    A = torch.from_numpy(np.concatenate(A))
    RET = torch.from_numpy(np.concatenate(RET))
    A = (A - A.mean()) / (A.std() + 1e-8)
    stats = {"kl": 0.0, "clipfrac": 0.0, "vloss": 0.0, "n": len(S)}
    for epoch in range(args.ppo_epochs):
        perm = torch.randperm(len(S))
        kls = []
        for s in range(0, len(S), args.minibatch):
            mb = perm[s:s + args.minibatch]
            lp, v, ent = model.log_prob(S[mb], U[mb])
            ratio = (lp - LP0[mb]).exp()
            pg = -torch.min(ratio * A[mb], ratio.clamp(1 - args.clip, 1 + args.clip) * A[mb]).mean()
            vl = ((v - RET[mb]) ** 2).mean()
            loss = (pg - args.ent_coef * ent.mean() if actor else 0.0) + args.vf_coef * vl
            for o in opts.values():
                o.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opts["critic"].step()
            if actor:
                opts["actor"].step()
            with torch.no_grad():
                kls.append(float((LP0[mb] - lp).mean()))
                stats["clipfrac"] = float(((ratio - 1).abs() > args.clip).float().mean())
                stats["vloss"] = float(vl)
        stats["kl"] = float(np.mean(kls))
        if actor and stats["kl"] > args.target_kl:
            break
    stats["epochs"] = epoch + 1
    return stats


def atomic_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def task_batches(pairs, it, sd, rng, args, stream=TRAIN_STREAM, extra=None):
    """Group (task, init) training episodes into per-task batches of at most args.batch_episodes."""
    by = {}
    for j, (tid, init) in enumerate(pairs):
        disturb = "none" if rng.random() < args.p_none else "object"
        by.setdefault(tid, []).append({"key": (stream, it * 1000 + j, tid, init), "init": init, "disturb": disturb})
    out = []
    for tid, eps in by.items():
        for s in range(0, len(eps), args.batch_episodes):
            out.append({"tid": tid, "episodes": eps[s:s + args.batch_episodes], "sd": sd, **(extra or {})})
    return out


def train(args):
    model = ResPolicy(log_std_init=args.log_std_init)
    opts = {"actor": torch.optim.Adam(list(model.pi.parameters()) + [model.log_std], lr=args.lr),
            "critic": torch.optim.Adam(model.v.parameters(), lr=args.lr_critic)}
    start = 0
    part = args.out + ".partial"
    if args.init_from and not os.path.exists(part):        # e.g. the BC checkpoint of an earlier run
        model.load_state_dict(torch.load(args.init_from, map_location="cpu")["model"])
        if args.log_std_init is not None:
            with torch.no_grad():
                model.log_std.fill_(args.log_std_init)
        print("initialised from %s" % args.init_from, flush=True)
    if os.path.exists(part):
        st = torch.load(part, map_location="cpu")
        model.load_state_dict(st["model"]); start = st["iter"] + 1
        for k in opts:
            opts[k].load_state_dict(st["opt"][k])
        random.setstate(st["py_rng"]); np.random.set_state(st["np_rng"]); torch.set_rng_state(st["torch_rng"])
        print("resuming at iteration %d" % start, flush=True)
    curve = open(args.out + ".curve.jsonl", "a" if start else "w")
    pool = make_pool(args)
    rng = np.random.default_rng(args.seed + start)
    combos = [(t, i) for t in range(N_TASKS) for i in range(args.train_inits)]
    if start == 0 and args.bc_script and not args.init_from:   # behaviour-cloning warm start from the scripted shift
        alpha, k = [float(x) for x in args.bc_script.split(",")]
        sd = {kk: v.clone() for kk, v in model.state_dict().items()}
        reps = max(1, args.bc_episodes // len(combos))
        bcb = task_batches([c for c in combos for _ in range(reps)], 0, sd, rng, args, stream=TRAIN_STREAM + 1,
                           extra={"bc": (alpha, int(k))})
        eps = [e for e in run_batches(pool, bcb) if len(e["LP"])]
        if not eps:
            raise SystemExit("behaviour cloning: no active decisions (no displaced object was ever detected)")
        S = torch.from_numpy(np.concatenate([e["S"] for e in eps]))
        U = torch.from_numpy(np.concatenate([e["U"] for e in eps]))
        RET = torch.from_numpy(np.concatenate([gae(e, args)[1] * 0 + e["success"] for e in eps]).astype(np.float32))
        for epoch in range(200):
            perm = torch.randperm(len(S))
            for b in range(0, len(S), 256):
                mb = perm[b:b + 256]
                mu, _, v = model(S[mb])
                loss = ((mu - U[mb]) ** 2).mean() + ((v - RET[mb]) ** 2).mean()
                for o in opts.values():
                    o.zero_grad()
                loss.backward()
                for o in opts.values():
                    o.step()
        print(json.dumps({"bc_episodes": len(eps), "bc_success": float(np.mean([e["success"] for e in eps])),
                          "bc_decisions": len(S), "bc_loss": float(loss)}), flush=True)
        atomic_save({"model": model.state_dict(), "iter": -1, "args": vars(args)}, args.out + ".bc")
    for it in range(start, args.iters):
        t0 = time.time()
        sd = {k: v.clone() for k, v in model.state_dict().items()}
        pick = rng.choice(len(combos), size=args.episodes, replace=False)
        eps = run_batches(pool, task_batches([combos[ci] for ci in pick], it, sd, rng, args))
        stats = ppo_update(model, opts, eps, args, actor=it >= args.critic_warmup)
        dist = [e for e in eps if e["disturb"] == "object"]
        act = [e for e in eps if len(e["LP"])]
        rec = {"iter": it, "episodes": len(eps), "success": float(np.mean([e["success"] for e in eps])),
               "active_episodes": len(act), "decisions": int(sum(len(e["LP"]) for e in act)),
               "success_disturbed": float(np.mean([e["success"] for e in dist])) if dist else None,
               "mean_abs_delta": float(np.mean(np.concatenate([np.abs(e["D"]).ravel() for e in act]))) if act else 0.0,
               "std": [float(x) for x in model.log_std.detach().exp()], "sec": time.time() - t0, **stats}
        print(json.dumps(rec), flush=True)
        curve.write(json.dumps(rec) + "\n"); curve.flush()
        st = {"model": model.state_dict(), "opt": {k: o.state_dict() for k, o in opts.items()}, "iter": it,
              "args": vars(args),
              "py_rng": random.getstate(), "np_rng": np.random.get_state(), "torch_rng": torch.get_rng_state()}
        atomic_save(st, part)
        if (it + 1) % args.save_every == 0 or it + 1 == args.iters:
            atomic_save({"model": model.state_dict(), "iter": it, "args": vars(args)}, args.out + ".it%03d" % it)
    atomic_save({"model": model.state_dict(), "iter": args.iters - 1, "args": vars(args)}, args.out)
    pool.close()


def evaluate(args):
    sd, scripted, vision, student = None, None, None, None
    if args.eval.startswith("scripted:"):                # scripted:alpha,chunks
        scripted = tuple(float(x) for x in args.eval.split(":", 1)[1].split(","))
        scripted = (scripted[0], int(scripted[1]))
    elif args.eval.startswith("vision:"):                # vision:alpha,chunks (needs --dispreg and the react_step detector)
        vision = tuple(float(x) for x in args.eval.split(":", 1)[1].split(","))
        vision = (vision[0], int(vision[1]))
    elif args.eval.startswith("student:"):               # student:ckpt (needs --dispreg and the react_step detector)
        st = torch.load(args.eval.split(":", 1)[1], map_location="cpu")
        student = {"sd": st["model"], "teacher_sd": None, "beta": 0.0, "window": st.get("window", 6),
                   "no_dhat": bool(st.get("no_dhat", False))}
    elif args.eval != "zero":
        sd = torch.load(args.eval, map_location="cpu")["model"]
    pool = make_pool(args)
    inits = range(args.eval_init_lo, args.eval_init_hi)
    batches = [{"tid": t, "align": True, "sd": sd, "greedy": True, "disabled": sd is None, "probe": scripted, "vision": vision,
                "student": student,
                "noise_salt": args.eval_noise_salt,
                "episodes": [{"key": (R.EVAL_STREAM, args.eval_seed, t, i), "init": i, "disturb": args.eval_disturb}
                             for i in inits]} for t in range(N_TASKS)]
    os.makedirs(os.path.dirname(os.path.abspath(args.eval_out)), exist_ok=True)
    eps = run_batches(pool, batches, per_episode_s=60.0)
    with open(args.eval_out + ".part", "w") as f:
        for e in sorted(eps, key=lambda e: (e["task_id"], e["init_state_id"])):
            row = {k: v for k, v in e.items() if k not in ("S", "U", "LP", "V", "X", "Y")}
            row["D"] = np.round(np.asarray(e["D"]), 4).tolist()
            f.write(json.dumps(row) + "\n")
    os.replace(args.eval_out + ".part", args.eval_out)
    rows = [json.loads(l) for l in open(args.eval_out)]
    print("eval %s: %d episodes, success %.1f%%" % (args.eval, len(rows), 100 * np.mean([r["success"] for r in rows])))
    pool.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="ckpt/res_v1.pt")
    p.add_argument("--ports", default="8000,8001")
    p.add_argument("--workers_per_port", type=int, default=6)
    p.add_argument("--iters", type=int, default=80)
    p.add_argument("--episodes", type=int, default=120, help="episodes per PPO iteration")
    p.add_argument("--train_inits", type=int, default=25)
    p.add_argument("--p_none", type=float, default=0.0, help="fraction of undisturbed training episodes")
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--lr_critic", type=float, default=1e-3)
    p.add_argument("--critic_warmup", type=int, default=2, help="first iterations: critic updates only")
    p.add_argument("--bc_script", default=None, help="alpha,chunks: behaviour-clone the scripted shift first")
    p.add_argument("--bc_episodes", type=int, default=250)
    p.add_argument("--init_from", default=None, help="checkpoint to start PPO from (skips BC)")
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--ppo_epochs", type=int, default=8)
    p.add_argument("--minibatch", type=int, default=256)
    p.add_argument("--target_kl", type=float, default=0.02)
    p.add_argument("--vf_coef", type=float, default=0.5)
    p.add_argument("--ent_coef", type=float, default=0.0)
    p.add_argument("--res_cost", type=float, default=0.01)
    p.add_argument("--log_std_init", type=float, default=-1.2)
    p.add_argument("--save_every", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval", default=None, help="checkpoint to evaluate greedily ('zero' = no residual)")
    p.add_argument("--eval_out", default="results_res/eval.jsonl")
    p.add_argument("--eval_seed", type=int, default=0)
    p.add_argument("--eval_init_lo", type=int, default=25)
    p.add_argument("--eval_init_hi", type=int, default=50)
    p.add_argument("--eval_disturb", default="object")
    p.add_argument("--eval_noise_salt", type=int, default=0)
    p.add_argument("--batch_episodes", type=int, default=6, help="training episodes per worker batch (one task)")
    # rollout settings (same meaning as in run_libero.py)
    p.add_argument("--suite", default="libero_10")
    p.add_argument("--n", type=int, default=2)
    p.add_argument("--h", type=int, default=10)
    p.add_argument("--schedule", default="static", choices=["static", "react_step"])
    p.add_argument("--react_setting", default="2,10")
    p.add_argument("--detector", default=None)
    p.add_argument("--dispreg", default=None, help="TorchScript displacement regressor (vision residual)")
    p.add_argument("--gate", default="cum", choices=["cum", "jump"],
                   help="privileged teacher gate: cumulative displacement since reset, or a sudden per-query jump")
    p.add_argument("--step_tau", type=float, default=None)
    p.add_argument("--disturb_trigger", default="approach")
    p.add_argument("--disturb_mag", default="0.04,0.06")
    p.add_argument("--disturb_window", default="40,160")
    p.add_argument("--drift_steps", type=int, default=60)
    p.add_argument("--push_steps", type=int, default=5)
    p.add_argument("--mock", action="store_true")
    p.add_argument("--init_file", default="pruned", choices=["pruned", "full"],
                   help="initial states: pruned = the benchmark's 50 (default, as before); full = all 100 of <task>.init, "
                        "evaluation only (one-time fresh-state test, experiments/bok/prereg_fresh50.md)")
    a = p.parse_args()
    if a.init_file == "full" and not a.eval:
        raise SystemExit("--init_file full is for evaluation only")
    if a.eval and not 0 <= a.eval_init_lo < a.eval_init_hi <= (100 if a.init_file == "full" else 50):
        raise SystemExit("--eval_init_lo/--eval_init_hi outside the loaded initial states (%s)" % a.init_file)
    base = R.build_parser().parse_args([])                    # every run_libero default (identical behaviour)
    for k, v in vars(a).items():
        setattr(base, k, v)
    base.disturb = "object"
    if a.eval:
        evaluate(base)
    else:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        train(base)


if __name__ == "__main__":
    main()
