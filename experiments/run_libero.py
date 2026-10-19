#!/usr/bin/env python3
"""Rollout runner for the ICEIC paper: static grid, two-phase schedules, latency coupling.

Runs in the LIBERO environment (see openpi/examples/libero). The policy is served by
serve_policy_dynamic.py, which accepts a per-query "num_steps" and "noise_seed".

Examples
  # static grid point, no latency coupling
  python run_libero.py --suite libero_10 --schedule static --n 5 --h 5 --seed 0 --out results/static_n5_h5_s0.jsonl
  # two-phase schedule with the gripper trigger
  python run_libero.py --schedule cheap2exp --trigger gripper --seed 0 --out results/c2e_grip_s0.jsonl
  # closed-loop latency check on the RTX 4060 profile
  python run_libero.py --schedule static --n 10 --h 5 --device rtx4060_profile --seed 0 --out results/lat_4060_n10_h5_s0.jsonl
  # save frames of one episode for Fig. 5
  python run_libero.py --schedule cheap2exp --trigger gripper --tasks 3 --trials 1 --frames rollouts/example
  # pipeline test without LIBERO / GPU
  python run_libero.py --mock --schedule cheap2exp --trigger gripper --trials 3 --out /tmp/mock.jsonl

Reproducibility: every episode has a key (stream, seed, task, initial state). The simulator is
reseeded from it before the episode and every query sends a noise seed derived from it and the
query index, so the randomness of an episode does not depend on run order, resuming, or other
clients (policy servers give bit-identical outputs). One environment instance serves all episodes
of a task, and MuJoCo state it carries between episodes can still change an episode slightly when
the episodes before it differ (e.g. after resuming); a run with a fixed episode order is exactly
reproducible. Output goes to <out>.part (one line per episode, flushed) and is renamed to <out>
once all episodes are done; rerunning resumes a .part file.
"""
import argparse
import hashlib
import json
import os
import pathlib
import time

import numpy as np

from scheduling import H, Schedule, cost_function, delay, gripper_toggle, mean_speed
from selectors_bok import select
from sched_features import features, quat2axisangle as _quat2axisangle

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]     # openpi settle action (gripper OPEN: never use it to stall)
LIBERO_ENV_RESOLUTION = 256
MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300,
             "libero_10": 520, "libero_90": 400}
RECV_TIMEOUT_S = 120.0                        # a query takes < 0.1 s; anything longer means a hung server
EVAL_STREAM = 0                               # episode-key stream of baselines and evaluations


# ----------------------------------------------------------------------------- seeds
def stream_seed(*key):
    """Deterministic 32-bit seed from a tuple of non-negative integers."""
    return int(np.random.SeedSequence([int(x) for x in key]).generate_state(1)[0])


def noise_seed(episode_key, query_idx):
    return stream_seed(1, *episode_key, query_idx)


def sim_seed(episode_key):
    return stream_seed(2, *episode_key)


CAND_STREAM, SELECT_STREAM = 6, 7      # streams 3 (scheduler sampling), 4 (disturbance) and 5 are taken
CONT_SALT_BASE = 1000                  # noise salts >= 1000 are reserved for branch continuations


def cand_seed(salt, episode_key, query_idx, k):
    """Noise seed of candidate k at a query. Candidate 0 is the ordinary query noise (so K=1 reproduces every
    earlier run); candidate k >= 1 does not depend on K, so the K=4 set is a prefix of the K=8 set."""
    if k == 0:
        return noise_seed(episode_key, query_idx) if not salt else stream_seed(1, salt, *episode_key, query_idx)
    return stream_seed(CAND_STREAM, salt, *episode_key, query_idx, k)


# ----------------------------------------------------------------------------- policy client
class ServerError(RuntimeError):
    """The policy server answered with a traceback instead of actions."""


class PolicyClient:
    """openpi's websocket client with a receive timeout and reconnection (servers may restart)."""

    def __init__(self, host, port, timeout_s=RECV_TIMEOUT_S):
        from openpi_client import websocket_client_policy
        self._make = lambda: websocket_client_policy.WebsocketClientPolicy(host, port)
        self._timeout = timeout_s
        self._p = self._make()

    def reconnect(self):
        try:
            self._p._ws.close()
        except Exception:
            pass
        self._p = self._make()          # blocks until the server accepts connections again

    def infer(self, obs):
        from openpi_client import msgpack_numpy
        p = self._p
        if not (hasattr(p, "_ws") and hasattr(p, "_packer")):   # other openpi-client versions
            return p.infer(obs)
        p._ws.send(p._packer.pack(obs))
        resp = p._ws.recv(timeout=self._timeout)
        if isinstance(resp, str):
            raise ServerError("error in inference server:\n" + resp)
        return msgpack_numpy.unpackb(resp)


def is_connection_error(e):
    return (isinstance(e, (ConnectionError, TimeoutError, EOFError, ServerError))
            or type(e).__name__.startswith(("ConnectionClosed", "InvalidMessage", "InvalidStatus")))


def failed_episode(err):
    """Record of an episode that raised twice; counted as a failure, as in openpi's LIBERO script."""
    return {"success": False, "episode_length": 0, "n_calls": 0, "total_nfe": 0, "stall_steps": 0,
            "initial_wait": 0, "cost_ms": 0.0, "calls": [], "error": repr(err)}


MAX_CONSECUTIVE_FAILED = 3           # more failed episodes in a row than this means a systematic error
_consecutive_failed = [0]


def run_safely(run, rebuild_env, policy, label, max_reconnects=5):
    """Run one episode. Connection problems: reconnect and rerun (the episode is determined by its
    key). Environment errors: rebuild the environment and rerun once, then record a failure. Several
    failed episodes in a row abort the run: that is a bug, not an unstable simulation."""
    reconnects, env_errors = 0, 0
    while True:
        try:
            res = run()
            _consecutive_failed[0] = 0
            return res
        except KeyboardInterrupt:
            raise
        except Exception as e:  # noqa: BLE001
            if is_connection_error(e):
                reconnects += 1
                if reconnects > max_reconnects or not hasattr(policy, "reconnect"):
                    raise
                print("%s: connection problem (%r); reconnecting" % (label, e), flush=True)
                time.sleep(5)
                policy.reconnect()
                continue
            env_errors += 1
            print("%s: episode error (%r); %s" % (label, e, "rebuilding env and retrying" if env_errors == 1
                                                   else "recorded as failure"), flush=True)
            rebuild_env()
            if env_errors > 1:
                _consecutive_failed[0] += 1
                if _consecutive_failed[0] >= MAX_CONSECUTIVE_FAILED:
                    raise RuntimeError("%d episodes in a row failed (last: %r); aborting" % (
                        _consecutive_failed[0], e)) from e
                return failed_episode(e)


# ----------------------------------------------------------------------------- output
class EpisodeLog:
    """One JSON line per episode in <out>.part, renamed to <out> when complete (finish()).
    An existing .part is resumed: its complete lines are kept and those episodes are skipped."""

    def __init__(self, out, key_fields=("task_id", "init_state_id")):
        self.out, self.part, self.key_fields = out, out + ".part", key_fields
        self.done, keep = set(), []
        if os.path.exists(self.part):
            for line in open(self.part):
                if not line.endswith("\n"):
                    break                                   # truncated tail of a killed run
                try:
                    rec = json.loads(line)
                except ValueError:
                    break
                keep.append(line)
                self.done.add(self.key(rec))
            print("resuming %s: %d episodes already done" % (self.part, len(keep)), flush=True)
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        with open(self.part, "w") as f:
            f.writelines(keep)
        self.f = open(self.part, "a")

    def key(self, rec):
        return tuple(rec[k] for k in self.key_fields)

    def write(self, rec):
        self.f.write(json.dumps(rec) + "\n")
        self.f.flush()
        self.done.add(self.key(rec))

    def finish(self):
        self.f.close()
        os.replace(self.part, self.out)


def open_log(out):
    """EpisodeLog for --out, or None. Exits if <out> is already complete."""
    if not out:
        return None
    if os.path.exists(out):
        raise SystemExit("%s is complete; delete it to rerun" % out)
    return EpisodeLog(out)


# ----------------------------------------------------------------------------- policy I/O
def make_element(obs, task_description, resize_size, image_tools, quat2axisangle):
    """Observation dict in the openpi LIBERO format (180-degree rotation as in openpi)."""
    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, resize_size, resize_size))
    wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, resize_size, resize_size))
    state = np.concatenate((obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]),
                            obs["robot0_gripper_qpos"]))
    return {"observation/image": img, "observation/wrist_image": wrist,
            "observation/state": state, "prompt": str(task_description)}, img


class _MockImageTools:
    @staticmethod
    def resize_with_pad(img, h, w):
        return img[:h, :w]

    @staticmethod
    def convert_to_uint8(img):
        return np.asarray(img).astype(np.uint8)


# ----------------------------------------------------------------------------- one episode
def disturbance_plan(args, episode_key):
    """When and how this episode is disturbed (None if --disturb none); drawn from the episode key."""
    kind = getattr(args, "disturb", "none")
    if kind == "none" or episode_key is None:
        return None
    rng = np.random.default_rng(stream_seed(4, *episode_key))
    lo, hi = [int(x) for x in args.disturb_window.split(",")]
    ang = float(rng.uniform(0, 2 * np.pi))
    return {"kind": kind, "t": args.num_steps_wait + int(rng.integers(lo, hi + 1)),
            "dir": [float(np.cos(ang)), float(np.sin(ang))],
            "mag": float(rng.uniform(*[float(x) for x in args.disturb_mag.split(",")]))}


def movable_objects(env):
    """(name, joint) of the objects of interest that are free bodies (fixtures have no free joint)."""
    inner = getattr(env, "env", None)
    objs = getattr(inner, "objects_dict", None) or {}
    out = []
    for name in getattr(inner, "obj_of_interest", []) or []:
        obj = objs.get(name)
        if obj is not None and getattr(obj, "joints", None):
            out.append((name, obj.joints[0]))
    return out


def approaching(env, eef, last_gripper, radius=0.12):
    """True if the open gripper is within `radius` of a movable object of interest it does not hold."""
    sim = getattr(getattr(env, "env", None), "sim", None)
    if sim is None or last_gripper > 0:
        return False
    for _, j in movable_objects(env):
        q = np.asarray(sim.data.get_joint_qpos(j), dtype=float)
        dist = np.linalg.norm(q[:3] - np.asarray(eef))
        if q.shape[0] >= 7 and 0.06 <= dist < radius:
            return True
    return False


def nearest_free_object(env, eef):
    """(name, joint) of the movable object of interest closest to the gripper but not held by it, or None."""
    sim = getattr(getattr(env, "env", None), "sim", None)
    if sim is None:
        return None
    cands = []
    for name, j in movable_objects(env):
        q = np.array(sim.data.get_joint_qpos(j), dtype=float)
        dist = np.linalg.norm(q[:3] - np.asarray(eef))
        if q.shape[0] >= 7 and dist >= 0.06:          # a free body that the gripper does not hold
            cands.append((dist, name, j))
    if not cands:
        return None
    _, name, j = min(cands, key=lambda c: c[0])
    return name, j


def shift_object(env, joint, dx, dy):
    """Teleport a free body horizontally by (dx, dy) metres and stop it."""
    sim = env.env.sim
    q = np.array(sim.data.get_joint_qpos(joint), dtype=float)
    q[0] += dx
    q[1] += dy
    sim.data.set_joint_qpos(joint, q)
    sim.data.set_joint_qvel(joint, np.zeros(6))
    sim.forward()


def apply_object_disturbance(env, plan, eef):
    """Move the movable object of interest closest to the gripper (but not held by it) by plan['mag']
    metres in the horizontal direction plan['dir']. Returns (object name, joint), or None."""
    pick = nearest_free_object(env, eef)
    if pick is None:
        return None
    name, j = pick
    shift_object(env, j, plan["mag"] * plan["dir"][0], plan["mag"] * plan["dir"][1])
    return name, j


def thumb128(img):
    """128x128 RGB (2x2 mean pooling of the 256x256 render), uint8."""
    x = np.asarray(img, dtype=np.float32)
    h, w = x.shape[0] // 2 * 2, x.shape[1] // 2 * 2
    return x[:h, :w].reshape(h // 2, 2, w // 2, 2, -1).mean((1, 3)).astype(np.uint8)


def camera_digest(obs):
    """32x32 and 64x64 grey thumbnails of the two cameras (cheap change detection)."""
    out = {}
    for key, name in (("agentview_image", "img"), ("robot0_eye_in_hand_image", "wrist")):
        x = np.asarray(obs[key], dtype=np.float32).mean(-1)
        for r, suffix in ((32, ""), (64, "64")):
            hh, ww = x.shape[0] // r * r, x.shape[1] // r * r
            out[name + suffix] = x[:hh, :ww].reshape(r, hh // r, r, ww // r).mean((1, 3))
    return out


def camera_change(a, b):
    """Per camera (0..1): mean absolute grey-level change of the 32x32 thumbnail ('img', 'wrist'), and the
    99th percentile and maximum block change of the 64x64 thumbnail ('img_p99', 'img_max', ...)."""
    out = {}
    for k in ("img", "wrist"):
        out[k] = float(np.abs(a[k] - b[k]).mean() / 255.0)
        if k + "64" in a and k + "64" in b:
            d = np.abs(a[k + "64"] - b[k + "64"]) / 255.0
            out[k + "_p99"], out[k + "_max"] = float(np.percentile(d, 99)), float(d.max())
    return out


_DETECTOR = {}


def detector_score(path, prev, cur, act, deef):
    """Learned disturbance detector (train_detector.py, TorchScript): logit for the frame pair prev -> cur."""
    import torch
    if path not in _DETECTOR:
        torch.set_num_threads(2)
        _DETECTOR[path] = torch.jit.load(path).eval()
    f = lambda d, k: np.clip(d[k], 0, 255).astype(np.uint8).astype(np.float32) / 255.0
    i0, i1, w0, w1 = f(prev, "img64"), f(cur, "img64"), f(prev, "wrist64"), f(cur, "wrist64")
    x = np.stack([i0, i1, i1 - i0, w0, w1, w1 - w0])[None].astype(np.float32)
    v = np.concatenate([np.asarray(act, np.float32), np.asarray(deef, np.float32) * 50.0])[None].astype(np.float32)
    with torch.no_grad():
        return float(_DETECTOR[path](torch.from_numpy(x), torch.from_numpy(v))[0])


def object_pos(env, joint):
    sim = getattr(getattr(env, "env", None), "sim", None)
    if sim is None or joint is None:
        return None
    return np.array(sim.data.get_joint_qpos(joint), dtype=float)[:3]


class BestOfK:
    """Candidate generation and selection settings of an episode (all off by default = today's behaviour).

    num_candidates K, selector name, window ('all', 'post:W' = the W queries from the first query after the
    disturbance, 'post:all'), cand_mode ('seq': K ordinary requests; 'batch': one best-of-K request to
    serve_policy_k.py). Branch runs (branch_r R >= 0): exactly one candidate, branch_k, at the R-th query after
    the disturbance; afterwards the noise salt becomes CONT_SALT_BASE + cont_salt (cont_salt 0: unchanged)."""

    def __init__(self, args):
        self.K = int(getattr(args, "num_candidates", 1) or 1)
        self.selector = getattr(args, "selector", "first")
        self.window = getattr(args, "select_window", "post:all")
        self.mode = getattr(args, "cand_mode", "seq")
        self.w_grip = float(getattr(args, "w_grip", 1.0))
        self.branch_r = getattr(args, "branch_r", None)
        self.branch_k = int(getattr(args, "branch_k", 0) or 0)
        self.cont_salt = int(getattr(args, "cont_salt", 0) or 0)
        self.embed = bool(getattr(args, "log_embed", 0))
        self.embed_at = getattr(args, "embed_at", "all") or "all"
        g = getattr(args, "aim_gain", None)
        self.aim_gain = np.array(json.load(open(g)), dtype=float) if g else None
        if self.branch_r is not None and self.K > 1:
            raise ValueError("branch runs execute one forced candidate; use --num_candidates 1")
        if self.branch_r is not None and self.branch_r < 0:
            raise ValueError("--branch_r must be >= 0 (queries after the disturbance)")
        if not (self.window in ("all", "post:all") or (self.window.startswith("post:") and
                                                        self.window[5:].isdigit())):
            raise ValueError("bad --select_window %r" % self.window)
        if self.K > 1 or self.branch_r is not None or self.embed:
            self.active = True
        else:
            self.active = False

    def in_window(self, jp):
        if self.K <= 1:
            return False
        if self.window == "all":
            return True
        if jp is None or jp < 0:
            return False
        w = self.window.split(":", 1)[1]
        return w == "all" or jp < int(w)

    def salt_at(self, noise_salt, jp):
        if self.branch_r is not None and jp is not None and jp > self.branch_r and self.cont_salt:
            return CONT_SALT_BASE + self.cont_salt
        return noise_salt


def run_episode(env, policy, task_description, schedule, cost, args, tools, frames_dir=None,
                cost_ms=None, k=1.0, episode_key=None, residual=None):
    """cost: latency function of the emulated device (None = no latency coupling).
    cost_ms: per-query cost model used for budget accounting (defaults to cost, else nfe).
    episode_key: (stream, seed, task, init state); sets the policy noise of every query and the
    disturbance (--disturb object|push|drift): one per episode, at a key-dependent step.
    residual: optional object with reset(env) and at_query(env, obs, chunk, info) -> 7-vector added to every
    action of the window that follows the query (residual RL on top of the base policy)."""
    image_tools, quat2axisangle = tools
    cost_ms = cost_ms or cost or (lambda n: float(n))          # fall back to counting steps
    spent = 0.0
    max_steps = MAX_STEPS[args.suite]
    t, stall_steps, initial_wait, total_nfe = 0, 0, 0, 0
    calls, done, last_gripper, obs = [], False, -1.0, None
    n_queries = 0
    plan = disturbance_plan(args, episode_key)
    dist = {"applied": False, "at": None, "object": None, "push_left": 0, "fresh": False,
            "drift_joint": None, "drift_left": 0, "drifted": 0, "joint": None, "z0": None, "lifted": False}
    bok = getattr(args, "_bok", None) or BestOfK(args)
    noise_salt = int(getattr(args, "noise_salt", 0) or 0)
    q_dist = [None]                       # index of the first query issued after the disturbance
    qinfo = {}                            # best-of-K fields of the last query (merged into its call record)
    cand_log = []                         # per selection query: candidates, scores, choice (--log_cands)
    exec_log = [] if getattr(args, "log_exec", 0) else None
    surp = {"prev": None, "detected_at": None, "value": None}   # camera-change detector (react_detect)
    surp_on = getattr(args, "log_surprise", 0) or getattr(schedule, "name", "") == "react_detect"
    frames = {"t": [], "img": [], "wrist": [], "eef": [], "act": []} if getattr(args, "log_frames", None) else None
    step_on = (getattr(args, "log_step_change", 0) or getattr(schedule, "name", "") == "react_step"
               or frames is not None)
    stepd = {"prev": None, "log": [], "fresh": False, "last": -10 ** 6, "dets": [], "prev_eef": None,
             "scores": []}                                                      # per-step change detector
    last_exec = [None]
    total_samples = [0]
    res_delta = [None]                    # residual added to every executed action of the current window
    want_ev = residual is not None and getattr(residual, "wants_event", False)   # vision residual: event frames
    evf = {} if getattr(args, "log_event_frames", None) else None   # frames right before/after the displacement

    obs = env.reset()
    obs = env.set_init_state(env_init_state) if (env_init_state := getattr(env, "_init_state", None)) is not None else obs
    for _ in range(args.num_steps_wait):          # let objects settle, as in openpi
        obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
        t += 1
    limit = max_steps + args.num_steps_wait
    if residual is not None:
        residual.reset(env)                       # after settling: reference object positions

    def query(n):
        nonlocal total_nfe, spent, n_queries
        q = n_queries
        if dist["applied"] and q_dist[0] is None:
            q_dist[0] = q
        jp = None if q_dist[0] is None else q - q_dist[0]
        element, img = make_element(obs, task_description, args.resize_size, image_tools, quat2axisangle)
        element["num_steps"] = int(n)
        if surp_on:
            surp["prev"] = camera_digest(obs)
        K, how = 1, None
        if episode_key is not None:
            salt = bok.salt_at(noise_salt, jp)
            if bok.branch_r is not None and jp == bok.branch_r:
                seeds, how = [cand_seed(noise_salt, episode_key, q, bok.branch_k)], "branch"
            elif bok.in_window(jp):
                K, how = bok.K, "select"
                seeds = [cand_seed(salt, episode_key, q, kk) for kk in range(K)]
            else:
                seeds = [cand_seed(salt, episode_key, q, 0)]
            element["noise_seed"] = seeds[0]
        n_queries += 1
        t0 = time.perf_counter()
        emb = None
        want_emb = bok.embed and (bok.embed_at == "all" or how == bok.embed_at)
        if K == 1 and not want_emb:
            out = policy.infer(element)
            ms = float(out.get("policy_timing", {}).get("infer_ms", (time.perf_counter() - t0) * 1000.0))
            cands = np.asarray(out["actions"])[None, :H]
            spent += float(cost_ms(n))
        elif bok.mode == "seq" and (K > 1 or bok.embed_at != "all"):
            cands, ms = [], 0.0
            for s_k in seeds:
                e = dict(element)
                e["noise_seed"] = s_k
                out = policy.infer(e)
                ms += float(out.get("policy_timing", {}).get("infer_ms", 0.0))
                cands.append(np.asarray(out["actions"])[:H])
            cands = np.stack(cands)
            spent += K * float(cost_ms(n))                  # K separate queries (batched mode is cheaper)
            if want_emb:                                    # data collection: one extra prefix-only pass
                e = dict(element)
                e.pop("noise_seed", None)
                e.update(num_samples=1, noise_seeds=[int(seeds[0])], exact=True, return_embedding=True)
                oe = policy.infer(e)
                if "emb" not in oe:
                    raise ServerError("server returned no embedding (serve_policy_k.py needed)")
                emb = (np.asarray(oe["emb"], dtype=np.float16), np.asarray(oe["emb_v"], dtype=np.float16))
        else:                                               # one best-of-K request (serve_policy_k.py)
            e = dict(element)
            e.pop("noise_seed", None)
            e.update(num_samples=K, noise_seeds=[int(x) for x in seeds], return_embedding=bok.embed,
                     exact=(K == 1))
            out = policy.infer(e)
            ms = float(out.get("policy_timing", {}).get("infer_ms", (time.perf_counter() - t0) * 1000.0))
            if "actions_k" not in out or np.asarray(out["actions_k"]).shape[0] != K:
                raise ServerError("server did not return %d candidates (serve_policy_k.py needed)" % K)
            cands = np.asarray(out["actions_k"])[:, :H]
            if bok.embed:
                emb = (np.asarray(out["emb"], dtype=np.float16), np.asarray(out["emb_v"], dtype=np.float16))
            spent += float(cost_ms(n)) + (K - 1) * float(getattr(args, "k_marginal_ms", 1.0) or 0.0)
        total_samples[0] += K
        idx, scores = 0, None
        if how == "select":
            sel_rng = np.random.default_rng(stream_seed(SELECT_STREAM, bok.salt_at(noise_salt, jp),
                                                        *episode_key, q))
            p_obj = object_pos(env, dist["joint"])
            sctx = {"rng": sel_rng, "last_action": last_exec[0], "eef": np.asarray(obs["robot0_eef_pos"], float),
                    "p_obj": p_obj, "obj_z0": dist["z0"], "gripper_open": last_gripper <= 0,
                    "obj_lifted": dist["lifted"]}
            if bok.aim_gain is not None:
                sctx["aim_gain"] = bok.aim_gain
            idx, scores = select(bok.selector, cands, sctx, w_grip=bok.w_grip)
        chunk = cands[idx]
        if residual is not None:
            res_delta[0] = np.asarray(residual.at_query(env, obs, chunk, {"t": t, "limit": limit, "q": q,
                                                                       "last_gripper": last_gripper,
                                                                       "event": stepd.pop("event", None)}), dtype=float)
        total_nfe += int(n) * K
        qinfo.clear()
        if getattr(args, "log_chunks", 0):
            qinfo["chunk"] = np.round(np.asarray(chunk, dtype=float), 4).tolist()
        if bok.active:                     # ch: hash of the executed chunk (replay checks of branch runs)
            qinfo.update(K=K, sel=int(idx), jp=jp, how=how,
                         ch=hashlib.sha1(np.ascontiguousarray(chunk, dtype=np.float64).tobytes()).hexdigest()[:12])
        if getattr(args, "log_cands", None) and (how in ("select", "branch") or emb is not None):
            p_obj = object_pos(env, dist["joint"])
            cand_log.append({"q": q, "t": t, "jp": -999 if jp is None else jp, "cands": cands.astype(np.float32),
                             "hashes": np.array([hashlib.sha1(np.ascontiguousarray(c_, dtype=np.float64).tobytes())
                                                 .hexdigest()[:12] for c_ in cands]),
                             "seeds": np.asarray(seeds, dtype=np.int64), "chosen": idx,
                             "scores": (np.zeros(K) if scores is None else scores).astype(np.float32),
                             "eef": np.asarray(obs["robot0_eef_pos"], np.float32),
                             "p_obj": np.full(3, np.nan, np.float32) if p_obj is None else p_obj.astype(np.float32),
                             "state": np.asarray(element["observation/state"], np.float32),
                             "last_action": (np.full(7, np.nan) if last_exec[0] is None else
                                             np.asarray(last_exec[0])).astype(np.float32),
                             "gripper_open": last_gripper <= 0, "lifted": dist["lifted"],
                             "emb": emb[0] if emb is not None else None, "emb_v": emb[1] if emb is not None else None})
        if frames_dir is not None:
            from PIL import Image
            Image.fromarray(img).save(os.path.join(frames_dir, "frame_%04d.png" % t))
        return chunk, ms

    def hold():
        return [0.0] * 6 + [float(last_gripper)]   # keep pose AND the current gripper command

    def execute(a):
        nonlocal obs, done, t, last_gripper
        a = list(map(float, a))
        if res_delta[0] is not None:                     # residual policy: bounded correction of the action
            a = [float(np.clip(x + dx, -1.0, 1.0)) for x, dx in zip(a, res_delta[0])]
        pre128 = None
        if want_ev:                                      # frames before this step (for a detection after it)
            pre128 = (thumb128(obs["agentview_image"]), thumb128(obs["robot0_eye_in_hand_image"]),
                      np.asarray(obs["robot0_eef_pos"], np.float32))
        if (plan is not None and not dist["applied"]
                and (t >= plan["t"] or (args.disturb_trigger == "approach" and t >= args.num_steps_wait + 20
                                        and approaching(env, obs["robot0_eef_pos"], last_gripper)))):
            dist["applied"], dist["at"], dist["fresh"] = True, t, True
            if evf is not None:
                evf.update(before_img=thumb128(obs["agentview_image"]), before_wrist=thumb128(obs["robot0_eye_in_hand_image"]),
                           eef_before=np.asarray(obs["robot0_eef_pos"], np.float32), action=np.asarray(a, np.float32))
            if plan["kind"] == "object":
                pick = apply_object_disturbance(env, plan, obs["robot0_eef_pos"])
                if pick is not None:
                    dist["object"], dist["joint"] = pick
            elif plan["kind"] == "drift":
                pick = nearest_free_object(env, obs["robot0_eef_pos"])
                if pick is not None:
                    dist["object"], dist["drift_joint"] = pick
                    dist["joint"] = dist["drift_joint"]
                    dist["drift_left"] = args.drift_steps
            else:
                dist["push_left"] = args.push_steps
            if dist["joint"] is not None:                # object height at the disturbance (priv_aim)
                dist["z0"] = float(object_pos(env, dist["joint"])[2])
        if dist["drift_left"] > 0:                       # moving object: plan['mag'] metres over drift_steps,
            if a[-1] > 0:                                # until the gripper closes
                dist["drift_left"] = 0
            else:
                step = plan["mag"] / args.drift_steps
                shift_object(env, dist["drift_joint"], step * plan["dir"][0], step * plan["dir"][1])
                dist["drift_left"] -= 1
                dist["drifted"] += 1
        if dist["push_left"] > 0:                        # external push: offsets the commanded translation
            a[0] += plan["mag"] * plan["dir"][0]
            a[1] += plan["mag"] * plan["dir"][1]
            dist["push_left"] -= 1
        if dist["z0"] is not None and not dist["lifted"]:
            p = object_pos(env, dist["joint"])
            dist["lifted"] = bool(p is not None and p[2] - dist["z0"] > 0.03)
        if exec_log is not None:                         # t, action (7), eef (3), displaced object (3, NaN before)
            po = object_pos(env, dist["joint"]) if dist["joint"] is not None else None
            exec_log.append([t] + a + [float(x) for x in obs["robot0_eef_pos"]] +
                            ([float(x) for x in po] if po is not None else [float("nan")] * 3))
        obs, _, done, _ = env.step(a)
        last_gripper = float(a[-1])
        last_exec[0] = a
        t += 1
        if evf is not None and "before_img" in evf and "after_img" not in evf:
            evf.update(after_img=thumb128(obs["agentview_image"]), after_wrist=thumb128(obs["robot0_eye_in_hand_image"]),
                       eef_after=np.asarray(obs["robot0_eef_pos"], np.float32),
                       d=np.asarray(plan["dir"], np.float32) * plan["mag"], t=np.asarray(t),
                       moved=np.asarray(dist["joint"] is not None))
        if getattr(args, "stop_after_event", 0) and dist["at"] is not None and t >= dist["at"] + args.stop_after_event:
            done, dist["stopped"] = True, True           # data collection only: end the episode early (not a success)
        if step_on:                                      # change between consecutive frames (after this step)
            d = camera_digest(obs)
            if frames is not None:                       # detector dataset: 64x64 grey frames after every step
                frames["t"].append(t)
                frames["img"].append(np.clip(d["img64"], 0, 255).astype(np.uint8))
                frames["wrist"].append(np.clip(d["wrist64"], 0, 255).astype(np.uint8))
                frames["eef"].append(np.asarray(obs["robot0_eef_pos"], np.float32))
                frames["act"].append(np.asarray(a, np.float32))
            eef_now = np.asarray(obs["robot0_eef_pos"], np.float32)
            if stepd["prev"] is not None:
                ch = camera_change(stepd["prev"], d)
                if getattr(args, "log_step_change", 0):
                    stepd["log"].append([t, round(ch["img"], 5), round(ch["wrist"], 5),
                                         round(ch["img_p99"], 5), round(ch["img_max"], 5),
                                         round(ch["wrist_p99"], 5), round(ch["wrist_max"], 5)])
                tau = getattr(args, "step_tau", None)
                sig = ch[getattr(args, "surprise_signal", "img")]
                if getattr(args, "detector", None):      # learned detector replaces the camera-change statistic
                    sig = detector_score(args.detector, stepd["prev"], d, a, eef_now - stepd["prev_eef"])
                    stepd["scores"].append([t, round(sig, 3)])
                if (tau is not None and sig > tau
                        and t - stepd["last"] >= int(getattr(args, "step_cooldown", 20))):
                    stepd["last"], stepd["fresh"] = t, True
                    stepd["dets"].append(t)
                    surp["detected_at"] = t
                    if pre128 is not None:               # hand the event frames to the residual at the next query
                        stepd["event"] = {"before_img": pre128[0], "before_wrist": pre128[1],
                                          "after_img": thumb128(obs["agentview_image"]),
                                          "after_wrist": thumb128(obs["robot0_eye_in_hand_image"]),
                                          "action": np.asarray(a, np.float32), "deef": eef_now - pre128[2], "t": t}
            stepd["prev"], stepd["prev_eef"] = d, eef_now

    def interrupted():
        """True once, right after a disturbance (react_now: oracle; react_step: per-step detector), for schedules
        that may cut the window short."""
        if getattr(schedule, "name", "") == "react_step":
            if stepd["fresh"]:
                stepd["fresh"] = False
                return True
            return False
        if dist["fresh"] and getattr(schedule, "interrupt", False):
            dist["fresh"] = False
            return True
        return False

    def ctx():
        if surp_on and surp["prev"] is not None:
            surp["value"] = camera_change(surp["prev"], camera_digest(obs))
            tau = getattr(args, "surprise_tau", None)
            sig = surp["value"][getattr(args, "surprise_signal", "img")]
            window = getattr(schedule, "react_window", 30)
            if (tau is not None and sig > tau and
                    (surp["detected_at"] is None or t >= surp["detected_at"] + window)):
                surp["detected_at"] = t
        return {"obs": obs, "t": t, "limit": limit, "cost_so_far_ms": spent, "calls": calls, "k": k,
                "disturbed_at": dist["at"], "detected_at": surp["detected_at"]}

    prev_action = [None]

    def feat(prev_chunk):
        if not args.log_features:
            return None
        f = features(prev_chunk, last_gripper, obs, t, limit, spent, None, prev_action[0], k)
        return [round(float(v), 4) for v in f]

    # first query: the robot waits d(n0) periods for it (initial_wait, not counted as a stall)
    f0 = feat(None)
    (n, h), crit = schedule.next_setting(None, last_gripper, ctx())
    chunk, ms = query(n)
    calls.append({"t": t, "num_steps": n, "replan_steps": h, "critical": crit,
                  "gripper_toggle": False, "mean_speed": None, "infer_ms": ms, "feat": f0, **qinfo})
    prev_action[0] = (n, h)
    if cost is not None:
        d = delay(cost(n))
        for _ in range(d):
            if t >= limit or done:
                break
            execute(hold())
            initial_wait += 1
    window = list(chunk[:h])

    while not done and t < limit:
        f2 = feat(chunk)
        (n2, h2), crit2 = schedule.next_setting(chunk, last_gripper, ctx())
        rec = {"num_steps": n2, "replan_steps": h2, "critical": crit2,
               "gripper_toggle": gripper_toggle(chunk, last_gripper), "mean_speed": mean_speed(chunk),
               "feat": f2}
        prev_action[0] = (n2, h2)
        if cost is None:                                          # synchronous, inference is free
            for a in window:
                execute(a)
                if done or t >= limit or interrupted():
                    break
            if done or t >= limit:
                break
            if getattr(schedule, "decide_at_query", False) or surp_on:   # re-decide from the state at the query
                cx = ctx()
                if getattr(schedule, "decide_at_query", False):
                    (n2, h2), crit2 = schedule.next_setting(chunk, last_gripper, cx)
                    rec.update(num_steps=n2, replan_steps=h2, critical=crit2)
                    prev_action[0] = (n2, h2)
                if surp_on and surp["value"] is not None:
                    rec["surprise"] = [round(surp["value"]["img"], 5), round(surp["value"]["wrist"], 5)]
                    rec["detected_at"] = surp["detected_at"]
            rec["t"] = t
            chunk, ms = query(n2)
            window = list(chunk[:h2])
        else:
            d2 = delay(cost(n2))
            async_ok = (d2 <= h2 <= H - d2) and len(window) >= d2
            if async_ok:                                          # naive asynchronous protocol
                new_chunk = None
                for i, a in enumerate(window):
                    if i == len(window) - d2:
                        rec["t"] = t
                        new_chunk, ms = query(n2)                  # computed now, "arrives" d2 steps later
                    execute(a)
                    if done or t >= limit:
                        break
                if done or t >= limit or new_chunk is None:
                    if new_chunk is not None:                     # issued and paid for, still in flight
                        rec["infer_ms"] = ms
                        rec.update(qinfo)
                        calls.append(rec)
                    break
                chunk = new_chunk
                window = list(chunk[d2:d2 + h2])                  # first d2 actions are stale
            else:                                                 # synchronous fallback with stalls
                for a in window:
                    execute(a)
                    if done or t >= limit:
                        break
                if done or t >= limit:
                    break
                rec["t"] = t
                chunk, ms = query(n2)
                for _ in range(d2):
                    if done or t >= limit:
                        break
                    execute(hold())
                    stall_steps += 1
                window = list(chunk[:h2])
        rec["infer_ms"] = ms
        rec.update(qinfo)
        calls.append(rec)

    out = {"success": bool(done) and not dist.get("stopped", False), "episode_length": t - args.num_steps_wait,
           "n_calls": len(calls), "total_nfe": total_nfe, "stall_steps": stall_steps,
           "initial_wait": initial_wait, "cost_ms": spent, "calls": calls}
    if bok.active:
        out["bok"] = {"selector": bok.selector, "num_candidates": bok.K, "cand_mode": bok.mode,
                      "select_window": bok.window, "noise_salt": noise_salt, "q_dist": q_dist[0],
                      "branch": None if bok.branch_r is None else {"r": bok.branch_r, "k": bok.branch_k,
                                                                   "c": bok.cont_salt},
                      "total_samples": total_samples[0]}
    if getattr(args, "log_cands", None):
        out["_cand_log"] = cand_log
    if exec_log is not None:
        out["exec"] = exec_log
    if getattr(args, "log_step_change", 0):
        out["step_change"] = stepd["log"]
    if frames is not None:
        out["_frames"] = frames
    if evf is not None and "after_img" in evf:
        out["_event_frames"] = evf
    if getattr(schedule, "name", "") == "react_step":
        out["detections"] = stepd["dets"]
        if getattr(args, "log_step_change", 0) and stepd["scores"]:
            out["detector_scores"] = stepd["scores"]
    if plan is not None:
        out["disturbance"] = {"kind": plan["kind"], "planned_t": plan["t"], "mag": plan["mag"], "dir": plan["dir"],
                              "applied_t": dist["at"], "object": dist["object"], "drifted": dist["drifted"]}
    return out


# ----------------------------------------------------------------------------- environments
def make_world(args):
    """(policy, tools, get_env, task_ids); get_env(task_id) -> (env, instruction, init_states)."""
    if args.mock:
        from mock import MockEnv, MockPolicy
        policy = MockPolicy(seed=args.seed)
        tools = (_MockImageTools, _quat2axisangle)
        task_ids = list(range(10)) if args.tasks == "all" else [int(x) for x in args.tasks.split(",")]

        def get_env(task_id):
            n_states = 100 if getattr(args, "init_file", "pruned") == "full" else 50
            return MockEnv(seed=args.seed * 100 + task_id), "mock task %d put the bowl in the drawer" % task_id, [None] * n_states
        return policy, tools, get_env, task_ids

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from openpi_client import image_tools
    policy = PolicyClient(args.host, args.port)
    tools = (image_tools, _quat2axisangle)
    suite = benchmark.get_benchmark_dict()[args.suite]()
    task_ids = list(range(suite.n_tasks)) if args.tasks == "all" else [int(x) for x in args.tasks.split(",")]

    def get_env(task_id):
        task = suite.get_task(task_id)
        bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=LIBERO_ENV_RESOLUTION,
                                 camera_widths=LIBERO_ENV_RESOLUTION)
        env.seed(args.seed)     # affects object positions even with fixed initial states (openpi note)
        for _ in range(int(getattr(args, "warm_reset", 0) or 0)):   # throwaway resets of a freshly built env
            env.reset()
        return env, task.language, load_init_states(suite, task_id, getattr(args, "init_file", "pruned"))
    return policy, tools, get_env, task_ids


def load_init_states(suite, task_id, which="pruned"):
    """Initial states of one task. 'pruned' (default): <task>.pruned_init, the 50 states the benchmark evaluates, exactly
    as before. 'full': <task>.init, all 100 states; its first 50 must equal .pruned_init. States 50-99 are used only by the
    one-time fresh-initial-state test of 2026-10-19 (experiments/bok/prereg_fresh50.md)."""
    states = suite.get_task_init_states(task_id)
    if which == "pruned":
        return states
    if which != "full":
        raise ValueError("init_file must be 'pruned' or 'full', not %r" % (which,))
    import torch
    from libero.libero import get_libero_path
    task = suite.get_task(task_id)
    name = task.init_states_file
    if not name.endswith(".pruned_init"):
        raise SystemExit("task %d: unexpected init-state file name %r" % (task_id, name))
    full = torch.load(os.path.join(get_libero_path("init_states"), task.problem_folder, name[:-len(".pruned_init")] + ".init"))
    if len(full) != 100 or not np.array_equal(np.asarray(full)[:len(states)], np.asarray(states)):
        raise SystemExit("task %d: .init is not 100 states with .pruned_init as its first 50" % task_id)
    return full


def save_cand_log(path, entries):
    """One npz per episode: per selection query q, t, jp, cands [K,H,7], seeds, chosen, scores, eef, p_obj, state
    (and emb, emb_v when requested). Written atomically."""
    arrs = {}
    for i, e in enumerate(entries):
        for k, v in e.items():
            if v is not None:
                arrs["%d_%s" % (i, k)] = np.asarray(v)
    arrs["n"] = np.asarray(len(entries))
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **arrs)
    os.replace(tmp, path)


class EnvSlot:
    """The environment of one task; rebuilt after an episode error."""

    def __init__(self, get_env, task_id):
        self.get_env, self.task_id = get_env, task_id
        self.env, self.description, self.init_states = get_env(task_id)

    def rebuild(self):
        try:
            self.env.close()
        except Exception:
            pass
        self.env, self.description, self.init_states = self.get_env(self.task_id)

    def close(self):
        try:
            self.env.close()
        except Exception:
            pass


# ----------------------------------------------------------------------------- main
def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", default="libero_10")
    ap.add_argument("--schedule", default="static",
                    choices=["static", "uniform_cheap", "uniform_expensive", "cheap2exp", "exp2cheap",
                             "react_next", "react_now", "react_detect", "react_step"])
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--h", type=int, default=5)
    ap.add_argument("--trigger", default=None, choices=[None, "gripper", "speed"])
    ap.add_argument("--v_min", type=float, default=None)
    ap.add_argument("--device", default="none",
                    choices=["none", "rtx5090", "rtx4090_profile", "rtx4060_profile"])
    ap.add_argument("--latency", default=None, help="latency.json measured on this GPU (for --device rtx5090)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tasks", default="all", help="comma-separated task ids or 'all'")
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--out", default=None, help="output .jsonl (one line per episode)")
    ap.add_argument("--frames", default=None, help="save frames + rollout.json of the first episode here")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--resize_size", type=int, default=224)
    ap.add_argument("--num_steps_wait", type=int, default=10)
    ap.add_argument("--mock", action="store_true", help="use mock env/policy (pipeline test)")
    ap.add_argument("--disturb", default="none", choices=["none", "object", "push", "drift"],
                    help="one disturbance per episode: move an object of interest, push the arm, "
                         "or make an object of interest slide until the gripper closes")
    ap.add_argument("--disturb_window", default="40,160", help="disturbance step after settling, uniform in [a,b]")
    ap.add_argument("--disturb_mag", default="0.04,0.06",
                    help="object: displacement in metres; push: added action units per step (uniform in [a,b])")
    ap.add_argument("--push_steps", type=int, default=5)
    ap.add_argument("--drift_steps", type=int, default=60, help="drift: steps over which the object slides mag metres")
    ap.add_argument("--noise_salt", type=int, default=0,
                    help="nonzero: redraw the policy noise only (same episodes and disturbances), for luck baselines")
    ap.add_argument("--disturb_trigger", default="time", choices=["time", "approach"],
                    help="approach: when the open gripper first comes within 12 cm of a movable object of interest "
                         "(at the latest at the planned step)")
    ap.add_argument("--react_setting", default="2,2", help="react_*: setting used right after the disturbance")
    ap.add_argument("--react_window", type=int, default=30, help="react_*: steps after the disturbance")
    g = ap.add_argument_group("best-of-K selection (all off by default)")
    g.add_argument("--num_candidates", type=int, default=1, help="K candidate chunks per query inside the window")
    g.add_argument("--selector", default="first",
                   choices=["first", "random", "medoid", "antimedoid", "continuity", "discontinuity", "priv_aim"])
    g.add_argument("--select_window", default="post:all",
                   help="'all', 'post:W' (W queries from the first query after the disturbance) or 'post:all'")
    g.add_argument("--cand_mode", default="seq", choices=["seq", "batch"],
                   help="seq: K ordinary requests; batch: one best-of-K request (serve_policy_k.py)")
    g.add_argument("--w_grip", type=float, default=1.0, help="medoid: weight of gripper sign disagreement")
    g.add_argument("--aim_gain", default=None, help="priv_aim: JSON 3x3 eef-displacement gain (fit from --log_exec)")
    g.add_argument("--k_marginal_ms", type=float, default=1.0, help="batch mode: cost of each extra candidate")
    g.add_argument("--branch_r", type=int, default=None,
                   help="branch run: force candidate --branch_k at the R-th query after the disturbance")
    g.add_argument("--branch_k", type=int, default=0)
    g.add_argument("--cont_salt", type=int, default=0, help="branch run: noise salt of the continuation (+1000)")
    g.add_argument("--log_cands", default=None, help="directory for per-episode candidate npz shards")
    g.add_argument("--log_embed", type=int, default=0, help="request prefix embeddings (serve_policy_k.py)")
    g.add_argument("--embed_at", default="all", choices=["all", "select", "branch"],
                   help="with --log_embed: the queries that get an embedding (seq mode: an extra prefix pass)")
    g.add_argument("--log_exec", type=int, default=0, help="log every executed action and eef position")
    g.add_argument("--episodes", default=None, help="JSON list of [task, init] pairs to run (canonical order)")
    g.add_argument("--log_surprise", type=int, default=0, help="log the camera-change signal at every query")
    g.add_argument("--log_chunks", type=int, default=0, help="log the executed chunk of every query")
    g.add_argument("--log_step_change", type=int, default=0, help="log the per-step camera change")
    g.add_argument("--log_frames", default=None, help="directory for per-episode 64x64 grey frame npz (detector data)")
    g.add_argument("--log_event_frames", default=None, help="directory: 128x128 RGB frames right before/after the disturbance")
    g.add_argument("--stop_after_event", type=int, default=0, help="end the episode N steps after the disturbance")
    g.add_argument("--step_tau", type=float, default=None, help="react_step: per-step camera-change threshold")
    g.add_argument("--step_cooldown", type=int, default=20, help="react_step: minimum steps between detections")
    g.add_argument("--detector", default=None, help="react_step: TorchScript detector (train_detector.py); "
                                                    "--step_tau is then a logit threshold")
    g.add_argument("--surprise_signal", default="img",
                   choices=["img", "wrist", "img_p99", "img_max", "wrist_p99", "wrist_max"])
    g.add_argument("--surprise_tau", type=float, default=None, help="react_detect: camera-change threshold")
    g.add_argument("--warm_reset", type=int, default=0, help="throwaway env.reset() calls after building an env")
    g.add_argument("--fresh_env", type=int, default=0,
                   help="rebuild the environment before every episode (exact replay of episode subsets)")
    ap.add_argument("--log_features", type=int, default=1,
                    help="log the scheduler state features of every query (offline dataset); 0 to disable")
    return ap


def main():
    args = build_parser().parse_args()

    lat = None
    if args.latency:
        j = json.load(open(args.latency))
        lat = dict(zip(j["n"], j["ms_mean"]))
    cost = cost_function(None if args.device == "none" else args.device, lat)
    cost_ms = cost_function("rtx5090", lat) if lat else None       # our GPU, for budget accounting
    k = cost(10) / cost_ms(10) if (cost is not None and cost_ms is not None) else 1.0
    schedule = Schedule(args.schedule, args.n, args.h, args.trigger, args.v_min,
                        tuple(int(x) for x in args.react_setting.split(",")), args.react_window)
    static = args.schedule == "static"

    args._bok = BestOfK(args)                 # validates the best-of-K flags before anything is written
    log = open_log(args.out)
    policy, tools, get_env, task_ids = make_world(args)
    only = None
    if args.episodes:
        only = {(int(a), int(b)) for a, b in json.load(open(args.episodes))}
        task_ids = [tid for tid in task_ids if any(a == tid for a, _ in only)]
    if args.log_cands:
        os.makedirs(args.log_cands, exist_ok=True)
    first = True
    for task_id in task_ids:
        slot = EnvSlot(get_env, task_id)
        used = False
        for ep in range(min(args.trials, len(slot.init_states))):
            if only is not None and (task_id, ep) not in only:
                continue
            if log is not None and (task_id, ep) in log.done:
                first = False
                continue
            fresh = {"used": used}
            used = True
            key = (EVAL_STREAM, args.seed, task_id, ep)
            frames_dir = None
            if args.frames and first:
                frames_dir = args.frames
                os.makedirs(frames_dir, exist_ok=True)

            def run():
                if args.fresh_env and fresh["used"]:     # every attempt (also retries) on a fresh env
                    slot.rebuild()
                fresh["used"] = True
                slot.env._init_state = slot.init_states[ep]
                slot.env.seed(sim_seed(key))
                return run_episode(slot.env, policy, slot.description, schedule, cost, args, tools, frames_dir,
                                   cost_ms=cost_ms, k=k, episode_key=key)
            res = run_safely(run, slot.rebuild, policy, "task %d ep %d" % (task_id, ep))
            evr = res.pop("_event_frames", None)
            if args.log_event_frames and evr is not None:
                os.makedirs(args.log_event_frames, exist_ok=True)
                tmp = os.path.join(args.log_event_frames, "t%d_e%d.tmp.npz" % (task_id, ep))
                np.savez_compressed(tmp, **evr)
                os.replace(tmp, os.path.join(args.log_event_frames, "t%d_e%d.npz" % (task_id, ep)))
            fr = res.pop("_frames", None)
            if args.log_frames and fr is not None:            # frames first, then the jsonl line (resume-safe)
                os.makedirs(args.log_frames, exist_ok=True)
                tmp = os.path.join(args.log_frames, "t%d_e%d.tmp.npz" % (task_id, ep))
                np.savez_compressed(tmp, **{k: np.asarray(v) for k, v in fr.items()})
                os.replace(tmp, os.path.join(args.log_frames, "t%d_e%d.npz" % (task_id, ep)))
            cl = res.pop("_cand_log", None)
            if args.log_cands and cl is not None:            # shard first, then the jsonl line (resume-safe)
                save_cand_log(os.path.join(args.log_cands, "t%d_e%d.npz" % (task_id, ep)), cl)
            rec = {"suite": args.suite, "task_id": task_id, "task_name": slot.description,
                   "init_state_id": ep, "seed": args.seed, "schedule_name": args.schedule,
                   "trigger": args.trigger, "num_steps": args.n if static else None,
                   "replan_steps": args.h if static else None,
                   "device": None if args.device == "none" else args.device, **res}
            if log is not None:
                log.write(rec)
            if frames_dir is not None:
                json.dump({"task_name": slot.description, "success": res["success"],
                           "episode_length": res["episode_length"], "schedule_name": args.schedule,
                           "trigger": args.trigger, "num_steps_wait": args.num_steps_wait,
                           "calls": res["calls"]},
                          open(os.path.join(frames_dir, "rollout.json"), "w"))
            first = False
            print("task %d ep %d: success=%s len=%d calls=%d nfe=%d stalls=%d" % (
                task_id, ep, res["success"], res["episode_length"], res["n_calls"],
                res["total_nfe"], res["stall_steps"]), flush=True)
        slot.close()
    if log is not None:
        log.finish()


if __name__ == "__main__":
    main()
