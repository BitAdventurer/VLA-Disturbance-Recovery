#!/usr/bin/env python3
"""qual_frames.py --task T --init I --arms base,student --port 8000 --out DIR
Replays single disturbed episodes with the reset-count alignment of the evaluations (identical episodes to the
result files) and saves every agent-view frame (frame_%04d.png), meta.json (result, calls, residuals) and a
montage.png per arm.  Run from the experiments directory with the LIBERO client python."""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.expanduser(os.environ.get("VLA_EXPERIMENTS", "~/disk2/vla-inference-scheduling/experiments")))
import run_libero as R      # noqa: E402
import residual_rl as RR    # noqa: E402

DET = dict(schedule="react_step", detector="bok/detector_v1_t24.pt", step_tau=1.628, react_setting="2,10")


def make_args(port):
    a = R.build_parser().parse_args(["--suite", "libero_10", "--n", "2", "--h", "10", "--disturb", "object",
                                     "--disturb_trigger", "approach", "--disturb_mag", "0.04,0.06",
                                     "--disturb_window", "40,160"])
    a.port, a.dispreg, a.gate = port, "bok/dispreg_v2_t24.pt", "jump"
    return a


def montage(out, meta, every=10):
    from PIL import Image, ImageDraw
    files = sorted(f for f in os.listdir(out) if f.startswith("frame_"))
    ts = [int(f[6:10]) for f in files]
    pick = [f for f, t in zip(files, ts) if t % every == 0]
    cols = 10
    rows = (len(pick) + cols - 1) // cols
    im0 = Image.open(os.path.join(out, pick[0]))
    w, h = im0.size
    sheet = Image.new("RGB", (cols * w, rows * h), "white")
    dr = ImageDraw.Draw(sheet)
    for i, f in enumerate(pick):
        im = Image.open(os.path.join(out, f))
        x, y = (i % cols) * w, (i // cols) * h
        sheet.paste(im, (x, y))
        dr.text((x + 4, y + 4), f[6:10], fill=(255, 255, 0))
    sheet.save(os.path.join(out, "montage.png"))


def run(arm, tid, init, out, port):
    a = make_args(port)
    if arm == "student":
        for k, v in DET.items():
            setattr(a, k, v)
    policy, tools, get_env, _ = R.make_world(a)
    slot = R.EnvSlot(get_env, tid)
    slot.resets = 0
    while slot.resets < init:                      # throwaway resets: same state as the sequential/evaluation runs
        slot.env.reset()
        slot.resets += 1
    key = (R.EVAL_STREAM, 0, tid, init)
    agent = None
    if arm == "student":
        st = torch.load("ckpt/student_v2.pt", map_location="cpu")
        net = RR.StudentNet()
        net.load_state_dict(st["model"])
        agent = RR.StudentAgent(net, None, tid, a.dispreg, beta=0.0, window=st.get("window", 6),
                                seed=R.stream_seed(10, *key), gate="jump", no_dhat=bool(st.get("no_dhat", False)))
    sched = R.Schedule(a.schedule, a.n, a.h, react_setting=tuple(int(x) for x in a.react_setting.split(",")))
    os.makedirs(out, exist_ok=True)
    slot.env._init_state = slot.init_states[init % len(slot.init_states)]
    slot.env.seed(R.sim_seed(key))
    slot.resets += 1
    res = R.run_episode(slot.env, policy, slot.description, sched, None, a, tools, out, episode_key=key, residual=agent)
    meta = {k: v for k, v in res.items() if k != "obs"}
    meta["task"] = slot.description
    meta["D"] = np.asarray(agent.D).tolist() if agent is not None else None
    meta["d_hat"] = [float(x) for x in agent.d_hat] if (agent is not None and agent.d_hat is not None) else None
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"),
              default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    d = res.get("disturbance") or {}
    print("%s task %d init %d: success %s len %d applied_t %s object %s calls %d" % (
        arm, tid, init, res.get("success"), res.get("episode_length"), d.get("applied_t"), d.get("object"),
        len(res.get("calls", []))), flush=True)
    montage(out, meta)
    slot.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--task", type=int, required=True)
    p.add_argument("--init", type=int, required=True)
    p.add_argument("--arms", default="base,student")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--out", default="results_res/qual")
    args = p.parse_args()
    for arm in args.arms.split(","):
        run(arm, args.task, args.init, os.path.join(args.out, "%s_t%d_e%d" % (arm, args.task, args.init)), args.port)
