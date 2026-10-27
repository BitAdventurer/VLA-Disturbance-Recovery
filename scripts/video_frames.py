#!/usr/bin/env python3
"""video_frames.py --task T --init I --arms base,student --port P --out DIR
Replays the Fig. 4 episodes exactly as qual_frames.py does (same reset alignment, seeds and modules) and also saves
the agent-view and wrist images after every control step (DIR/<arm>_t<T>_e<I>/steps/{agent,wrist}_%04d.png at the
env's 256 px, rotated 180 degrees as in openpi). The experiments code is not modified: run_episode is wrapped at
runtime. Run from the experiments directory with the LIBERO client python and a policy server, as for the
evaluations; then compose_video.py DIR OUT.mp4 [FPS] writes the side-by-side video."""
import argparse
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))      # qual_frames.py sits next to this file
import qual_frames as Q     # noqa: E402

R = Q.R
STEPS = {}
_run_episode = R.run_episode


def run_episode(env, *a, **k):
    out = STEPS["dir"]
    os.makedirs(out, exist_ok=True)
    n = [0]
    step = env.step

    def save(obs):
        for key, name in (("agentview_image", "agent"), ("robot0_eye_in_hand_image", "wrist")):
            Image.fromarray(np.ascontiguousarray(obs[key][::-1, ::-1])).save(os.path.join(out, "%s_%04d.png" % (name, n[0])))

    def stepped(action):
        res = step(action)
        n[0] += 1
        save(res[0])
        return res

    env.step = stepped
    try:
        return _run_episode(env, *a, **k)
    finally:
        env.step = step


R.run_episode = run_episode

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--task", type=int, required=True)
    p.add_argument("--init", type=int, required=True)
    p.add_argument("--arms", default="base,student")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    for arm in args.arms.split(","):
        d = os.path.join(args.out, "%s_t%d_e%d" % (arm, args.task, args.init))
        STEPS["dir"] = os.path.join(d, "steps")
        Q.run(arm, args.task, args.init, d, args.port)
