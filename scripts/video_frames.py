#!/usr/bin/env python3
"""video_frames.py --task T --init I --arms base,replan,student [--disturb object|none] [--mag 0.04,0.06] --port P --out DIR
Replays single evaluation episodes with qual_frames.py's reset alignment, seeds and modules and saves the agent-view
and wrist images after every control step (DIR/<arm>_t<T>_e<I>/steps/{agent,wrist}_%04d.png at the env's 256 px,
rotated 180 degrees as in openpi), plus meta.json. Arms: base = the policy alone, (n, h) = (2, 10); replan = the
learned detector replans at each detection (react_step), no residual; student = replan + the released residual
student. --disturb none gives undisturbed episodes; --mag sets the displacement range (0.08,0.12 for 8-12 cm).
--log_state also writes steps/states.jsonl (per step: end-effector position, gripper joints and the positions of the
movable objects) and steps/camera.json (the agent view's world-to-pixel matrix), and adds to meta.json the learned
detector's per-step scores (replan and student arms) and the student's residual and displacement estimate at every
query; compose_traj.py draws them.
The experiments code is not modified: run_episode is wrapped at runtime. Run from the experiments directory with the
LIBERO client python and a policy server, as for the evaluations; then compose_video.py writes the side-by-side
video."""
import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))      # qual_frames.py sits next to this file
import qual_frames as Q     # noqa: E402

R, RR = Q.R, Q.RR
STEPS = {}
_run_episode = R.run_episode


def run_episode(env, *a, **k):
    out = STEPS["dir"]
    os.makedirs(out, exist_ok=True)
    n = [0]
    step = env.step
    states = [] if STEPS.get("log_state") else None

    def save(obs):
        for key, name in (("agentview_image", "agent"), ("robot0_eye_in_hand_image", "wrist")):
            Image.fromarray(np.ascontiguousarray(obs[key][::-1, ::-1])).save(os.path.join(out, "%s_%04d.png" % (name, n[0])))
        if states is not None:           # simulator state is only read, so the episode is unchanged
            STEPS["shape"] = obs["agentview_image"].shape[:2]
            objs = {name: R.object_pos(env, j) for name, j in R.movable_objects(env)}
            states.append({"t": n[0], "eef": np.asarray(obs["robot0_eef_pos"], float).tolist(),
                           "gripper": np.asarray(obs["robot0_gripper_qpos"], float).tolist(),
                           "objects": {k: v.tolist() for k, v in objs.items() if v is not None}})

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
        if states:
            from robosuite.utils.camera_utils import get_camera_transform_matrix
            with open(os.path.join(out, "states.jsonl"), "w") as f:
                for s in states:
                    f.write(json.dumps(s) + "\n")
            h, w = STEPS["shape"]
            m = get_camera_transform_matrix(env.env.sim, "agentview", h, w)
            json.dump({"camera": "agentview", "height": int(h), "width": int(w), "world_to_pixel": m.tolist(),
                       "note": "pixel coordinates of the env image, before the 180-degree rotation of the saved frames"},
                      open(os.path.join(out, "camera.json"), "w"))


R.run_episode = run_episode


def run(arm, tid, init, out, port, disturb, mag, log_state=False):
    """qual_frames.run with a detector-only arm and a configurable disturbance."""
    a = Q.make_args(port)
    a.disturb, a.disturb_mag = disturb, mag
    a.log_step_change = int(log_state)              # only logs: detector scores per step
    if arm in ("replan", "student"):
        for k, v in Q.DET.items():
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
    qlog = []
    if agent is not None and log_state:            # residual and displacement estimate at every query
        at_query = agent.at_query

        def logged(env, obs, chunk, info):
            r = at_query(env, obs, chunk, info)
            qlog.append({"t": info.get("t"), "delta": np.asarray(agent.D[-1], float).tolist() if agent.D else None,
                         "d_hat": None if agent.d_hat is None else np.asarray(agent.d_hat, float).tolist()})
            return r
        agent.at_query = logged
    sched = R.Schedule(a.schedule, a.n, a.h, react_setting=tuple(int(x) for x in a.react_setting.split(",")))
    os.makedirs(out, exist_ok=True)
    slot.env._init_state = slot.init_states[init % len(slot.init_states)]
    slot.env.seed(R.sim_seed(key))
    slot.resets += 1
    STEPS["dir"] = os.path.join(out, "steps")
    STEPS["log_state"] = log_state
    res = R.run_episode(slot.env, policy, slot.description, sched, None, a, tools, out, episode_key=key, residual=agent)
    meta = {k: v for k, v in res.items() if k != "obs"}
    if qlog:
        meta["residual_log"] = qlog
    meta.update(task=slot.description, task_id=tid, init=init, arm=arm, disturb=disturb, mag=mag)
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"),
              default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print("%s task %d init %d: success %s len %d detections %s" % (arm, tid, init, res.get("success"),
          res.get("episode_length"), res.get("detections")), flush=True)
    slot.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--task", type=int, required=True)
    p.add_argument("--init", type=int, required=True)
    p.add_argument("--arms", default="base,student")
    p.add_argument("--disturb", default="object", choices=["object", "none"])
    p.add_argument("--mag", default="0.04,0.06")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--out", required=True)
    p.add_argument("--log_state", action="store_true")
    args = p.parse_args()
    for arm in args.arms.split(","):
        run(arm, args.task, args.init, os.path.join(args.out, "%s_t%d_e%d" % (arm, args.task, args.init)),
            args.port, args.disturb, args.mag, args.log_state)
