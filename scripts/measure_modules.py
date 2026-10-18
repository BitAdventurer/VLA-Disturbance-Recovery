#!/usr/bin/env python3
"""measure_modules.py OUT.json: CPU time of the modules the recovery pipeline adds beside the policy (ICEIC Sec. V-C).

Times each module the way deployment calls it (run_libero.detector_score, VisionShift.predict, StudentAgent.at_query):
input preparation from uint8 frames plus the forward pass, torch on the CPU with 1 and 2 threads (deployment sets 2).
Random inputs of the deployed shapes; 50 warm-up calls, then N timed calls (median, p90, mean in ms).
Not timed: rendering, the policy query itself (latency.json) and the rest of the control loop."""
import json
import os
import platform
import sys
import time

import numpy as np
import torch

EXP = os.path.expanduser(os.environ.get("VLA_EXPERIMENTS", "~/disk2/vla-inference-scheduling/experiments"))
sys.path.insert(0, EXP)
from residual_rl import StudentNet  # noqa: E402

DET = os.path.join(EXP, "bok/detector_v1_t24.pt")
REG = os.path.join(EXP, "bok/dispreg_v2_t24.pt")
STU = os.path.join(EXP, "ckpt/student_v2.pt")
rng = np.random.default_rng(0)


def timeit(fn, n, warm=50):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append(1e3 * (time.perf_counter() - t0))
    ts = np.array(ts)
    return dict(median_ms=round(float(np.median(ts)), 4), p90_ms=round(float(np.percentile(ts, 90)), 4),
                mean_ms=round(float(ts.mean()), 4), n=n)


def main(out):
    det = torch.jit.load(DET).eval()
    reg = torch.jit.load(REG).eval()
    stu = StudentNet()
    stu.load_state_dict(torch.load(STU, map_location="cpu", weights_only=False)["model"])   # as residual_rl --eval student:
    stu.eval()
    n_par = lambda m: int(sum(p.numel() for p in m.parameters()))

    frame = lambda s: rng.integers(0, 256, (s, s), dtype=np.uint8).astype(np.float64)       # grey 64x64 digest
    prev = {"img64": frame(64), "wrist64": frame(64)}
    cur = {"img64": frame(64), "wrist64": frame(64)}
    act, deef = rng.normal(size=7), rng.normal(size=3) * 0.01

    def det_call():                                     # = run_libero.detector_score after the first load
        f = lambda d, k: np.clip(d[k], 0, 255).astype(np.uint8).astype(np.float32) / 255.0
        i0, i1, w0, w1 = f(prev, "img64"), f(cur, "img64"), f(prev, "wrist64"), f(cur, "wrist64")
        x = np.stack([i0, i1, i1 - i0, w0, w1, w1 - w0])[None].astype(np.float32)
        v = np.concatenate([np.asarray(act, np.float32), np.asarray(deef, np.float32) * 50.0])[None].astype(np.float32)
        with torch.no_grad():
            return float(det(torch.from_numpy(x), torch.from_numpy(v))[0])

    rgb = lambda: rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)
    ev = dict(before_img=rgb(), after_img=rgb(), before_wrist=rgb(), after_wrist=rgb(), action=act, deef=deef)

    def reg_call():                                     # = VisionShift.predict
        x = np.concatenate([ev["before_img"], ev["after_img"], ev["before_wrist"], ev["after_wrist"]], -1)
        x = torch.as_tensor(x[None]).float().permute(0, 3, 1, 2) / 255.0
        x = torch.cat([x, x[:, 3:6] - x[:, 0:3], x[:, 9:12] - x[:, 6:9]], 1)
        v = torch.as_tensor(np.concatenate([ev["action"], ev["deef"] * 50.0])[None].astype(np.float32))
        with torch.no_grad():
            return reg(x, v)[0].numpy().astype(float)

    xs = rng.normal(size=36).astype(np.float32)

    def stu_call():                                     # = the MLP part of StudentAgent.at_query (features excluded)
        with torch.no_grad():
            return np.tanh(stu(torch.from_numpy(xs)[None])[0].numpy())

    res = dict(date=time.strftime("%Y-%m-%d %H:%M:%S %Z"), host_cpu=platform.processor() or platform.machine(),
               torch=torch.__version__, loadavg_before=list(os.getloadavg()),
               params=dict(detector=n_par(det), regressor=n_par(reg), student_mlp=n_par(stu)),
               when=dict(detector="every control step", regressor="once per detection",
                         student_mlp="each active query (up to 6 after a detection)"),
               threads={})
    try:
        res["host_cpu"] = [l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")][0]
    except Exception:
        pass
    for th in (1, 2):
        torch.set_num_threads(th)
        res["threads"][str(th)] = dict(detector=timeit(det_call, 1000), regressor=timeit(reg_call, 300),
                                       student_mlp=timeit(stu_call, 1000))
    res["loadavg_after"] = list(os.getloadavg())
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
