#!/usr/bin/env python3
"""Measure per-query latency c(n) on this GPU for Table I and the RTX 5090 panel of Fig. 2.

  python measure_latency.py --out latency.json          # against serve_policy_dynamic.py
  python measure_latency.py --mock --out /tmp/lat.json  # pipeline test

c(n) (ms_mean) is the client round trip, i.e. what the robot waits for: input transforms,
the model, output transforms, serialization and the local websocket. The model-only time
(openpi's policy_timing.infer_ms) and the whole server-side policy call are stored as well.
Run it with no other load on the GPU. Warm-up queries are discarded so JIT compilation does
not count.
"""
import argparse
import json
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default="1,2,3,5,10")
    ap.add_argument("--repeats", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--out", default="latency.json")
    ap.add_argument("--mock", action="store_true")
    args = ap.parse_args()

    if args.mock:
        from mock import MockPolicy
        policy = MockPolicy()
    else:
        from run_libero import PolicyClient
        policy = PolicyClient(args.host, args.port)

    rng = np.random.default_rng(0)
    element = {"observation/image": (rng.random((224, 224, 3)) * 255).astype(np.uint8),
               "observation/wrist_image": (rng.random((224, 224, 3)) * 255).astype(np.uint8),
               "observation/state": rng.random(8).astype(np.float32),
               "prompt": "put the black bowl in the bottom drawer of the cabinet and close it"}
    ns = [int(x) for x in args.steps.split(",")]
    res = {"n": ns, "ms_mean": [], "ms_std": [], "ms_model_mean": [], "ms_model_std": [], "ms_server_mean": []}
    q = 0
    for n in ns:
        element["num_steps"] = n
        for _ in range(args.warmup):
            element["noise_seed"] = q
            q += 1
            policy.infer(element)
        rt, model, server = [], [], []
        for _ in range(args.repeats):
            element["noise_seed"] = q
            q += 1
            t0 = time.perf_counter()
            out = policy.infer(element)
            rt.append((time.perf_counter() - t0) * 1000.0)
            model.append(float(out.get("policy_timing", {}).get("infer_ms", rt[-1])))
            if args.mock:                    # the mock answers instantly; use its simulated latency
                rt[-1] = model[-1]
            server.append(float(out.get("server_timing", {}).get("infer_ms", rt[-1])))
        res["ms_mean"].append(float(np.mean(rt)))
        res["ms_std"].append(float(np.std(rt)))
        res["ms_model_mean"].append(float(np.mean(model)))
        res["ms_model_std"].append(float(np.std(model)))
        res["ms_server_mean"].append(float(np.mean(server)))
        print("n=%2d  round trip %.1f +- %.1f ms   (server %.1f ms, model %.1f ms)" % (
            n, res["ms_mean"][-1], res["ms_std"][-1], res["ms_server_mean"][-1], res["ms_model_mean"][-1]))
    json.dump(res, open(args.out, "w"), indent=1)
    c1, c0 = np.polyfit(ns, res["ms_mean"], 1)
    print("linear fit (round trip): c0 = %.1f ms, c1 = %.2f ms/step -> saved to %s" % (c0, c1, args.out))


if __name__ == "__main__":
    main()
