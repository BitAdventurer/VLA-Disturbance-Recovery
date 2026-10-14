#!/usr/bin/env python3
"""Policy server that accepts a per-query "num_steps" and "noise_seed".

Runs in the openpi environment. Based on openpi/scripts/serve_policy.py; if the openpi
API differs in your version, adapt the three marked lines.

PyTorch path (required on Blackwell GPUs such as the RTX 5090, whose CUDA is not
supported by openpi's pinned JAX): convert the JAX checkpoint once, copy its norm stats
(the converter looks for assets next to, not inside, the checkpoint directory), then point
--checkpoint at the converted directory; create_trained_policy detects the format.

  python examples/convert_jax_model_to_pytorch.py --config_name pi05_libero \
      --checkpoint_dir <jax checkpoint> --output_path checkpoints/pi05_libero_pytorch
  cp -r <jax checkpoint>/assets checkpoints/pi05_libero_pytorch/assets
  python serve_policy_dynamic.py --checkpoint checkpoints/pi05_libero_pytorch --port 8000

Noise: every query may carry "noise_seed"; the flow-matching noise is then drawn from
numpy's default_rng(noise_seed), so results do not depend on which queries (of which
clients) the server answered before. Without it, the noise comes from the server's own
generator ("rng_seed" + "reset_only" reseeds it).

The first request after start triggers a long torch.compile; --warmup (default) runs
dummy queries for every num_steps used in the study before clients connect, so that
websocket clients do not time out and measured latencies exclude compilation.
"""
import argparse
import logging
import os
import time

import numpy as np


class DynamicStepsPolicy:
    """Wraps an openpi Policy so that each request may set the number of integration steps and the noise."""

    def __init__(self, policy, noise_shape):
        self._p = policy
        self._noise_shape = tuple(noise_shape)
        self._rng = np.random.default_rng(0)

    @property
    def metadata(self):
        return getattr(self._p, "metadata", {})

    def infer(self, obs):
        obs = dict(obs)
        seed = obs.pop("rng_seed", None)
        if seed is not None:
            self._reseed(int(seed))
        if obs.pop("reset_only", False):
            return {"ok": True}
        n = obs.pop("num_steps", None)
        if n is not None:
            kw = dict(getattr(self._p, "_sample_kwargs", {}) or {})   # [1] openpi Policy stores
            kw["num_steps"] = int(n)                                   #     sample kwargs here
            self._p._sample_kwargs = kw
        noise_seed = obs.pop("noise_seed", None)
        rng = self._rng if noise_seed is None else np.random.default_rng(int(noise_seed))
        noise = rng.standard_normal(self._noise_shape, dtype=np.float32)
        t0 = time.perf_counter()
        out = dict(self._p.infer(obs, noise=noise))
        out.setdefault("policy_timing", {})
        out["policy_timing"]["wall_ms"] = (time.perf_counter() - t0) * 1000.0
        out["policy_timing"].setdefault("infer_ms", out["policy_timing"]["wall_ms"])
        return out

    def _reseed(self, seed):
        self._rng = np.random.default_rng(seed)
        if getattr(self._p, "_is_pytorch_model", False):
            import torch
            torch.manual_seed(seed)                                    # [3] PyTorch port
        else:
            import jax
            self._p._rng = jax.random.key(seed)                        # [2] JAX policies


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="pi05_libero")
    ap.add_argument("--checkpoint", default="gs://openpi-assets/checkpoints/pi05_libero")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--warmup", default="1,2,3,5,10",
                    help="num_steps values to compile/warm up before serving ('' to skip)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    if os.path.exists(os.path.join(args.checkpoint, "model.safetensors")):
        # PyTorch checkpoint: openpi still imports JAX; keep it off the GPU (no Blackwell support,
        # and its default preallocation would take 75% of the memory next to the torch model).
        os.environ.setdefault("JAX_PLATFORMS", "cpu")

    from openpi.policies import policy_config as _policy_config
    from openpi.serving import websocket_policy_server
    from openpi.shared import download
    from openpi.training import config as _config

    train_config = _config.get_config(args.config)
    ckpt = download.maybe_download(args.checkpoint)
    policy = _policy_config.create_trained_policy(train_config, ckpt)
    wrapped = DynamicStepsPolicy(policy, (train_config.model.action_horizon, train_config.model.action_dim))
    if args.warmup:
        rng = np.random.default_rng(0)
        dummy = {"observation/image": (rng.random((224, 224, 3)) * 255).astype(np.uint8),
                 "observation/wrist_image": (rng.random((224, 224, 3)) * 255).astype(np.uint8),
                 "observation/state": rng.random(8).astype(np.float32),
                 "prompt": "put the black bowl in the bottom drawer of the cabinet and close it"}
        for n in [int(x) for x in args.warmup.split(",") if x]:
            t0 = time.perf_counter()
            wrapped.infer({**dummy, "num_steps": n, "noise_seed": 0})
            wrapped.infer({**dummy, "num_steps": n, "noise_seed": 1})
            logging.info("warm-up num_steps=%d done in %.1f s (includes compilation)",
                         n, time.perf_counter() - t0)
    logging.info("serving %s on port %d with dynamic num_steps", args.config, args.port)
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=wrapped, host="0.0.0.0", port=args.port, metadata=wrapped.metadata)
    server.serve_forever()


if __name__ == "__main__":
    main()
