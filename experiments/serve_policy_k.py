#!/usr/bin/env python3
"""Policy server that accepts a per-query "num_steps" and "noise_seed", plus best-of-K requests.

serve_policy_dynamic.py with one addition: a request carrying "num_samples" K returns K action chunks for one
observation from ONE prefix pass (the prefix KV cache is repeated to batch K for the action expert), plus pooled
prefix features for a critic (see DynamicStepsPolicy.infer_k). Requests without "num_samples" take the unchanged
legacy path, so old clients get byte-identical answers.

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
import hashlib
import logging
import os
import time

import numpy as np


class SharedPrefixSampler:
    """K flow-matching samples for ONE observation with ONE prefix pass (SigLIP x3 + Gemma-2B, 968 tokens).

    Mirrors PI0Pytorch.sample_actions (openpi pi0_pytorch.py:376-420) except that
      (a) the prefix KV cache is built at batch 1 and repeated to batch K before the Euler loop,
      (b) the loop runs num_steps times in Python (openpi's `while time >= -dt / 2` also runs exactly
          num_steps times, but its tensor-valued condition breaks the graph and syncs the GPU every step),
      (c) pooled prefix features for a critic are returned as well.
    The openpi repo is not modified; only its methods are called.
    """

    def __init__(self, model, compile_mode="max-autotune"):
        import torch
        self.m = model                                    # PI0Pytorch, already on the GPU and in eval mode
        if compile_mode and compile_mode != "none":
            # One graph per (K, num_steps) for _sample, one for _embed. Past the default cap of 8 graphs per
            # code object, dynamo silently runs new variants eagerly (several times slower).
            torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit, 64)
            self._sample = torch.compile(self._sample_impl, mode=compile_mode, dynamic=False)
            self._embed = torch.compile(self._embed_impl, mode=compile_mode, dynamic=False)
        else:
            self._sample, self._embed = self._sample_impl, self._embed_impl

    def _prefix(self, observation):
        """pi0_pytorch.py:384-400 at batch 1, keeping the final hidden states that openpi discards."""
        import torch
        from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
        m = self.m
        images, img_masks, lang_tokens, lang_masks, state = m._preprocess_observation(observation, train=False)
        prefix_embs, prefix_pad_masks, prefix_att_masks = m.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        att_4d = m._prepare_attention_masks_4d(make_att_2d_masks(prefix_pad_masks, prefix_att_masks))
        position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        m.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"
        (hidden, _), cache = m.paligemma_with_expert.forward(
            attention_mask=att_4d, position_ids=position_ids, past_key_values=None,
            inputs_embeds=[prefix_embs, None], use_cache=True)
        # Critic features: masked means over valid prefix tokens. Token layout: [base 256 | wrist 256 |
        # right wrist 256 (zero image, masked) | prompt 200 (padded)], L = 968.
        valid = prefix_pad_masks[0].to(torch.float32)[:, None]            # [L, 1]
        h = hidden[0].to(torch.float32)                                    # [L, 2048], final RMSNorm output
        L, n_lang = h.shape[0], lang_tokens.shape[1]
        n_img = (L - n_lang) // len(images)                               # 256 SigLIP tokens per camera

        def pool(x, a, b):
            w = valid[a:b]
            return (x[a:b] * w).sum(0) / w.sum().clamp(min=1.0)

        emb = torch.stack([pool(h, 0, L), pool(h, 0, n_img), pool(h, n_img, 2 * n_img), pool(h, L - n_lang, L)])
        emb_v = torch.stack([pool(cache[i][1][0, 0].to(torch.float32), 0, L) for i in range(len(cache))])
        return state, prefix_pad_masks, cache, emb, emb_v                 # emb [4, 2048], emb_v [18, 256]

    def _sample_impl(self, observation, noise, num_steps: int):
        import torch
        m = self.m
        K = noise.shape[0]
        state, prefix_pad_masks, cache, emb, emb_v = self._prefix(observation)
        if K > 1:
            cache.batch_repeat_interleave(K)              # every layer: [1, 1, 968, 256] -> [K, 1, 968, 256]
            prefix_pad_masks = prefix_pad_masks.expand(K, -1)
            state = state.expand(K, -1)                   # ignored by pi05 (embed_suffix), needed for pi0
        dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=noise.device)   # as pi0_pytorch.py:402-406
        time = torch.tensor(1.0, dtype=torch.float32, device=noise.device)
        x_t = noise
        for _ in range(num_steps):
            v_t = m.denoise_step(state, prefix_pad_masks, cache, x_t, time.expand(K))  # cache is read-only here
            x_t = x_t + dt * v_t
            time = time + dt
        return x_t, emb, emb_v

    def _embed_impl(self, observation):
        _, _, _, emb, emb_v = self._prefix(observation)
        return emb, emb_v

    def sample(self, observation, noise, num_steps):
        """noise: float32 [K, H, 32] on the model device. Returns numpy (x [K, H, 32] model space, emb, emb_v)."""
        import torch
        with torch.no_grad():
            torch.compiler.cudagraph_mark_step_begin()    # outputs of the previous call may be overwritten
            x, emb, emb_v = self._sample(observation, noise, int(num_steps))
            return x.cpu().numpy(), emb.cpu().numpy(), emb_v.cpu().numpy()   # copy out before the next replay

    def embed(self, observation):
        import torch
        with torch.no_grad():
            torch.compiler.cudagraph_mark_step_begin()
            emb, emb_v = self._embed(observation)
            return emb.cpu().numpy(), emb_v.cpu().numpy()


class DynamicStepsPolicy:
    """Wraps an openpi Policy so that each request may set the number of integration steps and the noise."""

    def __init__(self, policy, noise_shape, k_compile=None, version=""):
        self._p = policy
        self._noise_shape = tuple(noise_shape)
        self._rng = np.random.default_rng(0)
        self._sampler = None                              # SharedPrefixSampler, built on the first K request
        self._k_compile = k_compile                       # None: the model's own pytorch_compile_mode
        self.version = version

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
        if "num_samples" in obs:                          # best-of-K request; the path below is unchanged
            return self.infer_k(obs)
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

    def infer_k(self, obs):
        """K chunks for one observation.

        Request keys (popped): num_samples K; num_steps n; noise_seeds (K ints; preferred) or noise_seed (seeds
        k >= 1 are then derived); return_embedding (default True); exact (default False).
        K == 1 and (exact or not return_embedding): the legacy compiled path runs, bit-identical to infer() with
        noise_seed = noise_seeds[0]; with exact and return_embedding a second, prefix-only pass adds the
        embedding (data collection only: it doubles the prefix cost). Otherwise the shared-prefix path runs.
        Reply: actions_k [K, H, 7] unnormalized (what the env executes), actions_k_norm [K, H, 7] (model space,
        quantile-normalized), actions = actions_k[0] (old clients keep working), emb [4, 2048]
        (all / base / wrist / prompt token means), emb_v [18, 256], noise_seeds, policy_timing.
        """
        import jax
        import torch
        from openpi.models import model as _model
        p = self._p
        K = int(obs.pop("num_samples"))
        n = int(obs.pop("num_steps", (p._sample_kwargs or {}).get("num_steps", 10)))
        seeds, base = obs.pop("noise_seeds", None), obs.pop("noise_seed", None)
        want_emb = bool(obs.pop("return_embedding", True))
        exact = bool(obs.pop("exact", False))
        if seeds is None:
            if base is None:
                raise ValueError("infer_k needs noise_seeds or noise_seed")
            seeds = [int(base)] + [int(np.random.SeedSequence([int(base), k]).generate_state(1)[0])
                                   for k in range(1, K)]
        seeds = [int(s) for s in seeds]
        if len(seeds) != K:
            raise ValueError(f"len(noise_seeds)={len(seeds)} != num_samples={K}")
        # Row k is exactly the noise infer() would draw for noise_seed=seeds[k] (serve_policy_dynamic.py:59-60).
        noise = np.stack([np.random.default_rng(s).standard_normal(self._noise_shape, dtype=np.float32)
                          for s in seeds])
        t0 = time.perf_counter()
        if self._sampler is None:
            self._sampler = SharedPrefixSampler(
                p._model, compile_mode=self._k_compile or p._model.config.pytorch_compile_mode)

        def observation_of(o):                            # policy.py:70-71, :78, :90
            inputs = p._input_transform(jax.tree.map(lambda x: x, o))
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(p._pytorch_device)[None, ...], inputs)
            return inputs, _model.Observation.from_dict(inputs)

        emb = emb_v = None
        if K == 1 and (exact or not want_emb):
            kw = dict(p._sample_kwargs or {})
            kw["num_steps"] = n
            p._sample_kwargs = kw
            legacy = p.infer(dict(obs), noise=noise[0])   # unchanged compiled openpi path
            actions_k = np.asarray(legacy["actions"])[None]
            model_ms = float(legacy["policy_timing"]["infer_ms"])
            x_norm = None
            if want_emb:
                _, observation = observation_of(dict(obs))
                emb, emb_v = self._sampler.embed(observation)   # second prefix pass: data collection only
        else:
            inputs, observation = observation_of(obs)
            t1 = time.perf_counter()
            x, emb, emb_v = self._sampler.sample(observation, torch.from_numpy(noise).to(p._pytorch_device), n)
            model_ms = (time.perf_counter() - t1) * 1000.0
            state = np.asarray(inputs["state"][0].detach().cpu())
            # Unnormalize broadcasts over K (transforms.py:175-181); LiberoOutputs keeps [..., :7].
            actions_k = np.asarray(p._output_transform({"state": state, "actions": x})["actions"])
            x_norm = x[..., :actions_k.shape[-1]]
        out = {"actions": actions_k[0], "actions_k": actions_k, "noise_seeds": np.asarray(seeds, dtype=np.int64),
               "policy_timing": {"infer_ms": model_ms, "wall_ms": (time.perf_counter() - t0) * 1000.0},
               "server_version": self.version, "shared_path": not (K == 1 and (exact or not want_emb))}
        if x_norm is not None:
            out["actions_k_norm"] = x_norm
        if want_emb:
            out["emb"], out["emb_v"] = emb, emb_v
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
    ap.add_argument("--warmup_k", default="",
                    help="best-of-K sample counts to compile/warm up, e.g. '1,2,4,8,16' ('' to skip)")
    ap.add_argument("--warmup_k_steps", default="2", help="num_steps values for the K warm-up")
    ap.add_argument("--warmup_embed", type=int, default=1,
                    help="also compile the prefix-only embedding graph (K=1 exact requests with an embedding)")
    ap.add_argument("--k_compile", default=None,
                    choices=[None, "max-autotune", "max-autotune-no-cudagraphs", "default", "none"],
                    help="torch.compile mode of the shared-prefix path (default: the model's own mode)")
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
    import torch
    version = "%s|k_compile=%s|torch=%s" % (hashlib.sha256(open(__file__, "rb").read()).hexdigest()[:12],
                                            args.k_compile or "model", torch.__version__)
    wrapped = DynamicStepsPolicy(policy, (train_config.model.action_horizon, train_config.model.action_dim),
                                 k_compile=args.k_compile, version=version)
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
        for K in [int(x) for x in args.warmup_k.split(",") if x]:
            for n in [int(x) for x in args.warmup_k_steps.split(",") if x]:
                t0 = time.perf_counter()
                for s in (0, 1):
                    wrapped.infer({**dummy, "num_samples": K, "num_steps": n,
                                   "noise_seeds": list(range(s, s + K))})
                logging.info("warm-up K=%d num_steps=%d done in %.1f s (includes compilation)",
                             K, n, time.perf_counter() - t0)
        if args.warmup_embed:
            t0 = time.perf_counter()
            for s in (0, 1):
                wrapped.infer({**dummy, "num_samples": 1, "num_steps": 2, "noise_seeds": [s],
                               "exact": True, "return_embedding": True})
            logging.info("warm-up embedding pass done in %.1f s", time.perf_counter() - t0)
    logging.info("server_version %s", version)
    logging.info("serving %s on port %d with dynamic num_steps", args.config, args.port)
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=wrapped, host="0.0.0.0", port=args.port, metadata=wrapped.metadata)
    server.serve_forever()


if __name__ == "__main__":
    main()
