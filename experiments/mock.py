"""Mock environment and policy so the runner can be tested without LIBERO or a GPU."""
import numpy as np


class MockEnv:
    """Mimics the parts of LIBERO's OffScreenRenderEnv used by the runner."""

    def __init__(self, seed=0, success_step=None):
        self.rng = np.random.default_rng(seed)
        self.success_step = success_step or int(self.rng.integers(150, 400))
        self.t = 0

    def seed(self, s):
        self.rng = np.random.default_rng(s)

    def reset(self):
        self.t = 0
        return self._obs()

    def set_init_state(self, state):
        return self._obs()

    def step(self, action):
        self.t += 1
        done = self.t >= self.success_step
        return self._obs(), 0.0, done, {}

    def _obs(self):
        return {
            "agentview_image": (self.rng.random((256, 256, 3)) * 255).astype(np.uint8),
            "robot0_eye_in_hand_image": (self.rng.random((256, 256, 3)) * 255).astype(np.uint8),
            "robot0_eef_pos": self.rng.random(3), "robot0_eef_quat": np.array([0, 0, 0, 1.0]),
            "robot0_gripper_qpos": self.rng.random(2),
        }


class MockPolicy:
    """Returns random chunks; the gripper flips every few queries to exercise the trigger."""

    def __init__(self, H=10, seed=0):
        self.H, self.rng, self.calls = H, np.random.default_rng(seed), 0

    def infer(self, element):
        n = int(element.get("num_steps", 10))
        if "num_samples" in element:                     # best-of-K request (as serve_policy_k.py)
            ks = [self.infer({"num_steps": n, "noise_seed": s})["actions"] for s in element["noise_seeds"]]
            return {"actions": ks[0], "actions_k": np.stack(ks), "policy_timing": {"infer_ms": 40.0 + 2.0 * n},
                    "emb": np.zeros((4, 8), np.float32), "emb_v": np.zeros((18, 4), np.float32)}
        self.calls += 1
        if "noise_seed" in element:                      # a pure function of the seed (and n)
            rng = np.random.default_rng(int(element["noise_seed"]))
            chunk = rng.normal(0, 0.3, (self.H, 7))
            chunk[:, -1] = -1.0 if rng.random() < 0.6 else 1.0
        else:
            chunk = self.rng.normal(0, 0.3, (self.H, 7))
            chunk[:, -1] = -1.0 if (self.calls // 4) % 2 == 0 else 1.0
        return {"actions": chunk, "policy_timing": {"infer_ms": 40.0 + 2.0 * n}}
