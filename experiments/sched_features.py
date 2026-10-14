"""State features of the scheduler, shared by the rollout runner (logging), the offline
pretraining script, and the online RL trainer. Keep this the single definition."""
import math

import numpy as np

from scheduling import GRID_H, GRID_N, gripper_toggle, mean_speed

FEAT_DIM = 8 + 6 + 1 + 1 + 1 + len(GRID_N) + len(GRID_H) + 1 + 1
REMAINING_IDX = 8 + 6 + 1          # index of the remaining-budget feature
BUDGET_IDX = REMAINING_IDX + 1     # index of the budget feature (fraction of the default's cost)
K_IDX = FEAT_DIM - 2               # index of the slowdown factor k


def quat2axisangle(quat):
    quat = np.array(quat, dtype=float)
    quat[3] = float(np.clip(quat[3], -1.0, 1.0))
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def features(prev_chunk, last_gripper, obs, t, limit, cost_so_far_ms, budget_ms, prev_action, k=1.0,
             budget_frac=None):
    """25-dim state: proprio(8), previous-chunk stats(6), progress(1), remaining budget(1),
    budget(1), previous decision one-hot(6), slowdown factor(1), last gripper command(1).
    The remaining budget is (B - spent) / B on the target device; the budget is B as a fraction
    of the default setting's mean episode cost on that device (0.4 / 0.6 / 1.0).
    budget_ms=None / budget_frac=None (plain rollouts) log 1.0 for both; the offline pretraining
    randomizes them and online RL fills in the real values."""
    proprio = np.concatenate([np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
                              quat2axisangle(obs["robot0_eef_quat"]).astype(np.float32),
                              np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32)])
    if prev_chunk is None:
        chunk_feats = np.zeros(6, dtype=np.float32)
    else:
        c = np.asarray(prev_chunk, dtype=np.float32)
        chunk_feats = np.array([*np.abs(c[:, :3]).mean(0), c[:, -1].mean(), mean_speed(c),
                                float(gripper_toggle(c, last_gripper))], dtype=np.float32)
    progress = np.array([t / limit], dtype=np.float32)
    remaining = np.array([1.0 if not budget_ms else (budget_ms - cost_so_far_ms) / budget_ms],
                         dtype=np.float32)
    budget = np.array([1.0 if budget_frac is None else budget_frac], dtype=np.float32)
    prev = np.zeros(len(GRID_N) + len(GRID_H), dtype=np.float32)
    if prev_action is not None:                  # settings off the study grid (finer-grid runs) stay zero
        if prev_action[0] in GRID_N:
            prev[GRID_N.index(prev_action[0])] = 1.0
        if prev_action[1] in GRID_H:
            prev[len(GRID_N) + GRID_H.index(prev_action[1])] = 1.0
    return np.concatenate([proprio, chunk_feats, progress, remaining, budget, prev, [k], [last_gripper]]
                          ).astype(np.float32)
