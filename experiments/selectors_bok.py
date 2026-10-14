"""Choosing one of K candidate action chunks (best-of-K selection).

Every selector maps candidates [K, H, 7] (unnormalized LIBERO actions) and a context dict to (index, scores);
higher score = preferred, ties go to the lowest index. numpy only, and the global np.random is never used
(LIBERO seeds and uses it for object placement); randomness comes from ctx["rng"].

ctx keys: rng (np.random.Generator), last_action (7,) or None, eef (3,), and for the privileged selector
p_obj (3,) or None, obj_z0 (float), gripper_open (bool), aim_gain (3x3).
"""
import numpy as np

# (q99 - q01) / 2 of the pi05_libero action norm stats (dims 0-5: translation, rotation; 6: gripper)
SCALE = np.array([0.842, 0.828, 0.937, 0.128, 0.175, 0.253])
AIM_GAIN = 0.05 * np.eye(3)        # eef displacement per unit of translation action and step (fit with --log_exec)


def _pairwise(c, w_grip):
    """[K, K] distances between chunks: scaled L2 over dims 0-5 plus w_grip * gripper sign mismatch rate."""
    x = (c[:, :, :6] / SCALE).reshape(len(c), -1)
    d = np.sqrt(((x[:, None, :] - x[None, :, :]) ** 2).sum(-1))
    g = np.sign(c[:, :, 6])
    return d + w_grip * (g[:, None, :] != g[None, :, :]).mean(-1)


def first(c, ctx, **_):
    return np.zeros(len(c))


def random(c, ctx, **_):
    s = np.zeros(len(c))
    s[int(ctx["rng"].integers(len(c)))] = 1.0
    return s


def medoid(c, ctx, w_grip=1.0, **_):
    return -_pairwise(c, w_grip).sum(1)


def antimedoid(c, ctx, w_grip=1.0, **_):
    return _pairwise(c, w_grip).sum(1)


def continuity(c, ctx, **_):
    last = ctx.get("last_action")
    if last is None:
        return np.zeros(len(c))
    return -np.sqrt((((c[:, 0, :6] - np.asarray(last)[:6]) / SCALE) ** 2).sum(-1))


def discontinuity(c, ctx, **_):
    return -continuity(c, ctx)


def priv_aim(c, ctx, **_):
    """Privileged (diagnostic only): uses the true object position. Predicted eef path e_j = eef + G cumsum(a_:3);
    score = -min_j |e_j - p_obj|_xy, minus 0.05 if the chunk closes the gripper more than 5 cm (xy) from the
    object origin at the closing step. Inactive (all zeros -> candidate 0) while the gripper is closed and for
    good once the object has been lifted 3 cm (ctx obj_lifted)."""
    p = ctx.get("p_obj")
    if p is None or not ctx.get("gripper_open", False) or ctx.get("obj_lifted", False):
        return np.zeros(len(c))
    G = np.asarray(ctx.get("aim_gain", AIM_GAIN))
    path = np.asarray(ctx["eef"])[None, None, :] + np.cumsum(c[:, :, :3] @ G.T, axis=1)   # [K, H, 3]
    xy = np.sqrt(((path[:, :, :2] - np.asarray(p)[None, None, :2]) ** 2).sum(-1))           # [K, H]
    err = xy.min(1)
    closes = (c[:, :, 6] > 0)
    first_close = np.where(closes.any(1), closes.argmax(1), -1)
    close_err = np.where(first_close >= 0, xy[np.arange(len(c)), np.maximum(first_close, 0)], 0.0)
    # closing far from the object (beyond the rim of a bowl, ~5 cm from its origin) is penalised
    return -err - 0.05 * ((first_close >= 0) & (close_err > 0.05))


SELECTORS = {f.__name__: f for f in (first, random, medoid, antimedoid, continuity, discontinuity, priv_aim)}
PRIVILEGED = {"priv_aim"}


def select(name, cands, ctx, **kw):
    """(index, scores) for selector `name`; ties go to the lowest index."""
    scores = np.asarray(SELECTORS[name](np.asarray(cands, dtype=float), ctx, **kw), dtype=float)
    return int(np.argmax(scores)), scores
