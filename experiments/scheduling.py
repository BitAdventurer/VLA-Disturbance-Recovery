"""Scheduling logic shared by the runner and the tests. No LIBERO or openpi imports here.

Notation follows the paper: n = integration steps, h = execution horizon, H = chunk length,
c_dev(n) = per-query latency of the emulated device (ms), d(n) = ceil(c_dev(n) / dt).
"""
import math
import numpy as np

H = 10                      # chunk length of the pi05_libero checkpoint
DT_MS = 50.0                # LIBERO control period (20 Hz)
CHEAP, EXPENSIVE = (2, 10), (10, 2)
GRID_N, GRID_H = [2, 5, 10], [2, 5, 10]     # study grid

# Device profiles c(n) = c0 + c1 n inferred from FASTER (arXiv:2603.19199, Table 2).
PROFILES = {
    "rtx4090_profile": (62.1 - (80.0 - 62.1) / 9.0, (80.0 - 62.1) / 9.0),
    "rtx4060_profile": (238.6 - (303.3 - 238.6) / 9.0, (303.3 - 238.6) / 9.0),
}


def cost_function(device, latency_table=None):
    """Return n -> latency in ms for the emulated device, or None for no latency coupling."""
    if device in (None, "", "none"):
        return None
    if device == "rtx5090":
        if not latency_table:
            raise ValueError("device=rtx5090 needs --latency (measured table from measure_latency.py)")
        ns = sorted(latency_table)
        return lambda n: float(np.interp(n, ns, [latency_table[m] for m in ns]))
    c0, c1 = PROFILES[device]
    return lambda n: c0 + c1 * n


def delay(cost_ms, dt=DT_MS):
    """Control periods occupied by one query, Eq. (3)."""
    return int(math.ceil(cost_ms / dt))


def stall_free(n, h, cost):
    """Eq. (4)."""
    d = delay(cost(n))
    return d <= h <= H - d


def gripper_toggle(chunk, last_gripper):
    """True if the chunk commands a gripper state different from the current one."""
    g = np.asarray(chunk)[:, -1]
    return bool(np.any(np.sign(g) != np.sign(last_gripper)))


def mean_speed(chunk):
    """Mean commanded end-effector translation per step (action units)."""
    return float(np.linalg.norm(np.asarray(chunk)[:, :3], axis=1).mean())


class Schedule:
    """Chooses (n, h) for the next query.

    static           : always (n, h)
    uniform_cheap    : always CHEAP
    uniform_expensive: always EXPENSIVE
    cheap2exp        : CHEAP in transport, EXPENSIVE when the previous chunk is precision-critical
    exp2cheap        : the reverse
    react_next       : (n, h), switching to react_setting for react_window steps after a disturbance,
                       from the next query on (oracle timing, an upper reference for learned schedulers)
    react_now        : the same, but the current window is cut short when the disturbance hits
    react_detect     : react_next without the oracle: the switch is triggered at a query when the camera image
                       changed more than a threshold since the previous query (ctx["detected_at"])
    react_step       : react_now without the oracle: a per-step camera-change detector cuts the window short
                       and triggers the switch (ctx["detected_at"])
    """

    def __init__(self, name, n=None, h=None, trigger=None, v_min=None, react_setting=(2, 2), react_window=30):
        self.name, self.n, self.h, self.trigger, self.v_min = name, n, h, trigger, v_min
        self.react_setting, self.react_window = tuple(react_setting), react_window
        self.interrupt = name in ("react_now", "react_step")   # cut the current window short on a disturbance
        self.decide_at_query = name in ("react_next", "react_now", "react_detect", "react_step")
        if name in ("cheap2exp", "exp2cheap") and trigger not in ("gripper", "speed"):
            raise ValueError("two-phase schedules need --trigger gripper|speed")
        if trigger == "speed" and v_min is None:
            raise ValueError("--trigger speed needs --v_min")

    def critical(self, prev_chunk, last_gripper):
        if prev_chunk is None:
            return False                     # no previous chunk yet
        if self.trigger == "gripper":
            return gripper_toggle(prev_chunk, last_gripper)
        if self.trigger == "speed":
            return mean_speed(prev_chunk) < self.v_min
        return False

    def next_setting(self, prev_chunk, last_gripper, ctx=None):
        """ctx (optional): dict with obs, t, limit, cost_so_far_ms, calls, k, disturbed_at -- used by
        learned and reactive schedulers."""
        if self.name == "static":
            return (self.n, self.h), False
        if self.name in ("react_next", "react_now", "react_detect", "react_step"):
            # react_next/now are oracle-timed (they know when the disturbance hit); react_detect/step use detectors
            t_d = (ctx or {}).get("detected_at" if self.name in ("react_detect", "react_step") else "disturbed_at")
            if t_d is not None and ctx["t"] < t_d + self.react_window:
                return self.react_setting, True
            return (self.n, self.h), False
        if self.name == "uniform_cheap":
            return CHEAP, False
        if self.name == "uniform_expensive":
            return EXPENSIVE, False
        if prev_chunk is None:               # the first query of both two-phase schedules is cheap
            return CHEAP, False
        crit = self.critical(prev_chunk, last_gripper)
        if self.name == "cheap2exp":
            return (EXPENSIVE if crit else CHEAP), crit
        if self.name == "exp2cheap":
            return (CHEAP if crit else EXPENSIVE), crit
        raise ValueError(self.name)
