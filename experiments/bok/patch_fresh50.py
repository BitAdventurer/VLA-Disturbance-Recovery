#!/usr/bin/env python3
"""patch_fresh50.py DIR: default-off option for the one-time fresh-initial-state test (ICEIC, 2026-10-19).

run_libero.py  : load_init_states(suite, task_id, which) -- 'pruned' (default) returns suite.get_task_init_states as
                 before; 'full' loads <task>.init (100 states) and checks that its first 50 equal .pruned_init.
                 get_env passes getattr(args, "init_file", "pruned"); the mock world returns 100 placeholder states
                 for 'full'.
residual_rl.py : --init_file {pruned,full} (eval only); the eval range must lie inside the loaded states; the silent
                 modulo wrap `init_states[init_id % len]` becomes an explicit range error (identical for every valid id).
Every replacement must match exactly once."""
import os
import sys

D = sys.argv[1]


def patch(fn, pairs):
    p = os.path.join(D, fn)
    s = open(p).read()
    for i, (old, new) in enumerate(pairs):
        n = s.count(old)
        if n != 1:
            raise SystemExit("%s edit %d: expected 1 match, found %d:\n%s" % (fn, i, n, old[:200]))
        s = s.replace(old, new)
    open(p, "w").write(s)
    print("%s: %d edits" % (fn, len(pairs)))


RUN_LIBERO = [
(r'''            return MockEnv(seed=args.seed * 100 + task_id), "mock task %d put the bowl in the drawer" % task_id, [None] * 50
        return policy, tools, get_env, task_ids''',
 r'''            n_states = 100 if getattr(args, "init_file", "pruned") == "full" else 50
            return MockEnv(seed=args.seed * 100 + task_id), "mock task %d put the bowl in the drawer" % task_id, [None] * n_states
        return policy, tools, get_env, task_ids'''),
(r'''        return env, task.language, suite.get_task_init_states(task_id)
    return policy, tools, get_env, task_ids''',
 r'''        return env, task.language, load_init_states(suite, task_id, getattr(args, "init_file", "pruned"))
    return policy, tools, get_env, task_ids


def load_init_states(suite, task_id, which="pruned"):
    """Initial states of one task. 'pruned' (default): <task>.pruned_init, the 50 states the benchmark evaluates, exactly
    as before. 'full': <task>.init, all 100 states; its first 50 must equal .pruned_init. States 50-99 are used only by the
    one-time fresh-initial-state test of 2026-10-19 (experiments/bok/prereg_fresh50.md)."""
    states = suite.get_task_init_states(task_id)
    if which == "pruned":
        return states
    if which != "full":
        raise ValueError("init_file must be 'pruned' or 'full', not %r" % (which,))
    import torch
    from libero.libero import get_libero_path
    task = suite.get_task(task_id)
    name = task.init_states_file
    if not name.endswith(".pruned_init"):
        raise SystemExit("task %d: unexpected init-state file name %r" % (task_id, name))
    full = torch.load(os.path.join(get_libero_path("init_states"), task.problem_folder, name[:-len(".pruned_init")] + ".init"))
    if len(full) != 100 or not np.array_equal(np.asarray(full)[:len(states)], np.asarray(states)):
        raise SystemExit("task %d: .init is not 100 states with .pruned_init as its first 50" % task_id)
    return full'''),
]

RESIDUAL_RL = [
(r'''    def run():
        slot.env._init_state = slot.init_states[init_id % len(slot.init_states)]''',
 r'''    if not 0 <= init_id < len(slot.init_states):        # was a silent modulo wrap (50 -> 0); identical for valid ids
        raise IndexError("init %d outside the %d loaded initial states (states 50-99 need --init_file full)"
                         % (init_id, len(slot.init_states)))

    def run():
        slot.env._init_state = slot.init_states[init_id]'''),
(r'''    p.add_argument("--mock", action="store_true")
    a = p.parse_args()''',
 r'''    p.add_argument("--mock", action="store_true")
    p.add_argument("--init_file", default="pruned", choices=["pruned", "full"],
                   help="initial states: pruned = the benchmark's 50 (default, as before); full = all 100 of <task>.init, "
                        "evaluation only (one-time fresh-state test, experiments/bok/prereg_fresh50.md)")
    a = p.parse_args()
    if a.init_file == "full" and not a.eval:
        raise SystemExit("--init_file full is for evaluation only")
    if a.eval and not 0 <= a.eval_init_lo < a.eval_init_hi <= (100 if a.init_file == "full" else 50):
        raise SystemExit("--eval_init_lo/--eval_init_hi outside the loaded initial states (%s)" % a.init_file)'''),
]

patch("run_libero.py", RUN_LIBERO)
patch("residual_rl.py", RESIDUAL_RL)
