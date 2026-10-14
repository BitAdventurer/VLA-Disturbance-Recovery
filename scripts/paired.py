"""paired.py ref.jsonl other1[.part] other2 ... : success/cost on the episodes common to ref and each file."""
import json, sys, numpy as np
def load(p):
    d = {}
    for line in open(p):
        try: r = json.loads(line)
        except Exception: continue
        if "success" not in r or r.get("failed"): continue
        d[(r["task_id"], r["init_state_id"])] = r
    return d
ref = load(sys.argv[1]); rng = np.random.default_rng(0)
for p in sys.argv[2:]:
    o = load(p); ks = sorted(set(ref) & set(o))
    if not ks: print(p, "no overlap"); continue
    a = np.array([ref[k]["success"] for k in ks], float); b = np.array([o[k]["success"] for k in ks], float)
    ca = np.mean([ref[k]["cost_ms"] for k in ks]) / 1000; cb = np.mean([o[k]["cost_ms"] for k in ks]) / 1000
    bs = [np.mean((b - a)[rng.integers(0, len(ks), len(ks))]) for _ in range(2000)]
    print("%-45s n=%3d  ref %.1f%% (%.2fs)  this %.1f%% (%.2fs)  diff %+.1f [%+.1f, %+.1f]" % (
        p.split("/")[-1], len(ks), 100*a.mean(), ca, 100*b.mean(), cb, 100*(b-a).mean(),
        100*np.percentile(bs, 2.5), 100*np.percentile(bs, 97.5)))
