#!/usr/bin/env python3
"""fig_paper.py OUTDIR: publication figures + numbers for the ICEIC residual-recovery paper.

Writes OUTDIR/res_forest.pdf (held-out inits 25-49: every arm vs the (2,10) noise-draw pool),
OUTDIR/res_general.pdf (student v2 / teacher v4 across disturbance types) and OUTDIR/res_numbers.json
(every number used in the tables).  Arms with several noise draws are averaged per episode; all CIs are
task-stratified paired bootstraps (6000 resamples) against the reference draws on the same episodes."""
import glob
import hashlib
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

E = os.path.expanduser(os.environ.get("VLA_EXPERIMENTS", "~/disk2/vla-inference-scheduling/experiments"))
OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(0)

plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7.5, "xtick.labelsize": 6.5,
                     "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "pdf.fonttype": 42, "ps.fonttype": 42,
                     "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
                     "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"]})
# Okabe-Ito
BLUE, ORANGE, GREEN, SKY, VERM, PURPLE, GREY, YELLOW = ("#0072B2", "#E69F00", "#009E73", "#56B4E9", "#D55E00",
                                                         "#CC79A7", "#7F7F7F", "#F0E442")


def L(path):
    if isinstance(path, tuple):          # one draw split over several files (e.g. held-out and training initial states)
        parts = [L(q) for q in path]
        if any(x is None for x in parts):
            return None
        merged = {}
        for x in parts:
            merged.update(x)
        return merged
    p = path if os.path.exists(path) else path + ".part"
    if not os.path.exists(p):
        return None
    return {(r["task_id"], r["init_state_id"]): r for r in map(json.loads, open(p)) if "success" in r and not r.get("error")}


RD = lambda n: "%s/results_dist/%s.jsonl" % (E, n)
RR = lambda n: "%s/results_res/%s.jsonl" % (E, n)
RB = lambda n: "%s/results_bok/%s.jsonl" % (E, n)
G = lambda pat: sorted(glob.glob("%s/results_dist/%s.jsonl" % (E, pat)))


def load(paths):
    D = [d for d in (L(p) for p in paths) if d]
    missing = [p for p in paths if L(p) is None]
    return D, missing


def paired(arms, refs, keys=None, nboot=6000):
    """arm mean over draws per episode minus reference mean over draws; task-stratified bootstrap CI."""
    ks = sorted(k for k in set.intersection(*[set(d) for d in refs]) & set().union(*[set(d) for d in arms])
                if keys is None or keys(k))
    y = np.array([np.mean([d[k]["success"] for d in arms if k in d]) for k in ks])
    r = np.array([np.mean([d[k]["success"] for d in refs]) for k in ks])
    x = y - r
    tasks = np.array([k[0] for k in ks]); grp = [np.flatnonzero(tasks == t) for t in sorted(set(tasks))]
    # 2026-10-16: the bootstrap stream is seeded by the comparison itself (its per-episode differences and tasks), so a
    # comparison gets the same interval wherever and however often it is computed, and adding arms moves no other interval.
    seed = int.from_bytes(hashlib.sha256(np.round(x, 9).tobytes() + tasks.astype(np.int64).tobytes()).digest()[:8], "little")
    rl = np.random.default_rng(seed)
    bs = np.array([x[np.concatenate([g[rl.integers(0, len(g), len(g))] for g in grp])].mean() for _ in range(nboot)])
    act = float(np.mean([any(any(abs(v) > 1e-6 for v in dd) for dd in (d[k].get("D") or [])) for d in arms for k in ks if k in d]))
    per_task = {int(t): round(float(100 * x[tasks == t].mean()), 1) for t in sorted(set(tasks))}
    return dict(success=round(float(100 * y.mean()), 1), ref=round(float(100 * r.mean()), 1),
                delta=round(float(100 * x.mean()), 1), lo=round(float(100 * np.percentile(bs, 2.5)), 1),
                hi=round(float(100 * np.percentile(bs, 97.5)), 1), n=len(ks), draws=len(arms), ref_draws=len(refs),
                active=round(act, 3), per_task=per_task, wins=int((x > 0).sum()), losses=int((x < 0).sum()))


numbers = {}
pool, miss = load([RD("obj_static_n2_h10_%s" % s) for s in ("s0", "ns1_s0", "ns2_s0", "ns3_s0", "ns4_s0", "ns5_s0", "ns6_s0")])
assert len(pool) >= 5, miss
print("pool draws:", len(pool))
HO = lambda k: k[1] >= 25
numbers["pool_all"] = round(100 * float(np.mean([np.mean([d[k]["success"] for d in pool]) for k in pool[0]])), 1)
numbers["pool_heldout"] = round(100 * float(np.mean([np.mean([d[k]["success"] for d in pool]) for k in pool[0] if HO(k)])), 1)
numbers["pool_draws"] = [round(100 * float(np.mean([r["success"] for r in d.values()])), 1) for d in pool]

# ----------------------------------------------------------------------------------------------------------
# Fig. 2: forest plot on held-out inits (and the same arms on all inits for the table)
# ----------------------------------------------------------------------------------------------------------
# teacher draws: held-out runs (inits 25-49) merged per noise salt with the chain8 runs on the training inits 0-24
TEACHER_DRAWS = [(RR("eval_res_v4_it039" + sfx), RR("eval_res_v4_it039_tr" + sfx)) for sfx in ("", "_ns1", "_ns2", "_ns3", "_ns4")]
arms = [
    # (key, label, paths, group, hollow)
    ("static_2_5", "static $(2,5)$: replan 2$\\times$ as often", [RD("obj_static_n2_h5_s0")], "lever", False),
    ("static_2_2", "static $(2,2)$: replan 5$\\times$ as often", [RD("obj_static_n2_h2_s0")], "lever", False),
    ("static_10_10", "static $(10,10)$: 5$\\times$ the steps", [RD("obj_static_n10_h10_s0")], "lever", False),
    ("sel8_random", "best-of-8 chunks: random", [RB("sel8_random")], "lever", False),
    ("sel8_medoid", "best-of-8 chunks: medoid", [RB("sel8_medoid")], "lever", False),
    ("sel8_priv", "best-of-8: object aim (priv.)", [RB("sel8_priv")], "lever", False),
    ("react_next", "next query, then $n{=}10$ (oracle)", [RD("obj_react_next_n10h10_s0"), RD("obj_react_next_n10h10_ns1_s0")], "timing", False),
    ("react_now_2", "at the event, $(2,10)$ (oracle)", [RD("obj_react_now_n2h10_%s" % s) for s in ("s0", "ns1_s0", "ns2_s0")], "timing", False),
    ("react_now_10", "at the event, then $n{=}10$ (oracle)", [RD("obj_react_now_n10h10_%s" % s) for s in ("s0", "ns1_s0", "ns2_s0")], "timing", False),
    ("det_2", "detector $\\rightarrow$ replan, $(2,10)$", [RD("det_n2h10_%s" % s) for s in ("s0", "ns1_s0", "ns2_s0", "ns3_s0", "ns4_s0")], "timing", False),
    ("det_10", "detector $\\rightarrow$ replan, then $n{=}10$", [RD("det_n10h10_%s" % s) for s in ("s0", "ns1_s0", "ns2_s0")], "timing", False),
    ("scripted", "scripted shift $0.5\\,d$ (priv.)", [RR("probe_s05_k2")] + [(RR("probe_s05_k2_ns%d" % s), RR("probe_s05_k2_tr_ns%d" % s)) for s in (1, 2)], "residual", False),
    ("vision_scripted", "fixed-rule shift $0.5\\,\\widehat{d}$", [RR("vision_s05_k2"), RR("vision_s05_k2_ns1"), RR("vision_s05_k2_ns2")], "residual", False),
    ("teacher_bc", "teacher: BC only (priv.)", [RR("eval_res_v4_bc")], "residual", True),
    ("teacher_scratch", "teacher: PPO from scratch (priv.)", [RR("abl_scratch_it039"), RR("abl_scratch_it039_ns1")], "residual", True),
    ("teacher", "teacher: BC + PPO (priv.)", TEACHER_DRAWS, "residual", False),
    ("student_nodhat", "student without $\\widehat{d}$", [RR("abl_student_nodhat"), RR("abl_student_nodhat_ns1")], "residual", True),
    ("student", "residual student (vision)", [RR("student_v2"), RR("student_v2_ns1"), RR("student_v2_ns2"), RR("student_v2_ns3"), RR("student_v2_ns4")], "residual", False),
]
forest = {}
for key, lab, paths, grp, hollow in arms:
    D, miss = load(paths)
    if miss:
        print("MISSING for %s: %s" % (key, miss))
    if not D:
        continue
    forest[key] = dict(label=lab, group=grp, heldout=paired(D, pool, HO))
    if all(any(k[1] < 25 for k in d) for d in D):
        forest[key]["all"] = paired(D, pool)
numbers["forest"] = forest

col = {"lever": GREY, "timing": ORANGE, "residual": GREEN}
names = {"lever": "Inference levers", "timing": "Replan timing", "residual": "Residual recovery"}
rows = [a for a in arms if a[0] in forest]
fig, ax = plt.subplots(figsize=(3.45, 2.85))
ytick, ylab, header_rows = [], [], []
y = 0
last_grp = None
for key, lab, paths, grp, hollow in rows:
    if grp != last_grp:
        if last_grp is not None:
            y += 0.35
        ytick.append(y); ylab.append(names[grp]); header_rows.append((len(ytick) - 1, grp)); y += 1
    last_grp = grp
    y += 0.3 * ("widehat" in lab)     # room for the hat of d-hat below the row above
    r = forest[key]["heldout"]
    m, lo, hi = r["delta"], r["lo"], r["hi"]
    ax.errorbar(m, y, xerr=[[m - lo], [hi - m]], fmt="o", ms=3.0, color=col[grp], mfc="white" if hollow else col[grp],
                mew=0.9, elinewidth=0.9, capsize=1.6, capthick=0.7)
    tr = ax.get_yaxis_transform()
    ax.text(1.16, y, "%.1f" % r["success"], transform=tr, va="center", ha="right", fontsize=6)
    ax.text(1.43, y, ("%+.1f" % m).replace("-", "\u2212"), transform=tr, va="center", ha="right", fontsize=6, fontweight="bold" if lo > 0 or hi < 0 else "normal")
    ax.text(1.57, y, "%d" % r["draws"], transform=tr, va="center", ha="center", fontsize=6)
    ytick.append(y); ylab.append(lab)
    y += 1
ax.axvline(0, color="k", lw=0.6)
ax.set_yticks(ytick); ax.set_yticklabels(ylab)
for i, grp in header_rows:
    t = ax.get_yticklabels()[i]; t.set_color(col[grp]); t.set_fontweight("bold")
ax.invert_yaxis()
ax.set_ylim(y - 0.6, -1.3)
ax.set_xlim(-16, 18)
ax.set_xticks(range(-16, 17, 8))
ax.set_xlabel("gain over the base policy (points, 95% CI)")
tr = ax.get_yaxis_transform()
ax.text(1.16, -0.75, "succ.", transform=tr, ha="right", va="center", fontsize=6, fontweight="bold")
ax.text(1.43, -0.75, "gain", transform=tr, ha="right", va="center", fontsize=6, fontweight="bold")
ax.text(1.57, -0.75, "draws", transform=tr, ha="center", va="center", fontsize=6, fontweight="bold")
ax.grid(axis="x", alpha=0.25, lw=0.4)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.tick_params(axis="y", length=0)
plt.subplots_adjust(left=0.44, right=0.735, bottom=0.135, top=0.995)
fig.savefig(os.path.join(OUT, "res_forest.pdf"))
print("saved res_forest.pdf")

# ----------------------------------------------------------------------------------------------------------
# Fig. 3: generalization across disturbance types (all 500 episodes; teacher at 4-6 cm: held-out only)
# ----------------------------------------------------------------------------------------------------------
conds = [
    ("train", "4\u20136 cm\n(training)", [RD("obj_static_n2_h10_%s" % s) for s in ("s0", "ns1_s0", "ns2_s0", "ns3_s0", "ns4_s0", "ns5_s0", "ns6_s0")],
     dict(student=[RR("student_v2"), RR("student_v2_ns1"), RR("student_v2_ns2"), RR("student_v2_ns3"), RR("student_v2_ns4")],
          teacher=TEACHER_DRAWS,
          vision=[RR("vision_s05_k2"), RR("vision_s05_k2_ns1"), RR("vision_s05_k2_ns2")], nodhat=[RR("abl_student_nodhat"), RR("abl_student_nodhat_ns1")])),
    ("big", "8\u201312 cm", G("objbig_static_n2_h10*"),
     dict(student=[RR("gen_big_student_v2"), RR("gen_big_student_v2_ns1")], teacher=[RR("gen_big_teacher_v4"), RR("gen_big_teacher_v4_ns1")], vision=[RR("gen_big_vision_s05"), RR("gen_big_vision_s05_ns1"), RR("gen_big_vision_s05_ns2")],
          nodhat=[RR("gen_big_student_nodhat"), RR("gen_big_student_nodhat_ns1")])),
    ("late", "4\u20136 cm,\nlate", G("objlate_static_n2_h10*"), dict(student=[RR("gen_late_student_v2"), RR("gen_late_student_v2_ns1")], teacher=[RR("gen_late_teacher_v4"), RR("gen_late_teacher_v4_ns1")])),
    ("drift", "sliding\n6\u201310 cm", G("drift_static_n2_h10*"), dict(student=[RR("gen_drift_student_v2")], teacher=[RR("gen_drift_teacher_v4")])),
    ("push", "arm push", G("push_static_n2_h10*"), dict(student=[RR("gen_push_student_v2")])),
    ("none", "none", [RD("cal_none_s0")], dict(student=[RR("student_v2_none")], teacher=[(RR("eval_res_v4_it039_none"), RR("eval_res_v4_it039_tr_none"))],
                                                 vision=[RR("vision_s05_k2_none")], nodhat=[RR("abl_student_nodhat_none")],
                                                 det=[RD("det_none_n2h10_s0")])),
]
gen = {}
for ck, clab, refp, armd in conds:
    refs, miss = load(refp)
    if miss or not refs:
        print("MISSING baseline for %s: %s (found %d)" % (ck, miss, len(refs)))
    if not refs:
        continue
    g = dict(label=clab, ref_paths=[os.path.basename(p) for p in refp])
    for ak, paths in armd.items():
        D, miss = load(paths)
        if miss:
            print("MISSING for %s/%s: %s" % (ck, ak, miss))
        if not D:
            continue
        keys = None   # 2026-10-16: the teacher now has draws on all 500 episodes (chain8 B4)
        g[ak] = paired(D, refs, keys)
        g[ak]["heldout_only"] = bool(keys)
    g["baseline"] = round(100 * float(np.mean([np.mean([d[k]["success"] for d in refs]) for k in refs[0]])), 1)
    g["baseline_draws"] = len(refs)
    gen[ck] = g
numbers["general"] = gen

fig, ax = plt.subplots(figsize=(3.45, 1.9))
xs = np.arange(len(gen))
w = 0.26
labels = [gen[c]["label"] for c in gen]
for i, (ak, lab, c, off) in enumerate([("baseline", "$\\pi_{0.5}$ alone", GREY, -w),
                                        ("student", "+ student (vision only)", GREEN, 0),
                                        ("teacher", "+ teacher (privileged)", BLUE, w)]):
    vals, err_lo, err_hi, xpos = [], [], [], []
    for j, ck in enumerate(gen):
        g = gen[ck]
        if ak == "baseline":
            vals.append(g["baseline"]); err_lo.append(0); err_hi.append(0); xpos.append(j + off)
        elif ak in g:
            r = g[ak]
            vals.append(r["success"]); err_lo.append(r["delta"] - r["lo"]); err_hi.append(r["hi"] - r["delta"]); xpos.append(j + off)
            tag = ("%+.1f" % r["delta"]).replace("-", "\u2212") + ("\u2020" if r.get("heldout_only") else "")
            ax.text(j + off, r["success"] + err_hi[-1] + 1.2, tag, ha="center", va="bottom", fontsize=6.5, rotation=90,
                    color=c, fontweight="bold" if r["lo"] > 0 or r["hi"] < 0 else "normal")
            if r.get("heldout_only"):
                ax.plot([j + off - w / 2, j + off + w / 2], [r["ref"], r["ref"]], color="k", lw=0.7, ls=(0, (2, 1.2)), zorder=5)
    ax.bar(xpos, vals, w, color=c, label=lab, yerr=[err_lo, err_hi] if ak != "baseline" else None,
           error_kw=dict(elinewidth=0.6, capsize=1.5, capthick=0.5), edgecolor="none")
ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=6)
ax.set_ylim(0, 118); ax.set_yticks(range(0, 101, 25))
ax.set_ylabel("success (%)")
ax.legend(loc="lower left", frameon=False, ncol=3, handlelength=1.0, columnspacing=1.0, borderaxespad=0.0, bbox_to_anchor=(-0.02, 1.0))
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.grid(axis="y", alpha=0.25, lw=0.4)
plt.subplots_adjust(left=0.115, right=0.995, bottom=0.2, top=0.92)
fig.savefig(os.path.join(OUT, "res_general.pdf"))
print("saved res_general.pdf")

# ----------------------------------------------------------------------------------------------------------
# extra numbers for the text: alpha sweep, detector-timing + privileged d, undisturbed detector-only, teacher v3
# ----------------------------------------------------------------------------------------------------------
extra = {}
for key, paths in [("alpha05_k2", [RR("probe_s05_k2")]), ("alpha1_k2", [RR("probe_s1_k2")]), ("alpha1_k1", [RR("probe_s1_k1")]),
                   ("alpha15_k2", [RR("probe_s15_k2")]), ("det_timing_priv_d", [RR("probe_det_s05_k2")]),
                   ("jump_scripted", [RR("probe_jump_s05_k2")]), ("student_v1", [RR("student_v1"), RR("student_v1_ns1")]),
                   ("teacher_v3", [RR("eval_res_v3_it059"), RR("eval_res_v3_it059_ns1"), RR("eval_res_v3_it059_ns2")]),
                   ("teacher_scratch_it019", [RR("abl_scratch_it019")]), ("teacher_scratch_it039", [RR("abl_scratch_it039"), RR("abl_scratch_it039_ns1")])]:
    D, miss = load(paths)
    if D:
        extra[key] = paired(D, pool)
        extra[key + "_heldout"] = paired(D, pool, HO)
# student vs vision scripted / teacher (paired, same episodes)
Ds, _ = load([RR("student_v2"), RR("student_v2_ns1"), RR("student_v2_ns2"), RR("student_v2_ns3"), RR("student_v2_ns4")])
Dv, _ = load([RR("vision_s05_k2"), RR("vision_s05_k2_ns1"), RR("vision_s05_k2_ns2")])
Dt, _ = load([RR("eval_res_v4_it039"), RR("eval_res_v4_it039_ns1"), RR("eval_res_v4_it039_ns2"), RR("eval_res_v4_it039_ns3"), RR("eval_res_v4_it039_ns4")])
Dn, _ = load([RR("abl_student_nodhat"), RR("abl_student_nodhat_ns1")])
Dsc, _ = load([RR("probe_s05_k2"), RR("probe_s05_k2_ns1"), RR("probe_s05_k2_ns2")])
extra["student_vs_vision_scripted"] = paired(Ds, Dv)
extra["student_vs_teacher_heldout"] = paired(Ds, Dt, HO)
extra["teacher_vs_scripted_heldout"] = paired(Dt, Dsc, HO)
Dscr, _ = load([RR("abl_scratch_it039"), RR("abl_scratch_it039_ns1")])
extra["teacher_vs_scratch_heldout"] = paired(Dt, Dscr, HO)
Dbc, _ = load([RR("eval_res_v4_bc")])
extra["teacher_vs_bc_heldout"] = paired(Dt, Dbc, HO)
extra["scratch_vs_bc_heldout"] = paired(Dscr, Dbc, HO)
extra["nodhat_vs_student"] = paired(Dn, Ds)
Dbn, _ = load([RR("gen_big_student_nodhat"), RR("gen_big_student_nodhat_ns1")]); Dbs, _ = load([RR("gen_big_student_v2"), RR("gen_big_student_v2_ns1")]); Dbv, _ = load([RR("gen_big_vision_s05"), RR("gen_big_vision_s05_ns1"), RR("gen_big_vision_s05_ns2")])
extra["big_nodhat_vs_student"] = paired(Dbn, Dbs)
extra["big_student_vs_vision"] = paired(Dbs, Dbv)
# event-timed replan (then n=10, oracle) vs next-query replan (oracle), same held-out episodes (Sec. V-B)
Drn, _ = load([RD("obj_react_now_n10h10_%s" % s) for s in ("s0", "ns1_s0", "ns2_s0")])
Drx, _ = load([RD("obj_react_next_n10h10_s0"), RD("obj_react_next_n10h10_ns1_s0")])
extra["react_now10_vs_next_heldout"] = paired(Drn, Drx, HO)
# 2x2 with privileged d (Sec. V-C): detector replan + scripted shift vs the shift alone at scheduled queries
# (no event replan) and vs the detector replan alone, same episodes
Dpd, _ = load([RR("probe_det_s05_k2")])
Ddet, _ = load([RD("det_n2h10_%s" % s) for s in ("s0", "ns1_s0", "ns2_s0", "ns3_s0", "ns4_s0")])
extra["detshift_vs_shift_heldout"] = paired(Dpd, Dsc, HO)
extra["detshift_vs_det_heldout"] = paired(Dpd, Ddet, HO)
extra["detshift_vs_det"] = paired(Dpd, Ddet)
# chain7 (2026-10-16): three draws of the privileged replan + shift arm, and the teacher WITH the detector replan
Dpd3, _ = load([RR("probe_det_s05_k2"), RR("probe_det_s05_k2_ns1"), RR("probe_det_s05_k2_ns2")])
Dtd, _ = load([RR("eval_res_v4_it039_det"), RR("eval_res_v4_it039_det_ns1"), RR("eval_res_v4_it039_det_ns2")])
if len(Dpd3) == 3 and len(Dtd) == 3:
    extra["detshift3"] = paired(Dpd3, pool)
    extra["detshift3_heldout"] = paired(Dpd3, pool, HO)
    extra["detshift3_vs_shift_heldout"] = paired(Dpd3, Dsc, HO)
    extra["detshift3_vs_det_heldout"] = paired(Dpd3, Ddet, HO)
    extra["detshift3_vs_det"] = paired(Dpd3, Ddet)
    extra["teacher_det_heldout"] = paired(Dtd, pool, HO)
    extra["teacher_det_vs_teacher_heldout"] = paired(Dtd, Dt, HO)
    extra["student_vs_teacher_det_heldout"] = paired(Ds, Dtd, HO)
    extra["teacher_det_vs_detshift3_heldout"] = paired(Dtd, Dpd3, HO)
# undisturbed, det only + teacher none vs cal_none heldout
Dnone, _ = load([RD("cal_none_s0")])
numbers["undisturbed_base"] = round(100 * float(np.mean([r["success"] for r in Dnone[0].values()])), 1)
numbers["undisturbed_base_heldout"] = round(100 * float(np.mean([r["success"] for k, r in Dnone[0].items() if HO(k)])), 1)
# failure stats of the pool: all failures are timeouts? gripper toggles not stored; episode length
# ---- 2026-10-16 (deadline extension): numbers behind the correctness fixes -----------------------------------------
def det_quality(paths, keys=None):
    """Closed-loop detection per draw (analyze_det.py rule): hit = a detection at steps applied_t+1..applied_t+3; every
    other detection is a false detection, counted per disturbed episode."""
    res = []
    for p in paths:
        d = L(p)
        if not d:
            continue
        ks = [k for k in d if keys is None or keys(k)]
        ev = hit = fa = 0
        for k in ks:
            r = d[k]
            at = (r.get("disturbance") or {}).get("applied_t", r.get("applied_t"))
            dets = r.get("detections") or []
            if at is None:
                fa += len(dets)
                continue
            ev += 1
            hit += any(at + 1 <= t <= at + 3 for t in dets)
            fa += sum(1 for t in dets if not (at + 1 <= t <= at + 3))
        res.append(dict(hit=round(100 * hit / ev, 1), fa=round(fa / len(ks), 3), n=len(ks)))
    return res


DETP = G("det_n2h10_*") + G("det_n10h10_*")
extra["detect_heldout"] = det_quality(DETP, HO)
extra["detect_train_inits"] = det_quality(DETP, lambda k: not HO(k))
extra["detect_all"] = det_quality(DETP)
# share of displacements applied by the time fallback (no approach before the planned step), per base draw
POOLP = [RD("obj_static_n2_h10_%s" % s) for s in ("s0", "ns1_s0", "ns2_s0", "ns3_s0", "ns4_s0", "ns5_s0", "ns6_s0")]
_fb = []
for p in POOLP:
    ev = [r["disturbance"] for r in L(p).values() if (r.get("disturbance") or {}).get("applied_t") is not None]
    _fb.append(round(100 * sum(e["applied_t"] == e["planned_t"] for e in ev) / len(ev), 1))
extra["time_fallback_pct"] = _fb
# undisturbed static grid (results/, three runs each): success and inference ms per episode
RO = lambda n: "%s/results/%s.jsonl" % (E, n)
_grid = {}
for c in ("n2_h10", "n2_h5", "n2_h2", "n10_h5", "n10_h10"):
    ds = [L(RO("static_%s_s%d" % (c, s))) for s in (0, 1, 2)]
    if all(ds):
        _grid[c] = dict(success=[round(100 * float(np.mean([r["success"] for r in d.values()])), 1) for d in ds],
                        mean=round(100 * float(np.mean([np.mean([r["success"] for r in d.values()]) for d in ds])), 1),
                        cost_ms=round(float(np.mean([np.mean([r["cost_ms"] for r in d.values()]) for d in ds])), 1))
if "n10_h5" in _grid and "n2_h10" in _grid:
    _grid["cost_ratio_n10h5_vs_n2h10"] = round(_grid["n10_h5"]["cost_ms"] / _grid["n2_h10"]["cost_ms"], 2)
extra["undisturbed_grid"] = _grid


def shift_along_d(paths, keys):
    """Mean commanded end-effector shift along the true displacement (cm): per query, the xy residual projected on the
    unit displacement, times 0.012 m per action unit per step (ScriptedShift.GAIN) and 10 executed steps."""
    v = []
    for p in paths:
        for k, r in (L(p) or {}).items():
            if not keys(k) or not r.get("true_d") or r.get("D") is None:
                continue
            u = np.asarray(r["true_d"], float)
            if np.linalg.norm(u) < 1e-6:
                continue
            u = u / np.linalg.norm(u)
            v.append(100 * 0.012 * 10 * sum(float(np.dot(q[:2], u)) for q in r["D"]))
    return dict(cm=round(float(np.mean(v)), 2), n=len(v))


_stu = [RR("student_v2")] + [RR("student_v2_ns%d" % s) for s in (1, 2, 3, 4)]
_tea = [RR("eval_res_v4_it039")] + [RR("eval_res_v4_it039_ns%d" % s) for s in (1, 2, 3, 4)]
extra["shift_along_d_heldout"] = {
    "student_4_6": shift_along_d(_stu, HO),
    "student_8_12": shift_along_d([RR("gen_big_student_v2"), RR("gen_big_student_v2_ns1")], HO),
    "teacher_4_6": shift_along_d(_tea, HO),
    "teacher_8_12": shift_along_d([RR("gen_big_teacher_v4"), RR("gen_big_teacher_v4_ns1")], HO),
    "fixed_rule_4_6": shift_along_d([RR("vision_s05_k2"), RR("vision_s05_k2_ns1"), RR("vision_s05_k2_ns2")], HO),
    "fixed_rule_8_12": shift_along_d([RR("gen_big_vision_s05")], HO)}
# regressor estimate vs true displacement at 8-12 cm (first estimate per episode, fixed-rule run, all 500)
_dh, _dt = [], []
for r in (L(RR("gen_big_vision_s05")) or {}).values():
    if r.get("preds") and r.get("true_d"):
        _dh.append(100 * float(np.hypot(r["preds"][0][1], r["preds"][0][2])))
        _dt.append(100 * float(np.hypot(*r["true_d"])))
extra["dhat_8_12_cm"] = dict(median_dhat=round(float(np.median(_dh)), 1), median_true=round(float(np.median(_dt)), 1), n=len(_dh))
# share of the student's failures in which the residual was active (table_pertask.py rule), by split
_act = lambda r: any(any(abs(x) > 1e-6 for x in q) for q in (r.get("D") or []))
_f = [(k, r) for p in _stu for k, r in (L(p) or {}).items() if not r["success"]]
extra["student_fail_active_pct"] = {
    "all": round(100 * sum(_act(r) for k, r in _f) / len(_f), 1),
    "heldout": round(100 * sum(_act(r) for k, r in _f if HO(k)) / sum(1 for k, r in _f if HO(k)), 1)}
Dsn, _ = load([RR("probe_s05_k2_none")])
extra["scripted_undisturbed"] = paired(Dsn, Dnone)
# ---- chain8 B1: fixed-rule shift over longer windows, chosen on training initial states 0-24 (bok/prereg_ext.md)
_sel1 = json.load(open(E + "/bok/chain8_b1_stage1.json"))
_sel2 = json.load(open(E + "/bok/chain8_b1_stage2.json"))
_w = _sel2["winner"]
_wt = "a" + _w.replace(",", "_k")
extra["b1_selection"] = dict(stage1_train_inits=_sel1, stage2_two_draw_mean_train_inits=_sel2["two_draw_mean_train_inits"], winner=_w)
if _w != "0.5,2":
    Dvt, _ = load([RR("vision_%s" % _wt), RR("vision_%s_ns1" % _wt), RR("vision_%s_ns2" % _wt)])
    extra["vision_tuned"] = paired(Dvt, pool)
    extra["vision_tuned_heldout"] = paired(Dvt, pool, HO)
    extra["student_vs_vision_tuned"] = paired(Ds, Dvt)
    extra["student_vs_vision_tuned_heldout"] = paired(Ds, Dvt, HO)
    extra["vision_tuned_vs_vision"] = paired(Dvt, Dv)
    extra["vision_tuned_undisturbed"] = paired(load([RR("vision_%s_none" % _wt)])[0], Dnone)
    Dbvt, _ = load([RR("gen_big_vision_%s" % _wt), RR("gen_big_vision_%s_ns1" % _wt)])
    extra["big_vision_tuned"] = paired(Dbvt, load(G("objbig_static_n2_h10*"))[0])
    extra["big_student_vs_vision_tuned"] = paired(Dbs, Dbvt)
# ---- chain8 B6: openpi's default (10,5) under the disturbance, paired with (10,5) undisturbed on the same episodes
D105, _ = load([RD("obj_static_n10_h5_s0"), RD("obj_static_n10_h5_ns1_s0")])
D105u, _ = load([RO("static_n10_h5_s%d" % s) for s in (0, 1, 2)])
if D105 and D105u:
    extra["n10h5_disturbed"] = dict(success=[round(100 * float(np.mean([r["success"] for r in d.values()])), 1) for d in D105],
                                    mean=round(100 * float(np.mean([np.mean([r["success"] for r in d.values()]) for d in D105])), 1))
    extra["n10h5_disturbed_vs_undisturbed"] = paired(D105, D105u)
    extra["n2h10_disturbed_vs_undisturbed"] = paired(pool, load([RO("static_n2_h10_s%d" % s) for s in (0, 1, 2)])[0])
# ---- chain8 B2: DAgger seed repeats (seed 0 = released student_v2 / student_v2_nodhat)
_BIGREF = load(G("objbig_static_n2_h10*"))[0]
_seed = {0: dict(stu=[RR("student_v2")] + [RR("student_v2_ns%d" % k) for k in (1, 2, 3, 4)], none=[RR("student_v2_none")],
                 big=[RR("gen_big_student_v2"), RR("gen_big_student_v2_ns1")],
                 nod=[RR("abl_student_nodhat"), RR("abl_student_nodhat_ns1")],
                 nodbig=[RR("gen_big_student_nodhat"), RR("gen_big_student_nodhat_ns1")])}
for _k in (1, 2):
    _seed[_k] = dict(stu=[RR("student_v2_s%d" % _k), RR("student_v2_s%d_ns1" % _k)], none=[RR("student_v2_s%d_none" % _k)],
                     big=[RR("gen_big_student_v2_s%d" % _k)], nod=[RR("abl_student_nodhat_s%d" % _k)],
                     nodbig=[RR("gen_big_student_nodhat_s%d" % _k)])
_ds = {}
for _k, f in _seed.items():
    L5 = {n: load(p)[0] for n, p in f.items()}
    _ds[str(_k)] = dict(student=paired(L5["stu"], pool), student_heldout=paired(L5["stu"], pool, HO),
                        undisturbed=paired(L5["none"], Dnone), big=paired(L5["big"], _BIGREF),
                        nodhat=paired(L5["nod"], pool), nodhat_big=paired(L5["nodbig"], _BIGREF),
                        nodhat_vs_student=paired(L5["nod"], L5["stu"]), big_nodhat_vs_student=paired(L5["nodbig"], L5["big"]))
_m = lambda key: round(float(np.mean([_ds[k][key]["success"] for k in _ds])), 1)
_rng = lambda key: [min(_ds[k][key]["success"] for k in _ds), max(_ds[k][key]["success"] for k in _ds)]
extra["dagger_seeds"] = dict(per_seed=_ds, mean=dict((key, _m(key)) for key in ("student", "student_heldout", "undisturbed", "big", "nodhat", "nodhat_big")),
                             range=dict((key, _rng(key)) for key in ("student", "student_heldout", "undisturbed", "big", "nodhat", "nodhat_big")),
                             big_nodhat_minus_student=[_ds[k]["big_nodhat_vs_student"]["delta"] for k in sorted(_ds)],
                             nodhat_minus_student=[_ds[k]["nodhat_vs_student"]["delta"] for k in sorted(_ds)])
# ---- sources of two numbers quoted in IV-A / IV-B / Conclusion (2026-10-17 verification)
_tv3, _ = load([RR("eval_res_v3_it059_none")])   # teacher v3 with the cumulative 2-cm gate, undisturbed, held out
if _tv3:
    extra["teacher_cumgate_undisturbed_heldout"] = paired(_tv3, Dnone, HO)
extra["detector_auroc_heldout"] = round(json.load(open(E + "/bok/detector_v1.pt.json"))["auroc"], 4)
# ---- 2026-10-17 quality revision -----------------------------------------------------------------------------
# held-out student vs the fixed-rule shift 0.5 d-hat (contribution 3 and Sec. V-C lead with held-out numbers)
extra["student_vs_vision_scripted_heldout"] = paired(Ds, Dv, HO)
# trigger of each episode's displacement, by majority over the seven base draws: approach (gripper within 12 cm)
# or time fallback (applied at the planned step); gains split by it on the held-out episodes (Sec. V-F)
_trig = {}
for k in pool[0]:
    _ev = [d[k]["disturbance"] for d in pool if (d[k].get("disturbance") or {}).get("applied_t") is not None]
    _trig[k] = "fallback" if 2 * sum(e["applied_t"] == e["planned_t"] for e in _ev) > len(_ev) else "approach"
_APP = lambda k: _trig.get(k) == "approach"
_FB = lambda k: _trig.get(k) == "fallback"
extra["trigger_counts"] = dict(all_approach=sum(map(_APP, pool[0])), all_fallback=sum(map(_FB, pool[0])),
                               heldout_approach=sum(1 for k in pool[0] if HO(k) and _APP(k)),
                               heldout_fallback=sum(1 for k in pool[0] if HO(k) and _FB(k)))
extra["trigger_split_heldout"] = {
    name: dict(approach=paired(D, pool, lambda k: HO(k) and _APP(k)), fallback=paired(D, pool, lambda k: HO(k) and _FB(k)))
    for name, D in (("student", Ds), ("teacher", Dt), ("fixed_rule", Dv), ("detector_replan", Ddet))}
# policy-query cost of the detector replan (Sec. IV-A): query time per episode summed from the measured per-query
# latency (cost_ms), relative to the base pool on the same 500 episodes; detections per disturbed episode; queries
# per 100 control steps; mean episode length
Ddet10, _ = load([RD("det_n10h10_%s" % s) for s in ("s0", "ns1_s0", "ns2_s0")])
_ks = sorted(pool[0])
def _qcost(D):
    m = lambda f: float(np.mean([np.mean([f(d[k]) for d in D]) for k in _ks]))
    return dict(cost_ms=round(m(lambda r: r["cost_ms"]), 1), queries=round(m(lambda r: r["n_calls"]), 2),
                queries_per_100_steps=round(m(lambda r: 100.0 * r["n_calls"] / r["episode_length"]), 2),
                length=round(m(lambda r: r["episode_length"]), 1),
                detections=round(m(lambda r: len(r.get("detections") or [])), 2), draws=len(D))
_qb, _qd, _qd10 = _qcost(pool), _qcost(Ddet), _qcost(Ddet10)
extra["query_cost"] = dict(base=_qb, detector_replan=_qd, detector_replan_n10=_qd10,
                           ratio_detector=round(_qd["cost_ms"] / _qb["cost_ms"], 2),
                           ratio_detector_n10=round(_qd10["cost_ms"] / _qb["cost_ms"], 2),
                           queries_per_step_increase_pct=round(100 * (_qd["queries_per_100_steps"] / _qb["queries_per_100_steps"] - 1), 1))
# each static schedule's drop under the displacement from its own undisturbed success (undisturbed grid, three
# runs, vs the disturbed all-500 success); quoted as "29--33 points" in Sec. V-A
_gm = lambda c: float(np.mean(_grid[c]["success"]))
_dist = {"n2_h10": numbers["pool_all"], "n2_h5": forest["static_2_5"]["all"]["success"], "n2_h2": forest["static_2_2"]["all"]["success"],
         "n10_h10": forest["static_10_10"]["all"]["success"], "n10_h5": extra["n10h5_disturbed"]["mean"]}
extra["schedule_drop_vs_undisturbed"] = {c: dict(undisturbed=round(_gm(c), 1), disturbed=v, drop=round(_gm(c) - v, 1)) for c, v in _dist.items()}
numbers["extra"] = extra
# scripted alpha table
numbers["alpha"] = {k: extra[k] for k in ("alpha05_k2", "alpha1_k2", "alpha1_k1", "alpha15_k2") if k in extra}

json.dump(numbers, open(os.path.join(OUT, "res_numbers.json"), "w"), indent=1)
print(json.dumps({k: {kk: (vv["delta"], vv["lo"], vv["hi"], vv["success"], vv["draws"], vv["n"]) if isinstance(vv, dict) and "delta" in vv else vv
                      for kk, vv in v.items()} if isinstance(v, dict) else v for k, v in numbers.items() if k not in ("extra",)}, indent=0)[:6000])
print("--- extra")
for k, v in extra.items():
    if not (isinstance(v, dict) and all(x in v for x in ("success", "ref", "delta", "lo", "hi"))):
        print("%-32s %s" % (k, json.dumps(v)[:200]))
        continue
    print("%-32s %5.1f vs %5.1f  %+.1f [%+.1f, %+.1f]  draws %d n %d active %.2f" % (k, v["success"], v["ref"], v["delta"], v["lo"], v["hi"], v["draws"], v["n"], v["active"]))
