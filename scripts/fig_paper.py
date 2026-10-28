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
                     "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
                     # math labels in the text font, not matplotlib's default DejaVu
                     "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.sf": "Arial",
                     "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold", "mathtext.fallback": "stixsans"})
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
    ("react_next", "next query, then $n\\,{=}\\,10$ (oracle)", [RD("obj_react_next_n10h10_s0"), RD("obj_react_next_n10h10_ns1_s0")], "timing", False),
    ("react_now_2", "at the event, $(2,10)$ (oracle)", [RD("obj_react_now_n2h10_%s" % s) for s in ("s0", "ns1_s0", "ns2_s0")], "timing", False),
    ("react_now_10", "at the event, then $n\\,{=}\\,10$ (oracle)", [RD("obj_react_now_n10h10_%s" % s) for s in ("s0", "ns1_s0", "ns2_s0")], "timing", False),
    ("det_2", "detector $\\rightarrow$ replan, $(2,10)$", [RD("det_n2h10_%s" % s) for s in ("s0", "ns1_s0", "ns2_s0", "ns3_s0", "ns4_s0")], "timing", False),
    ("det_10", "detector $\\rightarrow$ replan, then $n\\,{=}\\,10$", [RD("det_n10h10_%s" % s) for s in ("s0", "ns1_s0", "ns2_s0")], "timing", False),
    ("scripted", "scripted shift $0.5\\,d$ (priv.)", [RR("probe_s05_k2")] + [(RR("probe_s05_k2_ns%d" % s), RR("probe_s05_k2_tr_ns%d" % s)) for s in (1, 2)], "residual", False),
    ("vision_scripted", "fixed-rule shift $0.5\\,\\hat{d}$", [RR("vision_s05_k2"), RR("vision_s05_k2_ns1"), RR("vision_s05_k2_ns2")], "residual", False),
    ("teacher_bc", "teacher: BC only (priv.)", [RR("eval_res_v4_bc")], "residual", True),
    ("teacher_scratch", "teacher: PPO from scratch (priv.)", [RR("abl_scratch_it039"), RR("abl_scratch_it039_ns1")], "residual", True),
    ("teacher", "teacher: BC + PPO (priv.)", TEACHER_DRAWS, "residual", False),
    ("student_nodhat", "student without $\\hat{d}$", [RR("abl_student_nodhat"), RR("abl_student_nodhat_ns1")], "residual", True),
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
    ("push", "arm push", G("push_static_n2_h10*"), dict(student=[RR("gen_push_student_v2")], teacher=[RR("gen_push_teacher_v4")])),
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
for i, (ak, lab, c, off) in enumerate([("baseline", "π$_{0.5}$ alone", GREY, -w),
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
# policy-query time of the detector replan (Sec. IV-A): cost_ms = the episode's queries times the per-query round-trip
# latency measured offline for each integration setting (latency.json), an accounting value (simulation is synchronous),
# not a timed control loop; relative to the base pool on the same 500 episodes; detections per disturbed episode;
# queries per 100 control steps; mean episode length
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
# ---- 2026-10-18 external review: scope of "same disturbance", undisturbed reference, 8-12 cm split at 10 cm --------
# (1) event identity: each method run vs the base draw with the same noise salt (event step, displaced object, vector)
def _ev(r):
    d = r.get("disturbance")
    if d is not None:
        if d.get("applied_t") is None:
            return None
        return d["applied_t"], d.get("object"), (d["mag"] * d["dir"][0], d["mag"] * d["dir"][1])
    if r.get("applied_t") is None:
        return None
    td = r.get("true_d")
    return r["applied_t"], r.get("object"), (None if td is None else (td[0], td[1]))
def _salt(path):
    q = path[0] if isinstance(path, tuple) else path
    b = os.path.basename(q)
    return int(b.split("_ns")[1][0]) if "_ns" in b else 0
def _same_event(a, b):
    if a is None or b is None:
        return a is None and b is None
    return a[0] == b[0] and a[1] == b[1] and (a[2] is None or b[2] is None or
                                              max(abs(a[2][0] - b[2][0]), abs(a[2][1] - b[2][1])) < 1e-4)
_evid = {}
for key, label, paths, group, hollow in arms:
    if key.startswith("static_"):
        continue
    for p in paths:
        d = L(p)
        if not d:
            continue
        ref = pool[_salt(p)]
        ks = [k for k in d if k in ref]
        _evid["%s|%s" % (key, os.path.basename(p[0] if isinstance(p, tuple) else p))] = dict(
            n=len(ks), same=sum(_same_event(_ev(d[k]), _ev(ref[k])) for k in ks), salt=_salt(p))
_pairs = []
for i in range(len(pool)):
    for j in range(i + 1, len(pool)):
        ks = [k for k in pool[i] if k in pool[j] and _ev(pool[i][k]) and _ev(pool[j][k])]
        dt = [abs(_ev(pool[i][k])[0] - _ev(pool[j][k])[0]) for k in ks]
        _pairs.append(dict(diff_pct=100.0 * float(np.mean([x > 0 for x in dt])), median_shift_if_diff=float(np.median([x for x in dt if x > 0])),
                           object_or_vector_diff=sum(not (_ev(pool[i][k])[1] == _ev(pool[j][k])[1]) for k in ks)))
_static = {}
for name in ("obj_static_n2_h5_s0", "obj_static_n2_h2_s0", "obj_static_n10_h10_s0", "obj_static_n10_h5_s0", "obj_static_n10_h5_ns1_s0"):
    d = L(RD(name))
    if not d:
        continue
    ref = pool[_salt(RD(name))]
    ks = [k for k in d if k in ref and _ev(d[k]) and _ev(ref[k])]
    dt = [abs(_ev(d[k])[0] - _ev(ref[k])[0]) for k in ks]
    _static[name] = dict(n=len(ks), diff_pct=round(100.0 * float(np.mean([x > 0 for x in dt])), 1),
                         median_shift_if_diff=float(np.median([x for x in dt if x > 0])),
                         object_diff=sum(_ev(d[k])[1] != _ev(ref[k])[1] for k in ks))
extra["event_identity"] = dict(
    method_runs=_evid, method_runs_all_identical=all(v["same"] == v["n"] for v in _evid.values()),
    base_pairs=dict(n_pairs=len(_pairs), diff_pct_mean=round(float(np.mean([q["diff_pct"] for q in _pairs])), 1),
                    diff_pct_range=[round(min(q["diff_pct"] for q in _pairs), 1), round(max(q["diff_pct"] for q in _pairs), 1)],
                    median_shift_if_diff=float(np.median([q["median_shift_if_diff"] for q in _pairs])),
                    object_diff_total=sum(q["object_or_vector_diff"] for q in _pairs)),
    static_schedules=_static,
    static_diff_pct_range=[min(v["diff_pct"] for v in _static.values()), max(v["diff_pct"] for v in _static.values())] if _static else None)
# (2) undisturbed: the base row of Table II (cal_none_s0) shares seed 0 with results/static_n2_h10_s0, so the
# independent undisturbed reference is the three grid draws s0-s2 (mean 95.1); gains against it and same-seed flips
_G3, _ = load([RO("static_n2_h10_s%d" % s) for s in (0, 1, 2)])
_u = dict(student=load([RR("student_v2_none")])[0], detector=load([RD("det_none_n2h10_s0")])[0],
          teacher=load([(RR("eval_res_v4_it039_none"), RR("eval_res_v4_it039_tr_none"))])[0],
          fixed_rule=load([RR("vision_s05_k2_none")])[0])
_cs = Dnone[0]; _g0 = L(RO("static_n2_h10_s0"))
extra["undisturbed_reference"] = dict(
    grid3_mean=_grid["n2_h10"]["mean"], same_seed_base=numbers["undisturbed_base"],
    seed0_files_differ=sum(_cs[k]["success"] != _g0[k]["success"] for k in _cs if k in _g0),
    vs_grid3={n: paired(D, _G3) for n, D in _u.items() if D},
    vs_same_seed_flips={n: dict(gained=paired(D, Dnone)["wins"], lost=paired(D, Dnone)["losses"]) for n, D in _u.items() if D})
# (3) 8-12 cm split at the regressor's 10 cm training limit, by the planned magnitude (same in every draw)
_b0 = L(RD("objbig_static_n2_h10_s0"))
_GT10 = lambda k: _b0[k]["disturbance"]["mag"] > 0.10
extra["big_split_10cm"] = dict(
    n_gt10=sum(map(_GT10, _b0)), n_gt10_heldout=sum(1 for k in _b0 if _GT10(k) and HO(k)), base_draws=len(_BIGREF),
    student_le10=paired(Dbs, _BIGREF, lambda k: not _GT10(k)), student_gt10=paired(Dbs, _BIGREF, _GT10),
    student_gt10_heldout=paired(Dbs, _BIGREF, lambda k: _GT10(k) and HO(k)),
    fixed_rule_gt10=paired(Dbv, _BIGREF, _GT10), fixed_rule_gt10_heldout=paired(Dbv, _BIGREF, lambda k: _GT10(k) and HO(k)))
# ---- 2026-10-19 one-time fresh-initial-state test (experiments/bok/prereg_fresh50.md): LIBERO-10 .init states 50-99,
# every arm through residual_rl --eval --init_file full; files in results_test/ (never read by the HO filter above).
# Pre-registered endpoints: P1 student - base (five-draw reference); S1 student - fixed rule; S2 fixed rule - base;
# S3 detector replan - base; S4 undisturbed student - undisturbed base (same salt: flips, episodes where the student
# acted); S5 (exploratory) P1 and S1 split by trigger; sensitivity: P1 without the three settle outliers.
# Explicit file lists (no globs); skipped until the primary files exist.
RT = lambda n: "%s/results_test/%s.jsonl" % (E, n)
_FRESH_KEYS = {(t, i) for t in range(10) for i in range(50, 100)}
_FRESH_SETTLE_OUTLIERS = {(1, 87), (6, 52), (7, 76)}
_fr_names = dict(base=["fresh_base_s0", "fresh_base_ns1", "fresh_base_ns2", "fresh_base_ns3", "fresh_base_ns4"],
                 student=["fresh_student_s0", "fresh_student_ns1", "fresh_student_ns2"],
                 fixed=["fresh_fixed_s0", "fresh_fixed_ns1", "fresh_fixed_ns2"],
                 det=["fresh_det_s0", "fresh_det_ns1"],
                 none_base=["fresh_none_base_s0"], none_student=["fresh_none_student_s0"])
if all(os.path.exists(RT(n)) for n in _fr_names["base"] + _fr_names["student"] + _fr_names["fixed"]):
    _fr = {k: load([RT(n) for n in v])[0] for k, v in _fr_names.items()}
    _fr_rows = {}
    for k, v in _fr_names.items():
        for n in v:
            d = L(RT(n))
            _fr_rows[n] = dict(rows=0 if d is None else len(d), complete=d is not None and set(d) == _FRESH_KEYS)
    fresh = dict(files=_fr_rows,
                 success={k: [round(100 * float(np.mean([r["success"] for r in d.values()])), 1) for d in D] for k, D in _fr.items()},
                 base_mean=round(100 * float(np.mean([np.mean([d[k]["success"] for d in _fr["base"]]) for k in _FRESH_KEYS
                                                      if all(k in d for d in _fr["base"])])), 1))
    fresh["P1_student_vs_base"] = paired(_fr["student"], _fr["base"])
    fresh["S1_student_vs_fixed"] = paired(_fr["student"], _fr["fixed"])
    fresh["S2_fixed_vs_base"] = paired(_fr["fixed"], _fr["base"])
    if _fr["det"]:
        fresh["S3_det_vs_base"] = paired(_fr["det"], _fr["base"])
    if _fr["none_base"] and _fr["none_student"]:
        fresh["S4_undisturbed_student_vs_base"] = paired(_fr["none_student"], _fr["none_base"])
    # S5: trigger of each fresh episode, by majority over the five base draws: time fallback if applied_t equals the
    # planned step recomputed from the key (run_libero.disturbance_plan with the residual_rl.py defaults), else approach
    sys.path.insert(0, E)
    import run_libero as _RL
    _pa = _RL.build_parser().parse_args([])
    _pa.disturb, _pa.disturb_window, _pa.disturb_mag = "object", "40,160", "0.04,0.06"
    _ftrig = {}
    for k in _FRESH_KEYS:
        ev = [d[k] for d in _fr["base"] if k in d and d[k].get("applied_t") is not None]
        if not ev:
            continue
        planned = _RL.disturbance_plan(_pa, tuple(ev[0]["key"]))["t"]
        _ftrig[k] = "fallback" if 2 * sum(r["applied_t"] == planned for r in ev) > len(ev) else "approach"
    fresh["S5_trigger_counts"] = dict(approach=sum(v == "approach" for v in _ftrig.values()),
                                      fallback=sum(v == "fallback" for v in _ftrig.values()))
    fresh["S5_trigger_split"] = {
        name: {tr: paired(A, B, lambda k, tr=tr: _ftrig.get(k) == tr) for tr in ("approach", "fallback")
               if any(v == tr for v in _ftrig.values())}                    # an empty group is left out
        for name, A, B in (("student_vs_base", _fr["student"], _fr["base"]), ("student_vs_fixed", _fr["student"], _fr["fixed"]))}
    fresh["P1_without_settle_outliers"] = paired(_fr["student"], _fr["base"], lambda k: k not in _FRESH_SETTLE_OUTLIERS)
    extra["fresh50"] = fresh
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
