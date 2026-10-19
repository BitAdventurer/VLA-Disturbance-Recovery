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
