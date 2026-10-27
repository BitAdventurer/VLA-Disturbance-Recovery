#!/usr/bin/env python3
"""verify_replays.py VIDEO_DIR [OUT.json]: checks that every replay written by video_frames.py reproduces the episode
reported in the evaluation result files (same success, episode length and detector firings), so the videos show the
evaluated episodes. Run from the experiments directory."""
import json
import os
import sys

RES = {  # condition -> arm -> result file of the s0 draw used by the paper
    "obj": dict(base="results_dist/obj_static_n2_h10_s0.jsonl", replan="results_dist/det_n2h10_s0.jsonl",
                student="results_res/student_v2.jsonl"),
    "none": dict(base="results_dist/cal_none_s0.jsonl", replan="results_dist/det_none_n2h10_s0.jsonl",
                 student="results_res/student_v2_none.jsonl"),
    "big": dict(base="results_dist/objbig_static_n2_h10_s0.jsonl", replan="results_dist/objbig_det_n2h10_s0.jsonl",
                student="results_res/gen_big_student_v2.jsonl"),
}


def load(f):
    return {(r["task_id"], r["init_state_id"]): r for r in map(json.loads, open(f)) if "success" in r and not r.get("error")}


V = sys.argv[1]
rows, ok_all = [], True
for cond, files in RES.items():
    cd = os.path.join(V, cond)
    if not os.path.isdir(cd):
        continue
    for d in sorted(os.listdir(cd)):
        mp = os.path.join(cd, d, "meta.json")
        if not os.path.exists(mp):
            continue
        m = json.load(open(mp))
        r = load(files[m["arm"]])[(m["task_id"], m["init"])]
        got = (float(m["success"]), int(m["episode_length"]), list(m.get("detections") or []))
        want = (float(r["success"]), int(r["episode_length"]), list(r.get("detections") or []) if "detections" in r else got[2])
        ok = got == want
        ok_all &= ok
        rows.append(dict(condition=cond, task=m["task_id"], init=m["init"], arm=m["arm"], success=got[0], length=got[1],
                         detections=got[2], matches_result_file=ok, result_file=files[m["arm"]]))
        print("%-5s task %d init %2d %-7s success %.0f len %3d det %-12s %s" % (cond, m["task_id"], m["init"], m["arm"],
              got[0], got[1], got[2], "OK" if ok else "MISMATCH %s" % (want,)))
print("all match" if ok_all else "SOME MISMATCH")
if len(sys.argv) > 2:
    json.dump(rows, open(sys.argv[2], "w"), indent=1)
