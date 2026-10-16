#!/usr/bin/env bash
# chain8 (2026-10-16, deadline extension to 10-30): pre-registered runs of bok/prereg_ext.md.
#  B4 Table II on all 500 (teacher and scripted shift on inits 0-24; cumulative-gate scripted shift undisturbed)
#  B1 fixed-rule shift over longer windows, selected on training initial states 0-24
#  B5 more draws for single-draw arms (8-12 cm, late trigger)
#  B2 DAgger seed repeats of the student and the no-d-hat student
#  B6 base policy at openpi's default (10,5) under the disturbance
# Uses ONE policy server of ours on port 8001 (tmux iceic8_srv; two servers filled the shared GPU to 32 of 32.6 GB). The vla-rss-2027-e01 work stream keeps its
# own server on 8000 (tmux rss_*): never touch it. Our servers are stopped at the end.
source ${VLA_ENV:-$HOME/disk2/vla-env}/env.sh
E=${VLA_EXPERIMENTS:-$HOME/disk2/vla-inference-scheduling/experiments}; cd $E; mkdir -p logs/res results_res results_dist ckpt
until grep -q "server 8001 up" logs/chain8_servers.log 2>/dev/null; do sleep 10; done
echo "$(date "+%F %T") server 8001 up, starting"
P="--ports 8001 --workers_per_port 6"
DET="--schedule react_step --detector bok/detector_v1_t24.pt --step_tau 1.628 --react_setting 2,10"
DR="--dispreg bok/dispreg_v2_t24.pt"
ALL="--eval_init_lo 0 --eval_init_hi 50"
TR="--eval_init_lo 0 --eval_init_hi 25"
BIG="--disturb_mag 0.08,0.12"
LATE="--disturb_trigger time --disturb_window 100,200 --disturb_mag 0.04,0.06"
G="--gate jump"
run() { dir=$1; name=$2; shift 2; [ -f $dir/$name.jsonl ] && { echo "skip $name"; return; }
  $LIBERO_PY residual_rl.py $P --eval_out $dir/$name.jsonl "$@" > logs/res/$name.log 2>&1 \
    && echo "$(date "+%F %T") done $name $(grep "^eval" logs/res/$name.log)" || echo "$(date "+%F %T") FAILED $name"; }
distill() { out=$1; shift; [ -f $out ] && { echo "skip $out"; return; }
  $LIBERO_PY student_distill.py --teacher ckpt/res_v4.pt.it039 $DR $DET $G --ports 8001 --workers_per_port 6 --reps 2 \
    --out $out "$@" > logs/res/$(basename $out).log 2>&1 && echo "$(date "+%F %T") distilled $out" || echo "$(date "+%F %T") DISTILL FAILED $out"; }

# ---- B4: Table II on all 500 episodes
for s in 0 1 2 3 4; do sfx=$([ $s = 0 ] && echo "" || echo "_ns$s"); run results_res eval_res_v4_it039_tr$sfx --eval ckpt/res_v4.pt.it039 $G $TR --eval_noise_salt $s; done
run results_res eval_res_v4_it039_tr_none --eval ckpt/res_v4.pt.it039 $G $TR --eval_disturb none
for s in 1 2; do run results_res probe_s05_k2_tr_ns$s --eval scripted:0.5,2 $TR --eval_noise_salt $s; done
run results_res probe_s05_k2_none --eval scripted:0.5,2 $ALL --eval_disturb none
echo "$(date "+%F %T") B4_COMPLETE"

# ---- B1: fixed-rule window, selected on training initial states 0-24
for AK in 0.5,4 1.0,4 0.5,6 0.75,6 1.0,6 1.5,6; do t=${AK/,/_k}; run results_res tune_vis_a${t}_tr --eval vision:$AK $DR $DET $TR; done
TOP2=$(python3 - <<"PY"
import json
def succ(p, lo=0, hi=25):
    v = [float(r["success"]) for r in map(json.loads, open(p)) if "success" in r and lo <= r["init_state_id"] < hi]
    return sum(v) / len(v), len(v)
cands = {"0.5,2": "results_res/vision_s05_k2.jsonl"}
for ak in ["0.5,4", "1.0,4", "0.5,6", "0.75,6", "1.0,6", "1.5,6"]:
    cands[ak] = "results_res/tune_vis_a%s_tr.jsonl" % ak.replace(",", "_k")
st = {ak: succ(p) for ak, p in cands.items()}
order = sorted(st, key=lambda ak: (-st[ak][0], int(ak.split(",")[1]), float(ak.split(",")[0])))
print(" ".join(order[:2]))
open("logs/chain8_b1_stage1.json", "w").write(json.dumps({ak: st[ak] for ak in order}))
PY
)
echo "$(date "+%F %T") B1 stage 1 top two: $TOP2 (logs/chain8_b1_stage1.json)"
for AK in $TOP2; do [ "$AK" = "0.5,2" ] && continue; t=${AK/,/_k}; run results_res tune_vis_a${t}_tr_ns1 --eval vision:$AK $DR $DET $TR --eval_noise_salt 1; done
WIN=$(python3 - $TOP2 <<"PY"
import json, sys
def succ(p):
    v = [float(r["success"]) for r in map(json.loads, open(p)) if "success" in r and r["init_state_id"] < 25]
    return sum(v) / len(v)
res = {}
for ak in sys.argv[1:]:
    if ak == "0.5,2":
        fs = ["results_res/vision_s05_k2.jsonl", "results_res/vision_s05_k2_ns1.jsonl"]
    else:
        t = ak.replace(",", "_k"); fs = ["results_res/tune_vis_a%s_tr.jsonl" % t, "results_res/tune_vis_a%s_tr_ns1.jsonl" % t]
    res[ak] = sum(succ(f) for f in fs) / 2
order = sorted(res, key=lambda ak: (-res[ak], int(ak.split(",")[1]), float(ak.split(",")[0])))
open("logs/chain8_b1_stage2.json", "w").write(json.dumps({"two_draw_mean_train_inits": res, "winner": order[0]}))
print(order[0])
PY
)
echo "$(date "+%F %T") B1 winner: $WIN (logs/chain8_b1_stage2.json)"
if [ "$WIN" != "0.5,2" ]; then
  t=${WIN/,/_k}
  for s in 0 1 2; do sfx=$([ $s = 0 ] && echo "" || echo "_ns$s"); run results_res vision_a${t}$sfx --eval vision:$WIN $DR $DET $ALL --eval_noise_salt $s; done
  run results_res vision_a${t}_none --eval vision:$WIN $DR $DET $ALL --eval_disturb none
  for s in 0 1; do sfx=$([ $s = 0 ] && echo "" || echo "_ns$s"); run results_res gen_big_vision_a${t}$sfx --eval vision:$WIN $DR $DET $ALL $BIG --eval_noise_salt $s; done
fi
echo "$(date "+%F %T") B1_COMPLETE"

# ---- B5: more draws for single-draw arms
for s in 1 2; do run results_res gen_big_vision_s05_ns$s --eval vision:0.5,2 $DR $DET $ALL $BIG --eval_noise_salt $s; done
run results_res gen_big_student_nodhat_ns1 --eval student:ckpt/student_v2_nodhat.pt $DR $DET $G $ALL $BIG --eval_noise_salt 1
run results_dist objbig_static_n2_h10_ns2_s0 --eval zero $ALL $BIG --eval_noise_salt 2
run results_res gen_late_teacher_v4_ns1 --eval ckpt/res_v4.pt.it039 $G $ALL $LATE --eval_noise_salt 1
run results_res gen_late_student_v2_ns1 --eval student:ckpt/student_v2.pt $DR $DET $G $ALL $LATE --eval_noise_salt 1
run results_dist objlate_static_n2_h10_ns1_s0 --eval zero $ALL $LATE --eval_noise_salt 1
echo "$(date "+%F %T") B5_COMPLETE"

# ---- B2: DAgger seed repeats
for S in 1 2; do
  distill ckpt/student_v2_s$S.pt --seed $S
  distill ckpt/student_v2_nodhat_s$S.pt --seed $S --no_dhat
  STU="--eval student:ckpt/student_v2_s$S.pt $DR $DET $G"
  NOD="--eval student:ckpt/student_v2_nodhat_s$S.pt $DR $DET $G"
  run results_res student_v2_s$S $STU $ALL
  run results_res student_v2_s${S}_ns1 $STU $ALL --eval_noise_salt 1
  run results_res student_v2_s${S}_none $STU $ALL --eval_disturb none
  run results_res gen_big_student_v2_s$S $STU $ALL $BIG
  run results_res abl_student_nodhat_s$S $NOD $ALL
  run results_res gen_big_student_nodhat_s$S $NOD $ALL $BIG
done
echo "$(date "+%F %T") B2_COMPLETE"

# ---- B6: openpi's default (10,5) under the disturbance
for s in 0 1; do sfx=$([ $s = 0 ] && echo "_s0" || echo "_ns${s}_s0"); run results_dist obj_static_n10_h5$sfx --eval zero --n 10 --h 5 $ALL --eval_noise_salt $s; done
echo "$(date "+%F %T") B6_COMPLETE"

tmux kill-session -t iceic8_srv 2>/dev/null; sleep 15
for p in 8001; do PID=$(ss -ltnp 2>/dev/null | grep ":$p " | sed -n "s/.*pid=\([0-9]*\).*/\1/p" | head -1); [ -n "$PID" ] && kill $PID && echo "killed leftover server pid $PID on $p"; done
echo "$(date "+%F %T") CHAIN8_COMPLETE"
