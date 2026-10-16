#!/usr/bin/env bash
# chain8c (2026-10-16): the seed-2 half of B2 in bok/prereg_ext.md (student and no-d-hat student distilled with
# student_distill.py --seed 2, and their evaluations) on a second server of ours (port 8002), in parallel with chain8 on
# 8001. chain8 runs seed 1 first and reaches seed 2 only after B1, B5 and the whole seed-1 block (6 h or more); it skips
# checkpoints and result files that already exist, so the two chains do not run the same item.
# Stops its own server when done. Never touches port 8000 or tmux rss_* (vla-rss-2027-e01).
source ${VLA_ENV:-$HOME/disk2/vla-env}/env.sh
E=${VLA_EXPERIMENTS:-$HOME/disk2/vla-inference-scheduling/experiments}; cd $E; mkdir -p logs/res results_res ckpt
until grep -q "server 8002 up" logs/chain8c_servers.log 2>/dev/null; do sleep 10; done
echo "$(date "+%F %T") server 8002 up, starting B2 seed 2"
DET="--schedule react_step --detector bok/detector_v1_t24.pt --step_tau 1.628 --react_setting 2,10"
DR="--dispreg bok/dispreg_v2_t24.pt"
ALL="--eval_init_lo 0 --eval_init_hi 50"
BIG="--disturb_mag 0.08,0.12"
G="--gate jump"
run() { dir=$1; name=$2; shift 2; [ -f $dir/$name.jsonl ] && { echo "skip $name"; return; }
  $LIBERO_PY residual_rl.py --ports 8002 --workers_per_port 6 --eval_out $dir/$name.jsonl "$@" > logs/res/$name.log 2>&1 \
    && echo "$(date "+%F %T") done $name $(grep "^eval" logs/res/$name.log)" || echo "$(date "+%F %T") FAILED $name"; }
distill() { out=$1; shift; [ -f $out ] && { echo "skip $out"; return; }
  $LIBERO_PY student_distill.py --teacher ckpt/res_v4.pt.it039 $DR $DET $G --ports 8002 --workers_per_port 6 --reps 2 \
    --out $out "$@" > logs/res/$(basename $out).log 2>&1 && echo "$(date "+%F %T") distilled $out" || echo "$(date "+%F %T") DISTILL FAILED $out"; }
S=2
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
tmux kill-session -t iceic8c_srv 2>/dev/null; sleep 15
PID=$(ss -ltnp 2>/dev/null | grep ":8002 " | sed -n "s/.*pid=\([0-9]*\).*/\1/p" | head -1)
[ -n "$PID" ] && kill $PID && echo "killed leftover server pid $PID on 8002"
echo "$(date "+%F %T") CHAIN8C_COMPLETE"
