#!/usr/bin/env bash
source ${VLA_ENV:-$HOME/disk2/vla-env}/env.sh
E=${VLA_EXPERIMENTS:-$HOME/disk2/vla-inference-scheduling/experiments}; cd $E
until grep -q "CHAIN4_COMPLETE" logs/chain4.log 2>/dev/null; do sleep 60; done
P="--ports 8000,8001 --workers_per_port 5 --eval_init_lo 0 --eval_init_hi 50"
DET="--schedule react_step --detector bok/detector_v1_t24.pt --step_tau 1.628 --react_setting 2,10"
run() { name=$1; shift; [ -f results_res/$name.jsonl ] && return; $LIBERO_PY residual_rl.py $P --eval_out results_res/$name.jsonl "$@" > logs/res/$name.log 2>&1 && echo "$(date "+%F %T") done $name $(grep "^eval" logs/res/$name.log)" || echo "$(date "+%F %T") FAILED $name"; }
run gen_big_student_nodhat --eval student:ckpt/student_v2_nodhat.pt --dispreg bok/dispreg_v2_t24.pt $DET --gate jump --disturb_mag 0.08,0.12
run abl_student_nodhat_none --eval student:ckpt/student_v2_nodhat.pt --dispreg bok/dispreg_v2_t24.pt $DET --gate jump --eval_disturb none
echo "$(date "+%F %T") CHAIN5_COMPLETE"
