#!/usr/bin/env bash
# Teacher v4 with the sudden-jump privileged gate (no false triggers on nudges): BC from the scripted shift, PPO,
# evals (disturbed held-out, undisturbed held-out), then distillation into student_v2 and its evals.
source ${VLA_ENV:-$HOME/disk2/vla-env}/env.sh
E=${VLA_EXPERIMENTS:-$HOME/disk2/vla-inference-scheduling/experiments}; cd $E; mkdir -p ckpt results_res logs/res
until grep -q "CHAIN2_COMPLETE\|DISTILL FAILED" logs/chain2.log 2>/dev/null; do sleep 60; done
P="--ports 8000,8001 --workers_per_port 6"
G="--gate jump"
$LIBERO_PY residual_rl.py --out ckpt/res_v4.pt $P $G --iters 40 --episodes 120 --batch_episodes 6 --critic_warmup 1 \
    --bc_script 0.5,2 --bc_episodes 500 --save_every 10 > logs/res/res_v4.train.log 2>&1 || { echo "TRAIN FAILED"; exit 1; }
echo "$(date "+%F %T") trained res_v4"
P5="--ports 8000,8001 --workers_per_port 5"
run() { name=$1; shift; [ -f results_res/$name.jsonl ] && return; $LIBERO_PY residual_rl.py $P5 --eval_out results_res/$name.jsonl "$@" > logs/res/$name.log 2>&1 && echo "$(date "+%F %T") done $name $(grep "^eval" logs/res/$name.log)" || echo "$(date "+%F %T") FAILED $name"; }
run eval_res_v4_bc --eval ckpt/res_v4.pt.bc $G
run eval_res_v4_it039 --eval ckpt/res_v4.pt.it039 $G
run eval_res_v4_it039_none --eval ckpt/res_v4.pt.it039 $G --eval_disturb none
run eval_res_v4_it039_ns1 --eval ckpt/res_v4.pt.it039 $G --eval_noise_salt 1
run probe_jump_s05_k2 --eval scripted:0.5,2 $G
run probe_jump_s05_k2_none --eval scripted:0.5,2 $G --eval_disturb none
DET="--schedule react_step --detector bok/detector_v1_t24.pt --step_tau 1.628 --react_setting 2,10"
$LIBERO_PY student_distill.py --teacher ckpt/res_v4.pt.it039 --dispreg bok/dispreg_v2_t24.pt $DET $G $P \
    --reps 2 --out ckpt/student_v2.pt > logs/res/student_v2.log 2>&1 && echo "$(date "+%F %T") distilled v2" || { echo "DISTILL2 FAILED"; exit 1; }
P5="--ports 8000,8001 --workers_per_port 5 --eval_init_lo 0 --eval_init_hi 50"
run student_v2 --eval student:ckpt/student_v2.pt --dispreg bok/dispreg_v2_t24.pt $DET $G
run student_v2_none --eval student:ckpt/student_v2.pt --dispreg bok/dispreg_v2_t24.pt $DET $G --eval_disturb none
echo "$(date "+%F %T") CHAIN3_COMPLETE"
