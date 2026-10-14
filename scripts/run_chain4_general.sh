#!/usr/bin/env bash
# Chain 4 (overnight): generalization of the learned residual to other disturbances, extra noise draws, ablations.
#  A. student_v2 / teacher_v4 / vision scripted on: big displacement (8-12 cm), late (time-triggered 100-200), drift, push
#     (baselines exist in results_dist for the same aligned episodes, seed 0)
#  B. extra noise draws of student_v2 (ns1, ns2) and teacher v4 (ns2) on the main condition
#  C. ablation: student without the displacement estimate (no_dhat) distilled from teacher v4
#  D. ablation: residual teacher trained by PPO from scratch (no BC), jump gate
set -o pipefail
source ${VLA_ENV:-$HOME/disk2/vla-env}/env.sh
E=${VLA_EXPERIMENTS:-$HOME/disk2/vla-inference-scheduling/experiments}; cd $E; mkdir -p ckpt results_res logs/res
grep -q "server 8000 up" logs/servers.log && grep -q "server 8001 up" logs/servers.log || { echo "servers not up"; exit 1; }
P="--ports 8000,8001 --workers_per_port 5 --eval_init_lo 0 --eval_init_hi 50"
G="--gate jump"
DET="--schedule react_step --detector bok/detector_v1_t24.pt --step_tau 1.628 --react_setting 2,10"
STU="--eval student:ckpt/student_v2.pt --dispreg bok/dispreg_v2_t24.pt $DET $G"
run() { name=$1; shift; [ -f results_res/$name.jsonl ] && return; $LIBERO_PY residual_rl.py $P --eval_out results_res/$name.jsonl "$@" > logs/res/$name.log 2>&1 && echo "$(date "+%F %T") done $name $(grep "^eval" logs/res/$name.log)" || echo "$(date "+%F %T") FAILED $name"; }
# A. generalization (student v2, privileged teacher v4, vision scripted)
BIG="--disturb_mag 0.08,0.12"
run gen_big_student_v2 $STU $BIG
run gen_big_teacher_v4 --eval ckpt/res_v4.pt.it039 $G $BIG
run gen_big_vision_s05 --eval vision:0.5,2 --dispreg bok/dispreg_v2_t24.pt $DET $BIG
LATE="--disturb_trigger time --disturb_window 100,200 --disturb_mag 0.04,0.06"
run gen_late_student_v2 $STU $LATE
run gen_late_teacher_v4 --eval ckpt/res_v4.pt.it039 $G $LATE
DR="--eval_disturb drift --disturb_mag 0.06,0.10 --drift_steps 40"
run gen_drift_student_v2 $STU $DR
run gen_drift_teacher_v4 --eval ckpt/res_v4.pt.it039 $G $DR
PU="--eval_disturb push --disturb_mag 0.3,0.5 --push_steps 5"
run gen_push_student_v2 $STU $PU
# B. extra noise draws (main condition)
run student_v2_ns1 $STU --eval_noise_salt 1
run eval_res_v4_it039_ns2 --eval ckpt/res_v4.pt.it039 $G --eval_noise_salt 2 --eval_init_lo 25
run student_v2_ns2 $STU --eval_noise_salt 2
# C. ablation: student without d_hat
P6="--ports 8000,8001 --workers_per_port 6"
[ -f ckpt/student_v2_nodhat.pt ] || $LIBERO_PY student_distill.py --teacher ckpt/res_v4.pt.it039 --dispreg bok/dispreg_v2_t24.pt $DET $G $P6 \
    --reps 2 --no_dhat --out ckpt/student_v2_nodhat.pt > logs/res/student_v2_nodhat.log 2>&1 && echo "$(date "+%F %T") distilled nodhat" || echo "DISTILL nodhat FAILED"
run abl_student_nodhat --eval student:ckpt/student_v2_nodhat.pt --dispreg bok/dispreg_v2_t24.pt $DET $G
# D. ablation: PPO from scratch (no BC)
[ -f ckpt/res_v5_scratch.pt ] || $LIBERO_PY residual_rl.py --out ckpt/res_v5_scratch.pt $P6 $G --iters 40 --episodes 120 --batch_episodes 6 \
    --critic_warmup 2 --save_every 10 > logs/res/res_v5_scratch.train.log 2>&1 && echo "$(date "+%F %T") trained res_v5_scratch" || echo "TRAIN scratch FAILED"
P="--ports 8000,8001 --workers_per_port 5"
run abl_scratch_it039 --eval ckpt/res_v5_scratch.pt.it039 $G
run abl_scratch_it019 --eval ckpt/res_v5_scratch.pt.it019 $G
echo "$(date "+%F %T") CHAIN4_COMPLETE"
