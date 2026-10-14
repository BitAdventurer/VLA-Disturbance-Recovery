#!/usr/bin/env bash
# chain6 (2026-10-04): extra noise draws to tighten the paper's intervals. Uses the running servers on 8000/8001.
#  A. residual evals (10 workers): student_v2 ns3/ns4 (all inits), teacher v4 ns3/ns4 (held-out), vision scripted ns2,
#     big-displacement student/teacher ns1, no-d_hat student ns1, PPO-from-scratch ns1 (held-out).
#  B. in parallel, two more draws of the (2,10) baseline pool via run_libero (sequential, one per port).
set -o pipefail
source ${VLA_ENV:-$HOME/disk2/vla-env}/env.sh
E=${VLA_EXPERIMENTS:-$HOME/disk2/vla-inference-scheduling/experiments}; cd $E
mkdir -p logs/res logs/reseed
P="--ports 8000,8001 --workers_per_port 5 --eval_init_lo 0 --eval_init_hi 50"
G="--gate jump"
DET="--schedule react_step --detector bok/detector_v1_t24.pt --step_tau 1.628 --react_setting 2,10"
STU="--eval student:ckpt/student_v2.pt --dispreg bok/dispreg_v2_t24.pt $DET $G"
NOD="--eval student:ckpt/student_v2_nodhat.pt --dispreg bok/dispreg_v2_t24.pt $DET $G"
BIG="--disturb_mag 0.08,0.12"
run() { name=$1; shift; [ -f results_res/$name.jsonl ] && { echo "skip $name"; return; }; $LIBERO_PY residual_rl.py $P --eval_out results_res/$name.jsonl "$@" > logs/res/$name.log 2>&1 && echo "$(date "+%F %T") done $name $(grep "^eval" logs/res/$name.log)" || echo "$(date "+%F %T") FAILED $name"; }

# B. baseline draws in the background (sequential clients, one per port)
OBJ="--schedule static --n 2 --h 10 --disturb object --disturb_trigger approach --disturb_mag 0.04,0.06 --seed 0"
base() { s=$1; port=$2; out=results_dist/obj_static_n2_h10_ns${s}_s0.jsonl; [ -f $out ] && return
  $LIBERO_PY run_libero.py --port $port --latency latency.json $OBJ --noise_salt $s --out $out > logs/reseed/obj_static_ns$s.log 2>&1 \
    && echo "$(date "+%F %T") done baseline ns$s" || echo "$(date "+%F %T") FAILED baseline ns$s"; }
base 5 8000 &
base 6 8001 &

# A. residual evals
run student_v2_ns3 $STU --eval_noise_salt 3
run student_v2_ns4 $STU --eval_noise_salt 4
run eval_res_v4_it039_ns3 --eval ckpt/res_v4.pt.it039 $G --eval_noise_salt 3 --eval_init_lo 25
run eval_res_v4_it039_ns4 --eval ckpt/res_v4.pt.it039 $G --eval_noise_salt 4 --eval_init_lo 25
run vision_s05_k2_ns2 --eval vision:0.5,2 --dispreg bok/dispreg_v2_t24.pt $DET --eval_noise_salt 2
run gen_big_student_v2_ns1 $STU $BIG --eval_noise_salt 1
run gen_big_teacher_v4_ns1 --eval ckpt/res_v4.pt.it039 $G $BIG --eval_noise_salt 1
run abl_student_nodhat_ns1 $NOD --eval_noise_salt 1
run abl_scratch_it039_ns1 --eval ckpt/res_v5_scratch.pt.it039 $G --eval_noise_salt 1 --eval_init_lo 25
wait
echo "$(date "+%F %T") CHAIN6_COMPLETE"
