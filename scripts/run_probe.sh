#!/usr/bin/env bash
# Scripted privileged residual probes, all 50 inits, salt 0, aligned resets (paired with results_dist s0 runs).
source ${VLA_ENV:-$HOME/disk2/vla-env}/env.sh
cd ${VLA_EXPERIMENTS:-$HOME/disk2/vla-inference-scheduling/experiments}; mkdir -p results_res logs/res
P="--ports 8000,8001 --workers_per_port 5 --eval_init_lo 0 --eval_init_hi 50"
DET="--schedule react_step --detector bok/detector_v1_t24.pt --step_tau 1.628 --react_setting 2,10"
run() { name=$1; shift; [ -f results_res/$name.jsonl ] && return; $LIBERO_PY residual_rl.py $P --eval_out results_res/$name.jsonl "$@" > logs/res/$name.log 2>&1 && echo "$(date "+%F %T") done $name $(grep "^eval" logs/res/$name.log)" || echo "$(date "+%F %T") FAILED $name"; }
run probe_s1_k2 --eval scripted:1.0,2
run probe_det_s1_k2 --eval scripted:1.0,2 $DET
run probe_s1_k1 --eval scripted:1.0,1
run probe_s05_k2 --eval scripted:0.5,2
run probe_s15_k2 --eval scripted:1.5,2
echo "$(date "+%F %T") PROBE_COMPLETE"
