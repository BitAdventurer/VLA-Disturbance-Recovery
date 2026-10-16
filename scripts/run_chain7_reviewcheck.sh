#!/usr/bin/env bash
# chain7 (2026-10-16, follow-up of the anticipated-review check):
#  (a) two more noise draws of the privileged "detector replan + scripted 0.5 d" arm (probe_det_s05_k2, all 500 episodes),
#  (b) the privileged teacher v4 WITH the detector replan (react_step), held-out initial states, three draws.
# Uses ONE policy server of ours on port 8001 (the vla-rss-2027-e01 work stream runs its own server on 8000: never touch it)
# and stops that server when done.
source ${VLA_ENV:-$HOME/disk2/vla-env}/env.sh
E=${VLA_EXPERIMENTS:-$HOME/disk2/vla-inference-scheduling/experiments}; cd $E; mkdir -p logs/res results_res
PORT=8001
until grep -q "server $PORT up" logs/chain7_servers.log 2>/dev/null; do sleep 10; done
echo "$(date "+%F %T") server $PORT up, starting"
DET="--schedule react_step --detector bok/detector_v1_t24.pt --step_tau 1.628 --react_setting 2,10"
ALL="--eval_init_lo 0 --eval_init_hi 50"
run() { name=$1; shift; [ -f results_res/$name.jsonl ] && { echo "skip $name"; return; }
  $LIBERO_PY residual_rl.py --ports $PORT --workers_per_port 5 --eval_out results_res/$name.jsonl "$@" > logs/res/$name.log 2>&1 \
    && echo "$(date "+%F %T") done $name $(grep "^eval" logs/res/$name.log)" || echo "$(date "+%F %T") FAILED $name"; }
run probe_det_s05_k2_ns1 --eval scripted:0.5,2 $DET $ALL --eval_noise_salt 1
run eval_res_v4_it039_det --eval ckpt/res_v4.pt.it039 --gate jump $DET
run probe_det_s05_k2_ns2 --eval scripted:0.5,2 $DET $ALL --eval_noise_salt 2
run eval_res_v4_it039_det_ns1 --eval ckpt/res_v4.pt.it039 --gate jump $DET --eval_noise_salt 1
run eval_res_v4_it039_det_ns2 --eval ckpt/res_v4.pt.it039 --gate jump $DET --eval_noise_salt 2
tmux kill-session -t iceic7_srv 2>/dev/null; sleep 15
PID=$(ss -ltnp 2>/dev/null | grep ":$PORT " | sed -n "s/.*pid=\([0-9]*\).*/\1/p" | head -1)
[ -n "$PID" ] && kill $PID && echo "killed leftover server pid $PID on $PORT"
echo "$(date "+%F %T") CHAIN7_COMPLETE"
