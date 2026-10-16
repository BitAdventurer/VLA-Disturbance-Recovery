#!/usr/bin/env bash
# chain8b (2026-10-16): B6 of bok/prereg_ext.md (base policy at openpi's default (10,5) under the disturbance) on a second
# server of ours (port 8002), in parallel with chain8 on 8001. B6 is the LAST block of run_chain8_extension.sh, which skips
# result files that already exist, so the two chains never run the same item (chain8 needs ~10 h before it reaches B6).
# Stops its own server when done. Never touches port 8000 or tmux rss_* (vla-rss-2027-e01).
source ${VLA_ENV:-$HOME/disk2/vla-env}/env.sh
E=${VLA_EXPERIMENTS:-$HOME/disk2/vla-inference-scheduling/experiments}; cd $E; mkdir -p logs/res results_dist
until grep -q "server 8002 up" logs/chain8b_servers.log 2>/dev/null; do sleep 10; done
echo "$(date "+%F %T") server 8002 up, starting B6"
ALL="--eval_init_lo 0 --eval_init_hi 50"
run() { dir=$1; name=$2; shift 2; [ -f $dir/$name.jsonl ] && { echo "skip $name"; return; }
  $LIBERO_PY residual_rl.py --ports 8002 --workers_per_port 5 --eval_out $dir/$name.jsonl "$@" > logs/res/$name.log 2>&1 \
    && echo "$(date "+%F %T") done $name $(grep "^eval" logs/res/$name.log)" || echo "$(date "+%F %T") FAILED $name"; }
for s in 0 1; do sfx=$([ $s = 0 ] && echo "_s0" || echo "_ns${s}_s0"); run results_dist obj_static_n10_h5$sfx --eval zero --n 10 --h 5 $ALL --eval_noise_salt $s; done
tmux kill-session -t iceic8b_srv 2>/dev/null; sleep 15
PID=$(ss -ltnp 2>/dev/null | grep ":8002 " | sed -n "s/.*pid=\([0-9]*\).*/\1/p" | head -1)
[ -n "$PID" ] && kill $PID && echo "killed leftover server pid $PID on 8002"
echo "$(date "+%F %T") CHAIN8B_COMPLETE"
