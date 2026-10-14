#!/usr/bin/env bash
# Follow-up disturbance probes (pi0.5, LIBERO-10, seed 0), run after run_dist_probe.sh; no new job after 09:45.
set -eo pipefail
source ${VLA_ENV:-$HOME/disk2/vla-env}/env.sh
E=${VLA_EXPERIMENTS:-$HOME/disk2/vla-inference-scheduling/experiments}; cd $E
while tmux has-session -t distprobe 2>/dev/null; do sleep 30; done
while ps -C python -o args | grep -q serve_policy; do sleep 10; done
mkdir -p results_dist logs/dist2
PIDS=()
for p in 8000 8001 8002; do
  $OPENPI_PY serve_policy_dynamic.py --checkpoint $CKPT --port $p > logs/dist2/server_$p.log 2>&1 & PIDS+=($!)
  until grep -q "with dynamic num_steps" logs/dist2/server_$p.log; do kill -0 ${PIDS[-1]} || { tail -20 logs/dist2/server_$p.log; exit 1; }; sleep 5; done
done
trap "kill ${PIDS[*]} 2>/dev/null" EXIT
echo "$(date "+%F %T") servers up"
Q=logs/dist2/queue.txt; : > $Q
add() { echo "$1|$2 --seed 0 --out results_dist/$1_s0.jsonl" >> $Q; }
PUSH="--disturb push --disturb_trigger approach --disturb_mag 0.3,0.5 --push_steps 5"
OBJ="--disturb object --disturb_trigger approach --disturb_mag 0.04,0.06"
BIG="--disturb object --disturb_trigger approach --disturb_mag 0.08,0.12"
LATE="--disturb object --disturb_trigger time --disturb_window 100,200 --disturb_mag 0.04,0.06"
# P2 push (long h=2 jobs first so the three workers stay balanced)
add push_static_n2_h2  "--schedule static --n 2 --h 2 $PUSH"
add push_static_n2_h5  "--schedule static --n 2 --h 5 $PUSH"
add push_static_n2_h10 "--schedule static --n 2 --h 10 $PUSH"
add push_react_now     "--schedule react_now --n 2 --h 10 $PUSH"
add push_react_next    "--schedule react_next --n 2 --h 10 $PUSH"
# P5 does n matter under displacement?
add obj_static_n10_h10 "--schedule static --n 10 --h 10 $OBJ"
# P3 strong displacement
add objbig_static_n2_h2  "--schedule static --n 2 --h 2 $BIG"
add objbig_static_n2_h10 "--schedule static --n 2 --h 10 $BIG"
add objbig_static_n2_h5  "--schedule static --n 2 --h 5 $BIG"
add objbig_react_now     "--schedule react_now --n 2 --h 10 $BIG"
# P4 late displacement (second subtask)
add objlate_static_n2_h2  "--schedule static --n 2 --h 2 $LATE"
add objlate_static_n2_h10 "--schedule static --n 2 --h 10 $LATE"
add objlate_react_now     "--schedule react_now --n 2 --h 10 $LATE"
worker() {
  local port=$1 line name args out
  while true; do
    [ "$(date +%H%M)" -ge 0945 ] && [ "$(date +%H%M)" -lt 1200 ] && { echo "$(date "+%F %T") worker $port: past 09:45, no new job"; return 0; }
    line=$(flock $Q.lock sh -c "head -1 $Q; sed -i 1d $Q")
    [ -z "$line" ] && return 0
    name=${line%%|*}; args=${line#*|}
    out=$(echo "$args" | sed -n "s/.*--out \([^ ]*\).*/\1/p")
    [ -f "$out" ] && continue
    if $LIBERO_PY run_libero.py --port $port --latency latency.json $args >> logs/dist2/$name.log 2>&1; then
      echo "$(date "+%F %T") done   $name"; else echo "$(date "+%F %T") FAILED $name"; fi
  done
}
WPIDS=()
for p in 8000 8001 8002; do worker $p & WPIDS+=($!); done
wait ${WPIDS[@]}
echo "$(date "+%F %T") DIST_MORE_COMPLETE"
