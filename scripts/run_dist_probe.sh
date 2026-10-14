#!/usr/bin/env bash
# Disturbance probe (pi0.5, LIBERO-10, seed 0): object displaced by 4-6 cm when the open gripper approaches it.
set -eo pipefail
source ${VLA_ENV:-$HOME/disk2/vla-env}/env.sh
E=${VLA_EXPERIMENTS:-$HOME/disk2/vla-inference-scheduling/experiments}; cd $E
mkdir -p results_dist logs/dist
while tmux has-session -t smokedist 2>/dev/null || tmux has-session -t pi0seeds 2>/dev/null; do sleep 30; done
while ps -C python -o args | grep -q serve_policy; do sleep 10; done
PIDS=()
for p in 8000 8001 8002; do
  $OPENPI_PY serve_policy_dynamic.py --checkpoint $CKPT --port $p > logs/dist/server_$p.log 2>&1 & PIDS+=($!)
  until grep -q "with dynamic num_steps" logs/dist/server_$p.log; do kill -0 ${PIDS[-1]} || { tail -20 logs/dist/server_$p.log; exit 1; }; sleep 5; done
done
trap "kill ${PIDS[*]} 2>/dev/null" EXIT
echo "$(date "+%F %T") servers up"
D="--disturb object --disturb_trigger approach --disturb_mag 0.04,0.06 --seed 0"
Q=logs/dist/queue.txt; : > $Q
echo "obj_n2_h2|--schedule static --n 2 --h 2 $D --out results_dist/obj_static_n2_h2_s0.jsonl" >> $Q
echo "obj_n2_h5|--schedule static --n 2 --h 5 $D --out results_dist/obj_static_n2_h5_s0.jsonl" >> $Q
echo "obj_n2_h10|--schedule static --n 2 --h 10 $D --out results_dist/obj_static_n2_h10_s0.jsonl" >> $Q
echo "obj_react_now|--schedule react_now --n 2 --h 10 $D --out results_dist/obj_react_now_s0.jsonl" >> $Q
echo "obj_react_next|--schedule react_next --n 2 --h 10 $D --out results_dist/obj_react_next_s0.jsonl" >> $Q
worker() {
  local port=$1 line name args out
  while true; do
    line=$(flock $Q.lock sh -c "head -1 $Q; sed -i 1d $Q")
    [ -z "$line" ] && return 0
    name=${line%%|*}; args=${line#*|}
    out=$(echo "$args" | sed -n "s/.*--out \([^ ]*\).*/\1/p")
    [ -f "$out" ] && continue
    if $LIBERO_PY run_libero.py --port $port --latency latency.json $args >> logs/dist/$name.log 2>&1; then
      echo "$(date "+%F %T") done   $name"; else echo "$(date "+%F %T") FAILED $name"; fi
  done
}
WPIDS=()
for p in 8000 8001 8002; do worker $p & WPIDS+=($!); done
wait ${WPIDS[@]}
echo "$(date "+%F %T") DIST_PROBE_COMPLETE"
