#!/usr/bin/env bash
# Noise reseeds of the react->(10,*) variants (open item from the overnight study); 2 servers, ports 8000-8001.
set -eo pipefail
source ${VLA_ENV:-$HOME/disk2/vla-env}/env.sh
E=${VLA_EXPERIMENTS:-$HOME/disk2/vla-inference-scheduling/experiments}; cd $E
mkdir -p results_dist logs/reseed
PIDS=()
for p in 8000 8001; do
  $OPENPI_PY serve_policy_dynamic.py --checkpoint $CKPT --port $p > logs/reseed/server_$p.log 2>&1 & PIDS+=($!)
  until grep -q "with dynamic num_steps" logs/reseed/server_$p.log; do kill -0 ${PIDS[-1]} || { tail -20 logs/reseed/server_$p.log; exit 1; }; sleep 5; done
done
trap "kill ${PIDS[*]} 2>/dev/null" EXIT
echo "$(date "+%F %T") servers up"
Q=logs/reseed/queue.txt; : > $Q
OBJ="--disturb object --disturb_trigger approach --disturb_mag 0.04,0.06"
add() { echo "$1|$2 --seed 0 --out results_dist/$1_s0.jsonl" >> $Q; }
for s in 1 2; do
  add obj_react_now_n10h10_ns$s "--schedule react_now --n 2 --h 10 --react_setting 10,10 $OBJ --noise_salt $s"
  add obj_react_now_n10h5_ns$s  "--schedule react_now --n 2 --h 10 --react_setting 10,5 $OBJ --noise_salt $s"
done
worker() {
  local port=$1 line name args out
  while true; do
    line=$(flock $Q.lock sh -c "head -1 $Q; sed -i 1d $Q")
    [ -z "$line" ] && return 0
    name=${line%%|*}; args=${line#*|}
    out=$(echo "$args" | sed -n "s/.*--out \([^ ]*\).*/\1/p")
    [ -f "$out" ] && continue
    if $LIBERO_PY run_libero.py --port $port --latency latency.json $args >> logs/reseed/$name.log 2>&1; then
      echo "$(date "+%F %T") done   $name"; else echo "$(date "+%F %T") FAILED $name"; fi
  done
}
WPIDS=()
for p in 8000 8001; do worker $p & WPIDS+=($!); done
wait ${WPIDS[@]}
echo "$(date "+%F %T") RESEED_COMPLETE"
