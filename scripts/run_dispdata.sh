#!/usr/bin/env bash
# Event frames for the displacement regressor: 6 disturbance seeds x 500 episodes, 2-10 cm, stop 3 steps after the event.
source ${VLA_ENV:-$HOME/disk2/vla-env}/env.sh
cd ${VLA_EXPERIMENTS:-$HOME/disk2/vla-inference-scheduling/experiments}; mkdir -p results_bok/event_frames logs/dispdata
B="--latency latency.json --schedule static --n 2 --h 10 --disturb object --disturb_trigger approach --disturb_mag 0.02,0.10 --stop_after_event 3"
PIDS=()
for s in 1 2 3 4 5 6; do
  port=$((8000 + s % 2))
  $LIBERO_PY run_libero.py --port $port $B --seed $s --log_event_frames results_bok/event_frames/s$s --out results_bok/dispdata_s$s.jsonl > logs/dispdata/s$s.log 2>&1 &
  PIDS+=($!)
done
wait ${PIDS[@]}
echo "$(date "+%F %T") DISPDATA_COMPLETE $(ls results_bok/event_frames/*/ | grep -c npz)"
