#!/usr/bin/env bash
# Legacy policy servers on the given ports (default 8000 8001), each waits until warmed up; runs until killed.
source ${VLA_ENV:-$HOME/disk2/vla-env}/env.sh
E=${VLA_EXPERIMENTS:-$HOME/disk2/vla-inference-scheduling/experiments}; cd $E; mkdir -p logs/servers
PIDS=()
for p in ${@:-8000 8001}; do
  $OPENPI_PY serve_policy_dynamic.py --checkpoint $CKPT --port $p > logs/servers/server_$p.log 2>&1 & PIDS+=($!)
  until grep -q "with dynamic num_steps" logs/servers/server_$p.log; do kill -0 ${PIDS[-1]} || { tail -20 logs/servers/server_$p.log; exit 1; }; sleep 5; done
  echo "$(date "+%F %T") server $p up"
done
trap "kill ${PIDS[*]} 2>/dev/null" EXIT
wait
