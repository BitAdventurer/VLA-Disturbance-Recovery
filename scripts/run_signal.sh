#!/usr/bin/env bash
# Best-of-K signal phase (bok/prereg_bok.md): branch runs (split oracle) and selection runs, seq mode.
# Workers: legacy servers on 8000-8001 (started here) + the serve_policy_k.py server on 8002 (tmux kserver,
# whose legacy path is byte-identical). Queue: logs/bok/queue.txt, one job per line "name|args".
set -eo pipefail
source ${VLA_ENV:-$HOME/disk2/vla-env}/env.sh
E=${VLA_EXPERIMENTS:-$HOME/disk2/vla-inference-scheduling/experiments}; cd $E
while tmux has-session -t reseed 2>/dev/null; do sleep 30; done
mkdir -p results_bok/cands logs/bok
PIDS=()
for p in 8000 8001; do
  $OPENPI_PY serve_policy_dynamic.py --checkpoint $CKPT --port $p > logs/bok/server_$p.log 2>&1 & PIDS+=($!)
  until grep -q "with dynamic num_steps" logs/bok/server_$p.log; do kill -0 ${PIDS[-1]} || { tail -20 logs/bok/server_$p.log; exit 1; }; sleep 5; done
done
trap "kill ${PIDS[*]} 2>/dev/null" EXIT
grep -q "with dynamic num_steps" logs/bok/server_k_8002.log || { echo "kserver on 8002 not serving"; exit 1; }
echo "$(date "+%F %T") servers up"
Q=logs/bok/queue.txt; : > $Q
OBJ="--latency latency.json --schedule static --n 2 --h 10 --disturb object --disturb_trigger approach --disturb_mag 0.04,0.06 --seed 0"
STRAT="--episodes bok/strat.json --fresh_env 1"
add() { echo "$1|$2 --out results_bok/$1.jsonl" >> $Q; }
sel() { add "$1" "$OBJ --cand_mode seq --select_window ${3:-post:7} --num_candidates $2 --log_cands results_bok/cands/$1 ${*:4}"; }
for kc in "0 1" "1 0" "1 1" "2 0" "2 1" "3 0" "3 1" "0 0"; do set -- $kc; add br0_k$1_c$2 "$OBJ $STRAT --branch_r 0 --branch_k $1 --cont_salt $2"; done
sel sel8_priv 8 post:7 --selector priv_aim --aim_gain bok/aim_gain.json
sel sel8_random 8 post:7 --selector random
sel sel8_medoid 8 post:7 --selector medoid
sel sel8_continuity 8 post:7 --selector continuity
for kc in "0 1" "1 0" "1 1" "2 0" "2 1" "3 0" "3 1" "0 0"; do set -- $kc; add br2_k$1_c$2 "$OBJ $STRAT --branch_r 2 --branch_k $1 --cont_salt $2"; done
sel sel16_priv 16 post:7 --selector priv_aim --aim_gain bok/aim_gain.json
sel sel8_antimedoid 8 post:7 --selector antimedoid
sel sel8_discontinuity 8 post:7 --selector discontinuity
sel sel4_priv_postall 4 post:all --selector priv_aim --aim_gain bok/aim_gain.json
worker() {
  local port=$1 line name args out
  while true; do
    [ -f logs/bok/STOP ] && { echo "$(date "+%F %T") worker $port: STOP file"; return 0; }
    line=$(flock $Q.lock sh -c "head -1 $Q; sed -i 1d $Q")
    [ -z "$line" ] && return 0
    name=${line%%|*}; args=${line#*|}
    out=$(echo "$args" | sed -n "s/.*--out \([^ ]*\).*/\1/p")
    [ -f "$out" ] && continue
    if $LIBERO_PY run_libero.py --port $port $args >> logs/bok/$name.log 2>&1; then
      echo "$(date "+%F %T") done   $name ($port)"; else echo "$(date "+%F %T") FAILED $name ($port)"; fi
  done
}
WPIDS=()
for p in 8000 8001 8002; do worker $p & WPIDS+=($!); done
wait ${WPIDS[@]}
echo "$(date "+%F %T") SIGNAL_COMPLETE"
