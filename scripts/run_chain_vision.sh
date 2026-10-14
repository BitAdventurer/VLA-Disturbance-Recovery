#!/usr/bin/env bash
# After the residual RL run: event-frame data -> displacement regressor v2 -> vision-only residual evaluations.
source ${VLA_ENV:-$HOME/disk2/vla-env}/env.sh
E=${VLA_EXPERIMENTS:-$HOME/disk2/vla-inference-scheduling/experiments}; cd $E; mkdir -p logs/res results_res
until grep -q "RES_TRAIN_COMPLETE\|TRAIN FAILED" logs/res_v3.log 2>/dev/null; do sleep 60; done
echo "$(date "+%F %T") RL run finished; collecting event frames"
bash ${VLA_SCRIPTS:-$HOME/disk2/vla-env/scripts}/run_dispdata.sh > logs/dispdata.log 2>&1
echo "$(date "+%F %T") $(tail -n 1 logs/dispdata.log)"
nice -n 5 $OPENPI_PY ${VLA_SCRIPTS:-$HOME/disk2/vla-env/scripts}/train_dispreg2.py --out bok/dispreg_v2.pt > logs/res/dispreg_v2.log 2>&1
echo "$(date "+%F %T") dispreg_v2: $(grep FINAL logs/res/dispreg_v2.log)"
$LIBERO_PY ${VLA_SCRIPTS:-$HOME/disk2/vla-env/scripts}/train_dispreg2.py export bok/dispreg_v2.pt.state.pt bok/dispreg_v2_t24.pt >> logs/res/dispreg_v2.log 2>&1
P="--ports 8000,8001 --workers_per_port 5 --eval_init_lo 0 --eval_init_hi 50"
DET="--schedule react_step --detector bok/detector_v1_t24.pt --step_tau 1.628 --react_setting 2,10"
run() { name=$1; shift; [ -f results_res/$name.jsonl ] && return; $LIBERO_PY residual_rl.py $P --eval_out results_res/$name.jsonl "$@" > logs/res/$name.log 2>&1 && echo "$(date "+%F %T") done $name $(grep "^eval" logs/res/$name.log)" || echo "$(date "+%F %T") FAILED $name"; }
run vision_s05_k2 --eval vision:0.5,2 --dispreg bok/dispreg_v2_t24.pt $DET
run probe_det_s05_k2 --eval scripted:0.5,2 $DET
run vision_s05_k2_ns1 --eval vision:0.5,2 --dispreg bok/dispreg_v2_t24.pt $DET --eval_noise_salt 1
run vision_s05_k2_none --eval vision:0.5,2 --dispreg bok/dispreg_v2_t24.pt $DET --eval_disturb none
echo "$(date "+%F %T") CHAIN_COMPLETE"
