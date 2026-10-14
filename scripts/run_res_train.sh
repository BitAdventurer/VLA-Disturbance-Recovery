#!/usr/bin/env bash
# Residual RL teacher: BC warm start from the scripted shift, PPO, then held-out evaluations of every checkpoint.
# Usage: run_res_train.sh NAME BC_SCRIPT [extra residual_rl.py args, e.g. the react_step detector base]
set -o pipefail
source ${VLA_ENV:-$HOME/disk2/vla-env}/env.sh
cd ${VLA_EXPERIMENTS:-$HOME/disk2/vla-inference-scheduling/experiments}; mkdir -p ckpt results_res logs/res
NAME=$1; BC=$2; shift 2
P="--ports 8000,8001 --workers_per_port 6"
$LIBERO_PY residual_rl.py --out ckpt/$NAME.pt $P --iters 60 --episodes 120 --batch_episodes 6 --critic_warmup 2 \
    --bc_script $BC --bc_episodes 500 --save_every 10 "$@" > logs/res/$NAME.train.log 2>&1 || { echo "TRAIN FAILED"; exit 1; }
echo "$(date "+%F %T") trained $NAME"
for ck in ckpt/$NAME.pt.bc ckpt/$NAME.pt.it009 ckpt/$NAME.pt.it029 ckpt/$NAME.pt.it059; do
  [ -f $ck ] || continue
  tag=$(basename $ck | sed "s/.pt//; s/\\./_/g")
  $LIBERO_PY residual_rl.py --eval $ck --eval_out results_res/eval_$tag.jsonl $P "$@" > logs/res/eval_$tag.log 2>&1
  echo "$(date "+%F %T") eval $tag $(grep "^eval" logs/res/eval_$tag.log)"
done
echo "$(date "+%F %T") RES_TRAIN_COMPLETE"
