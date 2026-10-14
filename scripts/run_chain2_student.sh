#!/usr/bin/env bash
# After the vision chain: (1) confirm the teacher vs the scripted shift with two more noise draws (held-out inits),
# (2) teacher on undisturbed episodes, (3) distill the best teacher into the vision-only student (DAgger), (4) student evals.
source ${VLA_ENV:-$HOME/disk2/vla-env}/env.sh
E=${VLA_EXPERIMENTS:-$HOME/disk2/vla-inference-scheduling/experiments}; cd $E; mkdir -p logs/res results_res
until grep -q "CHAIN_COMPLETE" logs/chain.log 2>/dev/null; do sleep 60; done
P="--ports 8000,8001 --workers_per_port 5"
DET="--schedule react_step --detector bok/detector_v1_t24.pt --step_tau 1.628 --react_setting 2,10"
run() { name=$1; shift; [ -f results_res/$name.jsonl ] && return; $LIBERO_PY residual_rl.py $P --eval_out results_res/$name.jsonl "$@" > logs/res/$name.log 2>&1 && echo "$(date "+%F %T") done $name $(grep "^eval" logs/res/$name.log)" || echo "$(date "+%F %T") FAILED $name"; }
BEST=$(python3 - <<"PY"
import json, os
best = None
for ck, name in [("ckpt/res_v2.pt.bc", "eval_res_v2_bc"), ("ckpt/res_v3.pt.it009", "eval_res_v3_it009"),
                 ("ckpt/res_v3.pt.it029", "eval_res_v3_it029"), ("ckpt/res_v3.pt.it059", "eval_res_v3_it059")]:
    p = "results_res/%s.jsonl" % name
    if not (os.path.exists(p) and os.path.exists(ck)):
        continue
    rows = [json.loads(l) for l in open(p)]
    s = sum(r["success"] for r in rows) / len(rows)
    if best is None or s > best[0]:
        best = (s, ck)
print(best[1] if best else "ckpt/res_v3.pt")
PY
)
echo "$(date "+%F %T") best teacher: $BEST"
TAG=$(basename $BEST | sed "s/.pt//; s/\\./_/g")
run eval_${TAG}_ns1 --eval $BEST --eval_noise_salt 1
run probe_s05_k2_ns1 --eval scripted:0.5,2 --eval_noise_salt 1
run eval_${TAG}_ns2 --eval $BEST --eval_noise_salt 2
run probe_s05_k2_ns2 --eval scripted:0.5,2 --eval_noise_salt 2
run eval_${TAG}_none --eval $BEST --eval_disturb none
$LIBERO_PY student_distill.py --teacher $BEST --dispreg bok/dispreg_v2_t24.pt $DET --ports 8000,8001 --workers_per_port 6 \
    --reps 2 --out ckpt/student_v1.pt > logs/res/student_v1.log 2>&1 && echo "$(date "+%F %T") distilled" || { echo "DISTILL FAILED"; exit 1; }
P="--ports 8000,8001 --workers_per_port 5 --eval_init_lo 0 --eval_init_hi 50"
run student_v1 --eval student:ckpt/student_v1.pt --dispreg bok/dispreg_v2_t24.pt $DET
run student_v1_ns1 --eval student:ckpt/student_v1.pt --dispreg bok/dispreg_v2_t24.pt $DET --eval_noise_salt 1
run student_v1_none --eval student:ckpt/student_v1.pt --dispreg bok/dispreg_v2_t24.pt $DET --eval_disturb none
echo "$(date "+%F %T") CHAIN2_COMPLETE"
