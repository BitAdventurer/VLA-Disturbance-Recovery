#!/usr/bin/env bash
# Fig. 3 arm-push bar for the privileged teacher (added 2026-10-28): the arm-push episodes of gen_push_student_v2
# (run_chain4_general.sh), evaluated with the teacher res_v4.pt.it039 and its jump gate on one policy server.
source ${VLA_ENV:-$HOME/disk2/vla-env}/env.sh
S=$(cd "$(dirname "$0")" && pwd)
E=${VLA_EXPERIMENTS:-$HOME/disk2/vla-inference-scheduling/experiments}; cd $E; mkdir -p results_res logs/res
bash $S/start_servers.sh 8000 > logs/res/push_teacher_server.log 2>&1 & SRV=$!
until grep -q "server 8000 up" logs/res/push_teacher_server.log; do kill -0 $SRV || { tail logs/res/push_teacher_server.log; exit 1; }; sleep 5; done
$LIBERO_PY residual_rl.py --ports 8000 --workers_per_port 6 --eval_init_lo 0 --eval_init_hi 50 \
    --eval_out results_res/gen_push_teacher_v4.jsonl --eval ckpt/res_v4.pt.it039 --gate jump \
    --eval_disturb push --disturb_mag 0.3,0.5 --push_steps 5 > logs/res/gen_push_teacher_v4.log 2>&1
echo "$(date "+%F %T") exit $?"; grep "^eval" logs/res/gen_push_teacher_v4.log
kill $SRV                                       # start_servers.sh stops its server on exit
