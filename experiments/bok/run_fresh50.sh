#!/usr/bin/env bash
# run_fresh50.sh PORT: the one-time fresh-initial-state test of the ICEIC paper (experiments/bok/prereg_fresh50.md).
# LIBERO-10 .init states 50-99 (never rolled out before); every arm through residual_rl.py --eval with the default-off
# --init_file full, run from the tested patched copy in ~/disk2/vla-env/run_fresh50 (the live experiments/ tree is not
# modified), one policy server, 10 rollout workers. Every arm-draw must give 500 rows with the 500 distinct
# (task, init 50..99) keys; an incomplete arm-draw gets at most two attempts over all invocations of this script, and
# earlier outputs and logs are kept. Outputs: experiments/results_test/<name>.jsonl, logs: experiments/logs/fresh50/.
# FRESH_MOCK=1 runs the same sequence in mock mode into a scratch folder (smoke test only).
set -o pipefail
PORT=${1:?usage: run_fresh50.sh PORT}
source ~/disk2/vla-env/env.sh
R=~/disk2/vla-env/run_fresh50; E=~/disk2/vla-inference-scheduling/experiments
OUT=$E/results_test; LOG=$E/logs/fresh50; MOCK=
if [ "${FRESH_MOCK:-0}" = 1 ]; then OUT=~/disk2/vla-env/tmp/fresh50_mock/results_test; LOG=~/disk2/vla-env/tmp/fresh50_mock/logs; MOCK=--mock; fi
cd $R; mkdir -p $OUT $LOG
for f in ~/disk2/vla-env/stage_fresh50/new/*.py; do       # run only the tested patched copy
  cmp -s "$f" "$R/$(basename "$f")" || { echo "$(basename "$f") in $R is not the tested patched copy"; exit 1; }
done
if [ -z "$MOCK" ]; then                                    # a missing server would block the clients for hours
  timeout 5 bash -c "exec 3<>/dev/tcp/127.0.0.1/$PORT" 2>/dev/null || { echo "no policy server on the given port"; exit 1; }
fi
P="--ports $PORT --workers_per_port 10 --init_file full --eval_init_lo 50 --eval_init_hi 100 $MOCK"
DET="--schedule react_step --detector bok/detector_v1_t24.pt --step_tau 1.628 --react_setting 2,10"
STU="--eval student:ckpt/student_v2.pt --dispreg bok/dispreg_v2_t24.pt $DET --gate jump"
FIX="--eval vision:0.5,2 --dispreg bok/dispreg_v2_t24.pt $DET"
BASE="--eval zero"

complete() {   # 500 rows, keys = 10 tasks x inits 50..99, no error rows
  $LIBERO_PY - "$1" << 'PY'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
keys = {(r["task_id"], r["init_state_id"]) for r in rows if "success" in r and not r.get("error")}
want = {(t, i) for t in range(10) for i in range(50, 100)}
sys.exit(0 if len(rows) == 500 and keys == want else 1)
PY
}
NFAIL=0
run() {
  name=$1; shift; out=$OUT/$name.jsonl
  if [ -f $out ] && complete $out; then echo "$(date "+%F %T") skip $name (complete)"; return; fi
  for attempt in 1 2; do
    log=$LOG/$name.a$attempt.log
    [ -f $log ] && continue                                # used by an earlier invocation: two attempts in total
    [ -f $out ] && mv -f $out $out.prev_a$((attempt - 1))  # keep a short earlier output
    rm -f $out.part
    $LIBERO_PY residual_rl.py $P --eval_out $out "$@" > $log 2>&1
    if [ -f $out ] && complete $out; then echo "$(date "+%F %T") done $name $(grep "^eval" $log)"; return; fi
    echo "$(date "+%F %T") $name attempt $attempt incomplete ($( [ -f $out ] && wc -l < $out || echo 0) rows)"
  done
  echo "$(date "+%F %T") FAILED $name"; NFAIL=$((NFAIL + 1))
}

echo "$(date "+%F %T") FRESH50 start"
# primary draws interleaved (reference, student, fixed rule), then the remaining reference draws, then secondary arms
run fresh_base_s0     $BASE --eval_noise_salt 0
run fresh_student_s0  $STU  --eval_noise_salt 0
run fresh_fixed_s0    $FIX  --eval_noise_salt 0
run fresh_base_ns1    $BASE --eval_noise_salt 1
run fresh_student_ns1 $STU  --eval_noise_salt 1
run fresh_fixed_ns1   $FIX  --eval_noise_salt 1
run fresh_base_ns2    $BASE --eval_noise_salt 2
run fresh_student_ns2 $STU  --eval_noise_salt 2
run fresh_fixed_ns2   $FIX  --eval_noise_salt 2
run fresh_base_ns3    $BASE --eval_noise_salt 3
run fresh_base_ns4    $BASE --eval_noise_salt 4
run fresh_det_s0      $BASE $DET --eval_noise_salt 0
run fresh_det_ns1     $BASE $DET --eval_noise_salt 1
run fresh_none_base_s0    $BASE --eval_disturb none --eval_noise_salt 0
run fresh_none_student_s0 $STU  --eval_disturb none --eval_noise_salt 0
echo "$(date "+%F %T") FRESH50_COMPLETE failed=$NFAIL"
[ $NFAIL -eq 0 ]
