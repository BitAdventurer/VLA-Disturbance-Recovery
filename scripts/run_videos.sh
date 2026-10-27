#!/usr/bin/env bash
# README videos: replays the chosen episodes (held-out start states 25-49, selection rules in the README) under the
# base policy, the detector-only replan and the residual student with one policy server, checks every replay against
# the result files (verify_replays.py) and writes one side-by-side MP4 and a still per episode (compose_episode.py).
# Usage: bash scripts/run_videos.sh [OUTDIR]   (default: videos/ next to scripts/)
source ${VLA_ENV:-$HOME/disk2/vla-env}/env.sh
S=$(cd "$(dirname "$0")" && pwd)
E=${VLA_EXPERIMENTS:-$HOME/disk2/vla-inference-scheduling/experiments}
O=$(realpath -m "${1:-$S/../videos}"); mkdir -p "$O"
bash $S/start_servers.sh 8000 > "$O/server.log" 2>&1 & SRV=$!
until grep -q "server 8000 up" "$O/server.log"; do kill -0 $SRV || { tail "$O/server.log"; exit 1; }; sleep 5; done
cd $E
rep() {  # condition task init disturb magnitude
  $LIBERO_PY $S/video_frames.py --task $2 --init $3 --arms base,replan,student --disturb $4 --mag $5 --port 8000 --out "$O/$1"
}
mk() {   # condition task init name label
  $LIBERO_PY $S/compose_episode.py "$O/$4.mp4" --label "$5" --thumb "$O/$4.jpg" \
    --dirs base:$O/$1/base_t$2_e$3,replan:$O/$1/replan_t$2_e$3,student:$O/$1/student_t$2_e$3
}
GALLERY="2:28 3:29 4:27 5:37 7:25 8:27 9:41 0:28"
for ti in $GALLERY; do rep obj ${ti%:*} ${ti#*:} object 0.04,0.06; done
rep none 1 28 none 0.04,0.06
rep big 0 37 object 0.08,0.12
kill $SRV                                      # start_servers.sh stops its server on exit
python3 $S/verify_replays.py "$O" "$O/replay_check.json"
for ti in $GALLERY; do t=${ti%:*}; i=${ti#*:}; mk obj $t $i task${t}_4-6cm_s$i "4–6 cm displacement, held-out start state $i"; done
mk none 1 28 task1_undisturbed_s28 "undisturbed, held-out start state 28"
mk big 0 37 task0_8-12cm_s37 "8–12 cm displacement, held-out start state 37"
