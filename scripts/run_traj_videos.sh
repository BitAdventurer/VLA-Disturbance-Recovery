#!/usr/bin/env bash
# README trajectory videos: the start state of Fig. 4 (task 6, held-out start state 37) under the base policy, the
# detector-only replan and the residual student, with the per-step state logged (video_frames.py --log_state), once
# with the 4-6 cm displacement and once undisturbed. compose_traj_panels.py then writes the panels with the wrist
# camera, compose_traj.py the panels with the signal plots, and ffmpeg the README previews (animated WebP, 840 px,
# 10 fps). These are new rollouts, not replays of an evaluated episode: under the policy server's compile-time kernel
# autotuning they can differ in detail from the paper's episode (README, "Run the experiments").
# The compose scripts draw their text in Liberation Sans (Ubuntu package fonts-liberation).
# Usage: bash scripts/run_traj_videos.sh [OUTDIR]   (default: videos/traj/ next to scripts/)
source ${VLA_ENV:-$HOME/disk2/vla-env}/env.sh
S=$(cd "$(dirname "$0")" && pwd)
E=${VLA_EXPERIMENTS:-$HOME/disk2/vla-inference-scheduling/experiments}
O=$(realpath -m "${1:-$S/../videos/traj}"); mkdir -p "$O"
bash $S/start_servers.sh 8000 > "$O/server.log" 2>&1 & SRV=$!
until grep -q "server 8000 up" "$O/server.log"; do kill -0 $SRV || { tail "$O/server.log"; exit 1; }; sleep 5; done
cd $E
for d in object none; do
  $LIBERO_PY $S/video_frames.py --task 6 --init 37 --arms base,replan,student --disturb $d --port 8000 --log_state \
    --out "$O/$d"
done
kill $SRV                                      # start_servers.sh stops its server on exit
V=$O/trajectory_t6_s37
$LIBERO_PY $S/compose_traj_panels.py "$O/object" "${V}_wrist.mp4"                    # steps 40-70 at 1/4 speed, else 2x
$LIBERO_PY $S/compose_traj.py "$O/object" "${V}_signals.mp4" --stride 2              # twice real time
$LIBERO_PY $S/compose_traj.py "$O/none" "${V}_undisturbed.mp4" --stride 1            # real time
for v in wrist signals; do
  ffmpeg -y -loglevel error -i "${V}_$v.mp4" -vf "fps=10,scale=840:-1:flags=lanczos" -c:v libwebp_anim -lossless 0 \
    -quality 50 -compression_level 6 -loop 0 -an "${V}_$v.webp"
done
