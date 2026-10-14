#!/usr/bin/env bash
# LIBERO client env as in openpi/examples/libero, except torch: CPU build >= 2.0 (scheduler training;
# the pinned 1.11+cu113 cannot run on sm_120 and the rollouts only need torch.load for init states).
set -euxo pipefail
source ${VLA_ENV:-$HOME/disk2/vla-env}/env.sh
cd $VLA_ENV/openpi
uv venv --python 3.8 examples/libero/.venv
PY=examples/libero/.venv/bin/python
grep -vE "^(torch|torchvision|torchaudio)==" examples/libero/requirements.txt > $VLA_ENV/tmp/libero_req_notorch.txt
uv pip install --python $PY -r $VLA_ENV/tmp/libero_req_notorch.txt -r third_party/libero/requirements.txt \
  "torch==2.4.1+cpu" --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
uv pip install --python $PY -e packages/openpi-client -e third_party/libero
# repo requirements beyond LIBERO (keep numpy 1.22.4 that LIBERO/robosuite pin)
uv pip install --python $PY "matplotlib==3.7.5" "pyflakes>=3.0" "numpy==1.22.4"
$PY -c "import numpy, torch, matplotlib, mujoco, robosuite; print(numpy.__version__, torch.__version__, matplotlib.__version__, mujoco.__version__, robosuite.__version__)"
echo SETUP_LIBERO_DONE
