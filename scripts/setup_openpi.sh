#!/usr/bin/env bash
set -euxo pipefail
source ${VLA_ENV:-$HOME/disk2/vla-env}/env.sh
cd $VLA_ENV/openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync
.venv/bin/python -c "import torch,transformers;print(\"before:\",torch.__version__,torch.version.cuda,transformers.__version__)"
# Blackwell (sm_120) needs the CUDA 12.8 build of the pinned torch 2.7.1.
uv pip install --python .venv/bin/python "torch==2.7.1" --index-url https://download.pytorch.org/whl/cu128 --reinstall-package torch
cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/
.venv/bin/python -c "import torch;print(\"after:\",torch.__version__,torch.version.cuda,torch.cuda.get_device_name(0),torch.cuda.get_arch_list());x=torch.randn(1024,1024,device=\"cuda\");print(\"matmul ok\",(x@x).sum().item()!=0)"
echo SETUP_OPENPI_DONE
