# Keep every cache/temp file under $VLA_ENV.
export VLA_ENV=${VLA_ENV:-$HOME/disk2/vla-env}
export XDG_CACHE_HOME=$VLA_ENV/cache
export UV_CACHE_DIR=$VLA_ENV/cache/uv
export UV_PYTHON_INSTALL_DIR=$VLA_ENV/python
export OPENPI_DATA_HOME=$VLA_ENV/cache/openpi
export HF_HOME=$VLA_ENV/cache/huggingface
export TORCHINDUCTOR_CACHE_DIR=$VLA_ENV/cache/torchinductor
export TRITON_CACHE_DIR=$VLA_ENV/cache/triton
export PIP_CACHE_DIR=$VLA_ENV/cache/pip
export MPLCONFIGDIR=$VLA_ENV/cache/matplotlib
export TMPDIR=$VLA_ENV/tmp
export PATH=$HOME/.local/bin:$PATH
# Copy instead of hardlink: openpi patches transformers in-place, which would otherwise alter the uv cache.
export UV_LINK_MODE=copy
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export LIBERO_CONFIG_PATH=$VLA_ENV/libero-config
export OPENPI_PY=$VLA_ENV/openpi/.venv/bin/python
export LIBERO_PY=$VLA_ENV/openpi/examples/libero/.venv/bin/python
export CKPT=$VLA_ENV/checkpoints/pi05_libero_pytorch
export PYTHONPATH=$VLA_ENV/openpi/third_party/libero${PYTHONPATH:+:$PYTHONPATH}
# openpi imports JAX even on the PyTorch path; keep it off the GPU (no Blackwell support, 75% preallocation).
export JAX_PLATFORMS=cpu
