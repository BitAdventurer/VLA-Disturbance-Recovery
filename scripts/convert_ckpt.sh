#!/usr/bin/env bash
set -euxo pipefail
source ${VLA_ENV:-$HOME/disk2/vla-env}/env.sh
SRC=$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_libero
DST=$VLA_ENV/checkpoints/pi05_libero_pytorch
cd $VLA_ENV/openpi
JAX_PLATFORMS=cpu .venv/bin/python examples/convert_jax_model_to_pytorch.py \
  --checkpoint_dir $SRC --config_name pi05_libero --output_path $DST
# The converter looks for assets next to (not inside) checkpoint_dir; copy the norm stats explicitly.
rm -rf $DST/assets && cp -r $SRC/assets $DST/assets
ls -la $DST; find $DST/assets -type f; cat $DST/config.json
echo CONVERT_DONE
