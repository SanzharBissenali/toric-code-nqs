#!/bin/bash
# One-time build of the GPU NQS (NetKet / JAX) environment on Perlmutter.
# Run this on a LOGIN node (not a compute node). Takes a few minutes.
#
#   bash nersc/setup_conda_gpu.sh
#
# Mirrors setup_conda_cpu.sh (tag 2d-final) but installs the CUDA12 JAX stack for the
# variational (NetKet) path. Versions are pinned to match requirements.txt
# (jax 0.5.2 / jaxlib 0.5.1) plus the CUDA plugin/pjrt at the jaxlib version.
set -euo pipefail

ENV_NAME=tc-nqs
PROJ="${PROJ:-}"   # e.g. export PROJ=mXXXX -> env in /global/common/software/$PROJ/conda

module load conda

if [[ -n "$PROJ" ]]; then
  TARGET="/global/common/software/$PROJ/conda/$ENV_NAME"
  echo "Creating env at $TARGET"
  conda create --yes --prefix "$TARGET" python=3.12
  conda activate "$TARGET"
else
  echo "Creating named env '$ENV_NAME' (default location)"
  conda create --yes --name "$ENV_NAME" python=3.12
  conda activate "$ENV_NAME"
fi

# NQS stack: CUDA12 JAX + NetKet/Flax/optax + wandb. The `[with-cuda]` extra on
# the plugin is REQUIRED: it pulls the nvidia-*-cu12 wheels (cuDNN, cuBLAS, CUDA
# runtime) so JAX uses its OWN bundled cuDNN. Without it, jaxlib falls back to the
# system cudatoolkit's cuDNN (a different version) -> CUDNN_STATUS_INTERNAL_ERROR
# on the first conv. With it, no `module load cudatoolkit` is needed.
pip install --no-cache-dir \
  jax==0.5.2 jaxlib==0.5.1 "jax-cuda12-plugin[with-cuda]==0.5.1" jax-cuda12-pjrt==0.5.1 \
  netket==3.16.1.post1 flax==0.10.4 optax \
  "numpy==2.1.3" "scipy==1.15.2" numba tqdm wandb

# Confirm JAX sees the GPU (run again on a compute node — login nodes have none).
python - <<'PY'
import jax
print("jax", jax.__version__, "devices:", jax.devices())
print("has gpu:", any("cuda" in str(d).lower() for d in jax.devices()))
PY
echo "Done. Activate later with:  module load conda && conda activate $ENV_NAME"
echo "Then authenticate wandb once on the LOGIN node:  wandb login"

# make tc3d importable from any cwd (tests, analysis scripts)
pip install -e "$HOME/toric-code-nqs" --no-deps
