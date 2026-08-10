#!/bin/bash
# Re-evaluate saved train.py checkpoints at a larger sample budget
# (analysis/eval_ckpt.py): honest error bars for A_v/B_p/Mx/Mz (+ O_FM/S2 with
# TOPO=1, + the ParaToric-matched electric FM with FM_PARATORIC=1). Writes
# {name}{SUFFIX}.json next to each artifact; pull those tiny files local.
#
#   DIRS="$PSCRATCH/tc_nqs/tune_rect/hx0.2_hz0.1" GLOB='*k3_dt0.02.json' \
#     FM_PARATORIC=1 SUFFIX=.fm65k sbatch -q debug -t 00:30:00 nersc/submit_eval_ckpt.sh
#   DIRS="$PSCRATCH/tc_nqs/tune_rect_L5/hx0.2_hz0.1 ..." TOPO=1 FM_PARATORIC=1 \
#     sbatch nersc/submit_eval_ckpt.sh
#
# One srun per DIRS entry, sequential — the JAX compile is shared via the cache,
# so batching dirs into one job amortizes it (same trick as tc3d.sweep).
#SBATCH --job-name=tc-evalckpt
#SBATCH --account=m5340_g
#SBATCH --qos=shared
#SBATCH --constraint=gpu
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail

module load conda
conda activate tc-nqs

REPO="${REPO:-$HOME/toric-code-nqs}"
cd "$REPO" || { echo "[submit] REPO not found: $REPO"; exit 1; }

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1
# eval_ckpt is extract-style (no training wrapper) -> set the cache explicitly
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$PSCRATCH/tc_nqs/jax_cache}"

: "${DIRS:?set DIRS=\"dir1 [dir2 ...]\" (checkpoint directories on \$PSCRATCH)}"
GLOB="${GLOB:-*.json}"
EVAL_SAMPLES="${EVAL_SAMPLES:-65536}"
EVAL_CHAINS="${EVAL_CHAINS:-16}"
SUFFIX="${SUFFIX:-.eval65k}"
FLAGS=""
[ "${TOPO:-0}" = "1" ] && FLAGS="$FLAGS --topological"
[ "${FM_PARATORIC:-0}" = "1" ] && FLAGS="$FLAGS --fm_paratoric"
# X-membrane families (dual-basis ckpts): default scores corner-rule + R=1 anchor;
# narrow with FM_MEMBRANE_R="pt" or "1" (comma list, see eval_ckpt.py --fm_membrane_R)
[ "${FM_MEMBRANE_PARATORIC:-0}" = "1" ] && FLAGS="$FLAGS --fm_membrane_paratoric"
[ -n "${FM_MEMBRANE_R:-}" ] && FLAGS="$FLAGS --fm_membrane_R ${FM_MEMBRANE_R}"
[ "${SKIP_EXISTING:-0}" = "1" ] && FLAGS="$FLAGS --skip_existing"
# explicit sampler seed (post-4b6a797 the loader honors it; distinct SEEDs give
# distinct streams — use pairs of jobs as replica regression checks)
[ -n "${SEED:-}" ] && FLAGS="$FLAGS --seed ${SEED}"

for d in $DIRS; do
  echo "[eval] dir=$d glob=$GLOB samples=$EVAL_SAMPLES chains=$EVAL_CHAINS suffix=$SUFFIX flags=$FLAGS"
  srun -n 1 python -u analysis/eval_ckpt.py --dir "$d" --glob "$GLOB" \
    --eval_samples "$EVAL_SAMPLES" --eval_chains "$EVAL_CHAINS" \
    --suffix "$SUFFIX" $FLAGS
done
