#!/bin/bash
# One-off L=7 depth smoke: full production shape, fixed (hx,hz)=(0.2,0.2), for the
# d3 (inv_hidden "2 2 2") vs d4 ("2 2 2 2") architecture comparison. Batch version
# of the interactive smoke so it can run unattended and be watched with tail -f.
#
#   DEPTH=d3 sbatch nersc/smoke_L7_depth.sh
#   DEPTH=d4 sbatch nersc/smoke_L7_depth.sh
#   tail -f $PSCRATCH/tc_nqs/smoke_L7/smoke_L7_d3.log
#
# n_iter is capped so the run FINISHES inside the 2 h wall -- the final Vscore line
# (train.py: "[train] done ... Vscore=...") only prints on completion; a wall-clock
# kill mid-training would skip it. At ~100 s/step (d3) / ~116 s/step (d4), 50 steps
# is ~85 / ~99 min. Reuses the shared JAX compile cache from the interactive runs.
#SBATCH --job-name=smoke-L7
#SBATCH --account=m5340_g
#SBATCH --qos=shared
#SBATCH --constraint=gpu
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --output=smoke_L7_%j.out
set -euo pipefail

module load conda
conda activate tc-nqs

REPO="${REPO:-$HOME/threed_TC/ThreeD_TC}"
cd "$REPO" || { echo "[smoke] REPO not found: $REPO"; exit 1; }

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$PSCRATCH/jax_cache}"
mkdir -p "$JAX_COMPILATION_CACHE_DIR"

DEPTH="${DEPTH:-d3}"
case "$DEPTH" in
  d3) INV="2 2 2" ;;
  d4) INV="2 2 2 2" ;;
  *)  echo "[smoke] DEPTH must be d3 or d4 (got '$DEPTH')"; exit 1 ;;
esac
N_ITER="${N_ITER:-50}"          # must finish inside --time so the final Vscore prints
OUT_DIR="${OUT_DIR:-$PSCRATCH/tc_nqs/smoke_L7}"; mkdir -p "$OUT_DIR"
NAME="smoke_L7_${DEPTH}"

echo "[smoke] $NAME  inv_hidden='$INV'  n_iter=$N_ITER  -> $OUT_DIR/$NAME"

srun -n 1 python -u -m Three_TC.train \
  --L 7 --bc OBC --model bosonic --arch ToricCNN_gridinv \
  --hx 0.2 --hz 0.2 \
  --noninv_channels 4 --n_noninv 2 --inv_hidden $INV --kernel_size 4 \
  --dt 0.01 --lr_min 0.001 --diag_shift 1e-2 --qgt dense \
  --n_iter "$N_ITER" --n_samples 8192 --n_chains 1024 --n_sweeps 48 --chunk_size 2048 \
  --no_wandb --out_dir "$OUT_DIR" --name "$NAME" \
  2>&1 | tee "$OUT_DIR/${NAME}.log"
