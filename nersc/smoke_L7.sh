#!/bin/bash
# General L=7 architecture/optimization smoke, at fixed (hx,hz)=(0.2,0.2). Probes
# the invariant-block kernel size, SR regularization (diag_shift), and inv depth
# so we can gauge their COST (per-step time, esp. the dense-QGT jump with n_params)
# and their effect on the descent. Batch, watch with tail -f.
#
#   KERNEL=6 sbatch nersc/smoke_L7.sh                     # bigger kernel, dense QGT
#   KERNEL=5 sbatch nersc/smoke_L7.sh
#   KERNEL=4 DIAG_SHIFT=1e-3 sbatch nersc/smoke_L7.sh     # optimization control
#   tail -f $PSCRATCH/tc_nqs/smoke_L7/smoke_L7_k6_ds1e-2.log
#
# The final Vscore line prints only on completion, so N_ITER is sized to finish
# inside --time. Dense QGT storage is n_params^2: k5 ~3.6 GB, k6 ~10 GB (both fit
# 40 GB), k7 ~25 GB (likely OOM -- use --qgt onthefly / SRt instead). Each new
# KERNEL is a fresh XLA compile (the cache only holds k4).
#SBATCH --job-name=smoke-L7
#SBATCH --account=m5340_g
#SBATCH --qos=shared
#SBATCH --constraint=gpu
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=06:00:00
#SBATCH --output=slurm_logs/smoke_L7_%j.out
set -euo pipefail

module load conda
conda activate tc-nqs

REPO="${REPO:-$HOME/threed_TC/ThreeD_TC}"
cd "$REPO" || { echo "[smoke] REPO not found: $REPO"; exit 1; }

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$PSCRATCH/jax_cache}"
mkdir -p "$JAX_COMPILATION_CACHE_DIR"

KERNEL="${KERNEL:-5}"           # k5 is the L=7 sweet spot (k6 converges no lower)
DIAG_SHIFT="${DIAG_SHIFT:-5e-3}"  # 1e-2 over-regularized the descent
INV="${INV:-2 2 2}"             # invariant-block widths (depth = number of entries)
QGT="${QGT:-dense}"             # srt/minsr for sample-space solve at large kernel
N_ITER="${N_ITER:-175}"         # must finish inside --time so the final Vscore prints
RESUME="${RESUME:-0}"           # 1 -> continue the same NAME from its step-checkpoint
OUT_DIR="${OUT_DIR:-$PSCRATCH/tc_nqs/smoke_L7}"; mkdir -p "$OUT_DIR"
SUF=""; [ "$QGT" != "dense" ] && SUF="_${QGT}"    # keep dense names; srt gets its own
NAME="smoke_L7_k${KERNEL}_ds${DIAG_SHIFT}${SUF}"

# Resume must pass the SAME architecture (KERNEL/INV/...) so the checkpoint weights
# reload into a matching vstate. Checkpoint every 10 steps so a 200-step run is
# itself resumable if it ever runs long.
RESUME_FLAG=""; TEE_FLAG=""
if [ "$RESUME" = "1" ]; then RESUME_FLAG="--resume"; TEE_FLAG="-a"; fi

echo "[smoke] $NAME  kernel=$KERNEL  inv_hidden='$INV'  diag_shift=$DIAG_SHIFT  "\
"qgt=$QGT  n_iter=$N_ITER  resume=$RESUME  -> $OUT_DIR/$NAME"

srun -n 1 python -u -m Three_TC.train \
  --L 7 --bc OBC --model bosonic --arch ToricCNN_gridinv \
  --hx 0.2 --hz 0.2 \
  --noninv_channels 4 --n_noninv 2 --inv_hidden $INV --kernel_size "$KERNEL" \
  --dt 0.01 --lr_min 0.001 --diag_shift "$DIAG_SHIFT" --qgt "$QGT" \
  --n_iter "$N_ITER" --n_samples 8192 --n_chains 1024 --n_sweeps 48 --chunk_size 2048 \
  --checkpoint_every 10 $RESUME_FLAG \
  --no_wandb --out_dir "$OUT_DIR" --name "$NAME" \
  2>&1 | tee $TEE_FLAG "$OUT_DIR/${NAME}.log"
