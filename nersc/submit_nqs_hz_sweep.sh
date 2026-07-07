#!/bin/bash
# hz phase-diagram sweep: one NQS (ToricCNN_gridinv, bosonic) per hz value at a
# FIXED hx, as a Slurm job ARRAY (array index -> hz), for one system size L.
# Mirrors the validated train invocation of submit_nqs_gridinv.sh and the
# array-index->hz idiom of submit_ed_sweep.sh.
#
#   L=3 sbatch --array=0-15 nersc/submit_nqs_hz_sweep.sh          # validate (fast)
#   L=4 sbatch --array=0-15 nersc/submit_nqs_hz_sweep.sh
#   L=6 sbatch --array=0-15 nersc/submit_nqs_hz_sweep.sh          # L=6 defaults: diag_shift=1e-2, n_iter=200
#
# The --array size MUST equal HZ_N (default 16). hz_i = HZ_MIN + i*(HZ_MAX-HZ_MIN)/(HZ_N-1).
# Re-submitting the same array continues any unfinished point (--resume is always on).
#
# AUTO_RESUBMIT=1 makes each array task requeue ITSELF (its own index only) ~180 s
# before the wall limit, so a long/self-healing point survives across jobs without
# a manual re-submit (opt-in; off by default). Change 2a guarantees the checkpoint
# it resumes from is sane. Independent of the in-run divergence guard.
#SBATCH --job-name=tc-hzsweep
#SBATCH --account=m5340_g
#SBATCH --qos=shared
#SBATCH --constraint=gpu
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=05:00:00
#SBATCH --array=0-15
#SBATCH --signal=B:USR1@180
#SBATCH --output=%x-%A_%a.out
set -euo pipefail

module load conda
conda activate tc-nqs

REPO="${REPO:-$HOME/threed_TC/ThreeD_TC}"
cd "$REPO" || { echo "[submit] REPO not found: $REPO — set REPO=<clone path>"; exit 1; }

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# ---- sweep definition (fixed hx, swept hz) -----------------------------------
L="${L:-3}"
HX="${HX:-0.2}"
HZ_MIN="${HZ_MIN:-0.0}"
HZ_MAX="${HZ_MAX:-0.9}"
HZ_N="${HZ_N:-16}"                 # MUST match the --array size
HZ=$(python -c "print(round($HZ_MIN + ${SLURM_ARRAY_TASK_ID}*($HZ_MAX-$HZ_MIN)/($HZ_N-1), 4))")

# ---- architecture / optimization (validated gridinv config) ------------------
BC="${BC:-OBC}"
NONINV="${NONINV:-4}"; N_NONINV="${N_NONINV:-2}"; INV="${INV:-2 2 2}"
# kernel_size per L: L-1 grows the receptive field for long-range entanglement,
# but held at 4 for L=6 so the ~(L-1)^3 conv cost doesn't explode — depth (the
# stacked layers) is expected to carry the rest. Override any with KERNEL=.
if [ -z "${KERNEL:-}" ]; then
  case "$L" in
    3) KERNEL=2 ;; 4) KERNEL=3 ;; 5) KERNEL=4 ;; 6) KERNEL=4 ;; *) KERNEL=4 ;;
  esac
fi
DT="${DT:-0.01}"; LR_MIN="${LR_MIN:-0.001}"
# Per-L defaults (env-overridable). L=6 reproducibly blew up at 1e-3 *after*
# convergence (ill-conditioned SR solve) across multiple hz, so it runs at 1e-2
# with a shorter n_iter (both points had converged by step ~90-150 — less time in
# the ill-conditioned post-convergence regime). Lower L keep 1e-3 / 300. The
# in-run divergence guard is the backstop, not a substitute.
case "$L" in
  6) DIAG_SHIFT="${DIAG_SHIFT:-1e-2}"; N_ITER="${N_ITER:-200}" ;;
  *) DIAG_SHIFT="${DIAG_SHIFT:-1e-3}"; N_ITER="${N_ITER:-300}" ;;
esac
N_SAMPLES="${N_SAMPLES:-8192}"; N_CHAINS="${N_CHAINS:-1024}"
N_SWEEPS="${N_SWEEPS:-48}"; QGT="${QGT:-dense}"; CKPT_EVERY="${CKPT_EVERY:-10}"
CHUNK="${CHUNK:-2048}"             # 2048 covers L=3..6 (memory, not speed)

KERNEL_FLAG=""; [ "$KERNEL" != "0" ] && KERNEL_FLAG="--kernel_size $KERNEL"
CHUNK_FLAG="";  [ -n "$CHUNK" ]      && CHUNK_FLAG="--chunk_size $CHUNK"

OUT_DIR="${OUT_DIR:-$PSCRATCH/tc_nqs/phase_hx${HX}/L${L}}"
mkdir -p "$OUT_DIR"
NAME="bosonic_gridinv_L${L}_hx${HX}_hz${HZ}"

# ---- auto-resubmit just before the wall limit (opt-in) -----------------------
# Requeue only THIS task's index so hz is recomputed identically; the checkpoint
# on $PSCRATCH is the hand-off (--resume below). Carry HZ_MIN/MAX/N so the array-
# index -> hz math is unchanged on the requeue.
RESUB_COUNT="${RESUB_COUNT:-0}"
MAX_RESUBMITS="${MAX_RESUBMITS:-8}"
requeue() {
  if [ "${AUTO_RESUBMIT:-0}" = "1" ] && [ "$RESUB_COUNT" -lt "$MAX_RESUBMITS" ]; then
    echo "[hzsweep] wall limit near — resubmitting task ${SLURM_ARRAY_TASK_ID} (resume #$((RESUB_COUNT+1)))"
    RESUB_COUNT=$((RESUB_COUNT+1)) L="$L" HX="$HX" HZ_MIN="$HZ_MIN" HZ_MAX="$HZ_MAX" HZ_N="$HZ_N" \
      BC="$BC" DT="$DT" LR_MIN="$LR_MIN" DIAG_SHIFT="$DIAG_SHIFT" NONINV="$NONINV" \
      N_NONINV="$N_NONINV" INV="$INV" KERNEL="$KERNEL" N_ITER="$N_ITER" N_SAMPLES="$N_SAMPLES" \
      N_CHAINS="$N_CHAINS" N_SWEEPS="$N_SWEEPS" QGT="$QGT" CKPT_EVERY="$CKPT_EVERY" CHUNK="$CHUNK" \
      OUT_DIR="$OUT_DIR" AUTO_RESUBMIT=1 MAX_RESUBMITS="$MAX_RESUBMITS" \
      sbatch --array="${SLURM_ARRAY_TASK_ID}" "$0"
  fi
  exit 0
}
trap requeue USR1

echo "[hzsweep] task ${SLURM_ARRAY_TASK_ID}/$((HZ_N-1)): L=$L $BC hx=$HX hz=$HZ "\
"diag_shift=$DIAG_SHIFT n_iter=$N_ITER (resume #$RESUB_COUNT) -> $OUT_DIR/$NAME"

# `srun ... &` + `wait` so the USR1 trap fires promptly (a foreground srun would
# swallow the signal until it returns).
srun -n 1 python -u -m Three_TC.train \
  --L "$L" --bc "$BC" --model bosonic --arch ToricCNN_gridinv \
  --hx "$HX" --hz "$HZ" \
  --noninv_channels "$NONINV" --n_noninv "$N_NONINV" --inv_hidden $INV $KERNEL_FLAG \
  --dt "$DT" --lr_min "$LR_MIN" --diag_shift "$DIAG_SHIFT" --qgt "$QGT" \
  --n_iter "$N_ITER" --n_samples "$N_SAMPLES" --n_chains "$N_CHAINS" \
  --n_sweeps "$N_SWEEPS" $CHUNK_FLAG \
  --checkpoint_every "$CKPT_EVERY" --resume \
  --out_dir "$OUT_DIR" --name "$NAME" \
  --wandb_group "hzsweep-L${L}-hx${HX}" --wandb_offline &
wait
