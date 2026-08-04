#!/bin/bash
# hx phase-diagram sweep: one NQS (ToricCNN_gridinv, bosonic) per hx value at a
# FIXED hz, as a Slurm job ARRAY (array index -> hx), for one system size L.
# The orthogonal cut to submit_nqs_hz_sweep.sh (which fixes hx, sweeps hz): this
# fixes hz (default 0.0) and sweeps hx -- the magnetic (sigma^x) transition,
# expected near hx~1.0. Everything else mirrors the hz sweep verbatim (per-L
# kernel/diag_shift/n_iter, compile cache, --resume, auto-resubmit).
#
#   L=4 sbatch --time=01:00:00 --array=0-14 nersc/submit_nqs_hx_sweep.sh
#   L=7 sbatch --time=06:00:00 --array=0-14 nersc/submit_nqs_hx_sweep.sh
#
# The --array size MUST equal HX_N (default 15). hx_i = HX_MIN + i*(HX_MAX-HX_MIN)/(HX_N-1).
# Re-submitting the same array continues any unfinished point (--resume is always on).
# Outputs land in a SEPARATE tree ($PSCRATCH/tc_nqs/phase_hz${HZ}/L${L}) so they never
# collide with the hx-cut (phase_hx*/) data.
#SBATCH --job-name=tc-hxsweep
#SBATCH --account=m5340_g
#SBATCH --qos=shared
#SBATCH --constraint=gpu
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=05:00:00
#SBATCH --array=0-14
#SBATCH --signal=B:USR1@180
#SBATCH --output=slurm_logs/%x-%A_%a.out
set -euo pipefail

module load conda
conda activate tc-nqs

REPO="${REPO:-$HOME/toric-code-nqs}"
cd "$REPO" || { echo "[submit] REPO not found: $REPO — set REPO=<clone path>"; exit 1; }

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Persistent XLA compilation cache (shared across L, keyed by HLO) — see hz sweep.
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$PSCRATCH/jax_cache}"
mkdir -p "$JAX_COMPILATION_CACHE_DIR"

# ---- sweep definition (fixed hz, swept hx) -----------------------------------
L="${L:-4}"
HZ="${HZ:-0.0}"
HX_MIN="${HX_MIN:-0.8}"
HX_MAX="${HX_MAX:-1.3}"
HX_N="${HX_N:-15}"                 # MUST match the --array size
HX=$(python -c "print(round($HX_MIN + ${SLURM_ARRAY_TASK_ID}*($HX_MAX-$HX_MIN)/($HX_N-1), 4))")

# ---- architecture / optimization (validated gridinv config; per-L, field-independent)
BC="${BC:-OBC}"
NONINV="${NONINV:-4}"; N_NONINV="${N_NONINV:-2}"; INV="${INV:-2 2 2}"
# kernel_size per L: L-1 grows the receptive field for LRE; L=6 held at 4 (cost),
# L=7 at 5 (smoke sweet spot). Override with KERNEL=.
if [ -z "${KERNEL:-}" ]; then
  case "$L" in
    3) KERNEL=2 ;; 4) KERNEL=3 ;; 5) KERNEL=4 ;; 6) KERNEL=4 ;; 7) KERNEL=5 ;; *) KERNEL=4 ;;
  esac
fi
DT="${DT:-0.01}"; LR_MIN="${LR_MIN:-0.001}"
# Per-L defaults (env-overridable): L=6,7 diag_shift 5e-3 + L=7 n_iter 175; L<=5 diag 1e-3.
case "$L" in
  7) DIAG_SHIFT="${DIAG_SHIFT:-5e-3}"; N_ITER="${N_ITER:-175}" ;;
  6) DIAG_SHIFT="${DIAG_SHIFT:-5e-3}"; N_ITER="${N_ITER:-150}" ;;
  *) DIAG_SHIFT="${DIAG_SHIFT:-1e-3}"; N_ITER="${N_ITER:-150}" ;;
esac
N_SAMPLES="${N_SAMPLES:-8192}"; N_CHAINS="${N_CHAINS:-1024}"
N_SWEEPS="${N_SWEEPS:-48}"; QGT="${QGT:-dense}"; CKPT_EVERY="${CKPT_EVERY:-10}"
CHUNK="${CHUNK:-2048}"             # 2048 covers L=3..7 (memory, not speed)

KERNEL_FLAG=""; [ "$KERNEL" != "0" ] && KERNEL_FLAG="--kernel_size $KERNEL"
CHUNK_FLAG="";  [ -n "$CHUNK" ]      && CHUNK_FLAG="--chunk_size $CHUNK"

OUT_DIR="${OUT_DIR:-$PSCRATCH/tc_nqs/phase_hz${HZ}/L${L}}"
mkdir -p "$OUT_DIR"
NAME="bosonic_gridinv_L${L}_hx${HX}_hz${HZ}"

# ---- auto-resubmit just before the wall limit (opt-in) -----------------------
# Requeue only THIS task's index so hx is recomputed identically; the $PSCRATCH
# checkpoint is the hand-off (--resume below). Carry HX_MIN/MAX/N and HZ so the
# array-index -> hx math is unchanged on the requeue.
RESUB_COUNT="${RESUB_COUNT:-0}"
MAX_RESUBMITS="${MAX_RESUBMITS:-8}"
WALLTIME="${WALLTIME:-}"
requeue() {
  if [ "${AUTO_RESUBMIT:-0}" = "1" ] && [ "$RESUB_COUNT" -lt "$MAX_RESUBMITS" ]; then
    echo "[hxsweep] wall limit near — resubmitting task ${SLURM_ARRAY_TASK_ID} (resume #$((RESUB_COUNT+1)))"
    local tflag=""; [ -n "$WALLTIME" ] && tflag="--time=$WALLTIME"
    RESUB_COUNT=$((RESUB_COUNT+1)) L="$L" HZ="$HZ" HX_MIN="$HX_MIN" HX_MAX="$HX_MAX" HX_N="$HX_N" \
      BC="$BC" DT="$DT" LR_MIN="$LR_MIN" DIAG_SHIFT="$DIAG_SHIFT" NONINV="$NONINV" \
      N_NONINV="$N_NONINV" INV="$INV" KERNEL="$KERNEL" N_ITER="$N_ITER" N_SAMPLES="$N_SAMPLES" \
      N_CHAINS="$N_CHAINS" N_SWEEPS="$N_SWEEPS" QGT="$QGT" CKPT_EVERY="$CKPT_EVERY" CHUNK="$CHUNK" \
      OUT_DIR="$OUT_DIR" AUTO_RESUBMIT=1 MAX_RESUBMITS="$MAX_RESUBMITS" WALLTIME="$WALLTIME" \
      sbatch $tflag --array="${SLURM_ARRAY_TASK_ID}" "$0"
  fi
  exit 0
}
trap requeue USR1

echo "[hxsweep] task ${SLURM_ARRAY_TASK_ID}/$((HX_N-1)): L=$L $BC hz=$HZ hx=$HX "\
"diag_shift=$DIAG_SHIFT n_iter=$N_ITER (resume #$RESUB_COUNT) -> $OUT_DIR/$NAME"

# `srun ... &` + `wait` so the USR1 trap fires promptly.
srun -n 1 python -u -m tc3d.train \
  --L "$L" --bc "$BC" --model bosonic --arch ToricCNN_gridinv \
  --hx "$HX" --hz "$HZ" \
  --noninv_channels "$NONINV" --n_noninv "$N_NONINV" --inv_hidden $INV $KERNEL_FLAG \
  --dt "$DT" --lr_min "$LR_MIN" --diag_shift "$DIAG_SHIFT" --qgt "$QGT" \
  --n_iter "$N_ITER" --n_samples "$N_SAMPLES" --n_chains "$N_CHAINS" \
  --n_sweeps "$N_SWEEPS" $CHUNK_FLAG \
  --checkpoint_every "$CKPT_EVERY" --resume \
  --out_dir "$OUT_DIR" --name "$NAME" \
  --wandb_group "hxsweep-L${L}-hz${HZ}" --wandb_offline &
wait
