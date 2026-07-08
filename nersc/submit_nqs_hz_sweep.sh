#!/bin/bash
# hz phase-diagram sweep: one NQS (ToricCNN_gridinv, bosonic) per hz value at a
# FIXED hx, as a Slurm job ARRAY (array index -> hz), for one system size L.
# Mirrors the validated train invocation of submit_nqs_gridinv.sh and the
# array-index->hz idiom of submit_ed_sweep.sh.
#
#   L=3 sbatch --array=0-15 nersc/submit_nqs_hz_sweep.sh          # validate (fast)
#   L=4 sbatch --array=0-15 nersc/submit_nqs_hz_sweep.sh
#   L=6 sbatch --array=0-15 nersc/submit_nqs_hz_sweep.sh          # L=6,7 default diag_shift=1e-2 (n_iter=150 all L)
#
# Usually launched via nersc/run_phase_campaign.sh, which sets HX/L/HZ_* per cut
# and passes a per-L --time (WALLTIME) so small-L jobs backfill sooner.
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
#SBATCH --output=slurm_logs/%x-%A_%a.out
set -euo pipefail

module load conda
conda activate tc-nqs

REPO="${REPO:-$HOME/threed_TC/ThreeD_TC}"
cd "$REPO" || { echo "[submit] REPO not found: $REPO — set REPO=<clone path>"; exit 1; }

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Persistent XLA compilation cache: the first-step compile is expensive at large
# L (get_conn over ~L^3 sites + dense QGT, ~15 min at L=7). Caching to $PSCRATCH
# lets every later hz point / array task / resume reuse the compiled kernels
# instead of recompiling. Safe to share across L (keyed by HLO, so shapes differ).
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$PSCRATCH/jax_cache}"
mkdir -p "$JAX_COMPILATION_CACHE_DIR"

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
# kernel_size per L: L-1 grows the receptive field for long-range entanglement.
# L=6 held at 4 (cost). L=7 uses 5 -- the smoke showed k4 too small and k5 the sweet
# spot (k6 converges no lower). Override any with KERNEL=.
if [ -z "${KERNEL:-}" ]; then
  case "$L" in
    3) KERNEL=2 ;; 4) KERNEL=3 ;; 5) KERNEL=4 ;; 6) KERNEL=4 ;; 7) KERNEL=5 ;; *) KERNEL=4 ;;
  esac
fi
DT="${DT:-0.01}"; LR_MIN="${LR_MIN:-0.001}"
# Per-L defaults (env-overridable). diag_shift: L=6,7 at 5e-3 (1e-2 over-regularized
# the descent; the in-run divergence guard is the backstop for the large-L SR blow-up
# that originally motivated 1e-2); L<=5 keep the proven 1e-3. n_iter: L=7 needs 175
# (converges ~150-175 at 882 sites), others 150.
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

OUT_DIR="${OUT_DIR:-$PSCRATCH/tc_nqs/phase_hx${HX}/L${L}}"
mkdir -p "$OUT_DIR"
NAME="bosonic_gridinv_L${L}_hx${HX}_hz${HZ}"

# ---- auto-resubmit just before the wall limit (opt-in) -----------------------
# Requeue only THIS task's index so hz is recomputed identically; the checkpoint
# on $PSCRATCH is the hand-off (--resume below). Carry HZ_MIN/MAX/N so the array-
# index -> hz math is unchanged on the requeue.
RESUB_COUNT="${RESUB_COUNT:-0}"
MAX_RESUBMITS="${MAX_RESUBMITS:-8}"
# Per-L wall limit set by the driver on the sbatch command line (overrides the
# #SBATCH --time directive). Propagate it through requeues, else the resubmit
# reverts to the 5 h fallback directive.
WALLTIME="${WALLTIME:-}"
requeue() {
  if [ "${AUTO_RESUBMIT:-0}" = "1" ] && [ "$RESUB_COUNT" -lt "$MAX_RESUBMITS" ]; then
    echo "[hzsweep] wall limit near — resubmitting task ${SLURM_ARRAY_TASK_ID} (resume #$((RESUB_COUNT+1)))"
    local tflag=""; [ -n "$WALLTIME" ] && tflag="--time=$WALLTIME"
    RESUB_COUNT=$((RESUB_COUNT+1)) L="$L" HX="$HX" HZ_MIN="$HZ_MIN" HZ_MAX="$HZ_MAX" HZ_N="$HZ_N" \
      BC="$BC" DT="$DT" LR_MIN="$LR_MIN" DIAG_SHIFT="$DIAG_SHIFT" NONINV="$NONINV" \
      N_NONINV="$N_NONINV" INV="$INV" KERNEL="$KERNEL" N_ITER="$N_ITER" N_SAMPLES="$N_SAMPLES" \
      N_CHAINS="$N_CHAINS" N_SWEEPS="$N_SWEEPS" QGT="$QGT" CKPT_EVERY="$CKPT_EVERY" CHUNK="$CHUNK" \
      OUT_DIR="$OUT_DIR" AUTO_RESUBMIT=1 MAX_RESUBMITS="$MAX_RESUBMITS" WALLTIME="$WALLTIME" \
      sbatch $tflag --array="${SLURM_ARRAY_TASK_ID}" "$0"
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
