#!/bin/bash
# 2D toric-code hz phase-diagram sweep: one NQS (Combo CNN, bosonic surface code)
# per hz value at a FIXED hx, as a Slurm job ARRAY (array index -> hz), for one
# system size L. The 2D analogue of submit_nqs_hz_sweep.sh — same array-index->hz
# idiom, --resume, XLA cache, and (opt-in) self-requeue, but driving the 2D
# trainer `python -m model.train_2d` on model/ geometry (N = 2L^2 - 2L qubits,
# so L up to 10 is cheap: N=24/60/112/180 for L=4/6/8/10).
#
# For the known analytic cut hx=hy=0 the σ^z transition is 2nd order, 3D-Ising,
# at hz_c ≈ 0.328 J (TC → (2+1)D TFIM). Center the coarse grid on that.
#
# Pass the per-L --time (must match the WALLTIME set per L below; shorter L backfills
# sooner on shared QOS). L>=10 needs chunk_size (set automatically below) or the minSR
# local-energy step OOMs a 40 GB GPU (L=12 unchunked wants ~149 GiB). Pair large L with
# AUTO_RESUBMIT=1: --resume + self-requeue continues from the last checkpoint if the
# (uncertain, ~53 s/step at L=10) step time overruns the wall.
#   L=6  HX=0.0 sbatch --time=00:25:00 --array=0-15 nersc/submit_2d_hz_sweep.sh   # coarse [0.20,0.42]
#   L=8  HX=0.0 sbatch --time=00:50:00 --array=0-15 nersc/submit_2d_hz_sweep.sh
#   L=10 HX=0.0 AUTO_RESUBMIT=1 sbatch --time=04:00:00 --array=0-15 nersc/submit_2d_hz_sweep.sh
#   L=12 HX=0.0 AUTO_RESUBMIT=1 sbatch --time=06:00:00 --array=0-15 nersc/submit_2d_hz_sweep.sh
#   # refine near the crossing once located:
#   L=8  HX=0.0 HZ_MIN=0.28 HZ_MAX=0.36 HZ_N=12 sbatch --time=00:50:00 --array=0-11 nersc/submit_2d_hz_sweep.sh
#   # extend the UPPER tail to hz=0.55 (the finite-L O_FM inflection sits near the old
#   # 0.42 edge, so the high-hz plateau is missing). Grid 0.43..0.55 step ~0.015 has NO
#   # collision with the <=0.42 points; drops into the SAME phase_hx0.0/L* dir, so
#   # analysis/fm_2d.py auto-merges old+new into one hz curve:
#   L=6  HX=0.0 HZ_MIN=0.43 HZ_MAX=0.55 HZ_N=9 sbatch --time=00:25:00 --array=0-8 nersc/submit_2d_hz_sweep.sh
#   L=8  HX=0.0 HZ_MIN=0.43 HZ_MAX=0.55 HZ_N=9 sbatch --time=00:50:00 --array=0-8 nersc/submit_2d_hz_sweep.sh
#   L=10 HX=0.0 HZ_MIN=0.43 HZ_MAX=0.55 HZ_N=9 AUTO_RESUBMIT=1 sbatch --time=04:00:00 --array=0-8 nersc/submit_2d_hz_sweep.sh
#   L=12 HX=0.0 HZ_MIN=0.43 HZ_MAX=0.55 HZ_N=9 AUTO_RESUBMIT=1 sbatch --time=06:00:00 --array=0-8 nersc/submit_2d_hz_sweep.sh
#
# FSS uses L in {6,8,10,12} (the bulk FM loop side R=L-3 gives R=3,5,7,9). L=4
# (R=1) is a single-plaquette probe, not a Wilson loop, so it is intentionally
# dropped from the size series.
#
# The --array size MUST equal HZ_N (default 16). hz_i = HZ_MIN + i*(HZ_MAX-HZ_MIN)/(HZ_N-1).
# Re-submitting the same array continues any unfinished point (--resume is always on).
# AUTO_RESUBMIT=1 makes each task requeue its own index ~120 s before the wall limit.
#SBATCH --job-name=tc2d-hzsweep
#SBATCH --account=m5340_g
#SBATCH --qos=shared
#SBATCH --constraint=gpu
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --array=0-15
#SBATCH --signal=B:USR1@120
#SBATCH --output=slurm_logs/%x-%A_%a.out
set -euo pipefail

module load conda
conda activate tc-nqs

# Same clone as the 3D pipeline: this repo holds both model/ (2D) and Three_TC/ (3D).
REPO="${REPO:-$HOME/threed_TC/ThreeD_TC}"
cd "$REPO" || { echo "[submit] REPO not found: $REPO — set REPO=<clone path>"; exit 1; }

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$PSCRATCH/jax_cache}"
mkdir -p "$JAX_COMPILATION_CACHE_DIR"

# ---- sweep definition (fixed hx, swept hz) -----------------------------------
L="${L:-6}"
HX="${HX:-0.0}"
HZ_MIN="${HZ_MIN:-0.20}"           # coarse window brackets hz_c≈0.328 (3D-Ising)
HZ_MAX="${HZ_MAX:-0.42}"
HZ_N="${HZ_N:-16}"                 # MUST match the --array size
HZ=$(python -c "print(round($HZ_MIN + ${SLURM_ARRAY_TASK_ID}*($HZ_MAX-$HZ_MIN)/($HZ_N-1), 4))")

# ---- architecture / optimization (Combo CNN; builders_2d + TRAIN_DEFAULTS) ----
# Capacity is NOT the bottleneck here (convergence is), so the architecture is
# FIXED across L (channels + non-invariant kernel) and we scale only the training
# effort (n_iter, n_samples). The invariant branch's hardwired (L-1)^2 receptive
# field + Wilson features do the heavy lifting; the non-invariant edge conv is a
# mild symmetry-breaking correction, so kernel=3 (well past its useful floor) is
# ample. QGT pinned to minSR for a UNIFORM solver across the FSS series (n_params
# ~ n_samples here, so it's near the dense/minSR boundary; pinning avoids a mixed
# solver whose diag_shift would regularize in slightly different spaces). Overridable.
BC="${BC:-OBC}"
ARCH="${ARCH:-Combo}"
NONINV="${NONINV:-1 16}"           # --channels_noninv (fixed across L)
INV="${INV:-16 8 1}"              # --channels_inv   (fixed across L)
KERNEL="${KERNEL:-3}"              # non-invariant edge-conv kernel, FIXED across L
RESCALE="${RESCALE:-1.0}"          # Wilson-nonlinearity rescale (no-op for real dtype)
QGT="${QGT:-minsr}"                # pinned: uniform solver across the L series
# WALLTIME below is the PER-RUN (per array task) estimate; it does NOT set the
# #SBATCH --time directive (parsed before this runs) — pass it on the sbatch line
# (`sbatch --time=$WALLTIME ...`, see header). It IS propagated through requeues.
# Anchored on a MEASURED L=6 ~3 s/step (sampling-bound; step-time ~ n_sweeps=2N),
# extrapolated to L=8/10/12 (~6/9/16 s/step) as ~ n_iter*step + compile. Leaned
# toward the estimate, NOT padded: a timed-out run resumes (--resume) and its
# latest .ckpt is still scorable by analysis/fm_2d.py, so under-shooting is cheap.
# CHUNK_DEF: chunk_size for the local-energy estimator (minSR's local_estimators
# materializes ~ n_samples x N x #offdiag-H-terms at once). L<=8 fits a 40 GB A100
# unchunked; L=10 is right at the edge (~35 GiB -> intermittent OOM) and L=12 needs
# ~149 GiB unchunked -> hard OOM. chunk_size=1024 caps that at <10 GiB for both.
# n_iter is convergence-sized (L=6/8 flatten by ~step 40; ~150-200 is ample) rather
# than the old 500/600, which at the MEASURED L=10 ~53 s/step could never finish.
# Walltimes anchored on that measured 53 s/step (L=10); pair large L with
# AUTO_RESUBMIT=1 so a step-time surprise self-continues from the last checkpoint.
case "$L" in
  6)  N_ITER="${N_ITER:-300}"; N_SAMPLES="${N_SAMPLES:-4096}"; DIAG_SHIFT="${DIAG_SHIFT:-1e-3}"; DT="${DT:-0.02}"; LR_MIN="${LR_MIN:-0.002}"; CHUNK_DEF="";   WALLTIME="${WALLTIME:-00:25:00}" ;;
  8)  N_ITER="${N_ITER:-400}"; N_SAMPLES="${N_SAMPLES:-8192}"; DIAG_SHIFT="${DIAG_SHIFT:-1e-3}"; DT="${DT:-0.02}"; LR_MIN="${LR_MIN:-0.002}"; CHUNK_DEF="";   WALLTIME="${WALLTIME:-00:50:00}" ;;
  10) N_ITER="${N_ITER:-200}"; N_SAMPLES="${N_SAMPLES:-8192}"; DIAG_SHIFT="${DIAG_SHIFT:-2e-3}"; DT="${DT:-0.02}"; LR_MIN="${LR_MIN:-0.002}"; CHUNK_DEF=1024; WALLTIME="${WALLTIME:-04:00:00}" ;;
  12) N_ITER="${N_ITER:-200}"; N_SAMPLES="${N_SAMPLES:-8192}"; DIAG_SHIFT="${DIAG_SHIFT:-3e-3}"; DT="${DT:-0.01}"; LR_MIN="${LR_MIN:-0.001}"; CHUNK_DEF=1024; WALLTIME="${WALLTIME:-06:00:00}" ;;
  *)  N_ITER="${N_ITER:-300}"; N_SAMPLES="${N_SAMPLES:-8192}"; DIAG_SHIFT="${DIAG_SHIFT:-1e-3}"; DT="${DT:-0.02}"; LR_MIN="${LR_MIN:-0.002}"; CHUNK_DEF=1024; WALLTIME="${WALLTIME:-04:00:00}" ;;
esac
N_CHAINS="${N_CHAINS:-}"           # empty -> train_2d auto (GPU=1024)
CHUNK="${CHUNK:-$CHUNK_DEF}"; CKPT_EVERY="${CKPT_EVERY:-25}"

KERNEL_FLAG="";  [ "$KERNEL" != "0" ] && KERNEL_FLAG="--kernel_size $KERNEL"
CHAINS_FLAG="";  [ -n "$N_CHAINS" ]   && CHAINS_FLAG="--n_chains $N_CHAINS"
CHUNK_FLAG="";   [ -n "$CHUNK" ]      && CHUNK_FLAG="--chunk_size $CHUNK"

OUT_DIR="${OUT_DIR:-$PSCRATCH/tc_nqs_2d/phase_hx${HX}/L${L}}"
mkdir -p "$OUT_DIR"
NAME="${ARCH}_L${L}_hx${HX}_hz${HZ}_${BC}"     # matches model/train_2d _run_name

# ---- auto-resubmit just before the wall limit (opt-in) -----------------------
RESUB_COUNT="${RESUB_COUNT:-0}"
MAX_RESUBMITS="${MAX_RESUBMITS:-8}"
WALLTIME="${WALLTIME:-}"
requeue() {
  if [ "${AUTO_RESUBMIT:-0}" = "1" ] && [ "$RESUB_COUNT" -lt "$MAX_RESUBMITS" ]; then
    echo "[hz2d] wall limit near — resubmitting task ${SLURM_ARRAY_TASK_ID} (resume #$((RESUB_COUNT+1)))"
    local tflag=""; [ -n "$WALLTIME" ] && tflag="--time=$WALLTIME"
    RESUB_COUNT=$((RESUB_COUNT+1)) L="$L" HX="$HX" HZ_MIN="$HZ_MIN" HZ_MAX="$HZ_MAX" HZ_N="$HZ_N" \
      BC="$BC" ARCH="$ARCH" NONINV="$NONINV" INV="$INV" RESCALE="$RESCALE" KERNEL="$KERNEL" DT="$DT" LR_MIN="$LR_MIN" \
      DIAG_SHIFT="$DIAG_SHIFT" QGT="$QGT" N_ITER="$N_ITER" N_SAMPLES="$N_SAMPLES" N_CHAINS="$N_CHAINS" \
      CHUNK="$CHUNK" CKPT_EVERY="$CKPT_EVERY" OUT_DIR="$OUT_DIR" AUTO_RESUBMIT=1 \
      MAX_RESUBMITS="$MAX_RESUBMITS" WALLTIME="$WALLTIME" \
      sbatch $tflag --array="${SLURM_ARRAY_TASK_ID}" "$0"
  fi
  exit 0
}
trap requeue USR1

echo "[hz2d] task ${SLURM_ARRAY_TASK_ID}/$((HZ_N-1)): L=$L $BC $ARCH hx=$HX hz=$HZ "\
"diag_shift=$DIAG_SHIFT n_iter=$N_ITER (resume #$RESUB_COUNT) -> $OUT_DIR/$NAME"

# `srun ... &` + `wait` so the USR1 trap fires promptly.
srun -n 1 python -u -m model.train_2d \
  --L "$L" --bc "$BC" --arch "$ARCH" \
  --hx "$HX" --hz "$HZ" \
  --channels_noninv $NONINV --channels_inv $INV --rescale "$RESCALE" $KERNEL_FLAG \
  --dt "$DT" --lr_min "$LR_MIN" --diag_shift "$DIAG_SHIFT" --qgt "$QGT" \
  --n_iter "$N_ITER" --n_samples "$N_SAMPLES" $CHAINS_FLAG $CHUNK_FLAG \
  --checkpoint_every "$CKPT_EVERY" --resume \
  --out_dir "$OUT_DIR" --name "$NAME" \
  --wandb_group "hz2d-L${L}-hx${HX}" --wandb_offline &
wait
