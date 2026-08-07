#!/bin/bash
# Large-L NQS run of the grid-conv invariant architecture (ToricCNN_gridinv) on a
# single Perlmutter A100. Unlike submit_nqs_sweep.sh (tag 2d-final) (an L=2 ED-validation grid
# array), this is ONE long run whose every hyperparameter is an environment
# variable, so you submit different configs without editing the file:
#
#   L=4 DT=0.01 DIAG_SHIFT=1e-3 N_NONINV=2 NONINV=4 INV="4 4" KERNEL=4 N_ITER=400 \
#       sbatch nersc/submit_nqs_gridinv.sh
#
#   # sign-full (h_y): a nonzero HY makes train.py auto-select complex128 + a
#   # complex jacobian (no --dtype flag exists); cold init is the default.
#   L=4 BC=PBC HX=0.0 HZ=0.0 HY=0.3 KERNEL=3 INV="2 2 2" N_NONINV=2 NONINV=4 \
#       DIAG_SHIFT=5e-3 N_ITER=500 sbatch nersc/submit_nqs_gridinv.sh
#
# Robustness to the queue wall clock is built in:
#   * --checkpoint_every writes weights + the energy curve to $PSCRATCH every few
#     steps, so a timed-out / pre-empted job never loses progress.
#   * --resume is ALWAYS passed; it is a no-op on the first run (no checkpoint yet)
#     and continues from the last checkpoint on every later run. So you can simply
#     re-`sbatch` the same command to keep going.
#   * AUTO_RESUBMIT=1 makes the job resubmit itself ~3 min before the wall limit
#     (Slurm --signal), so an L=6/8 run spanning several queue slots finishes
#     unattended. Bounded by MAX_RESUBMITS.
#
#SBATCH --job-name=tc-gridinv
#SBATCH --account=m5340_g
#SBATCH --qos=shared
#SBATCH --constraint=gpu
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=05:00:00
#SBATCH --signal=B:USR1@180          # USR1 to the batch script 180s before the limit
#SBATCH --output=%x-%j.out
set -euo pipefail

module load conda
conda activate tc-nqs                # built by setup_conda_gpu.sh

# where you cloned the repo; override at submit time with REPO=... if it moves
REPO="${REPO:-$HOME/toric-code-nqs}"
cd "$REPO" || { echo "[submit] REPO not found: $REPO — set REPO=<clone path>"; exit 1; }

# persistent XLA compile cache: first job pays the ~20-min cold compile once,
# every later job (and every AUTO_RESUBMIT chunk) reuses it.
# Defined BEFORE the requeue trap so an early USR1 can never hit it unbound.
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$PSCRATCH/tc_nqs/jax_cache}"
mkdir -p "$JAX_COMPILATION_CACHE_DIR"

# walltime for AUTO_RESUBMIT chunks; empty -> the #SBATCH --time directive.
# Without this, an `sbatch --time=...` override would silently reset on requeue.
WALLTIME="${WALLTIME:-}"

# ---- hyperparameters (override any at submit time via env vars) --------------
L="${L:-4}"                          # linear size; N = 3L³ (PBC) / 3L³-3L² (OBC)
BC="${BC:-OBC}"
HX="${HX:-0.0}"
HZ="${HZ:-0.0}"                      # HX=HZ=0 -> exact E0 anchor (see exact-h0-energies note)
HY="${HY:-0.0}"                      # nonzero -> non-stoquastic; train.py auto-selects complex128
MODEL="${MODEL:-bosonic}"            # 'fermionic' -> decorated B~_p, PBC only; sign-full at
                                     # ANY field, so builders auto-selects complex128 + pair moves
PHASE_HEAD="${PHASE_HEAD:-0}"        # 1 -> token-quadratic phase head (fTC sign class;
                                     # complex dtype only; changes the parameter tree)
DT="${DT:-0.02}"                     # initial learning rate
LR_MIN="${LR_MIN:-0.002}"           # cosine-decay floor (== DT for constant lr)
DIAG_SHIFT="${DIAG_SHIFT:-1e-3}"    # SR regularization (raise to slow symmetry breaking)
NONINV="${NONINV:-4}"               # pre-Wilson edge channels
N_NONINV="${N_NONINV:-2}"           # number of pre-Wilson (non-invariant) layers
NONINV_HIDDEN="${NONINV_HIDDEN:-}"  # per-layer pre-Wilson widths (space-sep, e.g. "8 4");
                                    # when set, overrides N_NONINV x NONINV
RADIUS_EDGE="${RADIUS_EDGE:-}"      # noninv GeoConv stencil radius; empty -> builder
                                    # default 1.05 (15-tap); 0.9 -> NN-only 9-tap
INV="${INV:-4 4}"                   # post-Wilson invariant grid-conv widths (space-sep)
KERNEL="${KERNEL:-0}"               # invariant grid-conv kernel; 0 -> auto = L (full span)
N_ITER="${N_ITER:-400}"
N_SAMPLES="${N_SAMPLES:-8192}"      # 8192 held constant across L (worked well in 2D at large N)
N_CHAINS="${N_CHAINS:-1024}"        # A100 default; scale with N_SAMPLES (>= a few hundred/chain)
N_SWEEPS="${N_SWEEPS:-48}"          # proposals/sample, FIXED (not the geo.N*2 auto) -> O(N) sampling
QGT="${QGT:-dense}"                 # use dense on GPU
CKPT_EVERY="${CKPT_EVERY:-10}"
# expect_and_grad evaluates the net on (n_samples x n_conn) configs at once, where
# n_conn ~ #vertices + N; in float64 that conv OOMs a 40GB A100 at ANY L>=4 (56GB
# at L=4). Chunking tiles the sample batch to fit (and at L>=6 also keeps the dot
# dimension under the int32 2^31 limit). Halve if a larger L still OOMs.
CHUNK="${CHUNK:-2048}"
EXACT_E0="${EXACT_E0:-}"            # exact anchor (h=0 PBC: -4L^3): final JSON gets delta=|E-E0|/|E0|
REF_E="${REF_E:-}"                  # benchmark energy (e.g. QMC): stream signed per-step gap
REF_SIG="${REF_SIG:-}"              # 1-sigma of REF_E (enables the (+/- N sig) annotation)
SEED="${SEED:-}"                    # sampler/init seed; empty -> train.py default (0)
EXTRA_ARGS="${EXTRA_ARGS:-}"        # verbatim extra train.py flags (e.g. guard knobs
                                    # "--spike_factor 100"); must NOT change the
                                    # parameter tree, or tag NAME yourself

OUT_DIR="${OUT_DIR:-$PSCRATCH/tc_nqs/gridinv}"
# INV is part of the identity: two runs differing only in --inv_hidden (e.g.
# "4 4 4" vs "4 4 4 2") must NOT share a name, or they clobber each other's
# checkpoint and --resume loads a mismatched parameter tree. "4 4 4" -> "4-4-4".
INV_TAG=$(echo "$INV" | tr ' ' '-')
# tag the name with hy ONLY when nonzero, so real-path (hy=0) run names stay
# byte-identical and existing gridinv checkpoints keep resuming
HY_TAG=""; [ "$HY" != "0.0" ] && HY_TAG="_hy${HY}"
# DUAL=1 -> Hadamard-conjugated basis (star tokens on the vertex grid); tag the
# name so dual and primal runs at the same point never share a checkpoint
DUAL="${DUAL:-0}"
DUAL_FLAG=""; DUAL_TAG=""
[ "$DUAL" = "1" ] && { DUAL_FLAG="--dual_basis"; DUAL_TAG="_dual"; }
# NONINV_HIDDEN / RADIUS_EDGE change the parameter tree / stencil -> part of the
# name identity (tags empty when unset, so existing run names stay byte-identical)
NH_TAG=""; [ -n "$NONINV_HIDDEN" ] && NH_TAG="_nh$(echo "$NONINV_HIDDEN" | tr ' ' '-')"
RE_TAG=""; [ -n "$RADIUS_EDGE" ]   && RE_TAG="_r${RADIUS_EDGE}"
SEED_TAG=""; [ -n "$SEED" ]        && SEED_TAG="_s${SEED}"
# model changes H AND the parameter dtype -> part of the name identity (empty for
# bosonic so every existing run name stays byte-identical)
MODEL_TAG=""; [ "$MODEL" != "bosonic" ] && MODEL_TAG="_${MODEL}"
# phase head adds parameters -> part of the name identity
PH_FLAG=""; PH_TAG=""
[ "$PHASE_HEAD" = "1" ] && { PH_FLAG="--phase_head"; PH_TAG="_ph"; }
NAME="${NAME:-gridinv${MODEL_TAG}${DUAL_TAG}_L${L}_${BC}_hx${HX}_hz${HZ}${HY_TAG}_n${N_NONINV}x${NONINV}${NH_TAG}_inv${INV_TAG}_k${KERNEL}${RE_TAG}${PH_TAG}${SEED_TAG}}"

# Perlmutter compute nodes usually cannot reach wandb.ai -> log offline and
# `wandb sync $OUT_DIR/wandb/offline-*` from a login node afterward. Set
# WANDB_OFFLINE=0 if your project IS reachable, or NO_WANDB=1 to disable entirely.
WB_FLAG="--wandb_offline"
[ "${WANDB_OFFLINE:-1}" = "0" ] && WB_FLAG=""
[ "${NO_WANDB:-0}" = "1" ]      && WB_FLAG="--no_wandb"

KERNEL_FLAG=""; [ "$KERNEL" != "0" ] && KERNEL_FLAG="--kernel_size $KERNEL"
CHUNK_FLAG="";  [ -n "$CHUNK" ]      && CHUNK_FLAG="--chunk_size $CHUNK"
NH_FLAG="";  [ -n "$NONINV_HIDDEN" ] && NH_FLAG="--noninv_hidden $NONINV_HIDDEN"
RE_FLAG="";  [ -n "$RADIUS_EDGE" ]   && RE_FLAG="--radius_edge $RADIUS_EDGE"
REF_FLAGS=""; [ -n "$REF_E" ]        && REF_FLAGS="--ref_E $REF_E${REF_SIG:+ --ref_sig $REF_SIG}"
EX_FLAG="";  [ -n "$EXACT_E0" ]      && EX_FLAG="--exact_E0 $EXACT_E0"
SEED_FLAG=""; [ -n "$SEED" ]         && SEED_FLAG="--seed $SEED"

# ---- auto-resubmit just before the wall limit (opt-in) -----------------------
RESUB_COUNT="${RESUB_COUNT:-0}"
MAX_RESUBMITS="${MAX_RESUBMITS:-8}"
requeue() {
  if [ "${AUTO_RESUBMIT:-0}" = "1" ] && [ "$RESUB_COUNT" -lt "$MAX_RESUBMITS" ]; then
    echo "[submit] wall limit near — resubmitting (resume #$((RESUB_COUNT+1)))"
    # carry every knob forward; the checkpoint on $PSCRATCH is the hand-off
    RESUB_COUNT=$((RESUB_COUNT+1)) L="$L" BC="$BC" HX="$HX" HY="$HY" HZ="$HZ" DT="$DT" \
      LR_MIN="$LR_MIN" DIAG_SHIFT="$DIAG_SHIFT" NONINV="$NONINV" N_NONINV="$N_NONINV" \
      NONINV_HIDDEN="$NONINV_HIDDEN" RADIUS_EDGE="$RADIUS_EDGE" MODEL="$MODEL" \
      PHASE_HEAD="$PHASE_HEAD" \
      REF_E="$REF_E" REF_SIG="$REF_SIG" EXACT_E0="$EXACT_E0" SEED="$SEED" \
      INV="$INV" KERNEL="$KERNEL" N_ITER="$N_ITER" N_SAMPLES="$N_SAMPLES" \
      N_CHAINS="$N_CHAINS" N_SWEEPS="$N_SWEEPS" QGT="$QGT" CKPT_EVERY="$CKPT_EVERY" CHUNK="$CHUNK" \
      OUT_DIR="$OUT_DIR" NAME="$NAME" DUAL="$DUAL" EXTRA_ARGS="$EXTRA_ARGS" \
      AUTO_RESUBMIT=1 MAX_RESUBMITS="$MAX_RESUBMITS" \
      WANDB_OFFLINE="${WANDB_OFFLINE:-1}" NO_WANDB="${NO_WANDB:-0}" \
      JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-}" WALLTIME="${WALLTIME:-}" \
      sbatch ${WALLTIME:+--time="$WALLTIME"} "$0"
  fi
  exit 0
}
trap requeue USR1

echo "[submit] $NAME  L=$L $BC  hx=$HX hz=$HZ hy=$HY  noninv=${N_NONINV}x${NONINV}${NONINV_HIDDEN:+ nh='$NONINV_HIDDEN'} inv='$INV' k=$KERNEL${RADIUS_EDGE:+ r=$RADIUS_EDGE}"
echo "[submit] dt=$DT lr_min=$LR_MIN diag_shift=$DIAG_SHIFT n_iter=$N_ITER  (resume #$RESUB_COUNT)"

# `srun ... &` + `wait` so the trap fires promptly on USR1 (a foreground srun
# would swallow the signal until it returns).
srun -n 1 python -u -m tc3d.train \
  --L "$L" --bc "$BC" --model "$MODEL" --arch ToricCNN_gridinv $DUAL_FLAG $PH_FLAG \
  --hx "$HX" --hy "$HY" --hz "$HZ" \
  --noninv_channels "$NONINV" --n_noninv "$N_NONINV" $NH_FLAG $RE_FLAG --inv_hidden $INV $KERNEL_FLAG \
  --dt "$DT" --lr_min "$LR_MIN" --diag_shift "$DIAG_SHIFT" --qgt "$QGT" \
  --n_iter "$N_ITER" --n_samples "$N_SAMPLES" --n_chains "$N_CHAINS" \
  --n_sweeps "$N_SWEEPS" $CHUNK_FLAG \
  --checkpoint_every "$CKPT_EVERY" --resume \
  --out_dir "$OUT_DIR" --name "$NAME" $REF_FLAGS $EX_FLAG $SEED_FLAG $EXTRA_ARGS \
  --wandb_group "${SLURM_JOB_NAME}" $WB_FLAG &
wait
