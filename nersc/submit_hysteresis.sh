#!/bin/bash
# Hysteresis sweep: train the NQS along a field grid TWICE, warm-starting each
# point from its neighbour (--init_from) -- once forward (up from the topological
# phase), once backward (down from the polarized phase). A first-order transition
# shows two branches (a loop); a second-order one shows a single coincident curve.
# This is NOT a Slurm array: the warm-start chain is a hard dependency, so points
# run SEQUENTIALLY in one job.
#
#   # first-order candidate: hx-sweep at hz=0
#   SWEEP=hx sbatch --time=05:00:00 nersc/submit_hysteresis.sh
#   # second-order candidate: hz-sweep at hx=0
#   SWEEP=hz sbatch --time=05:00:00 nersc/submit_hysteresis.sh
#
# Restartable: each point is skipped if its final {name}.json already exists, so a
# job that hits the wall limit just needs re-submitting (AUTO_RESUBMIT=1 does that
# automatically ~180 s before the limit). Outputs:
#   $PSCRATCH/tc_nqs/hyst_L${L}_${SWEEP}/{forward,backward}/{name}.json
#SBATCH --job-name=tc-hyst
#SBATCH --account=m5340_g
#SBATCH --qos=shared
#SBATCH --constraint=gpu
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=05:00:00
#SBATCH --signal=B:USR1@180
#SBATCH --output=slurm_logs/%x-%j.out
set -euo pipefail

module load conda
conda activate tc-nqs

REPO="${REPO:-$HOME/threed_TC/ThreeD_TC}"
cd "$REPO" || { echo "[hyst] REPO not found: $REPO — set REPO=<clone path>"; exit 1; }

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$PSCRATCH/jax_cache}"
mkdir -p "$JAX_COMPILATION_CACHE_DIR"

# ---- sweep definition --------------------------------------------------------
L="${L:-4}"
SWEEP="${SWEEP:-hx}"                # which field is swept: hx (1st-order) or hz (2nd-order)
if [ "$SWEEP" = "hx" ]; then
  FIXED="${FIXED:-0.0}"            # fixed hz
  MIN="${MIN:-0.55}"; MAX="${MAX:-1.25}"; STEP="${STEP:-0.025}"
elif [ "$SWEEP" = "hz" ]; then
  FIXED="${FIXED:-0.0}"            # fixed hx
  MIN="${MIN:-0.05}"; MAX="${MAX:-0.45}"; STEP="${STEP:-0.025}"
else
  echo "[hyst] SWEEP must be hx or hz (got '$SWEEP')"; exit 1
fi

# swept-field grid, ascending, rounded like the other sweeps (round(...,4))
GRID=$(python -c "import numpy as np; \
print(' '.join(f'{round(float(x),4)}' for x in np.arange($MIN, $MAX+1e-9, $STEP)))")

# ---- architecture / optimization (validated gridinv config; L=4 defaults) ----
BC="${BC:-OBC}"
NONINV="${NONINV:-4}"; N_NONINV="${N_NONINV:-2}"; INV="${INV:-2 2 2}"
if [ -z "${KERNEL:-}" ]; then
  case "$L" in
    3) KERNEL=2 ;; 4) KERNEL=3 ;; 5) KERNEL=4 ;; 6) KERNEL=4 ;; 7) KERNEL=5 ;; *) KERNEL=4 ;;
  esac
fi
DT="${DT:-0.01}"; LR_MIN="${LR_MIN:-0.001}"
DIAG_SHIFT="${DIAG_SHIFT:-1e-3}"
N_ITER_COLD="${N_ITER_COLD:-150}"  # first point of each chain: trained from scratch
N_ITER_WARM="${N_ITER_WARM:-100}"  # warm-started points converge faster
N_SAMPLES="${N_SAMPLES:-8192}"; N_CHAINS="${N_CHAINS:-1024}"
N_SWEEPS="${N_SWEEPS:-48}"; QGT="${QGT:-dense}"; CKPT_EVERY="${CKPT_EVERY:-10}"
CHUNK="${CHUNK:-2048}"

KERNEL_FLAG=""; [ "$KERNEL" != "0" ] && KERNEL_FLAG="--kernel_size $KERNEL"
CHUNK_FLAG="";  [ -n "$CHUNK" ]      && CHUNK_FLAG="--chunk_size $CHUNK"
BASE="${BASE:-$PSCRATCH/tc_nqs/hyst_L${L}_${SWEEP}}"

# ---- auto-resubmit the WHOLE job near the wall limit (skip-if-done continues) --
RESUB_COUNT="${RESUB_COUNT:-0}"; MAX_RESUBMITS="${MAX_RESUBMITS:-8}"; WALLTIME="${WALLTIME:-}"
requeue() {
  if [ "${AUTO_RESUBMIT:-0}" = "1" ] && [ "$RESUB_COUNT" -lt "$MAX_RESUBMITS" ]; then
    echo "[hyst] wall limit near — resubmitting (resume #$((RESUB_COUNT+1)))"
    local tflag=""; [ -n "$WALLTIME" ] && tflag="--time=$WALLTIME"
    RESUB_COUNT=$((RESUB_COUNT+1)) L="$L" SWEEP="$SWEEP" FIXED="$FIXED" MIN="$MIN" MAX="$MAX" \
      STEP="$STEP" BASE="$BASE" AUTO_RESUBMIT=1 MAX_RESUBMITS="$MAX_RESUBMITS" WALLTIME="$WALLTIME" \
      sbatch $tflag "$0"
  fi
  exit 0
}
trap requeue USR1

# ---- one directed chain ------------------------------------------------------
run_chain() {           # $1 = direction (forward|backward), $2... = ordered field values
  local dir="$1"; shift
  local OUT="$BASE/$dir"; mkdir -p "$OUT"
  local PREV=""
  for H in "$@"; do
    if [ "$SWEEP" = "hx" ]; then local HX="$H" HZ="$FIXED"; else local HX="$FIXED" HZ="$H"; fi
    local NAME="bosonic_gridinv_L${L}_hx${HX}_hz${HZ}"
    if [ -f "$OUT/$NAME.json" ]; then
      echo "[hyst:$dir] skip $SWEEP=$H (done: $OUT/$NAME.json)"; PREV="$OUT/$NAME"; continue
    fi
    local INIT=(); local NITER="$N_ITER_COLD"
    [ -n "$PREV" ] && { INIT=(--init_from "$PREV"); NITER="$N_ITER_WARM"; }
    echo "[hyst:$dir] $SWEEP=$H  (n_iter=$NITER${PREV:+  warm<-$(basename "$PREV")}) -> $OUT/$NAME"
    srun -n 1 python -u -m Three_TC.train \
      --L "$L" --bc "$BC" --model bosonic --arch ToricCNN_gridinv \
      --hx "$HX" --hz "$HZ" \
      --noninv_channels "$NONINV" --n_noninv "$N_NONINV" --inv_hidden $INV $KERNEL_FLAG \
      --dt "$DT" --lr_min "$LR_MIN" --diag_shift "$DIAG_SHIFT" --qgt "$QGT" \
      --n_iter "$NITER" --n_samples "$N_SAMPLES" --n_chains "$N_CHAINS" \
      --n_sweeps "$N_SWEEPS" $CHUNK_FLAG \
      --checkpoint_every "$CKPT_EVERY" --resume "${INIT[@]}" \
      --out_dir "$OUT" --name "$NAME" \
      --wandb_group "hyst-L${L}-${SWEEP}-${dir}" --wandb_offline &
    wait                           # `&`+wait so the USR1 trap fires promptly
    PREV="$OUT/$NAME"
  done
}

echo "[hyst] L=$L SWEEP=$SWEEP fixed=$FIXED grid=[$MIN..$MAX step $STEP]  -> $BASE"
echo "[hyst] forward (ascending, from topological):"
run_chain forward  $GRID
REV=$(echo "$GRID" | tr ' ' '\n' | tac | tr '\n' ' ')
echo "[hyst] backward (descending, from polarized):"
run_chain backward $REV
echo "[hyst] done. Extract each branch (login node):"
echo "  python analysis/check_convergence.py --dir $BASE/forward  --L $L --dump results/hyst_L${L}_${SWEEP}/forward.json"
echo "  python analysis/check_convergence.py --dir $BASE/backward --L $L --dump results/hyst_L${L}_${SWEEP}/backward.json"
