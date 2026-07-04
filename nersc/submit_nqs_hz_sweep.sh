#!/bin/bash
# hz phase-diagram sweep: one NQS (ToricCNN_gridinv, bosonic) per hz value at a
# FIXED hx, as a Slurm job ARRAY (array index -> hz), for one system size L.
# Mirrors the validated train invocation of submit_nqs_gridinv.sh and the
# array-index->hz idiom of submit_ed_sweep.sh.
#
#   L=3 sbatch --array=0-15 nersc/submit_nqs_hz_sweep.sh          # validate (fast)
#   L=4 sbatch --array=0-15 nersc/submit_nqs_hz_sweep.sh
#   L=6 sbatch --array=0-15 nersc/submit_nqs_hz_sweep.sh          # diag_shift auto->1e-2
#
# The --array size MUST equal HZ_N (default 16). hz_i = HZ_MIN + i*(HZ_MAX-HZ_MIN)/(HZ_N-1).
# Re-submitting the same array continues any unfinished point (--resume is always on).
#SBATCH --job-name=tc-hzsweep
#SBATCH --account=m5340_g
#SBATCH --qos=shared
#SBATCH --constraint=gpu
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=05:00:00
#SBATCH --array=0-15
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
NONINV="${NONINV:-4}"; N_NONINV="${N_NONINV:-2}"; INV="${INV:-2 2 2}"; KERNEL="${KERNEL:-4}"
DT="${DT:-0.01}"; LR_MIN="${LR_MIN:-0.001}"
# L>=6 destabilizes at diag_shift 1e-3 (last session) -> auto-raise to 1e-2.
if [ -z "${DIAG_SHIFT:-}" ]; then
  [ "$L" -ge 6 ] && DIAG_SHIFT=1e-2 || DIAG_SHIFT=1e-3
fi
N_ITER="${N_ITER:-400}"; N_SAMPLES="${N_SAMPLES:-8192}"; N_CHAINS="${N_CHAINS:-1024}"
N_SWEEPS="${N_SWEEPS:-48}"; QGT="${QGT:-dense}"; CKPT_EVERY="${CKPT_EVERY:-10}"
CHUNK="${CHUNK:-2048}"             # 2048 covers L=3..6 (memory, not speed)

KERNEL_FLAG=""; [ "$KERNEL" != "0" ] && KERNEL_FLAG="--kernel_size $KERNEL"
CHUNK_FLAG="";  [ -n "$CHUNK" ]      && CHUNK_FLAG="--chunk_size $CHUNK"

OUT_DIR="${OUT_DIR:-$PSCRATCH/tc_nqs/phase_hx${HX}/L${L}}"
mkdir -p "$OUT_DIR"
NAME="bosonic_gridinv_L${L}_hx${HX}_hz${HZ}"

echo "[hzsweep] task ${SLURM_ARRAY_TASK_ID}/$((HZ_N-1)): L=$L $BC hx=$HX hz=$HZ "\
"diag_shift=$DIAG_SHIFT n_iter=$N_ITER -> $OUT_DIR/$NAME"

srun -n 1 python -u -m Three_TC.train \
  --L "$L" --bc "$BC" --model bosonic --arch ToricCNN_gridinv \
  --hx "$HX" --hz "$HZ" \
  --noninv_channels "$NONINV" --n_noninv "$N_NONINV" --inv_hidden $INV $KERNEL_FLAG \
  --dt "$DT" --lr_min "$LR_MIN" --diag_shift "$DIAG_SHIFT" --qgt "$QGT" \
  --n_iter "$N_ITER" --n_samples "$N_SAMPLES" --n_chains "$N_CHAINS" \
  --n_sweeps "$N_SWEEPS" $CHUNK_FLAG \
  --checkpoint_every "$CKPT_EVERY" --resume \
  --out_dir "$OUT_DIR" --name "$NAME" \
  --wandb_group "hzsweep-L${L}-hx${HX}" --wandb_offline
