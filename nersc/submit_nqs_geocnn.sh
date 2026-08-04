#!/bin/bash
# Symmetry-UNAWARE control run: GeoCNN (geometry-exact KernelManager3D kernel but
# NO Wilson 4-product, so NOT A_v-invariant) on a single Perlmutter A100. This is
# the head-to-head baseline for the symmetry-aware ToricCNN_gridinv workhorse:
# every training/sampler knob below is pinned to match the gridinv phase-diagram
# runs (see data/tc_nqs/phase_hx*/L*/bosonic_gridinv_*.json), so the ONLY variable
# that differs between the two learning curves is the Wilson invariance. Sibling of
# submit_nqs_gridinv.sh; the only real change is --arch GeoCNN + --cnn_hidden.
#
#   L=4 BC=OBC HX=0.0 HZ=0.1 CNN_HIDDEN="8 8 8" sbatch nersc/submit_nqs_geocnn.sh
#
# Robustness (checkpoint/resume, AUTO_RESUBMIT) is identical to submit_nqs_gridinv.sh.
# The default walltime is 30 min — right for the L=4 comparison points; for larger L
# override at submit time with `sbatch --time=HH:MM:SS`.
#
#SBATCH --job-name=tc-geocnn
#SBATCH --account=m5340_g
#SBATCH --qos=shared
#SBATCH --constraint=gpu
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
#SBATCH --signal=B:USR1@180          # USR1 to the batch script 180s before the limit
#SBATCH --output=%x-%j.out
set -euo pipefail

module load conda
conda activate tc-nqs                # built by setup_conda_gpu.sh

REPO="${REPO:-$HOME/toric-code-nqs}"
cd "$REPO" || { echo "[submit] REPO not found: $REPO — set REPO=<clone path>"; exit 1; }

# ---- hyperparameters (pinned to the gridinv topological runs; override via env) --
L="${L:-4}"
BC="${BC:-OBC}"                      # GeoCNN supports OBC (masks out-of-box taps)
HX="${HX:-0.0}"
HZ="${HZ:-0.0}"
CNN_HIDDEN="${CNN_HIDDEN:-8 8 8}"    # GeoCNN edge-conv channel widths (the size knob);
                                     # width-1 readout auto-appended. L=4: 8 8 8=6555p, 12 12 12=14151p
DT="${DT:-0.01}"                     # matches gridinv runs
LR_MIN="${LR_MIN:-0.001}"           # cosine-decay floor (matches gridinv runs)
DIAG_SHIFT="${DIAG_SHIFT:-1e-3}"
N_ITER="${N_ITER:-150}"             # same training budget as the gridinv curves
N_SAMPLES="${N_SAMPLES:-8192}"
N_CHAINS="${N_CHAINS:-1024}"
N_SWEEPS="${N_SWEEPS:-48}"
N_DISCARD="${N_DISCARD:-8}"
QGT="${QGT:-dense}"
CKPT_EVERY="${CKPT_EVERY:-10}"
CHUNK="${CHUNK:-2048}"

OUT_DIR="${OUT_DIR:-$PSCRATCH/tc_nqs/arch_compare/L${L}}"
CNN_TAG=$(echo "$CNN_HIDDEN" | tr ' ' '-')          # "8 8 8" -> "8-8-8" (part of the run identity)
NAME="${NAME:-bosonic_geocnn_L${L}_${BC}_hx${HX}_hz${HZ}_cnn${CNN_TAG}}"

WB_FLAG="--wandb_offline"
[ "${WANDB_OFFLINE:-1}" = "0" ] && WB_FLAG=""
[ "${NO_WANDB:-0}" = "1" ]      && WB_FLAG="--no_wandb"

CHUNK_FLAG=""; [ -n "$CHUNK" ] && CHUNK_FLAG="--chunk_size $CHUNK"

# ---- auto-resubmit just before the wall limit (opt-in; off by default) ----------
RESUB_COUNT="${RESUB_COUNT:-0}"
MAX_RESUBMITS="${MAX_RESUBMITS:-8}"
requeue() {
  if [ "${AUTO_RESUBMIT:-0}" = "1" ] && [ "$RESUB_COUNT" -lt "$MAX_RESUBMITS" ]; then
    echo "[submit] wall limit near — resubmitting (resume #$((RESUB_COUNT+1)))"
    RESUB_COUNT=$((RESUB_COUNT+1)) L="$L" BC="$BC" HX="$HX" HZ="$HZ" \
      CNN_HIDDEN="$CNN_HIDDEN" DT="$DT" LR_MIN="$LR_MIN" DIAG_SHIFT="$DIAG_SHIFT" \
      N_ITER="$N_ITER" N_SAMPLES="$N_SAMPLES" N_CHAINS="$N_CHAINS" N_SWEEPS="$N_SWEEPS" \
      N_DISCARD="$N_DISCARD" QGT="$QGT" CKPT_EVERY="$CKPT_EVERY" CHUNK="$CHUNK" \
      OUT_DIR="$OUT_DIR" NAME="$NAME" AUTO_RESUBMIT=1 MAX_RESUBMITS="$MAX_RESUBMITS" \
      WANDB_OFFLINE="${WANDB_OFFLINE:-1}" NO_WANDB="${NO_WANDB:-0}" \
      sbatch "$0"
  fi
  exit 0
}
trap requeue USR1

echo "[submit] $NAME  L=$L $BC  hx=$HX hz=$HZ  cnn_hidden='$CNN_HIDDEN'"
echo "[submit] dt=$DT lr_min=$LR_MIN diag_shift=$DIAG_SHIFT n_iter=$N_ITER  (resume #$RESUB_COUNT)"

srun -n 1 python -u -m tc3d.train \
  --L "$L" --bc "$BC" --model bosonic --arch GeoCNN \
  --hx "$HX" --hz "$HZ" \
  --cnn_hidden $CNN_HIDDEN \
  --dt "$DT" --lr_min "$LR_MIN" --diag_shift "$DIAG_SHIFT" --qgt "$QGT" \
  --n_iter "$N_ITER" --n_samples "$N_SAMPLES" --n_chains "$N_CHAINS" \
  --n_sweeps "$N_SWEEPS" --n_discard "$N_DISCARD" $CHUNK_FLAG \
  --checkpoint_every "$CKPT_EVERY" --resume \
  --out_dir "$OUT_DIR" --name "$NAME" \
  --wandb_group "${SLURM_JOB_NAME}" $WB_FLAG &
wait
