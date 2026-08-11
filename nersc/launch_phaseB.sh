#!/bin/bash
# Phase B launcher — both QMC-validation cuts, frozen spec in CAMPAIGN.md §Phase B.
# Run ON PERLMUTTER from the repo root:  bash nersc/launch_phaseB.sh
#
# Anchor (hx,hz)=(0.2,0.1); up cut sweeps hz at hx=0.2 (15 pts), right cut sweeps
# hx at hz=0.1 (14 pts); L=4,5,6; tune-rect winner arch (dual-basis gridinv,
# nh 4->8, inv (8,8), k=L-1, dt=0.02, lr_min 2e-3); every point cold-start;
# --final_eval_rounds 8 gives the 65k-equivalent end-of-training observables.
# L=5 diag_shift: 1e-3 at weak hx, 3e-3 at hx>=0.65 (the tune-rect pick at hx=0.6).
# 87 runs total; AUTO_RESUBMIT chains anything that outlives its wall.
set -euo pipefail
cd "$(dirname "$0")/.."

UP="0.1 0.15 0.18 0.2 0.22 0.24 0.26 0.28 0.3 0.32 0.34 0.36 0.4 0.45 0.5"
RIGHT_LO="0.2 0.35 0.5"
RIGHT_HI="0.65 0.75 0.8 0.85 0.9 0.95 1.0 1.05 1.1 1.175 1.25"

common=(DUAL=1 NONINV_HIDDEN="4 8" INV="8 8" DT=0.02 LR_MIN=0.002
        FINAL_EVAL_ROUNDS=8 AUTO_RESUBMIT=1
        NAME_TEMPLATE="phaseB_dual_L{L}_hx{hx}_hz{hz}")

# ---- up cut: fixed hx=0.2, swept hz --------------------------------------
env "${common[@]}" SWEEP=hz HX=0.2 FIELD_VALUES="$UP" L=4 KERNEL=3 \
  DIAG_SHIFT=1e-3 N_ITER=150 CHUNK_POINTS=5 WALLTIME=04:00:00 \
  OUT_DIR="$PSCRATCH/tc_nqs/phaseB/up/L4" \
  sbatch --array=0-2 --time=04:00:00 nersc/submit_nqs_batch.sh

env "${common[@]}" SWEEP=hz HX=0.2 FIELD_VALUES="$UP" L=5 KERNEL=4 \
  DIAG_SHIFT=1e-3 N_ITER=200 CHUNK_POINTS=3 WALLTIME=05:00:00 \
  OUT_DIR="$PSCRATCH/tc_nqs/phaseB/up/L5" \
  sbatch --array=0-4 --time=05:00:00 nersc/submit_nqs_batch.sh

env "${common[@]}" SWEEP=hz HX=0.2 FIELD_VALUES="$UP" L=6 KERNEL=5 \
  DIAG_SHIFT=1e-3 N_ITER=250 CHUNK_POINTS=1 WALLTIME=05:00:00 \
  OUT_DIR="$PSCRATCH/tc_nqs/phaseB/up/L6" \
  sbatch --array=0-14 --time=05:00:00 nersc/submit_nqs_batch.sh

# ---- right cut: fixed hz=0.1, swept hx ------------------------------------
env "${common[@]}" SWEEP=hx HZ=0.1 FIELD_VALUES="$RIGHT_LO $RIGHT_HI" L=4 KERNEL=3 \
  DIAG_SHIFT=1e-3 N_ITER=150 CHUNK_POINTS=5 WALLTIME=04:00:00 \
  OUT_DIR="$PSCRATCH/tc_nqs/phaseB/right/L4" \
  sbatch --array=0-2 --time=04:00:00 nersc/submit_nqs_batch.sh

env "${common[@]}" SWEEP=hx HZ=0.1 FIELD_VALUES="$RIGHT_LO" L=5 KERNEL=4 \
  DIAG_SHIFT=1e-3 N_ITER=200 CHUNK_POINTS=3 WALLTIME=05:00:00 \
  OUT_DIR="$PSCRATCH/tc_nqs/phaseB/right/L5" \
  sbatch --array=0 --time=05:00:00 nersc/submit_nqs_batch.sh

env "${common[@]}" SWEEP=hx HZ=0.1 FIELD_VALUES="$RIGHT_HI" L=5 KERNEL=4 \
  DIAG_SHIFT=3e-3 N_ITER=200 CHUNK_POINTS=3 WALLTIME=05:00:00 \
  OUT_DIR="$PSCRATCH/tc_nqs/phaseB/right/L5" \
  sbatch --array=0-3 --time=05:00:00 nersc/submit_nqs_batch.sh

env "${common[@]}" SWEEP=hx HZ=0.1 FIELD_VALUES="$RIGHT_LO $RIGHT_HI" L=6 KERNEL=5 \
  DIAG_SHIFT=1e-3 N_ITER=250 CHUNK_POINTS=1 WALLTIME=05:00:00 \
  OUT_DIR="$PSCRATCH/tc_nqs/phaseB/right/L6" \
  sbatch --array=0-13 --time=05:00:00 nersc/submit_nqs_batch.sh

echo "[phaseB] all 7 submissions in — 45 array tasks / 87 training runs."
