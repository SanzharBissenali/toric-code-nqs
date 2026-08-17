#!/bin/bash
# Phase B RERUN launcher — the 2026-08-16 overnight disagreement-window reruns.
# Run ON PERLMUTTER from the repo root:  bash nersc/launch_phaseB_rerun.sh
#
# Reruns the QMC-disagreeing Phase B points (analysis/phaseB_summary.ipynb §7/§12,
# pulls recomputed 2026-08-16) with the cross-validated fixes from Runs A/C/D/E:
#   right cut (first-order): dt 0.02->0.01 (lr_min 0.002->0.001, plain cosine,
#     NO --warmup_frac — warmup destabilizes this cut), n_iter 500;
#     diag_shift 1e-3 at L4/L5 (the Run-D-verified combo), 3e-3 at L6
#     (production near-transition default, Run E's dt=0.01 rows).
#   up cut (second-order): schedule unchanged (dt=0.02), just longer — n_iter 500
#     (Run A: L4 converges by step ~250-350; L5/L6 500-step status is the test).
# Fresh OUT_DIRs + knob-tagged names: tc3d.sweep force-resumes any existing
# {name}.json/checkpoint, so reusing the phaseB/ dirs would warm-start the old
# 150/200/250-step checkpoints under a stretched schedule. Never reuse them.
# 36 runs / 27 array tasks; every point cold-start, seed 0, --final_eval_rounds 8.
set -euo pipefail
cd "$(dirname "$0")/.."

common=(DUAL=1 NONINV_HIDDEN="4 8" INV="8 8" FINAL_EVAL_ROUNDS=8 AUTO_RESUBMIT=1)

# ---- right cut: fixed hz=0.1, swept hx — dt=0.01, 500 steps ----------------
# L4: full window fill 0.75..1.0 (flagged 0.75/0.80 + regression check 0.85-1.0)
env "${common[@]}" SWEEP=hx HZ=0.1 FIELD_VALUES="0.75 0.8 0.85 0.9 0.95 1.0" \
  L=4 KERNEL=3 DT=0.01 LR_MIN=0.001 DIAG_SHIFT=1e-3 N_ITER=500 \
  NAME_TEMPLATE="phaseB2_dt01n500_L{L}_hx{hx}_hz{hz}" \
  CHUNK_POINTS=3 WALLTIME=03:00:00 \
  OUT_DIR="$PSCRATCH/tc_nqs/phaseB_rerun/right/L4" \
  sbatch --array=0-1 --time=03:00:00 nersc/submit_nqs_batch.sh

# L5: window fill 0.75-1.0 (0.80 lag unresolved n=1; 1.0 A_v +16.7sigma;
# 0.85 re-run as a production-grade repeat of Run D's verified fix)
env "${common[@]}" SWEEP=hx HZ=0.1 FIELD_VALUES="0.75 0.8 0.85 0.9 1.0" \
  L=5 KERNEL=4 DT=0.01 LR_MIN=0.001 DIAG_SHIFT=1e-3 N_ITER=500 \
  NAME_TEMPLATE="phaseB2_dt01n500_L{L}_hx{hx}_hz{hz}" \
  CHUNK_POINTS=1 WALLTIME=03:30:00 \
  OUT_DIR="$PSCRATCH/tc_nqs/phaseB_rerun/right/L5" \
  sbatch --array=0-4 --time=03:30:00 nersc/submit_nqs_batch.sh

# L6: flagged 0.80-0.95 (worst in campaign) + the 6 never-landed points
env "${common[@]}" SWEEP=hx HZ=0.1 \
  FIELD_VALUES="0.75 0.8 0.85 0.9 0.95 1.0 1.05 1.1 1.175 1.25" \
  L=6 KERNEL=5 DT=0.01 LR_MIN=0.001 DIAG_SHIFT=3e-3 N_ITER=500 \
  NAME_TEMPLATE="phaseB2_dt01n500_L{L}_hx{hx}_hz{hz}" \
  CHUNK_POINTS=1 WALLTIME=05:00:00 \
  OUT_DIR="$PSCRATCH/tc_nqs/phaseB_rerun/right/L6" \
  sbatch --array=0-9 --time=05:00:00 nersc/submit_nqs_batch.sh

# ---- up cut: fixed hx=0.2, swept hz — schedule unchanged, 500 steps --------
env "${common[@]}" SWEEP=hz HX=0.2 FIELD_VALUES="0.22 0.24 0.26 0.28 0.3 0.32 0.34" \
  L=4 KERNEL=3 DT=0.02 LR_MIN=0.002 DIAG_SHIFT=1e-3 N_ITER=500 \
  NAME_TEMPLATE="phaseB2_n500_L{L}_hx{hx}_hz{hz}" \
  CHUNK_POINTS=4 WALLTIME=03:00:00 \
  OUT_DIR="$PSCRATCH/tc_nqs/phaseB_rerun/up/L4" \
  sbatch --array=0-1 --time=03:00:00 nersc/submit_nqs_batch.sh

env "${common[@]}" SWEEP=hz HX=0.2 FIELD_VALUES="0.22 0.24 0.26 0.28 0.3" \
  L=5 KERNEL=4 DT=0.02 LR_MIN=0.002 DIAG_SHIFT=1e-3 N_ITER=500 \
  NAME_TEMPLATE="phaseB2_n500_L{L}_hx{hx}_hz{hz}" \
  CHUNK_POINTS=1 WALLTIME=03:30:00 \
  OUT_DIR="$PSCRATCH/tc_nqs/phaseB_rerun/up/L5" \
  sbatch --array=0-4 --time=03:30:00 nersc/submit_nqs_batch.sh

env "${common[@]}" SWEEP=hz HX=0.2 FIELD_VALUES="0.22 0.24 0.26" \
  L=6 KERNEL=5 DT=0.02 LR_MIN=0.002 DIAG_SHIFT=1e-3 N_ITER=500 \
  NAME_TEMPLATE="phaseB2_n500_L{L}_hx{hx}_hz{hz}" \
  CHUNK_POINTS=1 WALLTIME=05:00:00 \
  OUT_DIR="$PSCRATCH/tc_nqs/phaseB_rerun/up/L6" \
  sbatch --array=0-2 --time=05:00:00 nersc/submit_nqs_batch.sh

echo "[phaseB_rerun] all 6 submissions in — 27 array tasks / 36 training runs."
