#!/bin/bash
# launch_fermionic_ladder.sh — fermionic h=0 architecture-ladder campaign
#
# Three tiers at hx=hz=0, L in {2,3,4}, for the presentation figure
# (CNN  vs  approx-symm  vs  sign-head + approx-symm), multi-seed:
#
#   cnn      GeoCNN (symmetry-unaware control, complex), cold start, guard opened
#   asymm    ToricCNN_gridinv, complex, NO phase head, cold start, guard opened
#            (reproduces the positive-sector trap: -22.52 at L=2)
#   signhead gridinv + frozen analytic GF(2) head + flux penalty 6 + chains_up,
#            --init_from a prefit checkpoint (exact recipe, BLOG 2026-08-10)
#
# Stages (first arg):
#   prefit   login-node CPU: generate prefit_anaC_k2_L{2,3,4}_s{0,1,2} via
#            analysis/scripts/prefit_phase_head.py (--seed varies the real trunk init)
#   smoke    submit 3 tiny L=2 jobs (one per tier, seed 0, N_ITER=30, _smoke names)
#   full     submit the whole matrix (39 jobs, ~30 GPU-h shared QOS)
#
# DRY_RUN=1 prints the sbatch/python commands instead of running them.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO_DIR="$(pwd)"   # pin jobs + prefit to THIS checkout (the campaign branch), not ~/toric-code-nqs

STAGE="${1:?usage: launch_fermionic_ladder.sh prefit|smoke|full}"
OUT_DIR="${OUT_DIR:-$PSCRATCH/tc_nqs/fermionic_ladder}"
SEEDS=(0 1 2)
SBATCH_EXTRA="${SBATCH_EXTRA:-}"   # e.g. --dependency=afterok:<smoke ids>
run() { if [ "${DRY_RUN:-0}" = "1" ]; then echo "DRY: $*"; else eval "$*"; fi; }

e0()      { case "$1" in 2) echo -32;; 3) echo -108;; 4) echo -256;; esac; }
chunk()   { case "$1" in 2) echo 2048;; *) echo 256;; esac; }
wall()    { case "$1" in 2) echo 00:45:00;; 3) echo 01:15:00;; 4) echo 02:30:00;; esac; }
# cold tiers drop to 4096 samples at L>=3 (plateau precision is not the point);
# signhead keeps 8192 (resolves delta ~ 1e-7)
cold_ns() { case "$1" in 2) echo 8192;; *) echo 4096;; esac; }
cold_nc() { case "$1" in 2) echo 1024;; *) echo 512;;  esac; }

# opened guard for cold tiers: the stock 10x spike heuristic vetoes phase
# restructuring (notes/fermionic_next_steps.md); matches the historical
# -22.52 cold-start runs exactly
GUARD_OPEN="--spike_factor 1e6 --max_rollbacks 50"

if [ "$STAGE" = "prefit" ]; then
  # run on a login node under the tc-nqs conda env (CPU-only, seconds per L)
  run "mkdir -p '$OUT_DIR'"
  for L in 2 3 4; do for S in "${SEEDS[@]}"; do
    run "PYTHONPATH=$REPO_DIR python analysis/scripts/prefit_phase_head.py --L $L --kernel 2 --analytic_C --frozen \
      --seed $S --save $OUT_DIR/prefit_anaC_k2_L${L}_s${S} \
      > $OUT_DIR/prefit_anaC_k2_L${L}_s${S}.log 2>&1"
  done; done
  echo "[prefit] done — check CERTIFICATE + 'CNN sign accuracy = 1.000000' in each .log"
  exit 0
fi

submit_cnn() {  # $1=L $2=cnn_hidden(space-sep) $3=seed $4=n_iter $5=name
  run "REPO=$REPO_DIR L=$1 BC=PBC MODEL=fermionic ARCH=GeoCNN CNN_HIDDEN='$2' KERNEL=0 \
    N_ITER=$4 N_SAMPLES=$(cold_ns "$1") N_CHAINS=$(cold_nc "$1") CHUNK=$(chunk "$1") \
    EXACT_E0=$(e0 "$1") SEED=$3 EXTRA_ARGS='$GUARD_OPEN' \
    OUT_DIR=$OUT_DIR NAME=$5 WALLTIME=$(wall "$1") AUTO_RESUBMIT=1 \
    sbatch $SBATCH_EXTRA --time=$(wall "$1") nersc/submit_nqs_gridinv.sh"
}
submit_asymm() {  # $1=L $2=inv_hidden $3=seed $4=n_iter $5=name
  run "REPO=$REPO_DIR L=$1 BC=PBC MODEL=fermionic KERNEL=2 NONINV_HIDDEN='4 8' INV='$2' \
    N_ITER=$4 N_SAMPLES=$(cold_ns "$1") N_CHAINS=$(cold_nc "$1") CHUNK=$(chunk "$1") \
    EXACT_E0=$(e0 "$1") SEED=$3 EXTRA_ARGS='$GUARD_OPEN' \
    OUT_DIR=$OUT_DIR NAME=$5 WALLTIME=$(wall "$1") AUTO_RESUBMIT=1 \
    sbatch $SBATCH_EXTRA --time=$(wall "$1") nersc/submit_nqs_gridinv.sh"
}
submit_signhead() {  # $1=L $2=seed $3=n_iter $4=name
  # HARD GATE (audit 2026-08-18): a missing prefit checkpoint does NOT crash
  # train.py — it silently cold-starts with theta=0, i.e. a no-op sign head
  # still labeled phase_head_frozen. Refuse to submit rather than corrupt the tier.
  if [ "${DRY_RUN:-0}" != "1" ] && [ ! -f "$OUT_DIR/prefit_anaC_k2_L${1}_s${2}.mpack" ]; then
    echo "[launch] missing $OUT_DIR/prefit_anaC_k2_L${1}_s${2}.mpack — run the 'prefit' stage first" >&2
    exit 1
  fi
  run "REPO=$REPO_DIR L=$1 BC=PBC MODEL=fermionic PHASE_HEAD_FROZEN=1 KERNEL=2 \
    NONINV_HIDDEN='4 8' INV='8 8' \
    N_ITER=$3 N_SAMPLES=8192 N_CHAINS=1024 CHUNK=$(chunk "$1") \
    EXACT_E0=$(e0 "$1") SEED=$2 \
    EXTRA_ARGS='--flux_penalty 6.0 --chains_up --init_from $OUT_DIR/prefit_anaC_k2_L${1}_s${2}' \
    OUT_DIR=$OUT_DIR NAME=$4 WALLTIME=$(wall "$1") AUTO_RESUBMIT=1 \
    sbatch $SBATCH_EXTRA --time=$(wall "$1") nersc/submit_nqs_gridinv.sh"
}

if [ "$STAGE" = "smoke" ]; then
  submit_cnn      2 "4 4 4" 0 30 ladder_cnn_L2_ch444_s0_smoke
  submit_asymm    2 "8 8"   0 30 ladder_asymm_L2_inv88_s0_smoke
  submit_signhead 2 0       30 ladder_signhead_L2_s0_smoke
  exit 0
fi

if [ "$STAGE" = "full" ]; then
  for L in 2 3 4; do
    for S in "${SEEDS[@]}"; do
      # tier CNN: canonical width everywhere; wider-shallower variant at L=2/3
      submit_cnn "$L" "4 4 4" "$S" 200 "ladder_cnn_L${L}_ch444_s${S}"
      [ "$L" != "4" ] && submit_cnn "$L" "8 8" "$S" 200 "ladder_cnn_L${L}_ch88_s${S}"
      # tier approx-symm: canonical inv(8,8); narrow inv(2,2,2) variant at L=2/3
      submit_asymm "$L" "8 8" "$S" 200 "ladder_asymm_L${L}_inv88_s${S}"
      [ "$L" != "4" ] && submit_asymm "$L" "2 2 2" "$S" 200 "ladder_asymm_L${L}_inv222_s${S}"
      # tier sign-head: seed spread via the prefit trunk init
      submit_signhead "$L" "$S" 100 "ladder_signhead_L${L}_s${S}"
    done
  done
  exit 0
fi

echo "unknown stage: $STAGE" >&2; exit 1
