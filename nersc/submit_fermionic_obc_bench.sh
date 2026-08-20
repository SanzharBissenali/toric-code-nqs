#!/bin/bash
#SBATCH --job-name=tc-fobc-bench
#SBATCH --account=m5340_g
#SBATCH --qos=shared
#SBATCH --constraint=gpu
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out

# fTC L=2 OBC benchmark campaign: the 7-run matrix of
# notes/fermionic_obc_l2_benchmark_plan.md (4x frozen-head kappa=6 rectangle,
# 2x kappa=0 magnetic variants, 1x sign-blind control) + per-run snapshot
# evals (exact full-space vs committed ED referee + MC-sampled). Gates first:
# the OBC test suite and an on-cluster prefit regeneration (certificate).
set -u
module load conda
conda activate tc-nqs
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$PSCRATCH/tc_nqs/jax_cache}"
mkdir -p "$JAX_COMPILATION_CACHE_DIR"

REPO="${REPO:-$HOME/toric-code-nqs}"
# The conda env's tc3d is pip-installed editable against ANOTHER clone; shadow
# it so every invocation (tests, scripts, python -m) imports THIS checkout.
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
OUT="${OUT:-$PSCRATCH/tc_nqs/fermionic_obc_L2}"
EDREF="$REPO/results/fermionic_obc_L2"          # committed ED referee data
mkdir -p "$OUT"
cd "$REPO"

echo "== gate 1: fermionic OBC test suite =="
( cd tests && python test_fermionic_obc.py ) || { echo "GATE 1 FAILED"; exit 1; }

echo "== gate 2: analytic prefit regeneration (certificate) =="
PREFIT="$OUT/prefit_anaC_k2_L2_OBC"
if python analysis/scripts/prefit_phase_head.py --L 2 --bc OBC --kernel 2 \
     --analytic_C --frozen --seed 0 --cert 2000 --save "$PREFIT"; then
  echo "prefit regenerated + certified"
else
  echo "WARNING: prefit regen failed -- falling back to committed artifact"
  cp "$EDREF/prefit_anaC_k2_L2_OBC.mpack" "$PREFIT.mpack" || { echo "GATE 2 FAILED"; exit 1; }
fi

COMMON="--L 2 --bc OBC --model fermionic --arch ToricCNN_gridinv \
 --kernel_size 2 --noninv_hidden 4 8 --inv_hidden 8 8 \
 --noninv_channels 4 --n_noninv 2 --chains_up \
 --dt 0.02 --lr_min 0.002 --diag_shift 0.001 \
 --n_samples 8192 --n_chains 1024 --chunk_size 2048 \
 --n_iter 150 --snapshot_every 25 --checkpoint_every 50 \
 --seed 0 --no_wandb --out_dir $OUT"

run () {
  NAME=$1; shift
  echo "== run $NAME  $(date +%H:%M:%S) =="
  python -m tc3d.train $COMMON --name "$NAME" "$@" || { echo "RUN $NAME FAILED"; return 1; }
  python analysis/scripts/eval_snapshots.py --dir "$OUT" --glob "$NAME.json" \
      --rounds 4 --exact --ed_vectors "$EDREF/ed_vectors" \
      || echo "EVAL $NAME FAILED"
}

run gridinv_fermionic_L2_OBC_hx0.0_hz0.0_k2_phf_fp6 \
    --hx 0.0 --hz 0.0 --phase_head_frozen --flux_penalty 6.0 --init_from "$PREFIT"
run gridinv_fermionic_L2_OBC_hx0.0_hz0.2_k2_phf_fp6 \
    --hx 0.0 --hz 0.2 --phase_head_frozen --flux_penalty 6.0 --init_from "$PREFIT"
run gridinv_fermionic_L2_OBC_hx0.2_hz0.0_k2_phf_fp6 \
    --hx 0.2 --hz 0.0 --phase_head_frozen --flux_penalty 6.0 --init_from "$PREFIT"
run gridinv_fermionic_L2_OBC_hx0.2_hz0.2_k2_phf_fp6 \
    --hx 0.2 --hz 0.2 --phase_head_frozen --flux_penalty 6.0 --init_from "$PREFIT"
run gridinv_fermionic_L2_OBC_hx0.2_hz0.0_k2_phf_fp0 \
    --hx 0.2 --hz 0.0 --phase_head_frozen --init_from "$PREFIT"
run gridinv_fermionic_L2_OBC_hx0.2_hz0.2_k2_phf_fp0 \
    --hx 0.2 --hz 0.2 --phase_head_frozen --init_from "$PREFIT"
run gridinv_fermionic_L2_OBC_hx0.2_hz0.2_k2_noph_fp0 \
    --hx 0.2 --hz 0.2

echo "== CAMPAIGN COMPLETE  $(date +%H:%M:%S) =="
