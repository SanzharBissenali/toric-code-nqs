#!/bin/bash
# Stage-0 certification for the hy-AXIS campaign (feat/hy-axis-L4): does the
# pure-hy line (hx=hz=0) COLD-START at L=2 OBC?  Settles the recipes-§C
# "fidelity 5e-8" precedent (whose exact params were never recorded) with the
# tuned cert stack before any L=4 production point is submitted.
#
# One debug job trains SEVERAL points sequentially (each ~4-6 min at L=2:
# 400 steps, ~0.6 s/step) and scores each against the dense ED referee bank
# (analysis/scripts/ed_referee_hy.py ground vectors, generated beforehand into
# results/hy_l2_certification/) via analysis/scripts/hy_cert_fidelity.py.
#
# Lanes (env knobs; every value travels through the AUTO_RESUBMIT requeue):
#   baseline:  HYS="0.4 0.8 1.0 1.2" DS=1e-3 WF=0
#   rescue:    HYS="0.4 0.8 1.0 1.2" DS=3e-3 WF=0.1 GW=40
#   seeds:     HYS="1.0 1.2" SEED=1 (each lane)
#
#SBATCH --job-name=tc-hy-l2ax
#SBATCH --account=m5340_g
#SBATCH --qos=debug
#SBATCH --constraint=gpu
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
#SBATCH --signal=B:USR1@180
#SBATCH --output=%x-%j.out
set -u
module load conda
conda activate tc-nqs

REPO="${REPO:-$HOME/toric-code-nqs}"
# The conda env's tc3d is pip-installed editable against ANOTHER clone; shadow
# it so python -m tc3d.train imports THIS checkout.
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$PSCRATCH/tc_nqs/jax_cache}"
mkdir -p "$JAX_COMPILATION_CACHE_DIR"

OUT="${OUT:-$PSCRATCH/tc_nqs/hy_axis/l2_cert}"
mkdir -p "$OUT"
cd "$REPO"

echo "== import gate: verify tc3d resolves to \$REPO =="
python -c "
import tc3d, os
p = os.path.abspath(tc3d.__file__)
repo = os.path.abspath('$REPO')
print('tc3d:', p)
assert p.startswith(repo), f'tc3d NOT from {repo} -- PYTHONPATH shadow failed'
print('OK: tc3d resolves to this checkout')
"

HYS="${HYS:-0.4 0.8 1.0 1.2}"
DS="${DS:-1e-3}"
WF="${WF:-0}"            # --warmup_frac (0 = off)
GW="${GW:-5}"            # --guard_warmup (train.py default 5; raise with WF)
SEED="${SEED:-0}"
N_ITER="${N_ITER:-400}"

# ---- auto-resubmit just before the wall limit (opt-in; bounded) -------------
RESUB_COUNT="${RESUB_COUNT:-0}"
MAX_RESUBMITS="${MAX_RESUBMITS:-3}"
requeue() {
  if [ "${AUTO_RESUBMIT:-0}" = "1" ] && [ "$RESUB_COUNT" -lt "$MAX_RESUBMITS" ]; then
    echo "[submit] wall limit near -- resubmitting (resume #$((RESUB_COUNT+1)))"
    RESUB_COUNT=$((RESUB_COUNT+1)) AUTO_RESUBMIT=1 MAX_RESUBMITS="$MAX_RESUBMITS" \
      REPO="$REPO" OUT="$OUT" HYS="$HYS" DS="$DS" WF="$WF" GW="$GW" \
      SEED="$SEED" N_ITER="$N_ITER" sbatch "$0"
  fi
  exit 0
}
trap requeue USR1

for HY in $HYS; do
  NAME="hy_axis_l2cert_hy${HY}_ds${DS}_wf${WF}_s${SEED}"
  GS="results/hy_l2_certification/gs_L2_OBC_hx0.0_hy${HY}_hz0.0_dual.npz"
  if [ ! -f "$GS" ]; then
    echo "[cert] SKIP hy=$HY -- missing ED reference $GS (run ed_referee_hy.py first)"
    continue
  fi
  if [ -f "$OUT/$NAME.fidelity.json" ]; then
    echo "[cert] hy=$HY already scored -- skipping (idempotent resubmit)"
    continue
  fi
  echo "[cert] $NAME  L=2 OBC dual cold  hx=0 hy=$HY hz=0  ds=$DS wf=$WF seed=$SEED"
  # `srun ... &` + `wait` so the USR1 trap fires promptly (a foreground srun
  # would swallow the signal until it returns). --resume is always passed: a
  # no-op on the first submit, continues after a requeue.
  srun -n 1 python -u -m tc3d.train \
    --L 2 --bc OBC --dual_basis --hx 0.0 --hy "$HY" --hz 0.0 \
    --arch ToricCNN_gridinv --noninv_hidden 4 8 --inv_hidden 8 8 \
    --qgt dense --n_samples 2048 --n_chains 256 \
    --dt 0.02 --lr_min 0.002 --diag_shift "$DS" --n_iter "$N_ITER" \
    --warmup_frac "$WF" --guard_warmup "$GW" \
    --checkpoint_every 10 --snapshot_every 50 --resume --seed "$SEED" \
    --no_topological --no_wandb \
    --out_dir "$OUT" --name "$NAME" &
  wait $!
  srun -n 1 python -u analysis/scripts/hy_cert_fidelity.py \
    --json "$OUT/$NAME.json" --gs "$GS" &
  wait $!
done
echo "[cert] lane done: DS=$DS WF=$WF SEED=$SEED HYS=$HYS"
