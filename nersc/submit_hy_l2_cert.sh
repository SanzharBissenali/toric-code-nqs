#!/bin/bash
# One-off NQS training job for the bosonic h_y != 0 (sign-problem-full) L=2 OBC
# certification (feat/hy-ed-referee): dual-basis complex ToricCNN_gridinv at
# (hx=0.2, hy=0.2, hz=0.1), certified against the dense ED referee
# (analysis/scripts/ed_referee_hy.py, results/hy_l2_certification/).
#
# Debug-queue lane (user-approved venue change from local CPU, which ran
# ~65-70 s/step under contention): 30 min cap, checkpointed every 10 steps +
# --resume, so a timeout just needs a re-`sbatch` (or AUTO_RESUBMIT=1 for the
# USR1 self-chain, capped at MAX_RESUBMITS -- up to 3 chained debug submits).
#
#SBATCH --job-name=tc-hy-l2-cert
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

# REPO must be the feat/hy-ed-referee worktree, NOT ~/toric-code-nqs (the main
# clone serves other campaigns on a different branch -- never touch it).
REPO="${REPO:-$HOME/wt-hy-ed}"
# The conda env's tc3d is pip-installed editable against ANOTHER clone; shadow
# it so python -m tc3d.train imports THIS checkout (see the 2026-08-20
# fermionic-OBC-bench gate 1 postmortem: silent wrong-clone import -> stale
# KeyError).
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$PSCRATCH/tc_nqs/jax_cache}"
mkdir -p "$JAX_COMPILATION_CACHE_DIR"

OUT="${OUT:-$PSCRATCH/tc_nqs/hy_l2_cert}"
mkdir -p "$OUT"
cd "$REPO"

echo "== import gate: verify tc3d resolves to \$REPO, not the peer clone =="
python -c "
import tc3d, os
p = os.path.abspath(tc3d.__file__)
repo = os.path.abspath('$REPO')
print('tc3d:', p)
assert p.startswith(repo), f'tc3d NOT from {repo} -- PYTHONPATH shadow failed'
print('OK: tc3d resolves to the worktree')
"

NAME="${NAME:-hy_l2_cert_hx0.2_hy0.2_hz0.1}"

# ---- auto-resubmit just before the wall limit (opt-in; bounded) -------------
RESUB_COUNT="${RESUB_COUNT:-0}"
MAX_RESUBMITS="${MAX_RESUBMITS:-3}"
requeue() {
  if [ "${AUTO_RESUBMIT:-0}" = "1" ] && [ "$RESUB_COUNT" -lt "$MAX_RESUBMITS" ]; then
    echo "[submit] wall limit near -- resubmitting (resume #$((RESUB_COUNT+1)))"
    RESUB_COUNT=$((RESUB_COUNT+1)) AUTO_RESUBMIT=1 MAX_RESUBMITS="$MAX_RESUBMITS" \
      REPO="$REPO" OUT="$OUT" NAME="$NAME" sbatch "$0"
  fi
  exit 0
}
trap requeue USR1

echo "[submit] $NAME  L=2 OBC dual  hx=0.2 hy=0.2 hz=0.1  (resume #$RESUB_COUNT)"

# `srun ... &` + `wait` so the trap fires promptly on USR1 (a foreground srun
# would swallow the signal until it returns). --resume is always passed: a
# no-op on the first submit (no checkpoint yet), continues on every later one.
srun -n 1 python -u -m tc3d.train \
  --L 2 --bc OBC --dual_basis --hx 0.2 --hy 0.2 --hz 0.1 \
  --arch ToricCNN_gridinv --noninv_hidden 4 8 --inv_hidden 8 8 \
  --qgt dense --n_samples 2048 --n_chains 256 \
  --dt 0.02 --lr_min 0.002 --diag_shift 1e-3 --n_iter 400 \
  --checkpoint_every 10 --resume --seed 0 --no_topological --no_wandb \
  --out_dir "$OUT" --name "$NAME" &
wait
