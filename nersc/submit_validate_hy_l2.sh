#!/bin/bash
# L=2 sign-full (h_y) validation: train the COMPLEX NQS and score it against exact
# diagonalisation (energy ΔE/|E| AND subspace fidelity) — the guardrail that a complex
# log ψ actually captures the h_y sign sector before we trust any L=4 number.
#
# Runs Three_TC/validate_hy_l2.py, which needs BOTH a GPU (NQS training) and a big-RAM
# host (the 2^24 L=2 3D ED sparse build + k=12 eigsh). We mirror the proven
# submit_nqs_hx_sweep.sh shape: shared QOS, 1 GPU. On Perlmutter shared, host RAM
# scales with --cpus-per-task (~2 GB/cpu), so 32 cpus ≈ 64 GB — enough for pure h_y=0.5.
# For the harder pure h_y=1.0 ED (larger near-degenerate manifold) bump RAM at submit
# time:  sbatch --cpus-per-task=64 ... (≈128 GB).
#
# Usage (defaults do a fast one-point DEBUG smoke of pure h_y=0.5):
#   sbatch --qos=debug -t 00:30:00 -N 1 -G 4 --exclusive nersc/submit_validate_hy_l2.sh  # smoke
#   MINI_SWEEP=1 sbatch --qos=shared --time=01:00:00 nersc/submit_validate_hy_l2.sh
#   DIAG_SHIFT=5e-3 TAGS=pure_hy0.5,pure_hy1.0 \
#       sbatch --qos=shared --time=01:30:00 --cpus-per-task=64 nersc/submit_validate_hy_l2.sh
# (debug QOS forces a whole node = full RAM; shared gives a fractional slice, RAM ∝ cpus.)
#
# QOS/time/cpus are passed on the sbatch command line; every physics knob is an env var,
# so the same file serves the smoke and the production grid.
#SBATCH --job-name=tc-hyval-l2
#SBATCH --account=m5340_g
#SBATCH --qos=shared
#SBATCH --constraint=gpu
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=01:30:00
#SBATCH --output=%x-%j.out
set -euo pipefail

module load conda
conda activate tc-nqs

REPO="${REPO:-$HOME/threed_TC/ThreeD_TC}"
cd "$REPO" || { echo "[submit] REPO not found: $REPO"; exit 1; }

OUTDIR="${OUTDIR:-$PSCRATCH/tc_nqs/hy_validation_l2}"
mkdir -p "$OUTDIR"

MINI_SWEEP="${MINI_SWEEP:-0}"
DIAG_SHIFT="${DIAG_SHIFT:-5e-3}"
TAGS="${TAGS:-pure_hy0.5}"          # comma-sep grid tags; default = single pure-h_y point
N_ITER="${N_ITER:-150}"             # smoke default; production uses 500 (script default)
ARCH="${ARCH:-}"                    # empty → script default (ToricCNN_full); ToricCNN_gridinv
                                    #   to validate the large-L workhorse in the sign-full regime
OUT="${OUT:-$OUTDIR/validate_hy_l2_${SLURM_JOB_ID:-local}.json}"

echo "=== hy L=2 validation | arch=${ARCH:-default} mini_sweep=$MINI_SWEEP diag_shift=$DIAG_SHIFT tags=$TAGS n_iter=$N_ITER ==="
echo "=== REPO=$REPO  git=$(git rev-parse --short HEAD 2>/dev/null)  OUT=$OUT ==="

if [ "$MINI_SWEEP" = "1" ]; then
    # pick diag_shift by fidelity at pure h_y=0.5 (ED computed once, reused across shifts)
    ARGS=(--mini-sweep)
else
    ARGS=(--diag-shift "$DIAG_SHIFT" --tags "$TAGS")
fi
[ -n "${N_ITER:-}" ] && ARGS+=(--n-iter "$N_ITER")
[ -n "$ARCH" ] && ARGS+=(--arch "$ARCH")

srun python -m Three_TC.validate_hy_l2 "${ARGS[@]}" --out "$OUT"
echo "=== done → $OUT ==="
