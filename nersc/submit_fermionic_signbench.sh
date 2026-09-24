#!/bin/bash
#SBATCH --job-name=tc-signbench
#SBATCH --account=m5340_g
#SBATCH --qos=shared
#SBATCH --constraint=gpu
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=02:30:00
#SBATCH --output=%x-%A_%a.out
#
# Learned-vs-gated sign-head benchmark, 3D half (2D-TC docs/signhead_benchmark_plan.md
# section 5): fermionic TC on the 2x2x3 OBC box (N=20), 3 arms x 9 (h_x, h_z) points.
#   T   --sign_arm twobranch           psi = a A_triv + s_pt2 A_top (signed a, init 0.05)
#   Mp  --sign_arm mlp --sign_mlp_init psi = A tanh(MLP(eps, x)), MLP pretrained on ED signs
#   M   --sign_arm mlp                 same, random MLP init
#   H   --sign_frame pt2               control, not a benchmark arm: head-only (T at a = 0),
#                                      pt2 sign x one positive trunk via the framed H~ = S H S
# One real gridinv trunk recipe, one optimizer (dense SR), one sampler, one seed for all.
#
# Two phases (MODE), chained with afterok so the array only starts on a passing prep:
#   MODE=prep  (no array)  signbench_prep.py: dense-ED referee at the 9 points
#              (eigsh, 2^20), per-arm ceilings, the 9 M-pre MLP fits; fails on any gate.
#   MODE=runs  (--array=0-8, one task per point)  the ARMS in order, each followed by
#              eval_snapshots.py --exact (full 2^20 contraction of every snapshot).
#
#   jid=$(MODE=prep sbatch --parsable --time=01:00:00 nersc/submit_fermionic_signbench.sh)
#   MODE=runs sbatch --dependency=afterok:$jid --array=0-8%3 nersc/submit_fermionic_signbench.sh
#   # smoke (debug QOS, one point, 20 steps, all arms):
#   MODE=smoke sbatch -q debug -t 00:30:00 nersc/submit_fermionic_signbench.sh
#
# Knobs: ARMS (default "T Mp M"), SEED (0), NITER (300), OUT, HX_LIST/HZ_LIST order is
# row-major in hx: task i -> (HX[i/3], HZ[i%3]).
# Pull back (no weights):
#   rsync -av --exclude '*.mpack' --exclude 'ed_vectors' \
#     perlmutter:/pscratch/sd/s/sanzharb/tc_nqs/fermionic_signbench/ results/fermionic_signbench/
set -u
module load conda
conda activate tc-nqs
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$PSCRATCH/tc_nqs/jax_cache}"
mkdir -p "$JAX_COMPILATION_CACHE_DIR"

REPO="${REPO:-$HOME/toric-code-nqs-fsign}"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"    # shadow the env's editable tc3d
cd "$REPO"

MODE="${MODE:-runs}"
SEED="${SEED:-0}"
NITER="${NITER:-300}"
ARMS="${ARMS:-T Mp M}"
OUT="${OUT:-$PSCRATCH/tc_nqs/fermionic_signbench}"
BOX_ARGS="--L 2 --Lxyz 2 2 3 --bc OBC"
BOX="L2x2x3_OBC"
HX_LIST=(0.2 0.5 0.8)
HZ_LIST=(0.0 0.2 0.4)
mkdir -p "$OUT"
echo "== signbench MODE=$MODE  host=$(hostname)  git=$(git rev-parse --short HEAD) =="

if [ "$MODE" = "prep" ]; then
  python analysis/scripts/signbench_prep.py --Lxyz 2 2 3 --bc OBC \
      --hx "${HX_LIST[@]}" --hz "${HZ_LIST[@]}" --seed "$SEED" --out_dir "$OUT"
  exit $?
fi

if [ "$MODE" = "smoke" ]; then
  HX=0.5; HZ=0.2; NITER=20; OUT="$OUT/smoke"; mkdir -p "$OUT"
  python analysis/scripts/signbench_prep.py --Lxyz 2 2 3 --bc OBC --hx $HX --hz $HZ \
      --pretrain_steps 2000 --out_dir "$OUT" || exit 1
else
  TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
  HX="${HX_LIST[$(( TASK_ID / ${#HZ_LIST[@]} ))]}"
  HZ="${HZ_LIST[$(( TASK_ID % ${#HZ_LIST[@]} ))]}"
fi
TAG="${BOX}_hx${HX}_hz${HZ}"
[ -f "$OUT/ed_vectors/gs_${TAG}.npz" ] || { echo "no ED referee for $TAG -- run MODE=prep first"; exit 1; }

COMMON="$BOX_ARGS --model fermionic --hx $HX --hz $HZ \
 --arch ToricCNN_gridinv --kernel_size 2 --noninv_hidden 4 8 --inv_hidden 8 8 --chains_up \
 --dt 0.02 --lr_min 0.002 --diag_shift 0.001 --qgt dense \
 --n_samples 8192 --n_chains 1024 --chunk_size 2048 \
 --n_iter $NITER --snapshot_every 25 --checkpoint_every 25 --resume \
 --seed $SEED --no_wandb --no_topological --out_dir $OUT"
[ "$MODE" = "smoke" ] && COMMON="$COMMON --snapshot_every 10"

FAILED=0
for arm in $ARMS; do
  NAME="signbench_${TAG}_${arm}_s${SEED}"
  case $arm in
    T)  X="--sign_arm twobranch --mix_init 0.05 --sign_arm_head pt2" ;;
    Mp) X="--sign_arm mlp --sign_mlp_init $OUT/signmlp_${TAG}.mpack" ;;
    M)  X="--sign_arm mlp" ;;
    H)  X="--sign_frame pt2" ;;
    *)  echo "unknown arm $arm"; FAILED=$((FAILED + 1)); continue ;;
  esac
  if [ -f "$OUT/$NAME.snapshots.json" ]; then echo "== skip $NAME (evaluated) =="; continue; fi
  if [ ! -f "$OUT/$NAME.json" ]; then
    echo "== run $NAME  $(date +%H:%M:%S) =="
    python -m tc3d.train $COMMON --name "$NAME" $X \
        || { echo "RUN $NAME exited nonzero (guard divergence finalizes the last sane state)";
             [ -f "$OUT/$NAME.json" ] || { FAILED=$((FAILED + 1)); continue; }; }
  fi
  python analysis/scripts/eval_snapshots.py --dir "$OUT" --glob "$NAME.json" \
      --exact --exact_only --ed_vectors "$OUT/ed_vectors" --exact_chunk 16384 \
      || { echo "EVAL $NAME FAILED"; FAILED=$((FAILED + 1)); }
done
echo "== task done (hx=$HX hz=$HZ): $FAILED failed arm(s)  $(date +%H:%M:%S) =="
[ "$FAILED" -eq 0 ]
