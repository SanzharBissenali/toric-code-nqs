#!/bin/bash
#SBATCH -A m5340_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -c 32
#SBATCH -t 05:00:00
#SBATCH -J fspeed
#SBATCH -o %x_%j.out
#
# fTC in-vivo TIMING ladder: per-SR-step wall-clock (sample/grad/qgt/update/total,
# + t_head/n_head_configs when the run carries a sign head) vs system size, ONE ARM
# PER JOB so a single arm/L point can be resubmitted without disturbing the rest.
# Same GRIDINV recipe throughout (kernel L-1, noninv 4-8, inv 8-8, chains_up, dt
# 0.02/diag_shift 1e-3) -- only the head flags change across arms:
#
#   baseline  no head at all (GPU-only control; --dtype float64 forces the SAME
#             real trunk the framed arms use, so baseline is an apples-to-apples
#             timing floor, not a physically converged run -- see CAVEAT below)
#   cup       lattice cup-product sign head (exact ALL L, notes/fermionic_sign_geometry.md)
#   linear    linear-decoder sign head
#   vote      majority-vote sign head
#   pt2       2nd-order perturbation-theory sign head
#
# Guard OPEN on every arm (--spike_factor 1e6 --max_rollbacks 50, as the hx-ladder's sign-blind
# tiers): the no-head real trunk and the frozen-sign heads hit the variance wall by design; a
# timing run must not stop there.
# Head CLI: --sign_frame {cup,linear,vote,pt2} + --sign_k_cap + --sign_max_terms are wired in
# tc3d/train.py / tc3d/sign_frame.py::build_sign_fn / tc3d/sign_decoders.py (feat/signhead-speed);
# linear/vote/pt2 are OBC-only (their lit classes are undefined at PBC).
#
# Submit (one arm, one L; from this worktree, on the cluster):
#   sbatch --export=ALL,L=4,ARM=cup nersc/submit_fermionic_speed_ladder.sh
# Debug queue, short smoke (10 steps, enough to clear the compile step):
#   sbatch -q debug -t 00:30:00 --export=ALL,L=3,ARM=pt2,N_ITER=10 nersc/submit_fermionic_speed_ladder.sh
# Full ladder: submit ONE ARM PER JOB, independently (no afterok chains -- one TIMEOUT
# would cancel every downstream job). Keep <= 4 jobs in the queue at once (cluster
# charter); the coordinator paces submissions L=2 -> 6, baseline first, both field
# points ((0.5,0.2) all L; (0.2,0.0) at L<=4):
#   sbatch -t 00:30:00 --export=ALL,L=2,ARM=cup            nersc/submit_fermionic_speed_ladder.sh
#   sbatch -t 01:00:00 --export=ALL,L=3,ARM=pt2,HX=0.2,HZ=0.0 nersc/submit_fermionic_speed_ladder.sh
# Check:
#   squeue -u sanzharb
# Pull results back:
#   rsync -av --exclude '*.mpack' perlmutter:/pscratch/sd/s/sanzharb/tc_nqs/fermionic_speed/ results/fermionic_speed/
#
# Recommended --time per L (this system's fermionic decoration is N ~ 3L^3, so
# the invariant grid-conv + any host-side sign head both grow steeply with L):
#   L=2  0:30   L=3  1:00   L=4  1:30 (pt2: see pt2 note)   L=5  4:00   L=6  5:00
# pt2 is the enumeration head (2nd order on exact ties): its per-row cost grows steeply with N,
# so at L>=4 it is submitted with a reduced N_SAMPLES (512) and/or a walltime cut accepted --
# the notebook keeps cut runs that passed the plateau start and flags them.
#
# Approximate per-step wall-clock this ladder is expected to land on (baseline
# GPU-only trunk; a sign head adds t_head on top -- that addition IS the
# measurement this ladder exists to make):
#   L=2  N=12   ~1 s/step
#   L=3  N=54   ~7 s/step
#   L=4  N=144  ~25-35 s/step
#   L=5  N=300  ~1.5-2 min/step
#   L=6  N=540  ~4-5 min/step
#
# Env knobs (all `${VAR:-default}`, override at submit time via --export or plain
# `VAR=... sbatch ...`):
#   L            (required) linear size, 2..6 tuned; other L falls back to
#                CHUNK=256/N_ITER=60 with a warning -- pass CHUNK/N_ITER explicitly
#   ARM          (required) one of: baseline cup linear vote pt2
#   HX=0.5  HZ=0.2  SEED=0
#   N_ITER       default per L: 100 (L=2,3; plateau check), 60 (L=4,5), 40 (L=6)
#   CHUNK        default per L: 2048/256/256/64/32 for L=2/3/4/5/6
#   N_SAMPLES=4096  N_CHAINS=1024  K_CAP=8  MAX_TERMS=200000  CKPT_EVERY=5
#   OUT=$PSCRATCH/tc_nqs/fermionic_speed
#   REPO=$HOME/toric-code-nqs-fsign
#   BLAS_THREADS=32  OMP/OPENBLAS/MKL threads for the numpy head (=SLURM_CPUS_PER_TASK); 1 = serial head
#   THREADS=32       NUMBA_NUM_THREADS (unused by today's heads)
#   WANDB=0          0 -> --no_wandb (default); 1 -> --wandb_offline
#
# Walltime is a submit-time sbatch flag (-t / --time), NOT an env knob here --
# see the recommended-per-L table above and the debug-queue example.
set -uo pipefail

module load conda
conda activate tc-nqs
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$PSCRATCH/tc_nqs/jax_cache}"
mkdir -p "$JAX_COMPILATION_CACHE_DIR"

REPO="${REPO:-$HOME/toric-code-nqs-fsign}"
# The conda env's tc3d is pip-installed editable against ANOTHER clone; shadow
# it so every invocation (scripts, python -m) imports THIS checkout.
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO" || { echo "[fspeed] REPO not found: $REPO"; exit 1; }

# Threads: the sign heads (tc3d/sign_decoders.py) are pure numpy -- the hot ops are
# float32 GEMMs (b @ K) and small einsums, so the HEAD's parallelism is the BLAS pool.
# BLAS_THREADS defaults to the job's cores (32); BLAS_THREADS=1 measures a serial head.
# NUMBA_NUM_THREADS is exported for completeness (no numba in the head today).
export NUMBA_NUM_THREADS="${THREADS:-32}"
BLAS_THREADS="${BLAS_THREADS:-${SLURM_CPUS_PER_TASK:-32}}"
export OMP_NUM_THREADS="$BLAS_THREADS" OPENBLAS_NUM_THREADS="$BLAS_THREADS" MKL_NUM_THREADS="$BLAS_THREADS"

L="${L:?set L (required, 2..6)}"
ARM="${ARM:?set ARM (required: one of baseline cup linear vote pt2)}"
case "$ARM" in
  baseline|cup|linear|vote|pt2) ;;
  *) echo "[fspeed] bad ARM='$ARM' (need one of: baseline cup linear vote pt2)"; exit 1 ;;
esac

HX="${HX:-0.5}"
HZ="${HZ:-0.2}"
SEED="${SEED:-0}"

# per-L tuned defaults (N_ITER, CHUNK); override either explicitly for any L
case "$L" in
  2) _NITER_D=100; _CHUNK_D=2048 ;;
  3) _NITER_D=100; _CHUNK_D=256  ;;
  4) _NITER_D=60; _CHUNK_D=256  ;;
  5) _NITER_D=60; _CHUNK_D=64   ;;
  6) _NITER_D=40; _CHUNK_D=32   ;;
  *) echo "[fspeed] WARNING: no tuned N_ITER/CHUNK default for L=$L (only 2..6) -- using 60/256, override explicitly"
     _NITER_D=60; _CHUNK_D=256 ;;
esac
N_ITER="${N_ITER:-$_NITER_D}"
CHUNK="${CHUNK:-$_CHUNK_D}"
N_SAMPLES="${N_SAMPLES:-4096}"
N_CHAINS="${N_CHAINS:-1024}"
K_CAP="${K_CAP:-8}"
CKPT_EVERY="${CKPT_EVERY:-5}"   # .curve.json cadence: a walltime-cut run keeps steps >= 20 on disk
OUT="${OUT:-$PSCRATCH/tc_nqs/fermionic_speed}"
WANDB="${WANDB:-0}"
mkdir -p "$OUT"

# invariant grid-conv kernel: 2 at L=2 (matches the L=2 ladder/plane convention),
# else L-1 (kernel_size saturates at Lx-1 -- see l6-l7-diag-shift-default note)
KERNEL=2
[ "$L" != "2" ] && KERNEL=$((L - 1))

WANDB_FLAG="--no_wandb"
[ "$WANDB" = "1" ] && WANDB_FLAG="--wandb_offline"

HEAD_FLAGS=""
MAX_TERMS="${MAX_TERMS:-200000}"
[ "$ARM" != "baseline" ] && HEAD_FLAGS="--sign_frame $ARM --sign_k_cap $K_CAP --sign_max_terms $MAX_TERMS"

NAME="speed_L${L}_${ARM}_hx${HX}_hz${HZ}_s${SEED}"

echo "=== code $(git -C "$REPO" rev-parse --short HEAD) ($(git -C "$REPO" branch --show-current)) ==="
echo "=== NUMBA_NUM_THREADS=$NUMBA_NUM_THREADS  BLAS threads=${BLAS_THREADS:-1}  SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-?} ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "=== $NAME:  L=$L ARM=$ARM  hx=$HX hz=$HZ seed=$SEED  kernel=$KERNEL chunk=$CHUNK n_iter=$N_ITER ==="

COMMON="--L $L --bc OBC --model fermionic --hx $HX --hz $HZ \
 --arch ToricCNN_gridinv --kernel_size $KERNEL --noninv_hidden 4 8 --inv_hidden 8 8 \
 --noninv_channels 4 --n_noninv 2 --chains_up --dt 0.02 --lr_min 0.002 --diag_shift 0.001 \
 --n_samples $N_SAMPLES --n_chains $N_CHAINS --chunk_size $CHUNK --n_iter $N_ITER \
 --checkpoint_every $CKPT_EVERY --seed $SEED --dtype float64 --no_topological --qgt onthefly \
 --spike_factor 1e6 --max_rollbacks 50 $WANDB_FLAG \
 --out_dir $OUT --name $NAME"

if [ -f "$OUT/$NAME.json" ] && ! grep -q '"diverged": true' "$OUT/$NAME.json"; then
  echo "== skip $NAME ($OUT/$NAME.json exists) =="
else
  echo "== run $NAME  $(date +%H:%M:%S) =="
  echo "[fspeed] python -m tc3d.train $COMMON $HEAD_FLAGS"
  python -m tc3d.train $COMMON $HEAD_FLAGS; RC=$?
  if [ $RC -ne 0 ]; then
    echo "RUN $NAME FAILED (see guard/log lines above)"
  fi
fi

# --- plateau summary: mean total / t_head over steps 20..N_ITER-1 -------------
python3 - "$OUT/$NAME.json" "$OUT/$NAME.curve.json" "$N_ITER" <<'PYEOF'
import json, os, sys

final_path, curve_path, n_iter = sys.argv[1], sys.argv[2], int(sys.argv[3])
path = final_path if os.path.exists(final_path) else curve_path
if not os.path.exists(path):
    print(f"[fspeed] no timing data found ({final_path} or {curve_path})")
    raise SystemExit(0)

with open(path) as f:
    d = json.load(f)
timing = d.get("curve", {}).get("timing", [])
lo, hi = 20, n_iter - 1
window = [t for t in timing if lo <= t.get("step", -1) <= hi]
if not window:
    print(f"[fspeed] {path}: {len(timing)} timing entries logged, none in plateau "
          f"window [{lo},{hi}] -- run stopped before step {lo} (walltime cut / short debug run)")
    raise SystemExit(0)

def mean(key):
    vals = [t[key] for t in window if key in t]
    return sum(vals) / len(vals) if vals else None

m_total, m_head = mean("total"), mean("t_head")
line = (f"[fspeed] {path}: plateau mean over {len(window)} steps in [{lo},{hi}]: "
        f"total={m_total:.4f} s/step")
if m_head is not None:
    share = m_head / m_total if m_total else float("nan")
    line += f"  t_head={m_head:.4f} s/step ({share * 100:.1f}% of total)"
else:
    line += "  (no t_head field in the timing dict: baseline arm, or this sign head isn't wired into pop_head_stats)"
print(line)
PYEOF

# --- how far did this job get before any walltime cut? ------------------------
LOGFILE="${SLURM_JOB_NAME:-fspeed}_${SLURM_JOB_ID:-unknown}.out"
if [ -f "$LOGFILE" ]; then
  NSTEP=$(grep -c '\[t\] step' "$LOGFILE")
  echo "== $LOGFILE: $NSTEP '[t] step' lines logged (need >= 21 to clear the step-20 plateau floor) =="
else
  echo "== log file $LOGFILE not found -- not running under sbatch, or logs/ missing =="
fi

echo "== $NAME COMPLETE $(date +%H:%M:%S) (train rc=${RC:-skipped}) =="
exit "${RC:-0}"
