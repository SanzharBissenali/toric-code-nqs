#!/bin/bash
# BATCHED phase-diagram sweep: many field points per Slurm job, run in ONE Python
# process (tc3d.sweep) so the ~10 min JAX/XLA compile is paid ONCE per job and
# reused across the whole chunk — ~CHUNK_POINTS x fewer compiles than the per-point
# array (submit_nqs_hz_sweep.sh). Same validated ToricCNN_gridinv config, same
# per-point {name}.{json,mpack,curve.json} outputs, so all downstream extraction
# (check_convergence.py / fm.py / renyi.py) is untouched.
#
# The array index now maps to a CHUNK of CHUNK_POINTS field points (default 4), not
# a single point. --array size MUST equal ceil(N / CHUNK_POINTS) where N = HZ_N (hz
# sweep) or HX_N (hx sweep). Field grid is identical to the per-point sweeps:
#   hz sweep (SWEEP=hz): hz_j = HZ_MIN + j*(HZ_MAX-HZ_MIN)/(HZ_N-1), fixed hx=HX
#   hx sweep (SWEEP=hx): hx_j = HX_MIN + j*(HX_MAX-HX_MIN)/(HX_N-1), fixed hz=HZ
#
#   # hz cut, L=4, 16 points, 4 per job -> 4 chunks (array 0-3):
#   L=4 HX=0.2 HZ_N=16 CHUNK_POINTS=4 \
#     sbatch --array=0-3 --time=04:00:00 nersc/submit_nqs_batch.sh
#   # hx cut, L=4, 15 points, 4 per job -> 4 chunks (array 0-3):
#   SWEEP=hx L=4 HZ=0.0 HX_N=15 CHUNK_POINTS=4 \
#     sbatch --array=0-3 --time=04:00:00 nersc/submit_nqs_batch.sh
#
# Walltime: a chunk trains CHUNK_POINTS points sequentially, so request ~CHUNK_POINTS
# x the per-point walltime (see run_phase_campaign.sh:walltime_for), capped at the
# 5 h QOS limit; AUTO_RESUBMIT=1 chains the remainder across requeues.
#
# AUTO_RESUBMIT=1 makes each chunk requeue ITSELF (same array index -> same field
# values) ~180 s before the wall limit; tc3d.sweep skips points whose
# {name}.json already exists and resumes the in-flight one from its checkpoint, so a
# requeue continues the chunk without a manual re-submit (opt-in; off by default).
#SBATCH --job-name=tc-batch
#SBATCH --account=m5340_g
#SBATCH --qos=shared
#SBATCH --constraint=gpu
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=05:00:00
#SBATCH --array=0-3
#SBATCH --signal=B:USR1@180
#SBATCH --output=slurm_logs/%x-%A_%a.out
set -euo pipefail

module load conda
conda activate tc-nqs

REPO="${REPO:-$HOME/toric-code-nqs}"
cd "$REPO" || { echo "[batch] REPO not found: $REPO — set REPO=<clone path>"; exit 1; }

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Persistent XLA cache (shared across L, keyed by HLO) — a fresh job still reuses the
# compiled kernels from disk; the in-process reuse (this script's whole point) then
# amortises even the cache-lookup/link cost across the chunk's points.
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$PSCRATCH/tc_nqs/jax_cache}"
mkdir -p "$JAX_COMPILATION_CACHE_DIR"

# ---- sweep definition (SWEEP=hz|hx) ------------------------------------------
SWEEP="${SWEEP:-hz}"
CHUNK_POINTS="${CHUNK_POINTS:-4}"           # field points folded into one process
L="${L:-4}"
if [ "$SWEEP" = "hz" ]; then                # fixed hx, swept hz
  HX="${HX:-0.2}"; FIXED="$HX"
  GMIN="${HZ_MIN:-0.0}"; GMAX="${HZ_MAX:-0.9}"; GN="${HZ_N:-16}"
  OUT_DIR="${OUT_DIR:-$PSCRATCH/tc_nqs/phase_hx${HX}/L${L}}"
  WB_TAG="batch-hzsweep-L${L}-hx${HX}"
elif [ "$SWEEP" = "hx" ]; then              # fixed hz, swept hx (orthogonal cut)
  HZ="${HZ:-0.0}"; FIXED="$HZ"
  GMIN="${HX_MIN:-0.8}"; GMAX="${HX_MAX:-1.3}"; GN="${HX_N:-15}"
  OUT_DIR="${OUT_DIR:-$PSCRATCH/tc_nqs/phase_hz${HZ}/L${L}}"
  WB_TAG="batch-hxsweep-L${L}-hz${HZ}"
else
  echo "[batch] SWEEP must be hz or hx (got '$SWEEP')"; exit 1
fi
# W&B group: explicit WANDB_GROUP wins, else the Slurm job name (so a
# multi-plane campaign that submits per-(hy,hz,L,branch) job names, e.g.
# p3d_hy{hy}_m{hz}_L{L}_{up|dn}, gets one group per launch instead of every
# hy plane's up+dn chains colliding into the single WB_TAG below), else the
# WB_TAG default (unnamed submissions, unchanged from before).
WANDB_GROUP="${WANDB_GROUP:-${SLURM_JOB_NAME:-$WB_TAG}}"

# chunk index -> the field values in this chunk (same round(...,4) as the per-point
# sweeps, so names/points are byte-identical to the array runs). FIELD_VALUES
# (explicit space-separated list) overrides the uniform GMIN/GMAX/GN grid —
# non-uniform campaign grids (e.g. Phase B coarse+fine windows) pass it directly;
# --array size must then be ceil(len(FIELD_VALUES) / CHUNK_POINTS).
if [ -n "${FIELD_VALUES:-}" ]; then
  VALUES=$(python -c "
vals='$FIELD_VALUES'.split()
i=$SLURM_ARRAY_TASK_ID; c=$CHUNK_POINTS
print(' '.join(vals[i*c:(i+1)*c]))
")
else
  VALUES=$(python -c "
i=$SLURM_ARRAY_TASK_ID; c=$CHUNK_POINTS; n=$GN
lo=i*c; hi=min((i+1)*c, n)
print(' '.join(str(round($GMIN + j*($GMAX-$GMIN)/(n-1), 4)) for j in range(lo, hi)))
")
fi
if [ -z "$VALUES" ]; then
  echo "[batch] chunk ${SLURM_ARRAY_TASK_ID} is empty (array larger than ceil($GN/$CHUNK_POINTS)); nothing to do."
  exit 0
fi

# ---- architecture / optimization (validated gridinv config; == per-point sweep) --
BC="${BC:-OBC}"
NONINV="${NONINV:-4}"; N_NONINV="${N_NONINV:-2}"; INV="${INV:-2 2 2}"
if [ -z "${KERNEL:-}" ]; then
  case "$L" in
    3) KERNEL=2 ;; 4) KERNEL=3 ;; 5) KERNEL=4 ;; 6) KERNEL=4 ;; 7) KERNEL=5 ;; *) KERNEL=4 ;;
  esac
fi
DT="${DT:-0.01}"; LR_MIN="${LR_MIN:-0.001}"
case "$L" in
  7) DIAG_SHIFT="${DIAG_SHIFT:-5e-3}"; N_ITER="${N_ITER:-175}" ;;
  6) DIAG_SHIFT="${DIAG_SHIFT:-5e-3}"; N_ITER="${N_ITER:-150}" ;;
  *) DIAG_SHIFT="${DIAG_SHIFT:-1e-3}"; N_ITER="${N_ITER:-150}" ;;
esac
N_SAMPLES="${N_SAMPLES:-8192}"; N_CHAINS="${N_CHAINS:-1024}"
N_SWEEPS="${N_SWEEPS:-48}"; QGT="${QGT:-dense}"; CKPT_EVERY="${CKPT_EVERY:-10}"
SNAPSHOT_EVERY="${SNAPSHOT_EVERY:-0}"     # >0: keep {name}.step{N}.mpack snapshots
CHUNK="${CHUNK:-2048}"             # --chunk_size (memory; L>=6 int32-overflow guard)
HY="${HY:-0.0}"                    # fixed passthrough Y field (--hy; complex ansatz auto-derives)
QGT_SOLVER="${QGT_SOLVER:-}"       # dense-QGT linear solver override; empty -> train.py default
COMPUTE_DTYPE="${COMPUTE_DTYPE:-}" # speed lever: --compute_dtype float32 (QGT stays double)
INV_IMPL="${INV_IMPL:-}"           # speed lever: --inv_impl dense (unfolded-GEMM invariant block)

# Late-binding campaign defaults: speed knobs left EMPTY at sbatch time are filled at job
# START from $TC3D_DEFAULTS_FILE (KEY=VALUE lines), so already-queued jobs adopt new settings
# without losing queue age. AUTO_RESUBMIT re-passes the resolved values explicitly, so a
# requeue chain never changes knobs mid-run; an explicit env value always wins.
TC3D_DEFAULTS_FILE="${TC3D_DEFAULTS_FILE:-${PSCRATCH:-/nonexistent}/tc_nqs/phase3d/defaults.env}"
if [ -f "$TC3D_DEFAULTS_FILE" ]; then
  while IFS='=' read -r k v; do
    case "$k" in
      QGT_SOLVER|COMPUTE_DTYPE|INV_IMPL) [ -z "${!k}" ] && printf -v "$k" '%s' "$v" && echo "[submit] default from $TC3D_DEFAULTS_FILE: $k=$v" ;;
    esac
  done < <(grep -E '^(QGT_SOLVER|COMPUTE_DTYPE|INV_IMPL)=' "$TC3D_DEFAULTS_FILE")
fi
WANDB_PROJECT="${WANDB_PROJECT:-}" # empty -> train.py's own default project
EXTRA_ARGS="${EXTRA_ARGS:-}"       # verbatim extra tc3d.sweep flags (space-separated)

# ---- chain-runner knobs (one resume-safe job = one warm-started branch) ------
WARM_START="${WARM_START:-0}"      # 1 -> --warm_start: chain each point off the
                                    # previous point's converged weights
ANCHOR_OVERRIDES="${ANCHOR_OVERRIDES:-}"  # JSON dict, point i==0 ONLY, e.g.
                                    # '{"dt":0.02,"lr_min":0.002,"n_iter":500,"diag_shift":1e-3}'
INIT_FROM="${INIT_FROM:-}"         # external checkpoint BASE path (no .mpack);
                                    # under WARM_START, seeds point i==0 only

KERNEL_FLAG=""; [ "$KERNEL" != "0" ] && KERNEL_FLAG="--kernel_size $KERNEL"
CHUNK_FLAG="";  [ -n "$CHUNK" ]      && CHUNK_FLAG="--chunk_size $CHUNK"
SNAP_FLAG="";   [ "$SNAPSHOT_EVERY" != "0" ] && SNAP_FLAG="--snapshot_every $SNAPSHOT_EVERY"
QGT_SOLVER_FLAG="";    [ -n "$QGT_SOLVER" ]    && QGT_SOLVER_FLAG="--qgt_solver $QGT_SOLVER"
CD_FLAG="";            [ -n "$COMPUTE_DTYPE" ] && CD_FLAG="--compute_dtype $COMPUTE_DTYPE"
II_FLAG="";            [ -n "$INV_IMPL" ]      && II_FLAG="--inv_impl $INV_IMPL"
WANDB_PROJECT_FLAG=""; [ -n "$WANDB_PROJECT" ] && WANDB_PROJECT_FLAG="--wandb_project $WANDB_PROJECT"
WARM_START_FLAG="";    [ "$WARM_START" = "1" ] && WARM_START_FLAG="--warm_start"
INIT_FROM_FLAG="";     [ -n "$INIT_FROM" ]     && INIT_FROM_FLAG="--init_from $INIT_FROM"
# array, not a bare $VAR expansion: the JSON payload may contain spaces/braces
# that must survive as ONE argv token to --anchor_overrides.
ANCHOR_FLAG=(); [ -n "$ANCHOR_OVERRIDES" ] && ANCHOR_FLAG=(--anchor_overrides "$ANCHOR_OVERRIDES")

# ---- dual-basis winner arch + end-of-training pooled eval (Phase B) ----------
# DUAL=1 adds --dual_basis (tune-rect winner: Hadamard frame + star tokens);
# NONINV_HIDDEN="4 8" sets the noninv MLP widths; FINAL_EVAL_ROUNDS=K pools K
# fresh sampling rounds through the trained kernels for the final observables.
DUAL_FLAG="";  [ "${DUAL:-0}" = "1" ]         && DUAL_FLAG="--dual_basis"
NH_FLAG="";    [ -n "${NONINV_HIDDEN:-}" ]    && NH_FLAG="--noninv_hidden $NONINV_HIDDEN"
FER_FLAG="";   [ -n "${FINAL_EVAL_ROUNDS:-}" ] && FER_FLAG="--final_eval_rounds $FINAL_EVAL_ROUNDS"
# TOPO=0 skips the inline O_FM/S2 block (slow 16-chain eval clone; the fm.py /
# renyi.py extractors are authoritative). Pooled-eval runs skip it regardless.
TOPO_FLAG="";  [ "${TOPO:-1}" = "0" ]         && TOPO_FLAG="--no_topological"
# NB: the default must NOT live inside ${:-} — bash closes the expansion at the
# FIRST '}', so a brace-bearing default gets half-appended onto a supplied value.
if [ -z "${NAME_TEMPLATE:-}" ]; then
  NAME_TEMPLATE='bosonic_gridinv_L{L}_hx{hx}_hz{hz}'
fi

mkdir -p "$OUT_DIR"

# ---- auto-resubmit just before the wall limit (opt-in) -----------------------
# Requeue the SAME chunk index (so the field-value math is unchanged); the on-disk
# checkpoints + skip-if-{name}.json in tc3d.sweep are the hand-off. `env` (with a
# bash array) is used so multi-word values like INV="2 2 2" survive as one assignment.
RESUB_COUNT="${RESUB_COUNT:-0}"
MAX_RESUBMITS="${MAX_RESUBMITS:-8}"
WALLTIME="${WALLTIME:-}"
requeue() {
  if [ "${AUTO_RESUBMIT:-0}" = "1" ] && [ "$RESUB_COUNT" -lt "$MAX_RESUBMITS" ]; then
    echo "[batch] wall limit near — resubmitting chunk ${SLURM_ARRAY_TASK_ID} (resume #$((RESUB_COUNT+1)))"
    local tflag=""; [ -n "$WALLTIME" ] && tflag="--time=$WALLTIME"
    local -a E=(
      RESUB_COUNT="$((RESUB_COUNT+1))" REPO="$REPO" SWEEP="$SWEEP" CHUNK_POINTS="$CHUNK_POINTS" L="$L"
      BC="$BC" DT="$DT" LR_MIN="$LR_MIN" DIAG_SHIFT="$DIAG_SHIFT"
      NONINV="$NONINV" N_NONINV="$N_NONINV" INV="$INV" KERNEL="$KERNEL"
      N_ITER="$N_ITER" N_SAMPLES="$N_SAMPLES" N_CHAINS="$N_CHAINS" N_SWEEPS="$N_SWEEPS"
      QGT="$QGT" CKPT_EVERY="$CKPT_EVERY" SNAPSHOT_EVERY="$SNAPSHOT_EVERY"
      CHUNK="$CHUNK" OUT_DIR="$OUT_DIR"
      AUTO_RESUBMIT=1 MAX_RESUBMITS="$MAX_RESUBMITS" WALLTIME="$WALLTIME"
      DUAL="${DUAL:-0}" NONINV_HIDDEN="${NONINV_HIDDEN:-}"
      FINAL_EVAL_ROUNDS="${FINAL_EVAL_ROUNDS:-}" NAME_TEMPLATE="$NAME_TEMPLATE"
      FIELD_VALUES="${FIELD_VALUES:-}" TOPO="${TOPO:-1}"
      HY="$HY" QGT_SOLVER="$QGT_SOLVER" WANDB_PROJECT="$WANDB_PROJECT"
      COMPUTE_DTYPE="$COMPUTE_DTYPE" INV_IMPL="$INV_IMPL" TC3D_DEFAULTS_FILE=/dev/null   # requeue keeps its knobs (no late binding mid-chain)
      EXTRA_ARGS="$EXTRA_ARGS" WARM_START="$WARM_START"
      ANCHOR_OVERRIDES="$ANCHOR_OVERRIDES" INIT_FROM="$INIT_FROM"
      WANDB_GROUP="${WANDB_GROUP:-}"
    )
    if [ "$SWEEP" = "hz" ]; then
      E+=(HX="$FIXED" HZ_MIN="$GMIN" HZ_MAX="$GMAX" HZ_N="$GN")
    else
      E+=(HZ="$FIXED" HX_MIN="$GMIN" HX_MAX="$GMAX" HX_N="$GN")
    fi
    # SLURM_JOB_NAME is Slurm-provided (always set inside a real job); the
    # fallback only matters for local/harness testing of this trap.
    env "${E[@]}" sbatch $tflag --job-name="${SLURM_JOB_NAME:-tc-batch}" \
      --array="${SLURM_ARRAY_TASK_ID}" "$0"
  fi
  exit 0
}
trap requeue USR1

echo "[batch] chunk ${SLURM_ARRAY_TASK_ID}: SWEEP=$SWEEP L=$L $BC fixed=$FIXED "\
"values=[$VALUES] hy=$HY diag_shift=$DIAG_SHIFT n_iter=$N_ITER (resume #$RESUB_COUNT) -> $OUT_DIR"
echo "[batch] warm_start=$WARM_START anchor_overrides=${ANCHOR_OVERRIDES:-<none>} "\
"init_from=${INIT_FROM:-<none>} qgt_solver=${QGT_SOLVER:-<default>} compute_dtype=${COMPUTE_DTYPE:-<default>} inv_impl=${INV_IMPL:-<default>} wandb_project=${WANDB_PROJECT:-<default>} "\
"wandb_group=$WANDB_GROUP"

# `srun ... &` + `wait` so the USR1 trap fires promptly (a foreground srun would
# swallow the signal until it returns). One long-lived process loops over $VALUES.
srun -n 1 python -u -m tc3d.sweep \
  --field "$SWEEP" --field_values $VALUES --fixed_field_value "$FIXED" --hy "$HY" \
  --name_template "$NAME_TEMPLATE" \
  --L "$L" --bc "$BC" --model bosonic --arch ToricCNN_gridinv $DUAL_FLAG \
  --noninv_channels "$NONINV" --n_noninv "$N_NONINV" $NH_FLAG \
  --inv_hidden $INV $KERNEL_FLAG \
  --dt "$DT" --lr_min "$LR_MIN" --diag_shift "$DIAG_SHIFT" --qgt "$QGT" $QGT_SOLVER_FLAG \
  $CD_FLAG $II_FLAG \
  --n_iter "$N_ITER" --n_samples "$N_SAMPLES" --n_chains "$N_CHAINS" \
  --n_sweeps "$N_SWEEPS" $CHUNK_FLAG $FER_FLAG $TOPO_FLAG \
  --checkpoint_every "$CKPT_EVERY" $SNAP_FLAG \
  $WARM_START_FLAG "${ANCHOR_FLAG[@]}" $INIT_FROM_FLAG \
  --out_dir "$OUT_DIR" $WANDB_PROJECT_FLAG \
  --wandb_group "$WANDB_GROUP" --wandb_offline $EXTRA_ARGS &
wait
