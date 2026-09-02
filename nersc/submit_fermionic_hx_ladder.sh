#!/bin/bash
#SBATCH --job-name=tc-fhxladder
#SBATCH --account=m5340_g
#SBATCH --qos=shared
#SBATCH --constraint=gpu
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=03:00:00
#SBATCH --array=0-5%4
#SBATCH --output=%x-%A_%a.out
#
# fTC L=2 OBC architecture ladder along the magnetic line hz=0:
#   hx in {0.1, 0.2, 0.3, 0.5, 0.7, 1.0}, one array task per hx (max 4 concurrent).
# Each task runs up to 7 tiers sequentially, matching submit_fermionic_obc_bench.sh's
# recipe (same widths/dt/diag_shift/samples) at n_iter=300:
#   T1 plain    GeoCNN (complex), no head, cold, opened guard (launch_fermionic_ladder.sh
#               cold-tier recipe: canonical width "4 4 4", --spike_factor/--max_rollbacks)
#   T2 asymm    ToricCNN_gridinv, no head, cold, opened guard, canonical widths
#               (kernel 2, noninv_hidden 4 8, inv_hidden 8 8)
#   T3 anaC_k0  gridinv + frozen analytic-C head + --init_from the prefit, flux_penalty=0,
#               stock guard -- identical to the bench's phf_fp0 tier
#   T4 anaC_k6  same, flux_penalty=6.0 -- identical to the bench's phf_fp6 tier
#   T5 pt2sf    gridinv, cold, stock guard, formulation-B sign-framed H~ = S H S with
#               S from the committed pt2 lookup table (real trunk, no head, no penalty)
#   T6 votesf   same, S from the committed vote lookup table
#   T7 anaCsf   same, S = the analytic fTC h=0 sign (--sign_frame anaC) -- the
#               formulation-B control of the frozen-head T3/T4 tiers
#
# TIERS (env knob, default all seven) selects which tiers to run, space-separated
# tags from {plain asymm anaC_k0 anaC_k6 pt2sf votesf anaCsf}; e.g. to (re)run only
# the new sign-framed tiers without touching the existing four:
#   TIERS="pt2sf votesf anaCsf" sbatch nersc/submit_fermionic_hx_ladder.sh
# Each tier's own idempotent skip (snapshots.json / final json already present)
# still applies underneath TIERS, so re-submitting with a wider TIERS is safe.
#
# Task 0 additionally generates the dense-ED referee (all 6 hx at hz=0) and the
# prefit checkpoint once, gated against the committed L=2 OBC referee, plus (only
# when TIERS includes pt2sf/votesf) a cheap sanity check that the committed
# sign-table .npy files exist with 4096 (=2^12, L=2 OBC) entries; other tasks
# wait for these gates (see the leader/follower block below) rather than racing
# each other on the same shared output files.
#
# Submit   (from this worktree, on the cluster):
#   sbatch nersc/submit_fermionic_hx_ladder.sh
# Check:
#   squeue -u sanzharb
# Pull results back:
#   rsync -av perlmutter:/pscratch/sd/s/sanzharb/tc_nqs/fermionic_hx_ladder/ results/fermionic_hx_ladder/
set -u
module load conda
conda activate tc-nqs
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$PSCRATCH/tc_nqs/jax_cache}"
mkdir -p "$JAX_COMPILATION_CACHE_DIR"

REPO="${REPO:-$HOME/toric-code-nqs-fsign}"
# The conda env's tc3d is pip-installed editable against ANOTHER clone; shadow
# it so every invocation (scripts, python -m) imports THIS checkout.
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
OUT="${OUT:-$PSCRATCH/tc_nqs/fermionic_hx_ladder}"
EDREF="$REPO/results/fermionic_obc_L2"          # committed ED referee data
GATE0="$REPO/results/fermionic_gate0"           # committed sign-table lookups
SIGNTABLE_PT2="$GATE0/sign_table_pt2_2x2x2_OBC.npy"
SIGNTABLE_VOTE="$GATE0/sign_table_vote_2x2x2_OBC.npy"
mkdir -p "$OUT/ed_vectors"
cd "$REPO"

HX_LIST=(0.1 0.2 0.3 0.5 0.7 1.0)
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
HX="${HX_LIST[$TASK_ID]}"

# Which tiers to (re)run this submission -- default all seven; the coordinator
# can restrict this (e.g. TIERS="pt2sf votesf anaCsf") to add tiers to an
# already-complete ladder without disturbing the existing runs (each tier's
# own skip-if-done in run() also protects against accidental re-launch).
TIERS="${TIERS:-plain asymm anaC_k0 anaC_k6 pt2sf votesf anaCsf}"
has_tier () { [[ " $TIERS " == *" $1 "* ]]; }

PREFIT="$OUT/prefit_anaC_k2_L2_OBC"
ED_HXLINE="$OUT/ed_L2_OBC_hxline.json"
LEADER_DONE="$OUT/.gates_done"
LEADER_FAILED="$OUT/.gates_failed"

if [ "$TASK_ID" = "0" ]; then
  echo "== [leader] task 0: ED referee + prefit gates =="
  (
    set -e
    if [ ! -f "$ED_HXLINE" ]; then
      echo "== ED referee: dense OBC GS, hz=0, hx in ${HX_LIST[*]} =="
      python analysis/scripts/ed_electric_line.py --bc OBC \
          --hx "${HX_LIST[@]}" --hz 0.0 --out_dir "$OUT"
      cp "$OUT/ed_L2_OBC_rect.json" "$ED_HXLINE"
    else
      echo "== ED referee already present -- skip =="
    fi

    echo "== gate: hx=0.2 hz=0.0 E0 vs committed $EDREF/ed_L2_OBC_rect.json (tol 1e-10) =="
    python3 - "$ED_HXLINE" "$EDREF/ed_L2_OBC_rect.json" <<'PYEOF'
import json, sys
def e0(path, hx, hz):
    with open(path) as f:
        d = json.load(f)
    for p in d["points"]:
        if abs(p["hx"] - hx) < 1e-9 and abs(p["hz"] - hz) < 1e-9:
            return p["E0"]
    sys.exit(f"[ED gate] no point hx={hx} hz={hz} in {path}")
new, ref = e0(sys.argv[1], 0.2, 0.0), e0(sys.argv[2], 0.2, 0.0)
delta = abs(new - ref)
print(f"[ED gate] hx=0.2 hz=0.0: hxline E0={new:.12f}  committed E0={ref:.12f}  delta={delta:.3e}")
sys.exit(0 if delta < 1e-10 else 1)
PYEOF

    if [ -f "$PREFIT.mpack" ]; then
      echo "== prefit already present -- skip =="
    elif python analysis/scripts/prefit_phase_head.py --L 2 --bc OBC --kernel 2 \
         --analytic_C --frozen --seed 0 --cert 2000 --save "$PREFIT"; then
      echo "prefit regenerated + certified"
    else
      echo "WARNING: prefit regen failed -- falling back to committed artifact"
      cp "$EDREF/prefit_anaC_k2_L2_OBC.mpack" "$PREFIT.mpack"
    fi

    if has_tier pt2sf || has_tier votesf; then
      echo "== gate: sign-table files exist with 4096 (=2^12) entries =="
      python3 - "$SIGNTABLE_PT2" "$SIGNTABLE_VOTE" <<'PYEOF'
import sys
import numpy as np
for path in sys.argv[1:]:
    a = np.load(path)
    assert a.size == 4096, f"{path}: expected 4096 entries, got {a.size}"
    print(f"[sign-table gate] {path}: {a.size} entries OK")
PYEOF
    fi
  )
  if [ $? -eq 0 ]; then
    touch "$LEADER_DONE"
  else
    touch "$LEADER_FAILED"
    echo "[leader] GATES FAILED"
    exit 1
  fi
else
  echo "== [follower] task $TASK_ID: waiting on task 0's gates =="
  for i in $(seq 1 90); do
    [ -f "$LEADER_DONE" ] && break
    [ -f "$LEADER_FAILED" ] && { echo "[follower] leader gates failed -- aborting"; exit 1; }
    sleep 10
  done
  [ -f "$LEADER_DONE" ] || { echo "[follower] timed out waiting for leader gates"; exit 1; }
fi

COMMON="--L 2 --bc OBC --model fermionic --hx $HX --hz 0.0 \
 --dt 0.02 --lr_min 0.002 --diag_shift 0.001 \
 --n_samples 8192 --n_chains 1024 --chunk_size 2048 \
 --n_iter 300 --snapshot_every 25 --checkpoint_every 50 \
 --seed 0 --no_wandb --out_dir $OUT"
GUARD_OPEN="--spike_factor 1e6 --max_rollbacks 50"
GRIDINV_ARCH="--arch ToricCNN_gridinv --kernel_size 2 --noninv_hidden 4 8 \
 --inv_hidden 8 8 --noninv_channels 4 --n_noninv 2 --chains_up"

run () {
  NAME=$1; shift
  if [ -f "$OUT/$NAME.snapshots.json" ]; then
    echo "== skip $NAME (snapshots.json exists) =="
    return 0
  fi
  if [ -f "$OUT/$NAME.json" ]; then
    echo "== $NAME: final artifacts exist -- eval only =="
  else
    echo "== run $NAME  $(date +%H:%M:%S) =="
    if ! python -m tc3d.train $COMMON --name "$NAME" "$@"; then
      # guard DivergenceError still finalizes on the last sane state; its
      # snapshots + final JSON are a result worth evaluating, not discarding
      echo "RUN $NAME FAILED (see guard lines above)"
      [ -f "$OUT/$NAME.json" ] || return 1
    fi
  fi
  python analysis/scripts/eval_snapshots.py --dir "$OUT" --glob "$NAME.json" \
      --rounds 4 --exact --ed_vectors "$OUT/ed_vectors" \
      || echo "EVAL $NAME FAILED"
}

has_tier plain && run geocnn_fermionic_L2_OBC_hx${HX}_hz0.0_plain \
    --arch GeoCNN --cnn_hidden 4 4 4 $GUARD_OPEN

has_tier asymm && run gridinv_fermionic_L2_OBC_hx${HX}_hz0.0_k2_asymm \
    $GRIDINV_ARCH $GUARD_OPEN

has_tier anaC_k0 && run gridinv_fermionic_L2_OBC_hx${HX}_hz0.0_k2_anaC_k0 \
    $GRIDINV_ARCH --phase_head_frozen --flux_penalty 0 --init_from "$PREFIT"

has_tier anaC_k6 && run gridinv_fermionic_L2_OBC_hx${HX}_hz0.0_k2_anaC_k6 \
    $GRIDINV_ARCH --phase_head_frozen --flux_penalty 6.0 --init_from "$PREFIT"

has_tier pt2sf && run gridinv_fermionic_L2_OBC_hx${HX}_hz0.0_k2_pt2sf \
    $GRIDINV_ARCH --sign_frame table --sign_table "$SIGNTABLE_PT2"

has_tier votesf && run gridinv_fermionic_L2_OBC_hx${HX}_hz0.0_k2_votesf \
    $GRIDINV_ARCH --sign_frame table --sign_table "$SIGNTABLE_VOTE"

has_tier anaCsf && run gridinv_fermionic_L2_OBC_hx${HX}_hz0.0_k2_anaCsf \
    $GRIDINV_ARCH --sign_frame anaC

echo "== TASK $TASK_ID (hx=$HX) COMPLETE  $(date +%H:%M:%S) =="
