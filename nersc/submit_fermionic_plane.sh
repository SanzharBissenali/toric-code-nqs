#!/bin/bash
#SBATCH --job-name=tc-fplane
#SBATCH --account=m5340_g
#SBATCH --qos=shared
#SBATCH --constraint=gpu
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=03:00:00
#SBATCH --array=0-15%4
#SBATCH --output=%x-%A_%a.out
#
# fTC L=2 OBC (hx,hz)-PLANE campaign: HX_LIST x HZ_LIST = {0.0,0.2,0.5,1.0}^2,
# 16 points, one array task per (hx,hz) (max 4 concurrent). Task i -> (hx,hz) =
# (HX_LIST[i/4], HZ_LIST[i%4]) (bash integer arithmetic, row-major in hx).
#
# Each task runs up to 4 arms sequentially (same recipe as
# submit_fermionic_hx_ladder.sh's T2/T3/T5 tiers, generalized off the hz=0
# magnetic line onto the full plane, plus one new complex-trunk variant):
#   asymm    ToricCNN_gridinv (complex), no head, cold, opened guard
#            (canonical widths: kernel 2, noninv_hidden 4 8, inv_hidden 8 8)
#   anaC_k0  gridinv + frozen analytic-C head + --init_from the prefit,
#            flux_penalty=0, stock guard
#   pt2sf    gridinv, cold, stock guard, formulation-B sign-framed H~ = S H S
#            with S from the committed pt2 lookup table (real trunk)
#   pt2sfc   same as pt2sf but with a COMPLEX trunk (--dtype complex overrides
#            with_defaults' sign_frame-implied real dtype; H~ = S H S is real
#            either way -- this is a redundant but valid phase-capacity probe,
#            tests/test_sign_frame.py::test_explicit_dtype_forces_complex_trunk_
#            under_sign_frame is the unit witness for the override)
#
# ARMS (env knob, default all four at hy=0) selects which arms to (re)run this
# submission, space-separated tags from {asymm anaC_k0 pt2sf pt2sfc}, e.g.
# to add the new complex-trunk arm to an already-complete plane without
# touching the other three:
#   ARMS="pt2sfc" sbatch nersc/submit_fermionic_plane.sh
# Each arm's own idempotent skip (snapshots.json / final json already present)
# still applies underneath ARMS, so re-submitting with a wider ARMS is safe.
#
# HY (env knob, default 0.0) reruns this SAME (hx,hz) plane at a fixed nonzero
# h_y (tc3d.builders now allows --sign_frame + h_y != 0, requiring a complex
# trunk -- see tc3d/sign_frame.py). At HY != 0.0:
#   - OUT defaults to $PSCRATCH/tc_nqs/fermionic_plane_L2_hy${HY} (separate
#     from the hy=0 plane's output dir)
#   - COMMON/the ED referee both carry --hy $HY; run names carry _hy${HY}
#     before _k2 (gridinv_fermionic_L2_OBC_hx{hx}_hz{hz}_hy{hy}_k2_{arm})
#   - the hz=0 ladder import (below) is SKIPPED -- no hy!=0 ladder runs exist
#     to import from
#   - ARMS defaults to "asymm anaC_k0 pt2sfc" (pt2sf, the sign-framed REAL
#     trunk control, is dropped by default -- see the pt2sf run line for why)
#   HY=0.2 sbatch nersc/submit_fermionic_plane.sh
#
# Task 0 additionally:
#   (a) generates the dense-ED referee over the FULL 4x4 plane (16 points),
#       gated on E0(0,0) == -14 exactly (hy=0) or E0(hx=0,hz=0,hy) < -14
#       (concavity bound, hy!=0 -- -14 is only exact at hy=0);
#   (b) reuses fermionic_hx_ladder's prefit checkpoint if present on scratch,
#       else regenerates + certifies it (falls back to the committed artifact
#       on regen failure);
#   (c) (only when ARMS includes pt2sf/pt2sfc) checks the committed pt2
#       sign-table has 4096 (=2^12, L=2 OBC) entries;
#   (d) IMPORTS the hz=0 row already computed by submit_fermionic_hx_ladder.sh
#       (fermionic_hx_ladder's hx grid {0.1,0.2,0.3,0.5,0.7,1.0} overlaps this
#       plane's {0.0,0.2,0.5,1.0} at hx in {0.2, 0.5, 1.0}): for each such hx
#       and each of arms {asymm, anaC_k0, pt2sf} (pt2sf may not exist there
#       yet -- guarded by an existence check), copies the run's
#       {json,curve.json,snapshots.json,mpack,ckpt.mpack} into this OUT under
#       the IDENTICAL run name (both scripts name runs
#       gridinv_fermionic_L2_OBC_hx{hx}_hz{hz}_k2_{arm}, so hz=0.0 collides on
#       purpose), plus the matching exact_diag_fermionic_L2_OBC_hx{hx}_hz0.0.json
#       + ed_vectors npz, gated against the new plane referee's E0 at the same
#       point (delta < 1e-10). Imported runs are then picked up by each arm's
#       normal skip-if-done check below -- they are NOT recomputed.
# Other tasks wait for these gates (see the leader/follower block below)
# rather than racing each other on the same shared output files.
#
# Submit   (from this worktree, on the cluster):
#   sbatch nersc/submit_fermionic_plane.sh
#   HY=0.2 sbatch nersc/submit_fermionic_plane.sh          # hy!=0 plane, see HY above
# Check:
#   squeue -u sanzharb
# Pull results back (skip the large .mpack weights):
#   rsync -av --exclude '*.mpack' perlmutter:/pscratch/sd/s/sanzharb/tc_nqs/fermionic_plane_L2/ results/fermionic_plane_L2/
#   rsync -av --exclude '*.mpack' perlmutter:/pscratch/sd/s/sanzharb/tc_nqs/fermionic_plane_L2_hy0.2/ results/fermionic_plane_L2_hy0.2/
set -u
module load conda
conda activate tc-nqs
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$PSCRATCH/tc_nqs/jax_cache}"
mkdir -p "$JAX_COMPILATION_CACHE_DIR"

REPO="${REPO:-$HOME/toric-code-nqs-fsign}"
# The conda env's tc3d is pip-installed editable against ANOTHER clone; shadow
# it so every invocation (scripts, python -m) imports THIS checkout.
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

# HY (see the header comment) -- numeric zero-check via awk so "0", "0.0",
# "-0.0" etc. all count as the hy=0 (unchanged) branch regardless of how the
# caller spelled it.
HY="${HY:-0.0}"
HY_NONZERO=$(awk -v v="$HY" 'BEGIN{print (v+0!=0)}')
if [ "$HY_NONZERO" = "1" ]; then
  scontrol update JobId="${SLURM_JOB_ID:-}" JobName="tc-fplane-hy${HY}" 2>/dev/null || true
fi

OUT="${OUT:-}"
if [ -z "$OUT" ]; then
  if [ "$HY_NONZERO" = "1" ]; then
    OUT="$PSCRATCH/tc_nqs/fermionic_plane_L2_hy${HY}"
  else
    OUT="$PSCRATCH/tc_nqs/fermionic_plane_L2"
  fi
fi
EDREF="$REPO/results/fermionic_obc_L2"          # committed ED referee data (regen fallback)
GATE0="$REPO/results/fermionic_gate0"           # committed sign-table lookups
SIGNTABLE_PT2="$GATE0/sign_table_pt2_2x2x2_OBC.npy"
LADDER_OUT="$PSCRATCH/tc_nqs/fermionic_hx_ladder"   # completed magnetic-line ladder (hz=0 row)
mkdir -p "$OUT/ed_vectors"
cd "$REPO"

HX_LIST=(0.0 0.2 0.5 1.0)
HZ_LIST=(0.0 0.2 0.5 1.0)
NHZ=${#HZ_LIST[@]}
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
HX_IDX=$(( TASK_ID / NHZ ))
HZ_IDX=$(( TASK_ID % NHZ ))
HX="${HX_LIST[$HX_IDX]}"
HZ="${HZ_LIST[$HZ_IDX]}"

# Which arms to (re)run this submission -- default all four at hy=0, or the
# three complex-trunk arms at hy!=0 (pt2sf dropped by default there -- see its
# run line below); see the ARMS knob comment above. An explicit ARMS always
# wins over either default. Each arm's own skip-if-done in run() also
# protects against accidental re-launch when ARMS is widened later.
if [ "$HY_NONZERO" = "1" ]; then
  ARMS="${ARMS:-asymm anaC_k0 pt2sfc}"
else
  ARMS="${ARMS:-asymm anaC_k0 pt2sf pt2sfc}"
fi
has_arm () { [[ " $ARMS " == *" $1 "* ]]; }

PREFIT="$OUT/prefit_anaC_k2_L2_OBC"
ED_PLANE="$OUT/ed_L2_OBC_plane.json"
LEADER_DONE="$OUT/.gates_done"
LEADER_FAILED="$OUT/.gates_failed"
# Overlap of fermionic_hx_ladder's hx grid {0.1,0.2,0.3,0.5,0.7,1.0} with this
# plane's HX_LIST -- the only hx values whose hz=0 row can be imported verbatim.
IMPORT_HX_LIST=(0.2 0.5 1.0)
IMPORT_ARMS=(asymm anaC_k0 pt2sf)

if [ "$TASK_ID" = "0" ]; then
  echo "== [leader] task 0: ED referee + prefit + sign-table + hz=0 import gates =="
  (
    set -e
    if [ ! -f "$ED_PLANE" ]; then
      echo "== ED referee: dense OBC GS over the (hx,hz) plane (hy=$HY) =="
      HY_FLAG=()
      [ "$HY_NONZERO" = "1" ] && HY_FLAG=(--hy "$HY")
      python analysis/scripts/ed_electric_line.py --bc OBC \
          --hx "${HX_LIST[@]}" --hz "${HZ_LIST[@]}" "${HY_FLAG[@]}" --out_dir "$OUT"
      cp "$OUT/ed_L2_OBC_rect.json" "$ED_PLANE"
    else
      echo "== ED referee already present -- skip =="
    fi

    if [ "$HY_NONZERO" = "1" ]; then
      # h=0 is only exactly -14 AT hy=0.0 -- at hy!=0 fall back to the
      # concavity bound (any field lowers the ground energy: CLAUDE.md's
      # hy!=0 validation ladder, "E(h_y) < E(h_y=0) at the same hx,hz").
      echo "== gate: (hx=0,hz=0,hy=$HY) E0 < -14 (concavity bound; tol 1e-9) =="
      python3 - "$ED_PLANE" <<'PYEOF'
import json, sys
with open(sys.argv[1]) as f:
    d = json.load(f)
for p in d["points"]:
    if abs(p["hx"]) < 1e-9 and abs(p["hz"]) < 1e-9:
        E0 = p["E0"]
        print(f"[ED gate] (hx=0,hz=0,hy={p['hy']}) E0={E0:.12f}  (h=0 anchor -14)")
        sys.exit(0 if E0 < -14.0 - 1e-9 else 1)
sys.exit("[ED gate] no (hx=0,hz=0) point in the plane referee")
PYEOF
    else
      echo "== gate: (hx=0,hz=0) E0 == -14 exactly (tol 1e-10) =="
      python3 - "$ED_PLANE" <<'PYEOF'
import json, sys
with open(sys.argv[1]) as f:
    d = json.load(f)
for p in d["points"]:
    if abs(p["hx"]) < 1e-9 and abs(p["hz"]) < 1e-9:
        E0 = p["E0"]
        print(f"[ED gate] (0,0) E0={E0:.12f}")
        sys.exit(0 if abs(E0 - (-14.0)) < 1e-10 else 1)
sys.exit("[ED gate] no (0,0) point in the plane referee")
PYEOF
    fi

    if [ -f "$PREFIT.mpack" ]; then
      echo "== prefit already present -- skip =="
    elif [ -f "$LADDER_OUT/prefit_anaC_k2_L2_OBC.mpack" ]; then
      cp "$LADDER_OUT/prefit_anaC_k2_L2_OBC.mpack" "$PREFIT.mpack"
      echo "prefit reused from fermionic_hx_ladder"
    elif python analysis/scripts/prefit_phase_head.py --L 2 --bc OBC --kernel 2 \
         --analytic_C --frozen --seed 0 --cert 2000 --save "$PREFIT"; then
      echo "prefit regenerated + certified"
    else
      echo "WARNING: prefit regen failed -- falling back to committed artifact"
      cp "$EDREF/prefit_anaC_k2_L2_OBC.mpack" "$PREFIT.mpack"
    fi

    if has_arm pt2sf || has_arm pt2sfc; then
      echo "== gate: pt2 sign-table has 4096 (=2^12) entries =="
      python3 - "$SIGNTABLE_PT2" <<'PYEOF'
import sys
import numpy as np
a = np.load(sys.argv[1])
assert a.size == 4096, f"{sys.argv[1]}: expected 4096 entries, got {a.size}"
print(f"[sign-table gate] {sys.argv[1]}: {a.size} entries OK")
PYEOF
    fi

    if [ "$HY_NONZERO" = "1" ]; then
      echo "== hy=$HY != 0 -- no hy!=0 ladder runs exist to import; skipping hz=0 import =="
    else
      echo "== import hz=0 ladder artifacts: hx in ${IMPORT_HX_LIST[*]}, arms ${IMPORT_ARMS[*]} =="
      for hx in "${IMPORT_HX_LIST[@]}"; do
        ed_src="$LADDER_OUT/exact_diag_fermionic_L2_OBC_hx${hx}_hz0.0.json"
        if [ -f "$ed_src" ]; then
          cp "$ed_src" "$OUT/"
          npz_src="$LADDER_OUT/ed_vectors/exact_diag_fermionic_L2_OBC_hx${hx}_hz0.0.npz"
          [ -f "$npz_src" ] && cp "$npz_src" "$OUT/ed_vectors/"
          python3 - "$ed_src" "$ED_PLANE" "$hx" <<'PYEOF'
import json, sys
ed_src, ed_plane, hx = sys.argv[1], sys.argv[2], float(sys.argv[3])
with open(ed_src) as f:
    e_old = json.load(f)["E0"]
with open(ed_plane) as f:
    plane = json.load(f)
e_new = next(p["E0"] for p in plane["points"]
             if abs(p["hx"] - hx) < 1e-9 and abs(p["hz"]) < 1e-9)
delta = abs(e_old - e_new)
print(f"[ED import gate] hx={hx} hz=0.0: ladder E0={e_old:.12f}  plane E0={e_new:.12f}  delta={delta:.3e}")
sys.exit(0 if delta < 1e-10 else 1)
PYEOF
        else
          echo "  hx=$hx: no ladder ED referee point on scratch -- plane referee covers it fresh"
        fi
        for arm in "${IMPORT_ARMS[@]}"; do
          base="$LADDER_OUT/gridinv_fermionic_L2_OBC_hx${hx}_hz0.0_k2_${arm}"
          if [ -f "$base.snapshots.json" ] || [ -f "$base.json" ]; then
            for ext in json curve.json snapshots.json mpack ckpt.mpack; do
              [ -f "$base.$ext" ] && cp "$base.$ext" "$OUT/"
            done
            echo "  imported hx=$hx arm=$arm from fermionic_hx_ladder"
          else
            echo "  hx=$hx arm=$arm: not on scratch yet -- will run fresh"
          fi
        done
      done
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

# hy!=0: --hy $HY on every arm's Hamiltonian, and _hy${HY} in every run name
# (before _k2) -- the same "_hy{hy}" convention ed_electric_line.py/
# eval_snapshots.py's _ed_tag use for the ED referee filenames (matching cfg
# values, not the run-name string, are what actually drive that lookup; the
# tag here is for human/log traceability and to avoid same-named runs across
# the hy=0 and hy!=0 OUT dirs if ever pointed at the same OUT).
HY_FLAG=""
HYTAG=""
if [ "$HY_NONZERO" = "1" ]; then
  HY_FLAG="--hy $HY"
  HYTAG="_hy${HY}"
fi

COMMON="--L 2 --bc OBC --model fermionic --hx $HX --hz $HZ $HY_FLAG \
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

has_arm asymm && run gridinv_fermionic_L2_OBC_hx${HX}_hz${HZ}${HYTAG}_k2_asymm \
    $GRIDINV_ARCH $GUARD_OPEN

has_arm anaC_k0 && run gridinv_fermionic_L2_OBC_hx${HX}_hz${HZ}${HYTAG}_k2_anaC_k0 \
    $GRIDINV_ARCH --phase_head_frozen --flux_penalty 0 --init_from "$PREFIT"

# pt2sf (the sign_frame arm relying on with_defaults' REAL trunk default) is
# dropped from the default ARMS at hy!=0 (see the ARMS block above), but the
# run line stays so an explicit ARMS="pt2sf ..." can still force it: h_y!=0
# makes with_defaults default dtype=complex REGARDLESS of sign_frame (the
# `signfull` rule in tc3d.builders.with_defaults), so without an explicit
# --dtype float64 this "real trunk" arm silently trains an auto-promoted
# COMPLEX trunk -- identical to pt2sfc below, just under a confusing name.
# Forcing the real trunk explicitly (--dtype float64) is no longer possible
# at all: tc3d.builders now REJECTS sign_frame + hy!=0 + --dtype float64
# outright, because a real (positive) trunk cannot represent the residual
# complex phase left after S frames out the +-1 part of psi -- its overlap
# ceiling is exactly the head's own phase-optimized ceiling (S alone vs the
# ED ground state), already quantified per-point by gate 0's
# head_phase_ceiling (analysis/scripts/eval_snapshots.py). So there is no
# non-degenerate real-trunk control to run at hy!=0.
has_arm pt2sf && run gridinv_fermionic_L2_OBC_hx${HX}_hz${HZ}${HYTAG}_k2_pt2sf \
    $GRIDINV_ARCH --sign_frame table --sign_table "$SIGNTABLE_PT2"

has_arm pt2sfc && run gridinv_fermionic_L2_OBC_hx${HX}_hz${HZ}${HYTAG}_k2_pt2sfc \
    $GRIDINV_ARCH --sign_frame table --sign_table "$SIGNTABLE_PT2" --dtype complex

echo "== TASK $TASK_ID (hx=$HX, hz=$HZ, hy=$HY) COMPLETE  $(date +%H:%M:%S) =="
