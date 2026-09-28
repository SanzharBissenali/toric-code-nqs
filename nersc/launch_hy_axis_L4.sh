#!/bin/bash
# hy-AXIS L=4 campaign (feat/hy-axis-L4): the pure-h_y line, hx=hz=0,
# hy 0.0..1.5 in 0.1 steps, per notes/transition_mapping_recipes.md §C plus
# the hy-cuts 2026-08-29 lessons. Run ON PERLMUTTER from the repo root:
#   bash nersc/launch_hy_axis_L4.sh
#
# Stage 1 (user-approved 2026-08-29, gated on the L=2 OBC Stage-0 cert):
#   16 cold 500-step points into $BASE_OUT/cold/L4 (+ two −hy TR pairs, §C).
#   No QMC anywhere on this line — the only exact anchor is E0(h=0) = −172
#   (L=4 OBC, −(L³+3(L−1)²L)); every converged E(hy>0) must sit BELOW it
#   (concavity), and EXACT_E0 stamps delta into each final JSON.
#   hy=0.0 builds a REAL net by default → --force_complex for like-for-like
#   params (and so it can seed a warm chain; real→complex init_from breaks).
# Stage 2 (CHAINS=1; separate approval): warm-chain control lanes — up-chain
#   seeded from the cold hy=0.0 run, dn-chain from the cold hy=1.5 run
#   (§B links: dt 0.005, ds 3e-3, 200 steps, afterok) — the variational-energy
#   referee for "did cold converge" and the branch data near the expected
#   first-order feature at hy≈1.0.
# S2 comes from the snapshots afterwards (eval_snapshots.py --topological);
# nothing here computes O_FM/S2 in-train (final_eval_rounds>1 skips it).
set -euo pipefail
cd "$(dirname "$0")/.."

BASE_OUT="${BASE_OUT:-$PSCRATCH/tc_nqs/hy_axis}"
MANIFEST="${MANIFEST:-$BASE_OUT/manifest_$(date +%Y%m%d_%H%M%S).txt}"
mkdir -p "$BASE_OUT"

HYS="${HYS:-0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5}"
TR_HYS="${TR_HYS:--0.4 -1.0}"      # §C time-reversal pairs (E equal, ⟨σy⟩ antisymmetric)
CHAINS="${CHAINS:-0}"              # 1 -> also submit the warm-chain control lanes
UP_LINKS="0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5"
DN_LINKS="1.4 1.3 1.2 1.1 1.0 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1"

E0_ANCHOR="-172"                   # exact h=0 L=4 OBC energy; the line's one anchor

ARCH_ENV=(DUAL=1 NONINV_HIDDEN="4 8" INV="8 8" KERNEL=3 L=4 BC=OBC POST_S2_EVAL=1)
SNAP_ARGS="--snapshot_every 50 --final_eval_rounds 8"
STEM="gridinv_dual_L4_OBC_hx0.0_hz0.0"   # wrapper NAME stem at this (hx,hz)
TAIL="n2x4_nh4-8_inv8-8_k3"

submit() {  # submit <time> [--dep <jid>] <env pairs...>
  local t="$1"; shift
  local dep=""; [ "${1:-}" = "--dep" ] && { dep="--dependency=afterok:$2"; shift 2; }
  env "${ARCH_ENV[@]}" "$@" sbatch --parsable --time="$t" $dep nersc/submit_nqs_gridinv.sh
}

log() { echo "$1  $2" | tee -a "$MANIFEST"; }

# ---- Stage 1: cold lane (§A protocol at every hy; AUTO_RESUBMIT because ----
# ---- nothing past hy=0.4 has ever been timed) ------------------------------
declare -A CJID
for HY in $HYS; do
  extra="$SNAP_ARGS"
  ref_env=()
  if [ "$HY" = "0.0" ]; then
    extra="$SNAP_ARGS --force_complex"
    ref_env=(REF_E="$E0_ANCHOR" REF_SIG=0.01)   # exact anchor streams dE_ref
  fi
  jid=$(submit 03:30:00 HX=0.0 HZ=0.0 HY="$HY" \
    DT=0.02 LR_MIN=0.002 DIAG_SHIFT="${DIAG_SHIFT_COLD:-2e-3}" N_ITER=500 \
    EXACT_E0="$E0_ANCHOR" "${ref_env[@]}" EXTRA_ARGS="$extra" \
    AUTO_RESUBMIT=1 WALLTIME=03:30:00 OUT_DIR="$BASE_OUT/cold/L4")
  CJID[$HY]=$jid
  log "$jid" "cold hy=$HY 500 steps"
done

# ---- Stage 1: TR pairs (cold 300 steps, same dir; names carry _hy-X so ----
# ---- nothing clobbers; the extraction --hy filter separates them) ----------
for HY in $TR_HYS; do
  jid=$(submit 02:30:00 HX=0.0 HZ=0.0 HY="$HY" \
    DT=0.02 LR_MIN=0.002 DIAG_SHIFT="${DIAG_SHIFT_COLD:-2e-3}" N_ITER=300 \
    EXACT_E0="$E0_ANCHOR" EXTRA_ARGS="$SNAP_ARGS" \
    AUTO_RESUBMIT=1 WALLTIME=02:30:00 OUT_DIR="$BASE_OUT/cold/L4")
  log "$jid" "TR pair hy=$HY cold300"
done

[ "$CHAINS" = "1" ] || { echo "manifest: $MANIFEST (CHAINS=0 — control lanes not submitted)"; exit 0; }

# ---- Stage 2: warm-chain control lanes (§B links) --------------------------
chain() {  # chain <lane> <seed_hy> <seed_jid> <seed_dir> <links...>
  local lane="$1" prev_hy="$2" prev_jid="$3" prev_dir="$4"; shift 4
  local out="$BASE_OUT/$lane/L4"
  local prev_tag=""; [ "$prev_hy" != "0.0" ] && prev_tag="_hy${prev_hy}"
  local prev_name="${STEM}${prev_tag}_${TAIL}"
  for HY in "$@"; do
    local tag=""; [ "$HY" != "0.0" ] && tag="_hy${HY}"
    local name="${STEM}${tag}_${TAIL}_${lane}"
    local jid
    jid=$(submit 02:30:00 --dep "$prev_jid" HX=0.0 HZ=0.0 HY="$HY" \
      DT=0.005 LR_MIN=0.0005 DIAG_SHIFT=3e-3 N_ITER=200 \
      EXACT_E0="$E0_ANCHOR" NAME="$name" \
      EXTRA_ARGS="$SNAP_ARGS --init_from $prev_dir/$prev_name" \
      OUT_DIR="$out")
    log "$jid" "$lane hy=$HY link (from $prev_hy)"
    prev_jid=$jid; prev_name=$name; prev_dir=$out
  done
}
chain chain_up 0.0 "${CJID[0.0]}" "$BASE_OUT/cold/L4" $UP_LINKS
chain chain_dn 1.5 "${CJID[1.5]}" "$BASE_OUT/cold/L4" $DN_LINKS

echo "manifest: $MANIFEST"
