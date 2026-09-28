#!/bin/bash
# Sign-full (h_y != 0) L=4 transition campaign — both Phase-B cuts at hy in
# {0.2, 0.4}, per notes/transition_mapping_recipes.md §A/§B/§C. Run ON
# PERLMUTTER from the repo root:  bash nersc/launch_hy_cuts_L4.sh
#
# Coarse first pass ("test the waters", user-approved 2026-08-26):
#   up cut    (2nd order): fixed hx=0.2, hz in {0.1 .18 .22 .26 .3 .4} — cold,
#             500 steps (§A protocol).
#   right cut (1st order): fixed hz=0.1 — two cold anchors (hx=0.6, 1.25) +
#             afterok warm chains through the window at 0.05 steps (§B: never
#             cold inside the window; link = dt 0.005, ds 3e-3, 200 steps).
#   TR pairs  (§C ladder): one -hy run per cut per hy, 300 steps.
# Every run: --snapshot_every 50 --final_eval_rounds 8; REF_E streams the
# same-(hx,hz) hy=0 QMC value = the §C concavity bound (dE_ref<0 expected).
# hy=0.4 gated on the L=2 ED cert + debug smoke (jobs 57620514/57620515).
set -euo pipefail
cd "$(dirname "$0")/.."

HYS="${HYS:-0.2 0.4}"
BASE_OUT="${BASE_OUT:-$PSCRATCH/tc_nqs/hy_cuts}"
MANIFEST="${MANIFEST:-$BASE_OUT/manifest_$(date +%Y%m%d_%H%M%S).txt}"
mkdir -p "$BASE_OUT"

UP_HZ="${UP_HZ:-0.1 0.18 0.22 0.26 0.3 0.4}"   # coarse pass; leftover Phase-B pts via env
UP_ONLY="${UP_ONLY:-0}"                        # 1 -> submit only the up-cut loop
ANCHORS="0.6 1.25"
UP_LINKS="0.65 0.7 0.75 0.8 0.85 0.9 0.95 1.0 1.05 1.1 1.15"   # from hx=0.6 anchor
DN_LINKS="1.2 1.15 1.1 1.05 1.0 0.95 0.9 0.85 0.8"             # from hx=1.25 anchor

# hy=0 QMC references (L=4; highest-beta subset — beta=24 at hx 0.75..0.9)
ref_up()    { case "$1" in 0.1) echo "-173.4296 0.0128";; 0.18) echo "-174.2755 0.0128";;
              0.22) echo "-174.9665 0.0136";; 0.26) echo "-175.8767 0.0187";;
              0.3) echo "-177.5094 0.0294";; 0.4) echo "-185.4002 0.0229";;
              0.15) echo "-173.9005 0.0179";; 0.2) echo "-174.6200 0.0302";;
              0.24) echo "-175.3658 0.0200";; 0.28) echo "-176.5553 0.0332";;
              0.32) echo "-178.7722 0.0290";; 0.34) echo "-180.1635 0.0310";;
              0.36) echo "-181.8274 0.0291";; 0.45) echo "-190.4223 0.0390";;
              0.5) echo "-195.8930 0.0240";; *) echo "";; esac; }
ref_right() { case "$1" in 0.6) echo "-183.7065 0.0294";; 0.65) echo "-186.2562 0.0781";;
              0.75) echo "-193.1814 0.1339";; 0.8) echo "-197.8683 0.0708";;
              0.85) echo "-203.4860 0.0665";; 0.9) echo "-209.6384 0.0607";;
              0.95) echo "-215.8275 0.0369";; 1.0) echo "-222.2934 0.0391";;
              1.05) echo "-228.6304 0.0415";; 1.1) echo "-235.2636 0.0352";;
              1.25) echo "-255.2751 0.0371";; *) echo "";; esac; }

ARCH_ENV=(DUAL=1 NONINV_HIDDEN="4 8" INV="8 8" KERNEL=3 L=4 BC=OBC)
SNAP_ARGS="--snapshot_every 50 --final_eval_rounds 8"

submit() {  # submit <time> <extra sbatch args...> -- <env pairs...>
  local t="$1"; shift
  local dep=""; [ "${1:-}" = "--dep" ] && { dep="--dependency=afterok:$2"; shift 2; }
  env "${ARCH_ENV[@]}" "$@" sbatch --parsable --time="$t" $dep nersc/submit_nqs_gridinv.sh
}

log() { echo "$1  $2" | tee -a "$MANIFEST"; }

for HY in $HYS; do
  # ---- up cut: cold 500-step points (§A) ----------------------------------
  for HZ in $UP_HZ; do
    read -r RE RS <<<"$(ref_up "$HZ")"
    jid=$(submit 05:00:00 HX=0.2 HZ="$HZ" HY="$HY" \
      DT=0.02 LR_MIN=0.002 DIAG_SHIFT=1e-3 N_ITER=500 \
      REF_E="$RE" REF_SIG="$RS" EXTRA_ARGS="$SNAP_ARGS" \
      OUT_DIR="$BASE_OUT/up/hy$HY/L4")
    log "$jid" "up hy=$HY hz=$HZ cold500"
  done
  [ "$UP_ONLY" = "1" ] && continue

  # ---- right cut: cold anchors (§B step 1) --------------------------------
  RIGHT_OUT="$BASE_OUT/right/hy$HY/L4"
  declare -A AJID
  for HX in $ANCHORS; do
    read -r RE RS <<<"$(ref_right "$HX")"
    jid=$(submit 05:00:00 HX="$HX" HZ=0.1 HY="$HY" \
      DT=0.02 LR_MIN=0.002 DIAG_SHIFT=1e-3 N_ITER=500 \
      REF_E="$RE" REF_SIG="$RS" EXTRA_ARGS="$SNAP_ARGS" \
      OUT_DIR="$RIGHT_OUT")
    AJID[$HX]=$jid
    log "$jid" "right hy=$HY hx=$HX anchor cold500"
  done

  # ---- right cut: afterok warm chains (§B step 2) --------------------------
  chain() {  # chain <dirtag> <seed_hx> <seed_jid> <links...>
    local tag="$1" prev_hx="$2" prev_jid="$3"; shift 3
    local prev_name="gridinv_dual_L4_OBC_hx${prev_hx}_hz0.1_hy${HY}_n2x4_nh4-8_inv8-8_k3"
    for HX in "$@"; do
      local name="gridinv_dual_L4_OBC_hx${HX}_hz0.1_hy${HY}_n2x4_nh4-8_inv8-8_k3_${tag}"
      read -r RE RS <<<"$(ref_right "$HX")"
      local jid
      jid=$(submit 02:30:00 --dep "$prev_jid" HX="$HX" HZ=0.1 HY="$HY" \
        DT=0.005 LR_MIN=0.0005 DIAG_SHIFT=3e-3 N_ITER=200 \
        ${RE:+REF_E="$RE"} ${RS:+REF_SIG="$RS"} NAME="$name" \
        EXTRA_ARGS="$SNAP_ARGS --init_from $RIGHT_OUT/$prev_name" \
        OUT_DIR="$RIGHT_OUT")
      log "$jid" "right hy=$HY hx=$HX ${tag}-chain link (from $prev_hx)"
      prev_jid=$jid; prev_name=$name
    done
  }
  chain up 0.6 "${AJID[0.6]}" $UP_LINKS
  chain dn 1.25 "${AJID[1.25]}" $DN_LINKS

  # ---- TR pairs (§C: E(+hy) = E(-hy)), one per cut --------------------------
  # By design, NOT a bug: these -hy runs land in the SAME OUT_DIR as the +hy
  # population above (hy$HY names the dir after the positive $HY, the run's own
  # NAME/config still carries hy=-$HY) — names carry _hy-0.2 so nothing on disk
  # clobbers, and tc3d.fm/renyi's --hy filter (1e-9 tol, no None-means-any for
  # hy) separates the +/- populations again at extraction time.
  jid=$(submit 03:00:00 HX=0.2 HZ=0.26 HY="-$HY" \
    DT=0.02 LR_MIN=0.002 DIAG_SHIFT=1e-3 N_ITER=300 \
    EXTRA_ARGS="$SNAP_ARGS" OUT_DIR="$BASE_OUT/up/hy$HY/L4")
  log "$jid" "up hy=-$HY TR pair (hz=0.26) cold300"
  jid=$(submit 03:00:00 HX=0.6 HZ=0.1 HY="-$HY" \
    DT=0.02 LR_MIN=0.002 DIAG_SHIFT=1e-3 N_ITER=300 \
    EXTRA_ARGS="$SNAP_ARGS" OUT_DIR="$BASE_OUT/right/hy$HY/L4")
  log "$jid" "right hy=-$HY TR pair (hx=0.6) cold300"
done

echo "manifest: $MANIFEST"
