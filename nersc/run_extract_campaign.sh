#!/bin/bash
# Extraction campaign: for every (hx cut, system size L) submit ONE combined
# O_FM(R=1) + Rényi-S₂ extraction job (submit_extract_fm_s2.sh). The read-side
# analog of run_phase_campaign.sh -- run it after the training campaign to turn the
# trained checkpoints into the tiny fm_*/s2_* curve JSONs that
# analysis/vertical_line_hz.ipynb consumes.
#
# One job per (hx, L) -- NOT per hx -- because each L is a separate ~1-2 h
# all-or-nothing extraction per observable (mirrors the repo's one-L-per-job rule
# and backfills better on shared QOS). For true per-hx jobs instead, call
# submit_extract_fm_s2.sh directly with LS="4 5 6 7".
#
# Idempotent: SKIP_EXISTING=1 (default here) makes each job skip an L whose curve
# JSON already exists, so re-running only fills gaps. hx=0.0 is already extracted,
# so the default HX_VALS covers just the 5 remaining cuts.
#
#   nersc/run_extract_campaign.sh                            # 5 hx x 4 L = 20 jobs
#   DRYRUN=1 nersc/run_extract_campaign.sh                   # print sbatch lines, submit nothing
#   HX_VALS="0.2" L_VALS="6 7" nersc/run_extract_campaign.sh # top up a subset
#   HX_VALS="0.0 0.2 0.4 0.6 0.8 1.0" nersc/run_extract_campaign.sh  # include hx=0.0
set -euo pipefail

HX_VALS="${HX_VALS:-0.2 0.4 0.6 0.8 1.0}"
L_VALS="${L_VALS:-4 5 6 7}"
DRYRUN="${DRYRUN:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"     # idempotent top-up; set 0 to force recompute

SUBMIT="$(cd "$(dirname "$0")" && pwd)/submit_extract_fm_s2.sh"
[ -f "$SUBMIT" ] || { echo "[extract-campaign] submit script not found: $SUBMIT"; exit 1; }

# Extraction is eval-only and dominated by XLA kernel COMPILATION (a fixed cost the
# two observables largely share), not by the eval itself -- so running FM+S₂
# together is far from 2x. Measured: L=4,5 finish in <30 min, L=6,7 in ~1 h; these
# limits carry a comfortable margin.
walltime_for() {
  case "$1" in
    4) echo "01:00:00" ;;
    5) echo "01:00:00" ;;
    6) echo "02:00:00" ;;
    7) echo "02:00:00" ;;
    *) echo "02:00:00" ;;
  esac
}

n=0
for HX in $HX_VALS; do
  for L in $L_VALS; do
    TL="$(walltime_for "$L")"
    echo "[extract-campaign] hx=$HX L=$L  --time=$TL  (O_FM R=1 + S₂, SKIP_EXISTING=$SKIP_EXISTING)"
    n=$((n + 1))
    [ "$DRYRUN" = "1" ] && continue
    HX="$HX" LS="$L" SKIP_EXISTING="$SKIP_EXISTING" \
      sbatch --time="$TL" "$SUBMIT"
  done
done
if [ "$DRYRUN" = "1" ]; then
  echo "[extract-campaign] (dry-run) $n (hx,L) jobs listed; submitted nothing."
else
  echo "[extract-campaign] $n (hx,L) extraction jobs submitted."
fi
