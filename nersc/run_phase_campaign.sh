#!/bin/bash
# (hx,hz) phase-diagram campaign launcher: for every (hx cut, system size L) submit one
# hz-sweep array job via submit_nqs_hz_sweep.sh. Each array task trains one NQS at
# a fixed (L, hx, hz), saves weights + run JSON, and is --resume-safe.
#
# Idempotent: re-running just continues unfinished points (the submit script always
# resumes from its checkpoint). Pair with analysis/scripts/check_convergence.py --tree to
# find MISSING/flagged points, then re-run this (optionally narrowing HX_VALS/L_VALS)
# to top them up.
#
#   nersc/run_phase_campaign.sh                 # full 6 hx x 4 L campaign
#   DRYRUN=1 nersc/run_phase_campaign.sh        # print the 24 sbatch lines, submit nothing
#   HX_VALS="0.8 1.0" L_VALS=6 nersc/run_phase_campaign.sh   # top up a subset
#
# The hz window is fixed [HZ_MIN, HZ_MAX] at every hx (the derivative peak sits in
# 0.1-0.4 for L=4,5,6). If a QA scan shows no peak at high hx (hz_c drifts down as
# hx grows), re-run those cuts with a lower HZ_MIN.
set -euo pipefail

HX_VALS="${HX_VALS:-0.0 0.2 0.4 0.6 0.8 1.0}"
L_VALS="${L_VALS:-4 5 6 7}"
HZ_MIN="${HZ_MIN:-0.1}"; HZ_MAX="${HZ_MAX:-0.4}"; HZ_N="${HZ_N:-13}"   # 13 -> step 0.025
AUTO_RESUBMIT="${AUTO_RESUBMIT:-1}"    # fire-and-forget: survive the wall limit
DRYRUN="${DRYRUN:-0}"

SUBMIT="$(cd "$(dirname "$0")" && pwd)/submit_nqs_hz_sweep.sh"
[ -f "$SUBMIT" ] || { echo "[campaign] submit script not found: $SUBMIT"; exit 1; }

# submit from the repo root so slurm_logs/ (job --output target) exists,
# and propagate a non-default clone path to the jobs
REPO="${REPO:-$(dirname "$(dirname "$SUBMIT")")}"
cd "$REPO" || { echo "[campaign] REPO not found: $REPO"; exit 1; }
export REPO

# Fixed per-L wall time (150 iters x per-step + compile + margin, from the earlier
# timing: L6 ~43 s/step measured, L7 ~90 s, L4/L5 scaled). Sets the initial --time
# (overrides the #SBATCH directive) and propagates via WALLTIME through requeues.
walltime_for() {
  case "$1" in
    4) echo "01:00:00" ;;
    5) echo "01:30:00" ;;
    6) echo "03:00:00" ;;
    7) echo "06:00:00" ;;   # 175 iters x ~105 s/step (k5) ~= 5.1 h
    *) echo "06:00:00" ;;
  esac
}

last=$((HZ_N - 1))
n=0
for HX in $HX_VALS; do
  for L in $L_VALS; do
    TL="$(walltime_for "$L")"
    echo "[campaign] L=$L hx=$HX  hz[$HZ_MIN..$HZ_MAX]x$HZ_N (array 0-$last)  --time=$TL"
    n=$((n + 1))
    [ "$DRYRUN" = "1" ] && continue
    HX="$HX" L="$L" HZ_MIN="$HZ_MIN" HZ_MAX="$HZ_MAX" HZ_N="$HZ_N" \
      AUTO_RESUBMIT="$AUTO_RESUBMIT" WALLTIME="$TL" \
      sbatch --time="$TL" --array="0-$last" "$SUBMIT"
  done
done
echo "[campaign] ${DRYRUN:+(dry-run) }$n (hx,L) array jobs, $((n * HZ_N)) hz points total."
