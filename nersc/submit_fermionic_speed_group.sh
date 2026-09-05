#!/bin/bash
#SBATCH -A m5340_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -c 32
#SBATCH -t 02:00:00
#SBATCH -J fspeed
#SBATCH -o %x_%j.out
#
# Group wrapper for the speed ladder: run several ARMS sequentially on ONE node/GPU
# (same node => arm-to-arm comparison free of node-to-node noise). Used for the cheap
# sizes (L<=5); L=6 stays one arm per job via submit_fermionic_speed_ladder.sh.
# Env: L (required), ARMS (space-separated, default all five), HX, HZ, and every knob
# of submit_fermionic_speed_ladder.sh passes through unchanged.
#   sbatch -t 01:00:00 --export=ALL,L=2,ARMS="baseline cup linear vote pt2" nersc/submit_fermionic_speed_group.sh
set -uo pipefail
ARMS="${ARMS:-baseline cup linear vote pt2}"
REPO="${REPO:-$HOME/toric-code-nqs-fsign}"
cd "$REPO" || { echo "[fspeed-group] REPO not found: $REPO"; exit 1; }
echo "=== group job ${SLURM_JOB_ID:-?}: L=${L:?} ARMS=[$ARMS] hx=${HX:-0.5} hz=${HZ:-0.2} started $(date) ==="
RC_ALL=0
for ARM in $ARMS; do
  echo; echo "########## ARM=$ARM  $(date +%H:%M:%S) ##########"
  ARM="$ARM" bash nersc/submit_fermionic_speed_ladder.sh; rc=$?
  echo "########## ARM=$ARM finished rc=$rc  $(date +%H:%M:%S) ##########"
  [ $rc -ne 0 ] && RC_ALL=$rc
done
echo "=== group job done $(date) (worst rc=$RC_ALL) ==="
exit $RC_ALL
