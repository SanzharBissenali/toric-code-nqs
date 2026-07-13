#!/bin/bash
# Push all OFFLINE W&B runs from the finished campaign up to wandb.ai.
#
# Compute nodes trained with WANDB_MODE=offline (no network), leaving one
# `offline-run-*` dir per run under `<out_dir>/wandb/`. This walks the whole
# campaign tree and `wandb sync`s each. Run it from a LOGIN node (has network);
# `wandb login` once first.
#
#   bash nersc/sync_wandb.sh              # sync everything not yet synced
#   DRY_RUN=1 bash nersc/sync_wandb.sh    # list what WOULD sync, do nothing
#   FORCE=1   bash nersc/sync_wandb.sh    # re-sync even already-synced runs
#   BASE=/custom/root bash nersc/sync_wandb.sh
#
# Each run has a STABLE md5 run-id with resume="allow" (see train.py), so a
# re-sync updates the SAME wandb run and never duplicates -- FORCE is safe, and
# the "0/N" progress counter resetting between invocations is cosmetic. Already-
# synced runs carry a `.synced` sentinel (wandb --mark-synced) and are skipped.
set -uo pipefail

BASE="${BASE:-$PSCRATCH/tc_nqs}"
PROJECT="${PROJECT:-approx-sym-3D-TC}"
ENTITY="${ENTITY:-models-california-institute-of-technology-caltech}"

if ! command -v wandb >/dev/null 2>&1; then
  echo "[sync] 'wandb' not on PATH -- module load conda && conda activate tc-nqs" >&2
  exit 1
fi

mapfile -t RUNS < <(find "$BASE" -type d -name 'offline-run-*' 2>/dev/null | sort)
n=${#RUNS[@]}
echo "[sync] BASE=$BASE  project=$PROJECT  entity=$ENTITY"
echo "[sync] found $n offline runs"
[ "$n" -eq 0 ] && { echo "[sync] nothing to sync (check BASE, or runs were online)"; exit 0; }

done=0; skipped=0; failed=0
for i in "${!RUNS[@]}"; do
  d="${RUNS[$i]}"
  if [ -z "${FORCE:-}" ] && [ -f "$d/.synced" ]; then
    skipped=$((skipped + 1)); continue
  fi
  echo "[sync] ($((i + 1))/$n) $d"
  [ -n "${DRY_RUN:-}" ] && continue
  if wandb sync "$d" -p "$PROJECT" -e "$ENTITY" --mark-synced; then
    done=$((done + 1))
  else
    echo "[sync]   FAILED: $d" >&2; failed=$((failed + 1))
  fi
done
echo "[sync] done.  synced=$done  skipped(already)=$skipped  failed=$failed"
[ "$failed" -eq 0 ] || exit 1
