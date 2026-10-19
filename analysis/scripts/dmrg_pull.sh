#!/bin/bash
# Pull the DMRG L=3 OBC benchmark finals from Perlmutter into results/dmrg_L3_obc/{btc,ftc}
# (final-state JSONs + snapshots; no checkpoints/curves per the per-run commit policy), then
# print the comparison table.  Usage: bash analysis/scripts/dmrg_pull.sh [--curves]
set -euo pipefail
cd "$(dirname "$0")/../.."
HOST=perlmutter; REMOTE='/pscratch/sd/s/sanzharb/tc_nqs/dmrg_L3'
EXCL=(--exclude '*.mpack' --exclude 'smoke/' --exclude '*.step*' --exclude 'wandb/')
[ "${1:-}" = "--curves" ] || EXCL+=(--exclude '*.curve.json')
for side in btc ftc; do
  mkdir -p "results/dmrg_L3_obc/$side"
  rsync -a --prune-empty-dirs "${EXCL[@]}" "$HOST:$REMOTE/$side/" "results/dmrg_L3_obc/$side/"
done
.venv/bin/python analysis/scripts/dmrg_compare.py --runs results/dmrg_L3_obc/btc results/dmrg_L3_obc/ftc \
  --ref results/dmrg_L3_obc/dmrg_L3_obc_reference.json --csv results/dmrg_L3_obc/compare.csv
