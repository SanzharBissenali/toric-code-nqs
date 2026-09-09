#!/bin/bash
# Pull the phase3d campaign's outputs from Perlmutter, split by commit policy
# (CLAUDE.md "Data -- rsync back, commit summaries only"):
#   final {name}.json + {name}.snapshots.json + manifests/*.tsv + watch_state.json
#     -> results/phase3d/...   (small; meant to be committed)
#   {name}.curve.json (per-step learning curves; duplicates W&B, dominated repo
#     line growth historically)
#     -> data/tc_nqs/phase3d/...   (gitignored /data/ -- local only, never committed)
#
#   bash analysis/scripts/pull_phase3d.sh                # full pull, every plane
#   HY=0.0 bash analysis/scripts/pull_phase3d.sh          # one plane only
#   DRYRUN=1 bash analysis/scripts/pull_phase3d.sh        # rsync --dry-run (still SSHes)
#
# Run from the repo root. REMOTE_HOST="" (with REMOTE_BASE a local path) targets
# a plain local source tree instead of ssh -- used for offline testing; plane
# directories are discovered with `ls` (locally or over ssh) rather than handed
# to rsync as a wildcard, since remote-side glob expansion through rsync+ssh is
# inconsistent across versions.
set -euo pipefail
cd "$(dirname "$0")/../.."

REMOTE_HOST="${REMOTE_HOST-perlmutter}"
REMOTE_BASE="${REMOTE_BASE:-\$PSCRATCH/tc_nqs/phase3d}"   # expands server-side (ssh) by default
HY="${HY:-}"                                              # empty -> every hy{...} plane on disk
LOCAL_RESULTS="${LOCAL_RESULTS:-results/phase3d}"
LOCAL_DATA="${LOCAL_DATA:-data/tc_nqs/phase3d}"
DRYRUN="${DRYRUN:-0}"

grep -qx '/data/' .gitignore || { echo "[pull] WARNING: /data/ not in .gitignore -- refusing" >&2; exit 1; }

SRC_PREFIX=""; [ -n "$REMOTE_HOST" ] && SRC_PREFIX="$REMOTE_HOST:"
mkdir -p "$LOCAL_RESULTS" "$LOCAL_DATA"

RSYNC_FLAGS=(-avz --prune-empty-dirs)
[ "$DRYRUN" = "1" ] && RSYNC_FLAGS+=(--dry-run)

ls_dirs() {   # ls_dirs <path> -- basenames of subdirs, local or over ssh
  if [ -n "$REMOTE_HOST" ]; then
    ssh "$REMOTE_HOST" "ls -d $1/*/ 2>/dev/null" 2>/dev/null | xargs -n1 basename
  else
    ls -d "$1"/*/ 2>/dev/null | xargs -n1 basename
  fi
}

if [ -n "$HY" ]; then
  PLANES="hy$HY"
else
  PLANES=$(ls_dirs "$REMOTE_BASE" | grep '^hy' || true)
fi
[ -n "$PLANES" ] || { echo "[pull] no hy* planes found under $REMOTE_BASE"; exit 0; }
echo "[pull] planes: $PLANES"

echo "[pull] -> $LOCAL_RESULTS"
for plane in $PLANES; do
  # committable: exclude *.curve.json FIRST (it also matches *.json), then
  # include the run/snapshot JSONs, then drop everything else.
  rsync "${RSYNC_FLAGS[@]}" \
    --include='*/' \
    --exclude='*.curve.json' \
    --include='*.json' \
    --exclude='*' \
    "${SRC_PREFIX}${REMOTE_BASE}/${plane}/" "$LOCAL_RESULTS/${plane}/"
done
rsync "${RSYNC_FLAGS[@]}" \
  "${SRC_PREFIX}${REMOTE_BASE}/manifests/" "$LOCAL_RESULTS/manifests/" 2>/dev/null || true
# watch_state.json lives at the campaign root (one file, all planes -- see
# nersc/watch_phase3d.sh), not under hy{HY}/
rsync "${RSYNC_FLAGS[@]}" \
  "${SRC_PREFIX}${REMOTE_BASE}/watch_state.json" "$LOCAL_RESULTS/" 2>/dev/null || true

echo "[pull] -> $LOCAL_DATA (gitignored)"
for plane in $PLANES; do
  rsync "${RSYNC_FLAGS[@]}" \
    --include='*/' --include='*.curve.json' --exclude='*' \
    "${SRC_PREFIX}${REMOTE_BASE}/${plane}/" "$LOCAL_DATA/${plane}/"
done

echo "[pull] done."
