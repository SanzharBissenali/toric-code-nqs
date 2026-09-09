#!/bin/bash
# Hourly plane driver for the phase3d campaign (installed via scrontab, cron QOS).
# The launcher is idempotent and state-driven, so "re-run it" IS the plane agent:
#   1. submit whatever became launchable (L4 first, anchors, flanks, recentred fills,
#      chain link jobs once anchors are healthy, refinement) under MAX_QUEUE;
#   2. re-run the watcher over the union of this plane's manifests -> watch_state.json.
# Usage (one line per plane in scrontab):  HY=0.0 bash nersc/phase3d_cron_driver.sh
set -uo pipefail
HY="${HY:?set HY}"; MAX_QUEUE="${MAX_QUEUE:-40}"
export PATH="$HOME/.conda/envs/tc-nqs/bin:$PATH"
cd "$HOME/toric-code-nqs" && export PYTHONPATH="$HOME/toric-code-nqs"
BASE="$PSCRATCH/tc_nqs/phase3d"; LOG="$BASE/driver_hy$HY.log"
{
  echo "=== $(date -Is) hy=$HY ==="
  HY="$HY" MAX_QUEUE="$MAX_QUEUE" bash nersc/launch_phase3d.sh 2>&1 | grep -v '^\[launch\] env'
  ALL="$BASE/manifests/all_hy$HY.tsv"
  { printf 'jobid\thy\tcut\tL\trole\th\tname\tout_dir\tsubmitted_at\n'
    for m in "$BASE"/manifests/manifest_2026*.tsv; do awk -F'\t' -v hy="$HY" 'NR>1 && $2==hy' "$m"; done; } > "$ALL"
  MANIFEST="$ALL" BASE_OUT="$BASE" bash nersc/watch_phase3d.sh 2>&1 | awk '/^\[watch\]/ || n++ < 30'   # summary + first 30 flags
} >> "$LOG" 2>&1
