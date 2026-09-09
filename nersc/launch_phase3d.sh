#!/bin/bash
# 3D-TC phase3d campaign launcher: IDEMPOTENT and STATE-DRIVEN, one hy PLANE
# per invocation. Re-running `HY=<hy> bash nersc/launch_phase3d.sh` at any time
# submits exactly what has become launchable given (a) the manifests under
# $BASE_OUT/manifests (never resubmits a (hy,cut,L,role,h) already there) and
# (b) the finals on disk + the locator fits -- nothing else. Supersedes the
# earlier two-wave (LS=4 then LS="5 6") design entirely.
#
# ALL decision logic (priority order, dedup, recentring, the three physics
# refusals) lives in ONE python entry point: `analysis/scripts/phase3d_grid.py
# plan --bash`. This script only turns its TAB-separated lines into sbatch
# calls + manifest rows, and enforces the MAX_QUEUE ceiling (the plan step
# doesn't see the live Slurm queue -- that's cluster-only, so it's measured
# here and passed down as --max_new, guarded off entirely under DRYRUN).
#
# Priority order (see phase3d_grid.plan's docstring): L4 (all 7 electric +
# chain combined anchor+links) -> L5/6 chain anchors (separate cold jobs) ->
# L5/6 electric flanks+centre (seed) -> recentred electric fills (gated on the
# L4 fit) -> L5/6 chain link jobs (union window, singleton on the anchor) ->
# refine. Deferred-by-ceiling jobs are printed; the next re-run tops them up.
#
# Run ON PERLMUTTER (needs sbatch/squeue) from the repo root:
#   HY=0.0 bash nersc/launch_phase3d.sh                      # top up everything launchable
#   HY=0.2 LS=5 bash nersc/launch_phase3d.sh                 # LS is now just a display filter
#   DRYRUN=1 HY=0.0 bash nersc/launch_phase3d.sh             # print only, no sbatch/squeue
#
# CUTS default = all 10 minus the two pre-existing (electric_hx0.2, magnetic_hz0.1
# already run at hy=0 L4-6 and hy=0.2/0.4 L4 -- see CLAUDE.md track 1).
#
# INTERFACE ASSUMPTION (coded against a wrapper change landing in parallel, not
# yet in this worktree as of 2026-09-09): nersc/submit_nqs_batch.sh is gaining
# WARM_START (bool -> --warm_start), ANCHOR_OVERRIDES (JSON, point-0 only),
# INIT_FROM (-> --init_from, forwarded under --warm_start per A1), HY, WANDB_PROJECT,
# WANDB_GROUP and an EXTRA_ARGS passthrough (mirroring submit_nqs_gridinv.sh's).
# submit_nqs_gridinv.sh already has POST_S2_EVAL/POST_S2_SECTOR/POST_S2_ROUNDS.
# NEITHER wrapper's requeue() re-passes --job-name on an AUTO_RESUBMIT chunk (it
# reverts to the script's #SBATCH default) -- a caveat for the singleton
# dependency on a chain link job if its anchor times out mid-run; flagged, not
# fixed here (out of this task's scope: never edit submit_nqs_*.sh).
#
# KNOWN GAP: tc3d.sweep has neither --exact_E0 nor --ref_E/--ref_sig, so chain
# LINK jobs (batch wrapper) get neither -- only electric and chain ANCHOR jobs
# (both gridinv, single-point) do.
set -euo pipefail
cd "$(dirname "$0")/.."

HY="${HY:?set HY to one plane value, e.g. HY=0.0}"
LS="${LS:-}"                                            # optional display filter only
BASE_OUT="${BASE_OUT:-${PSCRATCH:-}/tc_nqs/phase3d}"     # ${PSCRATCH:-} -- unset on the Mac, set -u safe
DRYRUN="${DRYRUN:-0}"
MAX_QUEUE="${MAX_QUEUE:-40}"

DEFAULT_CUTS="electric_hx0.0 electric_hx0.5 electric_hx0.8 magnetic_hz0.0 magnetic_hz0.2 magnetic_hz0.4 magnetic_hz0.7 magnetic_hz1.0"
CUTS="${CUTS:-$DEFAULT_CUTS}"

GRIDPY="analysis/scripts/phase3d_grid.py"
PY="${PY:-python3}"
command -v "$PY" >/dev/null 2>&1 || PY=python
[ -f "$GRIDPY" ] || { echo "[launch] missing $GRIDPY"; exit 1; }

# manifests are written locally; under DRYRUN never touch $PSCRATCH (guard
# every cluster-only path/command -- this must run on the Mac with no cluster
# access at all).
if [ "$DRYRUN" = "1" ]; then
  MANIFEST_DIR="${TMPDIR:-/tmp}/phase3d_dryrun"
else
  MANIFEST_DIR="$BASE_OUT/manifests"
fi
mkdir -p "$MANIFEST_DIR"
MANIFEST="$MANIFEST_DIR/manifest_$(date +%Y%m%d_%H%M%S).tsv"
printf "jobid\thy\tcut\tL\trole\th\tname\tout_dir\tsubmitted_at\n" > "$MANIFEST"

# queue ceiling: cluster-only (squeue), so fully skipped under DRYRUN. The
# plan step is handed the REMAINING budget, not the raw MAX_QUEUE, so it
# truncates exactly like a real run would.
if [ "$DRYRUN" = "1" ]; then
  CURRENT_Q=0
else
  CURRENT_Q=$(squeue -u "$USER" -h 2>/dev/null | wc -l | tr -d ' ')
fi
MAX_NEW=$((MAX_QUEUE - CURRENT_Q))
[ "$MAX_NEW" -lt 0 ] && MAX_NEW=0

now() { date -u +%Y-%m-%dT%H:%M:%SZ; }
manifest_row() { printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$@" >> "$MANIFEST"; }

RESULTS_DIR="$BASE_OUT/hy$HY"
PLAN=$("$PY" "$GRIDPY" plan --hy "$HY" --results "$RESULTS_DIR" --manifests "$MANIFEST_DIR" \
  --max_new "$MAX_NEW" --cuts "$CUTS" --bash)

n_specs=0
while IFS=$'\t' read -r jobname wrapper walltime array dep out_dir_rel role cut L h_list env_str; do
  [ -z "$jobname" ] && continue
  [ -n "$LS" ] && { case " $LS " in *" $L "*) ;; *) continue ;; esac; }
  n_specs=$((n_specs + 1))

  wrapper_path="nersc/submit_nqs_gridinv.sh"
  [ "$wrapper" = "batch" ] && wrapper_path="nersc/submit_nqs_batch.sh"
  out_dir="$BASE_OUT/$out_dir_rel"

  # env_str is a sequence of shlex-quoted KEY=VALUE tokens from a trusted
  # source (this script's own python) -- eval-ing them into a bash array is
  # the standard, safe idiom for that (see e.g. submit_nqs_batch.sh's `env
  # "${E[@]}"` pattern).
  eval "ENVARR=($env_str)"
  ENVARR+=("OUT_DIR=$out_dir" "WALLTIME=$walltime")
  # INIT_FROM (chain link jobs) arrives as a bare checkpoint basename -- only
  # bash knows BASE_OUT, so the BASE path (no extension, per --init_from's
  # contract) is assembled here.
  for i in "${!ENVARR[@]}"; do
    case "${ENVARR[$i]}" in
      INIT_FROM=*) ENVARR[$i]="INIT_FROM=$out_dir/${ENVARR[$i]#INIT_FROM=}" ;;
    esac
  done

  sbatch_args=(--parsable --job-name="$jobname" --time="$walltime")
  [ "$array" != "-" ] && sbatch_args+=(--array="$array")
  [ "$dep" != "-" ] && sbatch_args+=(--dependency="$dep")
  echo "[launch] env ${ENVARR[*]} sbatch ${sbatch_args[*]} $wrapper_path" >&2

  if [ "$DRYRUN" = "1" ]; then
    jid="DRY"
  elif [ $((CURRENT_Q + n_specs)) -gt "$MAX_QUEUE" ]; then
    echo "[launch] MAX_QUEUE=$MAX_QUEUE reached -- NOT submitting $jobname" >&2
    jid="SKIPPED"
  else
    jid=$(env "${ENVARR[@]}" sbatch "${sbatch_args[@]}" "$wrapper_path")
  fi
  for h in $h_list; do
    manifest_row "$jid" "$HY" "$cut" "$L" "$role" "$h" "$jobname" "$out_dir" "$(now)"
  done
done <<<"$PLAN"

echo "[launch] hy=$HY  cuts=[$CUTS]  DRYRUN=$DRYRUN MAX_QUEUE=$MAX_QUEUE (queue was $CURRENT_Q)"
echo "[launch] $n_specs job(s) this run."
echo "[launch] manifest: $MANIFEST"
