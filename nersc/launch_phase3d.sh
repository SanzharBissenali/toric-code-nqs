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
# L4 fit) -> L5/6 chain link jobs (union window, singleton on the anchor --
# queued as soon as its anchor is merely SUBMITTED, via afterok:<anchor
# jobid>,singleton, so it accrues queue age while the anchor is still
# pending/running; only a FINISHED-but-unhealthy anchor still holds it) ->
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

# 2026-09-17 plan (notes/phase3d_L4_plan.md): + electric_hx0.65, magnetic_hz0.25, magnetic_hz0.85;
# electric_hx0.2 / magnetic_hz0.1 stay out (run by the older hy_cuts_L4 campaign at hy 0/0.2/0.4).
DEFAULT_CUTS="electric_hx0.0 electric_hx0.5 electric_hx0.65 electric_hx0.8 magnetic_hz0.0 magnetic_hz0.2 magnetic_hz0.25 magnetic_hz0.4 magnetic_hz0.7 magnetic_hz0.85 magnetic_hz1.0"
# 2026-09-19 (user): magnetic_hz0.1 also on the planes the older campaign never covered, so every plane's envelope
# has the same h_z rungs.
case "${HY:-0.0}" in 0.0|0.2|0.4|y) ;; *) DEFAULT_CUTS="$DEFAULT_CUTS magnetic_hz0.1 electric_hx0.2";; esac
CUTS_SET=0; [ -n "${CUTS:-}" ] && CUTS_SET=1
CUTS="${CUTS:-$DEFAULT_CUTS}"

GRIDPY="analysis/scripts/phase3d_grid.py"
# Prefer the tc-nqs conda python: the login-node system python3 is too old for the
# planner's fit modules (`from __future__ import annotations` -> SyntaxError).
PY="${PY:-$HOME/.conda/envs/tc-nqs/bin/python}"
command -v "$PY" >/dev/null 2>&1 || PY=python3
command -v "$PY" >/dev/null 2>&1 || PY=python
[ -f "$GRIDPY" ] || { echo "[launch] missing $GRIDPY"; exit 1; }

# manifests are written locally; under DRYRUN never touch $PSCRATCH (guard
# every cluster-only path/command -- this must run on the Mac with no cluster
# access at all).
if [ "$DRYRUN" = "1" ]; then
  MANIFEST_DIR="${TMPDIR:-/tmp}/phase3d_dryrun"
  # a truthful dry run dedupes against the REAL manifests: read a fresh copy of them
  # (never the originals -- the DRY rows below land in this temp dir only)
  rm -rf "$MANIFEST_DIR"; mkdir -p "$MANIFEST_DIR"
  if [ -n "${PSCRATCH:-}" ] && [ -d "$BASE_OUT/manifests" ]; then
    cp "$BASE_OUT"/manifests/manifest_*.tsv "$MANIFEST_DIR"/ 2>/dev/null || true
  fi
else
  MANIFEST_DIR="$BASE_OUT/manifests"
fi
mkdir -p "$MANIFEST_DIR"

# queue ceiling: cluster-only (squeue), so fully skipped under DRYRUN. The
# plan step is handed the REMAINING budget, not the raw MAX_QUEUE, so it
# truncates exactly like a real run would. Jobs held on a dependency (reason
# "Dependency"/"DependencyNeverSatisfied" -- an early-queued L5/6 chain link
# waiting on its anchor's afterok) are excluded from the count: they aren't
# competing for a runslot yet, so MAX_QUEUE stays a ceiling on
# runnable+running jobs, not on everything merely sitting in squeue.
if [ "$DRYRUN" = "1" ]; then
  CURRENT_Q=0
else
  CURRENT_Q=$(squeue -u "$USER" -h -o "%r" 2>/dev/null | grep -vc '^Dependency' | tr -d ' ')
fi
MAX_NEW=$((MAX_QUEUE - CURRENT_Q))
[ "$MAX_NEW" -lt 0 ] && MAX_NEW=0

now() { date -u +%Y-%m-%dT%H:%M:%SZ; }
# CRUCIAL fix: plan() reads every manifest_*.tsv under MANIFEST_DIR to dedupe --
# it must run BEFORE this script creates any file there. The old code named
# the manifest with 1-second resolution and truncated it (>) right away: two
# invocations landing in the same second raced, and the second one's `>`
# wiped the first run's already-written rows out from under it, which then
# resubmitted everything the first run had just launched. Fix: call plan()
# first; only if there's something to log, create a manifest whose name is
# unique even within the same second ($$ + $RANDOM), and only ever append.
MANIFEST=""
manifest_row() {
  if [ -z "$MANIFEST" ]; then
    MANIFEST="$MANIFEST_DIR/manifest_$(date +%Y%m%d_%H%M%S)_$$_${RANDOM}.tsv"
    printf "jobid\thy\tcut\tL\trole\th\tname\tout_dir\tsubmitted_at\n" > "$MANIFEST"
  fi
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$@" >> "$MANIFEST"
}

RESULTS_DIR="$BASE_OUT/hy$HY"
if [ "$HY" = "y" ]; then                                  # y-cut pseudo-plane (sweep hy at fixed hx,hz)
  RESULTS_DIR="$BASE_OUT/ycuts"
  [ "${CUTS_SET:-0}" = "1" ] || CUTS=""                   # planner picks its own y-cut list
fi
if [ -n "${RETRY_FINAL:-}" ]; then
  # Retry ONE landed single-point run (electric cold point / chain anchor) with
  # knob overrides, e.g. a GENUINE DIVERGENCE at the default diag_shift:
  #   RETRY_FINAL=<final.json> RETRY_SET="DIAG_SHIFT=5e-3" HY=0.0 bash nersc/launch_phase3d.sh
  # The planner parks the old outputs in <out_dir>/redo_<oldjid>/, drops the
  # point's manifest rows, and hands back one plan line; submission + the new
  # manifest row go through the normal loop below (MAX_QUEUE is not applied).
  SETS=(); for kv in ${RETRY_SET:-}; do SETS+=(--set "$kv"); done
  PLAN=$("$PY" "$GRIDPY" retry --final "$RETRY_FINAL" --manifests "$MANIFEST_DIR" "${SETS[@]}")
elif [ -n "${PLAN_FILE:-}" ]; then
  # Hand-made plan lines (phase3d_grid._bash_line format), e.g. extra electric points a review asked for:
  #   PLAN_FILE=/tmp/extra.tsv HY=1.0 bash nersc/launch_phase3d.sh   (dedup against the manifests is the caller's job)
  PLAN=$(cat "$PLAN_FILE")
else
  PLAN=$("$PY" "$GRIDPY" plan --hy "$HY" --results "$RESULTS_DIR" --manifests "$MANIFEST_DIR" \
    --max_new "$MAX_NEW" --cuts "$CUTS" --bash)
fi

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
  else
    jid=$(env "${ENVARR[@]}" sbatch "${sbatch_args[@]}" "$wrapper_path")
  fi
  for h in $h_list; do
    manifest_row "$jid" "$HY" "$cut" "$L" "$role" "$h" "$jobname" "$out_dir" "$(now)"
  done
  # POST_S2=1: end-of-training S2 (+O_FM string) on every point of a chain job, as a follow-up eval job that
  # waits (singleton on the chain's own job name) for the chain and its AUTO_RESUBMIT chunks to finish; the
  # branch glob picks only this chain's runs. Writes <name>.finaleval_electric.json (what phase3d_status reads).
  if [ "${POST_S2:-0}" = "1" ] && [ "$DRYRUN" != "1" ] && [ "$L" = "4" ] && [ "$wrapper" = "batch" ] && { [ "$role" = "chain_up" ] || [ "$role" = "chain_dn" ]; }; then
    br="${role#chain_}"; br="${br%%_*}"
    ejid=$(DIR="$out_dir" GLOB="*_k3_${br}.json" SECTOR=electric SUFFIX=.finaleval_electric.json LAST_ONLY=1 \
           sbatch --parsable -J "$jobname" --dependency=singleton nersc/submit_eval_hy_axis.sh)
    echo "[launch] post-S2 eval job $ejid (singleton after $jobname, glob *_k3_${br}.json)" >&2
  fi
done <<<"$PLAN"

echo "[launch] hy=$HY  cuts=[$CUTS]  DRYRUN=$DRYRUN MAX_QUEUE=$MAX_QUEUE (queue was $CURRENT_Q)"
echo "[launch] $n_specs job(s) this run."
if [ -n "$MANIFEST" ]; then
  echo "[launch] manifest: $MANIFEST"
else
  echo "[launch] nothing to submit -- no manifest written."
fi
