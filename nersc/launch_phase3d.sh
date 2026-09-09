#!/bin/bash
# 3D-TC phase3d campaign launcher (Stage 1, sparse grid): ONE hy PLANE per
# invocation, L in {4,5,6}, up to 10 cuts (4 electric 2nd-order + 3 magnetic +
# 3 tail 1st-order). Generalizes nersc/launch_hy_cuts_L4.sh; the grid/QMC-ref
# source of truth is analysis/scripts/phase3d_grid.py -- never hand-derive a
# field value here, always ask the grid script.
#
# Run ON PERLMUTTER (needs sbatch/squeue) from the repo root:
#   HY=0.0 bash nersc/launch_phase3d.sh                        # full default plane
#   HY=0.2 LS="4 5" CUTS="electric_hx0.0 electric_hx0.8" bash nersc/launch_phase3d.sh
#   DRYRUN=1 HY=0.0 bash nersc/launch_phase3d.sh                # print only, no sbatch/squeue
#
# CUTS default = all 10 minus the two pre-existing (electric_hx0.2, magnetic_hz0.1
# already run at hy=0 L4-6 and hy=0.2/0.4 L4 -- see CLAUDE.md track 1).
#
# INTERFACE ASSUMPTION (coded against a wrapper change landing in parallel, not
# yet in this worktree as of 2026-09-09): nersc/submit_nqs_batch.sh is gaining
# WARM_START (bool -> --warm_start), ANCHOR_OVERRIDES (JSON string of
# dt/lr_min/n_iter/diag_shift applied to ONLY the first/anchor point of
# FIELD_VALUES; every later link uses the plain DT/LR_MIN/DIAG_SHIFT/N_ITER env,
# per notes/transition_mapping_recipes.md section B), HY (-> --hy passthrough),
# WANDB_PROJECT (-> --wandb_project; sweep.py already accepts the flag, only the
# wrapper's env wiring is new), WANDB_GROUP (defaults to the Slurm job name;
# passed explicitly here so groups never collide across hy/branch even before
# wb_regroup.py runs) and a generic EXTRA_ARGS passthrough (mirroring
# submit_nqs_gridinv.sh's). If any of these names/semantics differ once landed,
# only submit_chain() below needs updating -- point math stays in phase3d_grid.py.
# submit_nqs_gridinv.sh also now has its own WANDB_PROJECT env knob -- used for
# electric jobs instead of an EXTRA_ARGS --wandb_project copy (one mechanism only).
#
# KNOWN GAP (not part of the parallel change): tc3d.sweep has neither
# --exact_E0 nor --ref_E/--ref_sig, so chain (batch-wrapper) jobs get neither
# EXACT_E0 nor REF_E/REF_SIG -- only electric (gridinv, single-point) jobs do.
set -euo pipefail
cd "$(dirname "$0")/.."

HY="${HY:?set HY to one plane value, e.g. HY=0.0}"
LS="${LS:-4 5 6}"
BASE_OUT="${BASE_OUT:-${PSCRATCH:-}/tc_nqs/phase3d}"   # ${PSCRATCH:-} -- unset on the Mac, set -u safe
DRYRUN="${DRYRUN:-0}"
MAX_QUEUE="${MAX_QUEUE:-40}"
WANDB_PROJECT_VAL="tc3d-phase3d"
SNAP_ARGS="--snapshot_every 50 --final_eval_rounds 8"

DEFAULT_CUTS="electric_hx0.0 electric_hx0.5 electric_hx0.8 magnetic_hz0.0 magnetic_hz0.2 magnetic_hz0.4 magnetic_hz0.7 magnetic_hz1.0"
CUTS="${CUTS:-$DEFAULT_CUTS}"

GRIDPY="analysis/scripts/phase3d_grid.py"
PY="${PY:-python3}"
command -v "$PY" >/dev/null 2>&1 || PY=python
[ -f "$GRIDPY" ] || { echo "[launch] missing $GRIDPY"; exit 1; }

# manifests are written locally; under DRYRUN never touch $PSCRATCH (guard every
# cluster-only path/command, per the A3 task's Mac-safety rule)
if [ "$DRYRUN" = "1" ]; then
  MANIFEST_DIR="${TMPDIR:-/tmp}/phase3d_dryrun"
else
  MANIFEST_DIR="$BASE_OUT/manifests"
fi
mkdir -p "$MANIFEST_DIR"
MANIFEST="$MANIFEST_DIR/manifest_$(date +%Y%m%d_%H%M%S).tsv"
printf "jobid\thy\tcut\tL\trole\th\tname\tout_dir\tsubmitted_at\n" > "$MANIFEST"

# queue ceiling: cluster-only (squeue), so it's fully skipped under DRYRUN --
# CURRENT_Q=0 there means the ceiling only ever binds on the printed job COUNT,
# which is exactly what the local test exercises.
if [ "$DRYRUN" = "1" ]; then
  CURRENT_Q=0
else
  CURRENT_Q=$(squeue -u "$USER" -h 2>/dev/null | wc -l | tr -d ' ')
fi
SUBMITTED_JOBS=0
now() { date -u +%Y-%m-%dT%H:%M:%SZ; }

# ---- per-L knobs, shared by electric and chain paths -----------------------
kernel_for() { echo $(($1 - 1)); }
diag_shift_for() { [ "$1" -ge 5 ] && echo "3e-3" || echo "1e-3"; }
chunk_env_for() { [ "$1" -le 5 ] && echo "CHUNK=2048" || echo ""; }   # L6: wrapper default
resubmit_for()  { [ "$1" -ge 5 ] && echo "1" || echo "0"; }           # electric only

walltime_for() {   # L hy -> HH:MM:SS (never > 05:00:00)
  local L="$1" hy="$2" nz=0
  [ "$hy" != "0.0" ] && nz=1
  case "$L" in
    4) [ "$nz" = 1 ] && echo "03:00:00" || echo "01:30:00" ;;
    5) [ "$nz" = 1 ] && echo "05:00:00" || echo "03:30:00" ;;
    *) echo "05:00:00" ;;
  esac
}

exact_e0_for() { case "$1" in 4) echo "-172";; 5) echo "-365";; 6) echo "-666";; esac; }

# reference lookup (glob+parse via phase3d_grid.py, hy-independent QMC refs)
ref_lookup() { "$PY" "$GRIDPY" --ref_lookup --hx "$1" --hz "$2" --L "$3" 2>/dev/null; }

# ---- ceiling-gated submit: prints the full command always (to stderr, so
# `jid=$(do_sbatch ...)` only ever captures the jobid line); under a real run
# (DRYRUN=0) stops sbatch-ing once CURRENT_Q+SUBMITTED_JOBS hits MAX_QUEUE ------
do_sbatch() {   # do_sbatch <jobname> <wrapper> <walltime> <array-or-empty> <env pairs...>
  local jobname="$1" wrapper="$2" tl="$3" arr="$4"; shift 4
  local -a sbatch_args=(--parsable --job-name="$jobname" --time="$tl")
  [ -n "$arr" ] && sbatch_args+=(--array="$arr")
  echo "[launch] env $* sbatch ${sbatch_args[*]} $wrapper" >&2
  if [ "$DRYRUN" = "1" ]; then
    echo "DRY"; return
  fi
  if [ $((CURRENT_Q + SUBMITTED_JOBS)) -ge "$MAX_QUEUE" ]; then
    echo "[launch] MAX_QUEUE=$MAX_QUEUE reached -- NOT submitting $jobname" >&2
    echo "SKIPPED"; return
  fi
  local jid
  jid=$(env "$@" sbatch "${sbatch_args[@]}" "$wrapper")
  SUBMITTED_JOBS=$((SUBMITTED_JOBS + 1))
  echo "$jid"
}

manifest_row() { printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$@" >> "$MANIFEST"; }

# ---- electric (2nd order): cold, one point = one submit_nqs_gridinv.sh job ---
submit_electric() {   # cut hx L
  local cut="$1" hx="$2" L="$3"
  local kernel ds chunk_e resub tl
  kernel=$(kernel_for "$L"); ds=$(diag_shift_for "$L")
  chunk_e=$(chunk_env_for "$L"); resub=$(resubmit_for "$L"); tl=$(walltime_for "$L" "$HY")
  local out_dir="$BASE_OUT/hy${HY}/${cut}/L${L}"
  local hz_points; hz_points=$("$PY" "$GRIDPY" --emit --cuts "$cut" --L "$L" --hy "$HY" 2>/dev/null)
  read -r _kind _hx_echo hzs <<<"$hz_points"
  for hz in $hzs; do
    local re rs extra="$SNAP_ARGS"
    read -r re rs <<<"$(ref_lookup "$hx" "$hz" "$L")"
    local exact="$(exact_e0_for "$L")"
    local jobname="p3d_hy${HY}_e${hx}_L${L}"
    local -a envs=(DUAL=1 NONINV_HIDDEN="4 8" INV="8 8" KERNEL="$kernel" BC=OBC
      L="$L" HX="$hx" HZ="$hz" HY="$HY" DT=0.02 LR_MIN=0.002 N_ITER=500
      DIAG_SHIFT="$ds" CKPT_EVERY=10 EXACT_E0="$exact"
      OUT_DIR="$out_dir" EXTRA_ARGS="$extra" AUTO_RESUBMIT="$resub"
      WANDB_PROJECT="$WANDB_PROJECT_VAL")
    [ -n "$chunk_e" ] && envs+=("$chunk_e")
    [ -n "$re" ] && envs+=(REF_E="$re" REF_SIG="$rs")
    local jid; jid=$(do_sbatch "$jobname" nersc/submit_nqs_gridinv.sh "$tl" "" \
      "${envs[@]}" WALLTIME="$tl")
    manifest_row "$jid" "$HY" "$cut" "$L" "cold" "$hz" "$jobname" "$out_dir" "$(now)"
  done
}

# ---- magnetic/tail (1st order): warm chains, one branch = one submit_nqs_batch.sh job --
submit_chain() {   # cut hz L branch(up|dn)
  local cut="$1" hz="$2" L="$3" branch="$4"
  local kernel ds tl
  kernel=$(kernel_for "$L"); ds=$(diag_shift_for "$L"); tl=$(walltime_for "$L" "$HY")
  local out_dir="$BASE_OUT/hy${HY}/${cut}/L${L}"
  local line; line=$("$PY" "$GRIDPY" --emit --cuts "$cut" --L "$L" --hy "$HY" 2>/dev/null | grep "^${branch} ")
  read -r _branch _hz anchor links <<<"$line"
  local field_values="$anchor $links"
  local n_pts; n_pts=$(echo "$field_values" | wc -w | tr -d ' ')
  local anchor_ov="{\"dt\":0.02,\"lr_min\":0.002,\"n_iter\":500,\"diag_shift\":$ds}"
  local name_tpl="gridinv_dual_L{L}_OBC_hx{hx}_hz{hz}_hy{hy}_n2x4_nh4-8_inv8-8_k${kernel}_${branch}"
  local extra="$SNAP_ARGS"
  local jobname="p3d_hy${HY}_m${hz}_L${L}_${branch}"
  local -a envs=(DUAL=1 NONINV_HIDDEN="4 8" INV="8 8" KERNEL="$kernel" BC=OBC
    L="$L" SWEEP=hx HZ="$hz" HY="$HY" FIELD_VALUES="$field_values" CHUNK_POINTS="$n_pts"
    WARM_START=1 ANCHOR_OVERRIDES="$anchor_ov" NAME_TEMPLATE="$name_tpl"
    DT=0.005 LR_MIN=0.0005 DIAG_SHIFT=3e-3 N_ITER=200 CKPT_EVERY=10
    OUT_DIR="$out_dir" EXTRA_ARGS="$extra" WANDB_PROJECT="$WANDB_PROJECT_VAL"
    WANDB_GROUP="$jobname" AUTO_RESUBMIT=1)
  [ "$L" -le 5 ] && envs+=(CHUNK=2048)
  local jid; jid=$(do_sbatch "$jobname" nersc/submit_nqs_batch.sh "$tl" "0" \
    "${envs[@]}" WALLTIME="$tl")
  local role="chain_${branch}"
  for h in $anchor $links; do
    manifest_row "$jid" "$HY" "$cut" "$L" "$role" "$h" "$jobname" "$out_dir" "$(now)"
  done
}

# ---- drive every requested (cut, L) ----------------------------------------
n_electric=0; n_chain=0
for cut in $CUTS; do
  for L in $LS; do
    case "$cut" in
      electric_hx*)
        hx="${cut#electric_hx}"
        submit_electric "$cut" "$hx" "$L"
        n_electric=$((n_electric + 1))
        ;;
      magnetic_hz*)
        hz="${cut#magnetic_hz}"
        submit_chain "$cut" "$hz" "$L" up
        submit_chain "$cut" "$hz" "$L" dn
        n_chain=$((n_chain + 1))
        ;;
      *) echo "[launch] unknown cut id: $cut" >&2; exit 1 ;;
    esac
  done
done

echo "[launch] hy=$HY  cuts=[$CUTS]  L=[$LS]  DRYRUN=$DRYRUN MAX_QUEUE=$MAX_QUEUE"
echo "[launch] $n_electric electric (cut,L) cells x 7 pts, $n_chain magnetic/tail (cut,L) cells x 2 chains"
echo "[launch] $SUBMITTED_JOBS jobs submitted this run (queue was $CURRENT_Q before)."
echo "[launch] manifest: $MANIFEST"
