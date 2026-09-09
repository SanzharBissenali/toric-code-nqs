#!/bin/bash
# Generalized phase3d campaign watcher (from ~/hy_cuts_watch.sh, which only
# grepped sacct + log markers): reads a launch_phase3d.sh manifest TSV, cross-
# references sacct state + the per-run slurm log (GENUINE DIVERGENCE, "warm
# start: loaded" per chain link) + the per-run final JSON (E0 vs the h=0 OBC
# bound), and writes $BASE_OUT/watch_state.json:
#   {run_name: {state, diverged, warm_loaded, E0, bound, above_bound, last_step, log}}
# Prints a one-line-per-flag summary; exits nonzero if anything is flagged.
#
# Run ON PERLMUTTER (needs sacct + the slurm log files on disk), from the repo
# root (gridinv logs land in cwd; batch logs in slurm_logs/):
#   MANIFEST=$PSCRATCH/tc_nqs/phase3d/manifests/manifest_*.tsv bash nersc/watch_phase3d.sh
#
# LOCAL=1 skips sacct/cluster lookups entirely -- job state comes only from
# SACCT_FILE (a pre-baked `sacct -X -P --format=JobID,State` dump) if given,
# else every job is UNKNOWN. For offline testing against a hand-built fake
# manifest + fake log directory; runs anywhere (no cluster access needed).
set -uo pipefail

MANIFEST="${MANIFEST:?set MANIFEST=<path to a launch_phase3d.sh manifest tsv>}"
[ -f "$MANIFEST" ] || { echo "[watch] not found: $MANIFEST" >&2; exit 1; }
BASE_OUT="${BASE_OUT:-$(cd "$(dirname "$MANIFEST")/.." && pwd)}"   # manifests/.. -> phase3d/
LOGDIR="${LOGDIR:-slurm_logs}"
LOCAL="${LOCAL:-0}"
SACCT_FILE="${SACCT_FILE:-}"     # LOCAL=1: pre-baked sacct dump (JobID|State lines)
PY="${PY:-python3}"; command -v "$PY" >/dev/null 2>&1 || PY=python

JOBIDS=$(tail -n +2 "$MANIFEST" | awk -F'\t' '{print $1}' | sort -u \
         | grep -v '^DRY$\|^SKIPPED$' || true)

STATE_FILE=$(mktemp)
FOLLOWUP_FILE=$(mktemp)
: > "$FOLLOWUP_FILE"
if [ "$LOCAL" = "1" ]; then
  [ -n "$SACCT_FILE" ] && cp "$SACCT_FILE" "$STATE_FILE" || : > "$STATE_FILE"
else
  if [ -n "$JOBIDS" ]; then
    JOBIDS_CSV=$(echo "$JOBIDS" | tr '\n' ',' | sed 's/,$//')
    sacct -X -j "$JOBIDS_CSV" --format=JobID,State%30 -n -P > "$STATE_FILE" 2>/dev/null || : > "$STATE_FILE"
  else
    : > "$STATE_FILE"
  fi
  # TIMEOUT-without-a-follow-up: for every TIMEOUT'd job, search by NAME (not
  # jobid -- AUTO_RESUBMIT's requeue is a brand-new jobid) for a later Submit.
  while IFS='|' read -r jid st; do
    [ "$st" = "TIMEOUT" ] || continue
    name=$(awk -F'\t' -v j="$jid" '$1==j{print $7; exit}' "$MANIFEST")
    [ -n "$name" ] || continue
    later=$(sacct -u "$USER" --name="$name" -h -P --format=JobID,Submit 2>/dev/null \
            | awk -F'|' -v me="$jid" '$1!=me' | wc -l | tr -d ' ')
    printf "%s\t%s\n" "$jid" "$later" >> "$FOLLOWUP_FILE"
  done < "$STATE_FILE"
fi

mkdir -p "$BASE_OUT" 2>/dev/null || true
WATCH_JSON="$BASE_OUT/watch_state.json"

"$PY" - "$MANIFEST" "$STATE_FILE" "$FOLLOWUP_FILE" "$LOGDIR" "$WATCH_JSON" <<'PYEOF'
import sys, os, csv, json, glob, math
from collections import Counter

manifest, state_file, followup_file, logdir, out_json = sys.argv[1:6]

BAD_STATES = ("FAILED", "TIMEOUT", "OUT_OF", "NODE_FAIL", "CANCELLED", "DEADLINE",
              "DependencyNeverSatisfied")

def anchor_obc(L):
    return -(L ** 3 + 3 * (L - 1) ** 2 * L)

def load_pairs(path):
    d = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|") if "|" in line else line.split("\t")
            if len(parts) >= 2:
                d[parts[0]] = parts[1]
    return d

states = load_pairs(state_file)
followup_count = load_pairs(followup_file)     # jobid -> "N later jobs with this name"

_CHAIN_ANCHORS = {0.0: (0.6, 1.25), 0.1: (0.6, 1.25), 0.2: (0.6, 1.25),
                  0.4: (0.5, 1.3), 0.7: (0.6, 1.5), 1.0: (0.8, 1.7)}

def run_name(row):
    """Reconstruct tc3d's own {name}.json basename -- mirrors submit_nqs_gridinv.sh's
    NAME auto-construction (electric, and the L5/6 chain ANCHOR -- a separate cold
    gridinv job with no branch suffix, since the A3 planner redesign) / the
    NAME_TEMPLATE this campaign submits for chain LINK jobs, given the mandatory
    DUAL=1 NONINV_HIDDEN="4 8" INV="8 8" KERNEL=L-1 arch env on every job (see
    nersc/launch_phase3d.sh)."""
    L, hy, h, role, cut = row["L"], row["hy"], row["h"], row["role"], row["cut"]
    kernel = int(L) - 1
    if cut.startswith("electric_hx"):
        hx, hz = cut[len("electric_hx"):], h
        hy_tag = "" if float(hy) == 0.0 else f"_hy{hy}"
        return f"gridinv_dual_L{L}_OBC_hx{hx}_hz{hz}{hy_tag}_n2x4_nh4-8_inv8-8_k{kernel}"
    hz, hx = cut[len("magnetic_hz"):], h
    branch = "up" if role == "chain_up" else "dn"
    lo, hi = _CHAIN_ANCHORS.get(round(float(hz), 4), (None, None))
    anchor = lo if branch == "up" else hi
    is_anchor_pt = anchor is not None and abs(float(hx) - anchor) < 1e-9
    if int(L) >= 5 and is_anchor_pt:      # separate cold job, plain (unsuffixed) name
        hy_tag = "" if float(hy) == 0.0 else f"_hy{hy}"
        return f"gridinv_dual_L{L}_OBC_hx{hx}_hz{hz}{hy_tag}_n2x4_nh4-8_inv8-8_k{kernel}"
    return f"gridinv_dual_L{L}_OBC_hx{hx}_hz{hz}_hy{hy}_n2x4_nh4-8_inv8-8_k{kernel}_{branch}"

def find_log(jobname, jobid, role):
    # gridinv (cold electric): %x-%j.out in the submission cwd (repo root).
    # batch (chains): slurm_logs/%x-%A_%a.out, single-task array -> _0.
    pats = ([f"{jobname}-{jobid}.out"] if role == "cold" else
             [os.path.join(logdir, f"{jobname}-{jobid}_0.out")])
    pats += [f"{jobname}-{jobid}.out", os.path.join(logdir, f"{jobname}-{jobid}_0.out")]
    for pat in pats:
        hits = glob.glob(pat)
        if hits:
            return hits[0]
    return None

rows = list(csv.DictReader(open(manifest), delimiter="\t"))
pts_per_job = Counter(r["jobid"] for r in rows)

job_log_cache = {}
def job_log_facts(jobid, jobname, role, n_points):
    key = jobid
    if key in job_log_cache:
        return job_log_cache[key]
    log = find_log(jobname, jobid, role)
    facts = {"log": log, "diverged_log": False, "warm_ok": None, "n_warm": 0}
    if log and os.path.exists(log):
        txt = open(log, errors="replace").read()
        facts["diverged_log"] = "GENUINE DIVERGENCE" in txt
        facts["n_warm"] = txt.count("warm start: loaded")
        if n_points > 1:      # chain job: every link but the anchor must warm-load
            facts["warm_ok"] = facts["n_warm"] >= (n_points - 1)
    job_log_cache[key] = facts
    return facts

out = {}
for row in rows:
    jobid, name = row["jobid"], row["name"]
    state = states.get(jobid, jobid if jobid in ("DRY", "SKIPPED") else "UNKNOWN")
    if state == "TIMEOUT" and followup_count.get(jobid, "0") == "0":
        state = "TIMEOUT_NO_FOLLOWUP"
    facts = job_log_facts(jobid, name, row["role"], pts_per_job[jobid])
    rn = run_name(row)
    jpath = os.path.join(row["out_dir"], f"{rn}.json")
    E0 = last_step = None
    diverged = facts["diverged_log"]
    bound = anchor_obc(int(row["L"]))
    if os.path.exists(jpath):
        try:
            d = json.load(open(jpath))
            diverged = bool(d.get("diverged", diverged))
            E0 = (d.get("observables") or {}).get("E0")
            steps = (d.get("curve") or {}).get("step")
            last_step = steps[-1] if steps else d.get("completed_steps")
        except (OSError, json.JSONDecodeError):
            pass
    above_bound = bool(E0 is not None and math.isfinite(E0) and E0 > bound)
    warm_loaded = None if row["role"] == "cold" else facts["warm_ok"]
    out[rn] = {"state": state, "diverged": diverged, "warm_loaded": warm_loaded,
               "E0": E0, "bound": bound, "above_bound": above_bound,
               "last_step": last_step, "log": facts["log"]}

with open(out_json, "w") as f:
    json.dump(out, f, indent=1)

def flagged(v):
    return (v["diverged"] or v["above_bound"] or v["warm_loaded"] is False
            or any(b in (v["state"] or "") for b in BAD_STATES))

n_flag = sum(1 for v in out.values() if flagged(v))
print(f"[watch] {len(out)} runs across {len(pts_per_job)} jobs -> {out_json}")
print(f"[watch] flagged: {n_flag}")
for rn in sorted(out):
    v = out[rn]
    if flagged(v):
        print(f"  ! {rn}: state={v['state']} diverged={v['diverged']} "
              f"above_bound={v['above_bound']} (E0={v['E0']} bound={v['bound']}) "
              f"warm_loaded={v['warm_loaded']}")
sys.exit(1 if n_flag else 0)
PYEOF
status=$?
rm -f "$STATE_FILE" "$FOLLOWUP_FILE"
exit "$status"
