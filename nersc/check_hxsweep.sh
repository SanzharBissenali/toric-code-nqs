#!/bin/bash
# Convergence triage for the hx-sweeps (FIXED hz, varying hx; runs live under
# $PSCRATCH/tc_nqs/phase_hz${HZ}/L${L}/, cf. submit_nqs_hx_sweep.sh). Prints one
# per-hx table per L — final energy, energy spread, and Vscore of every run, with
# unphysical/diverged/blown-up runs flagged. Reads the run JSONs only (no NetKet /
# no weights), so it runs anywhere the JSONs are (login node or after an rsync pull).
#
#   HZ=0.0 LS="4 5 6 7" bash nersc/check_hxsweep.sh
#   HZ=0.0 LS=6 TRACE=1 bash nersc/check_hxsweep.sh   # + energy trace of flagged runs
#   BASE=./results VSCORE_MAX=1.0 bash nersc/check_hxsweep.sh   # on pulled-local JSONs
#
# Exits nonzero if ANY L has a flagged run (so it can gate the O_FM^m extraction).
set -uo pipefail

REPO="${REPO:-$HOME/toric-code-nqs}"
[ -d "$REPO" ] && cd "$REPO"

HZ="${HZ:-0.0}"
LS="${LS:-4 5 6 7}"
BASE="${BASE:-$PSCRATCH/tc_nqs/phase_hz${HZ}}"
VSCORE_MAX="${VSCORE_MAX:-1.0}"
TRACE_ARG=(); [ -n "${TRACE:-}" ] && TRACE_ARG=(--trace)

# check_convergence.py uses f-strings (Python >=3.6). A bare login node defaults to
# python2 -> SyntaxError; prefer python3. Override with PY=<interpreter>.
PY="${PY:-}"
if [ -z "$PY" ]; then
  command -v python3 >/dev/null 2>&1 && PY=python3 || PY=python
fi
if ! "$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3,6) else 1)' 2>/dev/null; then
  echo "[check] '$PY' is Python <3.6 (check_convergence.py needs f-strings). Either" >&2
  echo "        'module load conda && conda activate tc-nqs', or pass PY=<python3>." >&2
  exit 1
fi

bad=0
for L in $LS; do
  DIR="$BASE/L${L}"
  if [ ! -d "$DIR" ]; then echo "[check] skip L=$L (no $DIR)"; continue; fi
  echo "============================================================"
  # --field hx forces the hx column even if a dir happens to hold one run;
  # check_convergence.py auto-detects otherwise. Nonzero exit => this L has flags.
  "$PY" analysis/scripts/check_convergence.py --dir "$DIR" --L "$L" --field hx \
    --vscore-max "$VSCORE_MAX" ${TRACE_ARG[@]+"${TRACE_ARG[@]}"} || bad=1
done
echo "============================================================"
[ "$bad" -eq 0 ] && echo "[check] all L clean." || echo "[check] some L flagged (see '<-- CHECK')."
exit "$bad"
