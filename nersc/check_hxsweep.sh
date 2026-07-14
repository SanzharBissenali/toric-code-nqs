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

REPO="${REPO:-$HOME/threed_TC/ThreeD_TC}"
[ -d "$REPO" ] && cd "$REPO"

HZ="${HZ:-0.0}"
LS="${LS:-4 5 6 7}"
BASE="${BASE:-$PSCRATCH/tc_nqs/phase_hz${HZ}}"
VSCORE_MAX="${VSCORE_MAX:-1.0}"
TRACE_ARG=(); [ -n "${TRACE:-}" ] && TRACE_ARG=(--trace)

bad=0
for L in $LS; do
  DIR="$BASE/L${L}"
  if [ ! -d "$DIR" ]; then echo "[check] skip L=$L (no $DIR)"; continue; fi
  echo "============================================================"
  # --field hx forces the hx column even if a dir happens to hold one run;
  # check_convergence.py auto-detects otherwise. Nonzero exit => this L has flags.
  python analysis/check_convergence.py --dir "$DIR" --L "$L" --field hx \
    --vscore-max "$VSCORE_MAX" ${TRACE_ARG[@]+"${TRACE_ARG[@]}"} || bad=1
done
echo "============================================================"
[ "$bad" -eq 0 ] && echo "[check] all L clean." || echo "[check] some L flagged (see '<-- CHECK')."
exit "$bad"
