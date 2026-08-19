#!/bin/bash
# Cluster-side ENERGY extraction for the energy-kink diagnostic (1st vs 2nd order).
# Reads the final energy + conjugate magnetization from each run JSON and dumps a
# compact per-L curve. PURE JSON reads (no NetKet) -> run on a LOGIN NODE, no GPU:
#
#   ssh sanzharb@perlmutter.nersc.gov          # NERSC user is sanzharb, not sanzhar123
#   cd $HOME/toric-code-nqs                # (= $REPO)
#   # second-order lines (hz-sweep at fixed hx):
#   for HX in 0.0 0.2 0.4 0.6 0.8 1.0; do HX=$HX LS="4 5 6 7" bash nersc/extract_energy.sh; done
#   # first-order line (hx-sweep at fixed hz=0):
#   SWEEP=hz0.0 LS="4 5 6 7" bash nersc/extract_energy.sh
#
# Produces $BASE/energy_L${L}_${SWEEP}.json. Pull each family into its own local dir:
#   rsync -avz '<host>:$PSCRATCH/tc_nqs/phase_hx${HX}/energy_L*_hx${HX}.json' results/energy_hx${HX}/
#   rsync -avz '<host>:$PSCRATCH/tc_nqs/phase_hz0.0/energy_L*_hz0.0.json'     results/energy_hz0.0/
set -uo pipefail   # NOT -e: check_convergence exits 1 when a point is flagged; that's fine

REPO="${REPO:-$HOME/toric-code-nqs}"
cd "$REPO"

LS="${LS:-}"
if [ -z "$LS" ]; then echo "[energy] set LS, e.g.  LS='4 5 6 7'"; exit 1; fi

# SWEEP picks the run tree + output tag. Default: an hx-sweep of the hz-sweep family.
if [ -n "${SWEEP:-}" ]; then
  BASE="${BASE:-$PSCRATCH/tc_nqs/phase_${SWEEP}}"; TAG="$SWEEP"
else
  HX="${HX:-0.2}"; BASE="${BASE:-$PSCRATCH/tc_nqs/phase_hx${HX}}"; TAG="hx${HX}"
fi

for L in $LS; do
  DIR="$BASE/L${L}"
  OUT="$BASE/energy_L${L}_${TAG}.json"
  if [ ! -d "$DIR" ]; then echo "[energy] skip L=$L (no $DIR)"; continue; fi
  echo "[energy] L=$L  <- $DIR"
  python -u analysis/scripts/check_convergence.py --dir "$DIR" --L "$L" --dump "$OUT" || true
done
echo "[energy] done. Pull: rsync -avz '<host>:$BASE/energy_L*_${TAG}.json' results/energy_${TAG}/"
