#!/bin/bash
# Combined O_FM (fixed R=1 loop) + Rényi-S₂ extraction for one hx cut. Runs BOTH
# order parameters from the SAME trained checkpoints in a single job -- they read
# the same $PSCRATCH/tc_nqs/phase_hx${HX}/L${L}/ tree, so one queue slot yields
# both curves per L and the outputs land together. Thin driver over extract_fm.sh
# + extract_s2.sh: it does NOT reimplement their per-L / loop-side logic, it fixes
# the FM loop to R=1 (the canonical `bulkR1` tag read by
# analysis/vertical_line_hz.ipynb) and calls each in turn. Run on a GPU node:
#
#   salloc -N 1 -C gpu --gpus 1 -q interactive -A m5340_g -t 02:00:00
#   module load conda && conda activate tc-nqs
#   HX=0.2 LS="4" bash nersc/extract_fm_s2.sh          # one L (usual: via submit wrapper)
#   HX=0.2 LS="4 5 6 7" bash nersc/extract_fm_s2.sh    # all L in one job (per-hx style)
#
# Both fm.py and renyi.py AUTO-DISCOVER the hz grid by globbing the checkpoint dir,
# so no hz range is needed here. Produces, per L, in $PSCRATCH/tc_nqs/phase_hx${HX}/:
#   fm_L${L}_hx${HX}_bulkR1.json     (O_FM, fixed R=1 perimeter-4 loop, xy/xz/yz avg)
#   s2_L${L}_hx${HX}_s2plaq.json     (S₂ of the central plaquette, xy/xz/yz avg)
# Pull each into its OWN local dir (results/phase_hx${HX}_bulkR1/ and
# results/phase_hx${HX}_s2plaq/) -- see CLAUDE.md; mixing tags double-counts an L.
#
# SKIP_EXISTING=1 skips an L whose output JSON already exists (idempotent top-up on
# re-run); default 0 = recompute/overwrite, matching extract_fm.sh/extract_s2.sh.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
HX="${HX:-0.2}"
LS="${LS:-}"
if [ -z "$LS" ]; then
  echo "[extract_fm_s2] set LS to the size(s) to extract, e.g.  HX=0.2 LS=4 bash nersc/extract_fm_s2.sh"
  exit 1
fi
export SKIP_EXISTING="${SKIP_EXISTING:-0}"

echo "[extract_fm_s2] hx=$HX  LS='$LS'  -> O_FM (R=1) + Rényi S₂  (SKIP_EXISTING=$SKIP_EXISTING)"
rc=0
echo "[extract_fm_s2] === O_FM (fixed R=1) ==="
if ! HX="$HX" LS="$LS" R=1 bash "$HERE/extract_fm.sh"; then
  echo "[extract_fm_s2] !! O_FM extraction FAILED for hx=$HX (continuing to S₂)"; rc=1
fi
echo "[extract_fm_s2] === Rényi S₂ ==="
if ! HX="$HX" LS="$LS" bash "$HERE/extract_s2.sh"; then
  echo "[extract_fm_s2] !! S₂ extraction FAILED for hx=$HX"; rc=1
fi
echo "[extract_fm_s2] done hx=$HX (exit $rc)."
exit $rc
