#!/bin/bash
# Cluster-side O_FM extraction: after an hz-sweep array finishes, compute the
# Fredenhagen-Marcu order-parameter curve + transition fit for each L and dump a
# tiny JSON per L. Run on a GPU interactive node (fast; needs NetKet):
#
#   salloc -N 1 -C gpu --gpus 1 -q interactive -A m5340_g -t 00:30:00
#   module load conda && conda activate tc-nqs
#   HX=0.2 LS="3 4 5 6" bash nersc/extract_fm.sh
#
# Produces $BASE/fm_L${L}_hx${HX}.json for each L (arrays + fit, no weights) —
# pull just these local for analysis/plot_phase_diagram.py.
set -euo pipefail

REPO="${REPO:-$HOME/threed_TC/ThreeD_TC}"
cd "$REPO"

HX="${HX:-0.2}"
LS="${LS:-}"                      # one L per job (see warning below); no all-L default
if [ -z "$LS" ]; then
  echo "[extract] set LS to the size to extract, one per job, e.g.  LS=5 bash nersc/extract_fm.sh"
  exit 1
fi
if [ "$(echo "$LS" | wc -w)" -gt 1 ]; then
  echo "[extract] WARNING: LS='$LS' stacks multiple sizes in one run. Each L is ~1-2h and the"
  echo "          extraction is all-or-nothing (the JSON is written only after all 16 points),"
  echo "          so one timeout loses the whole set. Strongly prefer one job per L." >&2
fi
SECTOR="${SECTOR:-electric}"     # electric = hz sweep (sigma^z loop)
EVAL_SAMPLES="${EVAL_SAMPLES:-8192}"
BASE="${BASE:-$PSCRATCH/tc_nqs/phase_hx${HX}}"

for L in $LS; do
  DIR="$BASE/L${L}"
  OUT="$BASE/fm_L${L}_hx${HX}.json"
  if [ ! -d "$DIR" ]; then echo "[extract] skip L=$L (no $DIR)"; continue; fi
  echo "[extract] L=$L  <- $DIR"
  python -m Three_TC.fm --dir "$DIR" --L "$L" --hx "$HX" \
    --sector "$SECTOR" --eval_samples "$EVAL_SAMPLES" --out "$OUT"
done
echo "[extract] done. Pull: rsync -avz <host>:$BASE/fm_L*_hx${HX}.json ./results/phase_hx${HX}/"
