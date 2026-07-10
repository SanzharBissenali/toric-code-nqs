#!/bin/bash
# Cluster-side O_FM extraction: after an hz-sweep array finishes, compute the
# Fredenhagen-Marcu order-parameter curve + transition fit for each L and dump a
# tiny JSON per L. Run on a GPU interactive node (fast; needs NetKet):
#
#   salloc -N 1 -C gpu --gpus 1 -q interactive -A m5340_g -t 00:30:00
#   module load conda && conda activate tc-nqs
#   HX=0.2 LS="4 5 6" bash nersc/extract_fm.sh
#
# PLACEMENT=bulk (default): largest bulk-centered square, averaged over the xy/xz/yz
# planes (needs L>=4; L<=3 are skipped). PLACEMENT=boundary: the legacy z=0 largest loop
# (reproduces the old fm_L*_hx*.json curves at any L).
#
# Produces $BASE/fm_L${L}_hx${HX}_${PLACEMENT}.json for each L (arrays + fit, no weights).
# Pull each placement into its OWN local dir (analysis/plot_phase_diagram.py globs fm_L*.json,
# so mixing placements in one dir would double-count an L).
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
PLACEMENT="${PLACEMENT:-bulk}"   # bulk = bulk-centered square, xy/xz/yz averaged; boundary = legacy
PLANES="${PLANES:-xy,xz,yz}"     # orientations to average (bulk only)
R="${R:-}"                       # bulk loop side: empty = largest (L-3, grows with L);
                                 # set (e.g. R=1 = perimeter-4 plaquette) for a size-
                                 # independent operator -> output suffixed R${R}
BASE="${BASE:-$PSCRATCH/tc_nqs/phase_hx${HX}}"

# Fixed-R curves get their OWN filename tag so they never mix with the L-3 curves
# (plot/FSS globs fm_L*.json; sharing a suffix would double-count an L).
TAG="$PLACEMENT"; RARG=()
if [ -n "$R" ]; then TAG="${PLACEMENT}R${R}"; RARG=(--R "$R"); fi

for L in $LS; do
  DIR="$BASE/L${L}"
  OUT="$BASE/fm_L${L}_hx${HX}_${TAG}.json"
  if [ ! -d "$DIR" ]; then echo "[extract] skip L=$L (no $DIR)"; continue; fi
  if [ "$PLACEMENT" = "bulk" ] && [ "$L" -lt 4 ]; then
    echo "[extract] skip L=$L (bulk-centered loop needs L>=4; use PLACEMENT=boundary for small L)"
    continue
  fi
  echo "[extract] L=$L  placement=$PLACEMENT${R:+ R=$R}  <- $DIR"
  python -u -m Three_TC.fm --dir "$DIR" --L "$L" --hx "$HX" \
    --sector "$SECTOR" --eval_samples "$EVAL_SAMPLES" \
    --placement "$PLACEMENT" --planes "$PLANES" "${RARG[@]}" --out "$OUT"
done
echo "[extract] done. Pull: rsync -avz <host>:$BASE/fm_L*_hx${HX}_${TAG}.json ./results/phase_hx${HX}_${TAG}/"
