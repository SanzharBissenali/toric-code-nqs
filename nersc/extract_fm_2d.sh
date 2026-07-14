#!/bin/bash
# Cluster-side 2D O_FM extraction: after the hz-sweep (electric) and/or hx-sweep
# (magnetic) arrays finish, compute the Fredenhagen-Marcu string order-parameter
# curve + transition fit for each L and dump a tiny JSON per L. The 2D analogue of
# nersc/extract_fm.sh, driving `python -m analysis.fm_2d` (model/ geometry, NetKet).
# Run on a GPU interactive node (fast; scores trained checkpoints, no training):
#
#   salloc -N 1 -C gpu --gpus 1 -q interactive -A m5340_g -t 02:00:00
#   module load conda && conda activate tc-nqs
#   SECTOR=electric LS="6 8 10 12" bash nersc/extract_fm_2d.sh   # hz cut, sigma^z string
#   SECTOR=magnetic LS="6 8 10 12" bash nersc/extract_fm_2d.sh   # hx cut, sigma^x dual string
#
# One tiny JSON per L: $BASE/fm2d_L${L}_${SECTOR}.json (arrays + fit, no weights).
# Pull each SECTOR into its OWN local dir (the notebook globs fm2d_L*_${SECTOR}.json):
#   rsync -avz '<host>:$BASE/fm2d_L*_electric.json' ./results/tc2d_electric/
#   rsync -avz '<host>:$BASE/fm2d_L*_magnetic.json' ./results/tc2d_magnetic/
set -euo pipefail

REPO="${REPO:-$HOME/threed_TC/ThreeD_TC}"
cd "$REPO"

SECTOR="${SECTOR:-electric}"        # electric = hz sweep (fix hx); magnetic = hx sweep (fix hz)
FIXED="${FIXED:-0.0}"               # value of the NON-swept field (the analytic cuts are at 0.0)
ARCH="${ARCH:-Combo}"
BC="${BC:-OBC}"
EVAL_SAMPLES="${EVAL_SAMPLES:-8192}"
# Chunk the eval forward pass at large L: vs.expect runs the Combo forward over all
# EVAL_SAMPLES at once, which OOMs a 40 GB GPU at L>=10 (N>=180) just like training's
# local-energy step did. Empty = auto (1024 at L>=10, none below); a number forces that
# chunk; 0 forces no chunk. If 1024 still OOMs, drop it (EVAL_CHUNK=512 or 256).
EVAL_CHUNK="${EVAL_CHUNK:-}"
LS="${LS:-}"                        # sizes to extract, e.g. LS="6 8 10 12"
if [ -z "$LS" ]; then
  echo "[extract2d] set LS to the sizes, e.g.  SECTOR=electric LS='6 8 10 12' bash nersc/extract_fm_2d.sh"
  exit 1
fi

# Where the training arrays wrote their checkpoints. electric (hz sweep) lives under
# phase_hx${FIXED}; magnetic (hx sweep) under phase_hz${FIXED}. Override BASE if your
# launch used a different tree.
if [ "$SECTOR" = "electric" ]; then
  BASE="${BASE:-$PSCRATCH/tc_nqs_2d/phase_hx${FIXED}}"
else
  BASE="${BASE:-$PSCRATCH/tc_nqs_2d/phase_hz${FIXED}}"
fi

for L in $LS; do
  DIR="$BASE/L${L}"
  OUT="$BASE/fm2d_L${L}_${SECTOR}.json"
  if [ ! -d "$DIR" ]; then echo "[extract2d] skip L=$L (no $DIR)"; continue; fi
  if [ "$L" -lt 4 ]; then
    echo "[extract2d] skip L=$L (bulk FM string needs L>=4, R=L-3>=1)"; continue
  fi
  # chunk the eval forward pass at large L (override with EVAL_CHUNK=<n>; 0 disables)
  if   [ "$EVAL_CHUNK" = "0" ]; then CHUNK_ARG=""                       # explicit: no chunk
  elif [ -n "$EVAL_CHUNK" ];    then CHUNK_ARG="--eval_chunk $EVAL_CHUNK"
  elif [ "$L" -ge 10 ];         then CHUNK_ARG="--eval_chunk 1024"      # auto for N>=180
  else                               CHUNK_ARG=""; fi
  echo "[extract2d] L=$L  sector=$SECTOR  fixed=$FIXED  ${CHUNK_ARG:-(no chunk)}  <- $DIR"
  python -u -m analysis.fm_2d --dir "$DIR" --L "$L" --sector "$SECTOR" \
    --fixed "$FIXED" --arch "$ARCH" --bc "$BC" \
    --eval_samples "$EVAL_SAMPLES" $CHUNK_ARG --out "$OUT"
done
echo "[extract2d] done. Pull: rsync -avz '<host>:$BASE/fm2d_L*_${SECTOR}.json' ./results/tc2d_${SECTOR}/"
