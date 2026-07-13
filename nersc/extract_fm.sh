#!/bin/bash
# Cluster-side O_FM extraction: after an hz-sweep array finishes, compute the
# Fredenhagen-Marcu order-parameter curve + transition fit for each L and dump a
# tiny JSON per L. Run on a GPU interactive node (fast; needs NetKet):
#
#   salloc -N 1 -C gpu --gpus 1 -q interactive -A m5340_g -t 00:30:00
#   module load conda && conda activate tc-nqs
#   HX=0.2 LS="4 5 6" bash nersc/extract_fm.sh
#
# PLACEMENT=bulk (default): bulk-centered square, averaged over the xy/xz/yz planes
# (needs L>=4; L<=3 are skipped). PLACEMENT=boundary: the legacy z=0 largest loop
# (reproduces the old fm_L*_hx*.json curves at any L).
#
# Loop side (bulk only) -- pick ONE:
#   (default)   largest-in-bulk, R=L-3 (aspect ratio R/L drifts -> 1 with L).
#   R=<int>     a fixed side at every L (e.g. R=1 = perimeter-4 plaquette).
#   R_FRAC=<f>  a fixed ASPECT RATIO R/L=f: R=round(L*f) per L (e.g. 0.5 = L/2). This
#               is the clean FSS choice -- constant R/L keeps different L self-similar
#               and centers the loop away from the OBC surface. L too small to host it
#               in the bulk (need R<=L-3, e.g. L=4 at f=0.5) are skipped. R_FRAC wins over R.
#
# Produces $BASE/fm_L${L}_hx${HX}_${TAG}.json for each L (arrays + fit, no weights), where
# TAG encodes the loop choice (bulk / bulkR${R} / bulkRf${R_FRAC}). Pull each TAG into its
# OWN local dir (analysis/plot_phase_diagram.py globs fm_L*.json, so mixing would double-count).
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
EVAL_CHAINS="${EVAL_CHAINS:-16}" # override n_chains at eval: GPU runs saved 1024
                                 # (~8 samples/chain -> NaN autocorr error); 16 = long
                                 # chains so error_of_mean is valid. Set empty to keep saved.
PLACEMENT="${PLACEMENT:-bulk}"   # bulk = bulk-centered square, xy/xz/yz averaged; boundary = legacy
PLANES="${PLANES:-xy,xz,yz}"     # orientations to average (bulk only)
R="${R:-}"                       # fixed bulk loop side (int); empty = largest (L-3)
R_FRAC="${R_FRAC:-}"             # fixed aspect ratio R/L: R=round(L*R_FRAC) per L; wins over R
BASE="${BASE:-$PSCRATCH/tc_nqs/phase_hx${HX}}"

# Each loop choice gets its OWN filename tag so curves never mix in a glob (plot/FSS
# globs fm_L*.json; sharing a tag would double-count an L). TAG is fixed across L.
TAG="$PLACEMENT"
if   [ -n "$R_FRAC" ]; then TAG="${PLACEMENT}Rf${R_FRAC}"
elif [ -n "$R" ];      then TAG="${PLACEMENT}R${R}"; fi
CARG=(); [ -n "$EVAL_CHAINS" ] && CARG=(--eval_chains "$EVAL_CHAINS")

for L in $LS; do
  DIR="$BASE/L${L}"
  OUT="$BASE/fm_L${L}_hx${HX}_${TAG}.json"
  if [ ! -d "$DIR" ]; then echo "[extract] skip L=$L (no $DIR)"; continue; fi
  if [ "$PLACEMENT" = "bulk" ] && [ "$L" -lt 4 ]; then
    echo "[extract] skip L=$L (bulk-centered loop needs L>=4; use PLACEMENT=boundary for small L)"
    continue
  fi
  # resolve this L's loop side (R_FRAC -> round(L*frac); else fixed R; else default L-3)
  Rside=""; RARG=()
  if   [ -n "$R_FRAC" ]; then Rside=$(python -c "print(int(round($L*$R_FRAC)))")
  elif [ -n "$R" ];      then Rside="$R"; fi
  if [ -n "$Rside" ]; then
    if [ "$Rside" -gt "$((L - 3))" ]; then
      echo "[extract] skip L=$L (R=$Rside > L-3=$((L - 3)); loop leaves the bulk at this aspect ratio)"
      continue
    fi
    RARG=(--R "$Rside")
  fi
  echo "[extract] L=$L  placement=$PLACEMENT${Rside:+ R=$Rside (R/L=$(python -c "print(f'{$Rside/$L:.2f}')"))}${EVAL_CHAINS:+ n_chains=$EVAL_CHAINS}  <- $DIR"
  python -u -m Three_TC.fm --dir "$DIR" --L "$L" --hx "$HX" \
    --sector "$SECTOR" --eval_samples "$EVAL_SAMPLES" "${CARG[@]}" \
    --placement "$PLACEMENT" --planes "$PLANES" "${RARG[@]}" --out "$OUT"
done
echo "[extract] done. Pull: rsync -avz <host>:$BASE/fm_L*_hx${HX}_${TAG}.json ./results/phase_hx${HX}_${TAG}/"
