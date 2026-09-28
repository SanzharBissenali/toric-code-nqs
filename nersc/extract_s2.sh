#!/bin/bash
# Cluster-side S₂ extraction: after an hz-sweep array finishes, compute the second
# Rényi entropy S₂(A) = −ln Tr(ρ_A²) of the bulk-centered 4-qubit plaquette patch
# for each L and dump a tiny JSON per L. An INDEPENDENT, local transition locator to
# cross-check the Fredenhagen–Marcu pipeline (nersc/extract_fm.sh). Run on a GPU
# interactive node (fast; needs NetKet):
#
#   salloc -N 1 -C gpu --gpus 1 -q interactive -A m5340_g -t 00:30:00
#   module load conda && conda activate tc-nqs
#   HX=0.0 LS="4" bash nersc/extract_s2.sh          # one L per job (see warning below)
#
# Patch = one central unit plaquette (4 coplanar edges = a single B_p), averaged over
# the xy/xz/yz orientations on the same wavefunction (PLANES). It is centered and
# strictly interior for L>=4 (needs L>=4; L<=3 are skipped). S₂ is a LOCAL,
# susceptibility-like transition locator (its dS₂/dhz peaks at h_c(L)) — NOT a TEE /
# topological diagnostic. Exact limits: S₂=3ln2 at hz=0 (stabilizer GS), 0 at hz→∞.
#
# HY=<float> (default 0.0) fixes the sign-full cut (tc3d.renyi --hy); mirrors extract_fm.sh
# -- a dir holding several hy populations (e.g. +hy campaign + its -hy TR-pair runs sharing
# one OUT_DIR by design) needs one call per HY. Nonzero HY appends _hy${HY} to the filename.
#
# Produces $BASE/s2_L${L}_hx${HX}${HY_TAG}_s2plaq.json for each L (raw S₂ curve; the peak
# extraction + FSS live in the archived templates _archive/analysis_archive/{vertical_line_hz,xz_cut}.ipynb). Pull into its OWN local dir
# results/phase_hx${HX}${HY_TAG}_s2plaq/ (a distinct TAG so it never mixes with FM curves).
set -euo pipefail

REPO="${REPO:-$HOME/toric-code-nqs}"
cd "$REPO"

HX="${HX:-0.0}"
HY="${HY:-0.0}"                   # fix the sign-full cut (tc3d.renyi --hy); NOT a sweepable field
LS="${LS:-}"                      # one L per job (see warning below); no all-L default
if [ -z "$LS" ]; then
  echo "[extract_s2] set LS to the size to extract, one per job, e.g.  LS=4 bash nersc/extract_s2.sh"
  exit 1
fi
if [ "$(echo "$LS" | wc -w)" -gt 1 ]; then
  echo "[extract_s2] WARNING: LS='$LS' stacks multiple sizes in one run. Each L is ~1-2h and the"
  echo "             extraction is all-or-nothing (the JSON is written only after all points),"
  echo "             so one timeout loses the whole set. Strongly prefer one job per L." >&2
fi
EVAL_SAMPLES="${EVAL_SAMPLES:-8192}"
EVAL_CHAINS="${EVAL_CHAINS:-16}" # override n_chains at eval: GPU runs saved 1024
                                 # (~8 samples/chain -> NaN autocorr error); 16 = long chains.
PLANES="${PLANES:-xy,xz,yz}"     # central-plaquette orientations to average
BASE="${BASE:-$PSCRATCH/tc_nqs/phase_hx${HX}}"
TAG="s2plaq"
HY_TAG=""; [ "$HY" != "0.0" ] && HY_TAG="_hy${HY}"

for L in $LS; do
  DIR="$BASE/L${L}"
  OUT="$BASE/s2_L${L}_hx${HX}${HY_TAG}_${TAG}.json"
  if [ "${SKIP_EXISTING:-0}" = "1" ] && [ -f "$OUT" ]; then
    echo "[extract_s2] skip L=$L (exists: $OUT; SKIP_EXISTING=1)"; continue; fi
  if [ ! -d "$DIR" ]; then echo "[extract_s2] skip L=$L (no $DIR)"; continue; fi
  if [ "$L" -lt 4 ]; then
    echo "[extract_s2] skip L=$L (bulk-centered plaquette patch needs L>=4)"
    continue
  fi
  echo "[extract_s2] L=$L  hx=$HX  hy=$HY  planes=$PLANES  n_chains=$EVAL_CHAINS  <- $DIR"
  python -u -m tc3d.renyi --dir "$DIR" --L "$L" --hx "$HX" --hy "$HY" \
    --eval_samples "$EVAL_SAMPLES" --eval_chains "$EVAL_CHAINS" \
    --planes "$PLANES" --out "$OUT"
done
echo "[extract_s2] done. Pull: rsync -avz <host>:$BASE/s2_L*_hx${HX}${HY_TAG}_${TAG}.json ./results/phase_hx${HX}${HY_TAG}_${TAG}/"
