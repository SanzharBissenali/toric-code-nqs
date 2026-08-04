#!/bin/bash
# Magnetic O_FM ('t Hooft membrane) + Rényi-S₂ extraction for the hx-driven line:
# the fixed-hz run trees (phase_hz*/L*) are hx-SWEEPS, so this is the e↔m MIRROR of
# extract_fm_s2.sh (which is electric / hz-sweep). It calls tc3d.fm/renyi
# directly (not extract_fm.sh, which hardcodes --hx + electric) with:
#   fm.py    --sector magnetic --field hx  (off-diagonal σ^x cube membrane, telescoped)
#   renyi.py --field hx                     (central-plaquette S₂, sector-blind)
#
# Loop side R: aspect-½ where it fits the strict bulk (R in [1, L-3]), else clamped.
#   L=4 -> R=1 (aspect-½ = R2 leaves the bulk at L=4); L=5 -> R=2; L=6 -> 3; L=7 -> 4.
# Run under sbatch on a GPU node (see submit_extract_hxline.sh):
#   L=4 sbatch nersc/submit_extract_hxline.sh
#
# Per hz, writes into $PSCRATCH/tc_nqs/phase_hz${HZ}/:
#   fm_L${L}_hz${HZ}_memR${R}.json   (magnetic O_FM, R fixed, membrane orientations avg)
#   s2_L${L}_hz${HZ}_s2plaq.json     (central-plaquette S₂, xy/xz/yz avg)
# Pull each into its OWN local dir (results/phase_hz${HZ}_memR${R}/, .../s2plaq/) — mixing
# tags double-counts an L in the plot/FSS globs (see CLAUDE.md).
#
# SKIP_EXISTING=1 (default) makes a timeout-resubmit idempotent: an hz whose output
# already exists is skipped. Set 0 to force recompute.
set -euo pipefail

REPO="${REPO:-$HOME/toric-code-nqs}"
cd "$REPO"

HZS="${HZS:-0.0 0.1 0.2 0.3 0.4 0.5 0.7 0.9 1.0 1.1}"
L="${L:-4}"
EVAL_SAMPLES="${EVAL_SAMPLES:-8192}"
EVAL_CHAINS="${EVAL_CHAINS:-16}"   # long chains -> valid autocorr (GPU runs saved 1024)
PLANES="${PLANES:-xy,xz,yz}"
ASPECT="${ASPECT:-0.5}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

# R = clamp(round(L*aspect), 1, L-3): aspect-½ where the bulk holds it, else the
# largest bulk-fitting side. L=4 -> 1, L=5 -> 2, L=6 -> 3, L=7 -> 4.
R="${R:-$(python3 -c "print(max(1, min(round($L*$ASPECT), $L-3)))")}"

echo "[hxline] L=$L R=$R planes=$PLANES eval_samples=$EVAL_SAMPLES eval_chains=$EVAL_CHAINS"
echo "[hxline] hzs='$HZS'  SKIP_EXISTING=$SKIP_EXISTING"
rc=0
for HZ in $HZS; do
  DIR="$PSCRATCH/tc_nqs/phase_hz${HZ}/L${L}"
  if [ ! -d "$DIR" ]; then echo "[hxline] skip hz=$HZ (no $DIR)"; continue; fi
  FMOUT="$PSCRATCH/tc_nqs/phase_hz${HZ}/fm_L${L}_hz${HZ}_memR${R}.json"
  S2OUT="$PSCRATCH/tc_nqs/phase_hz${HZ}/s2_L${L}_hz${HZ}_s2plaq.json"

  if [ "$SKIP_EXISTING" = 1 ] && [ -f "$FMOUT" ]; then
    echo "[hxline] skip FM hz=$HZ (exists: $FMOUT)"
  else
    echo "[hxline] === FM (magnetic, R=$R) hz=$HZ -> $FMOUT ==="
    if ! python3 -u -m tc3d.fm --dir "$DIR" --L "$L" --sector magnetic --field hx \
        --R "$R" --placement bulk --planes "$PLANES" \
        --eval_samples "$EVAL_SAMPLES" --eval_chains "$EVAL_CHAINS" --out "$FMOUT"; then
      echo "[hxline] !! FM hz=$HZ FAILED"; rc=1
    fi
  fi

  if [ "$SKIP_EXISTING" = 1 ] && [ -f "$S2OUT" ]; then
    echo "[hxline] skip S2 hz=$HZ (exists: $S2OUT)"
  else
    echo "[hxline] === S₂ hz=$HZ -> $S2OUT ==="
    if ! python3 -u -m tc3d.renyi --dir "$DIR" --L "$L" --field hx \
        --planes "$PLANES" --eval_samples "$EVAL_SAMPLES" --eval_chains "$EVAL_CHAINS" \
        --out "$S2OUT"; then
      echo "[hxline] !! S₂ hz=$HZ FAILED"; rc=1
    fi
  fi
done
echo "[hxline] done (exit $rc). Pull: rsync -avz 'perlmutter:$PSCRATCH/tc_nqs/phase_hz*/fm_L${L}_hz*_memR${R}.json' results/xz_line_L${L}_memR${R}/"
exit $rc
