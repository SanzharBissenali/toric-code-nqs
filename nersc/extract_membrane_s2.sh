#!/bin/bash
# Combined O_FM^m ('t Hooft MEMBRANE) + Rényi-S₂ extraction for the h_x-sweep at
# fixed h_z. The magnetic (m-condensation) mirror of nersc/extract_fm_s2.sh: it
# reads the hx-sweep checkpoints ($PSCRATCH/tc_nqs/phase_hz${HZ}/L${L}/, from
# submit_nqs_hx_sweep.sh -- hz FIXED, hx swept) and emits BOTH order parameters
# per L from the SAME wavefunctions in one job. Run on a GPU node:
#
#   salloc -N 1 -C gpu --gpus 1 -q interactive -A m5340_g -t 02:00:00
#   module load conda && conda activate tc-nqs
#   HZ=0.0 LS="5" bash nersc/extract_membrane_s2.sh        # one L (usual)
#   HZ=0.0 LS="4 5 6 7" bash nersc/extract_membrane_s2.sh  # all L in one job
#
# Both fm.py and renyi.py AUTO-DISCOVER the hx grid by globbing the dir (no hx range
# needed); --field hx + no --hx = sweep over ALL hx cuts in the dir. Produces per L,
# in $PSCRATCH/tc_nqs/phase_hz${HZ}/:
#   fm_L${L}_hz${HZ}_memA0.5.json   (O_FM^m, aspect-½ cube membrane, v0/v1/v2 avg,
#                                    telescoped estimator + B3 health; SECTOR=magnetic)
#   s2_L${L}_hz${HZ}_s2plaq.json    (S₂ of the central plaquette, xy/xz/yz avg)
# S₂ is sector-blind (its dS₂/dh_x inflection locates h_c independently); the membrane
# is the m-condensation order parameter (rises 0->1). Along h_x the electric string
# O_FM^e is the NULL test (≈0 throughout) -- extract it separately with SECTOR=electric
# if you want the null on the same tree.
#
# Pull each tag into its OWN local dir (see CLAUDE.md -- mixing tags double-counts an L):
#   results/phase_hz${HZ}_memA0.5/   and   results/phase_hz${HZ}_s2plaq/
# consumed by the hx-sweep analysis notebook (the horizontal-line mirror of
# analysis/vertical_line_hz.ipynb).
#
# NOTE: the aspect-½ cube membrane EXCLUDES L=4 (R=⌊L/2⌋=2 > L-3=1 leaves the bulk),
# so L=4 gets S₂ only; the membrane runs for L>=5. SKIP_EXISTING=1 skips an L whose
# output JSON already exists (idempotent top-up); default 0 = recompute/overwrite.
set -euo pipefail

REPO="${REPO:-$HOME/toric-code-nqs}"
cd "$REPO"

HZ="${HZ:-0.0}"
LS="${LS:-}"
if [ -z "$LS" ]; then
  echo "[memb_s2] set LS to the size(s) to extract, e.g.  HZ=0.0 LS=5 bash nersc/extract_membrane_s2.sh"
  exit 1
fi
if [ "$(echo "$LS" | wc -w)" -gt 1 ]; then
  echo "[memb_s2] WARNING: LS='$LS' stacks multiple sizes in one run. Each L is ~1-2h and each"
  echo "          extraction is all-or-nothing (JSON written only at the end), so one timeout"
  echo "          loses that L. Prefer one job per L (the submit wrapper does this)." >&2
fi

EVAL_SAMPLES="${EVAL_SAMPLES:-8192}"
EVAL_CHAINS="${EVAL_CHAINS:-16}"   # long chains -> valid error_of_mean (GPU runs saved 1024)
PLANES="${PLANES:-xy,xz,yz}"       # S₂ central-plaquette orientations to average
R="${R:-}"                          # override membrane cube side (int); empty = aspect-½ ⌊L/2⌋
BASE="${BASE:-$PSCRATCH/tc_nqs/phase_hz${HZ}}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
MEM_TAG="memA0.5"; [ -n "$R" ] && MEM_TAG="memR${R}"
S2_TAG="s2plaq"

echo "[memb_s2] hz=$HZ  LS='$LS'  -> O_FM^m (membrane) + Rényi S₂  (SKIP_EXISTING=$SKIP_EXISTING)"
rc=0
for L in $LS; do
  DIR="$BASE/L${L}"
  if [ ! -d "$DIR" ]; then echo "[memb_s2] skip L=$L (no $DIR)"; continue; fi
  if [ "$L" -lt 4 ]; then echo "[memb_s2] skip L=$L (bulk patch/membrane need L>=4)"; continue; fi

  # --- O_FM^m: cube membrane, magnetic sector, sweep hx (no --hx = match all) ---
  MEM_OUT="$BASE/fm_L${L}_hz${HZ}_${MEM_TAG}.json"
  RARG=(); [ -n "$R" ] && RARG=(--R "$R")
  if [ "$L" -lt 5 ] && [ -z "$R" ]; then
    echo "[memb_s2] L=$L: skip membrane (aspect-½ excludes L=4; set R=<int> to force a smaller cube)"
  elif [ "$SKIP_EXISTING" = "1" ] && [ -f "$MEM_OUT" ]; then
    echo "[memb_s2] L=$L: skip membrane (exists: $MEM_OUT)"
  else
    echo "[memb_s2] === L=$L  O_FM^m (membrane, ${MEM_TAG}) <- $DIR ==="
    if ! python -u -m tc3d.fm --dir "$DIR" --L "$L" --sector magnetic \
        --field hx --bc OBC --placement bulk \
        --eval_samples "$EVAL_SAMPLES" --eval_chains "$EVAL_CHAINS" \
        "${RARG[@]+"${RARG[@]}"}" --out "$MEM_OUT"; then
      echo "[memb_s2] !! membrane extraction FAILED for L=$L (continuing)"; rc=1
    fi
  fi

  # --- S₂: central plaquette, sweep hx ---
  S2_OUT="$BASE/s2_L${L}_hz${HZ}_${S2_TAG}.json"
  if [ "$SKIP_EXISTING" = "1" ] && [ -f "$S2_OUT" ]; then
    echo "[memb_s2] L=$L: skip S₂ (exists: $S2_OUT)"
  else
    echo "[memb_s2] === L=$L  Rényi S₂ (${S2_TAG}) <- $DIR ==="
    if ! python -u -m tc3d.renyi --dir "$DIR" --L "$L" --field hx \
        --eval_samples "$EVAL_SAMPLES" --eval_chains "$EVAL_CHAINS" \
        --planes "$PLANES" --out "$S2_OUT"; then
      echo "[memb_s2] !! S₂ extraction FAILED for L=$L (continuing)"; rc=1
    fi
  fi
done
echo "[memb_s2] done (exit $rc). Pull:"
echo "  rsync -avz <host>:$BASE/fm_L*_hz${HZ}_${MEM_TAG}.json ./results/phase_hz${HZ}_${MEM_TAG}/"
echo "  rsync -avz <host>:$BASE/s2_L*_hz${HZ}_${S2_TAG}.json  ./results/phase_hz${HZ}_${S2_TAG}/"
exit $rc
