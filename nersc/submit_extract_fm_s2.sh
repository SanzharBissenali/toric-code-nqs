#!/bin/bash
# Batch wrapper: submit the combined O_FM(R=1) + Rényi-S₂ extraction for ONE
# (hx, L). Mirrors submit_extract_fm.sh but runs BOTH order parameters from the
# same checkpoints in a single job (see nersc/extract_fm_s2.sh). One L per job on
# purpose: the per-L JSON is written all-or-nothing, so stacking sizes risks a
# timeout losing them. Extraction is fast (L=4,5 <30 min; L=6,7 ~1 h -- compile-
# dominated, so FM+S₂ together is far from 2x).
#
#   HX=0.2 LS="6" sbatch --time=02:00:00 nersc/submit_extract_fm_s2.sh   # one (hx,L)
#   HX=0.2 LS="4 5 6 7" sbatch --time=06:00:00 nersc/submit_extract_fm_s2.sh  # per-hx style
#   squeue --me
#
# Usually launched via nersc/run_extract_campaign.sh (the whole hx x L grid).
# Env knobs (HX, HY, LS, SKIP_EXISTING, EVAL_SAMPLES, EVAL_CHAINS, BASE) pass straight
# through to extract_fm_s2.sh -> extract_fm.sh/extract_s2.sh (sbatch --export=ALL). HY
# (default 0.0) fixes the sign-full cut and tags both output filenames _hy${HY} when nonzero.
#SBATCH --job-name=tc-fms2extract
#SBATCH --account=m5340_g
#SBATCH --qos=shared
#SBATCH --constraint=gpu
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail

module load conda
conda activate tc-nqs

REPO="${REPO:-$HOME/toric-code-nqs}"
cd "$REPO" || { echo "[submit] REPO not found: $REPO — set REPO=<clone path>"; exit 1; }

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1          # else stdout progress block-buffers and the job looks hung

echo "[submit] fm+s2 extraction: HX=${HX:-0.2}  HY=${HY:-0.0}  LS=${LS:-<unset>}  SKIP_EXISTING=${SKIP_EXISTING:-0}"
bash nersc/extract_fm_s2.sh
