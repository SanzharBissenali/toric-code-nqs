#!/bin/bash
# Batch wrapper for the O_FM (Fredenhagen-Marcu) extraction — the string/loop
# order-parameter sweep. It rebuilds each trained NQS, reloads weights, and
# evaluates <S>/sqrt(|<W>|) + <sz> over `eval_samples` per (L, hz); at L>=5 this
# runs ~2 h for the set, longer than a 1 h interactive node allows. So submit it:
#
#   LS="5" sbatch nersc/submit_extract_fm.sh                # ONE L per job (LS required)
#   LS="6" sbatch --time=05:00:00 nersc/submit_extract_fm.sh
#   squeue --me
#
# One L per job on purpose: each L is ~1-2h and the extraction is all-or-nothing
# (JSON written only at the end), so stacking sizes risks a timeout losing all of
# them. extract_fm.sh errors if LS is unset and warns if given more than one L.
#
# Writes $PSCRATCH/tc_nqs/phase_hx<HX>/fm_L<L>_hx<HX>.json (one per L); pull those
# tiny files local and run analysis/plot_phase_diagram.py. Env knobs (HX, LS,
# SECTOR, EVAL_SAMPLES, BASE) pass straight through to nersc/extract_fm.sh — sbatch
# propagates the submitting shell's environment (--export=ALL default).
#
# Unfinished (timed-out) runs are handled: extract_fm.sh -> fm_sweep falls back to
# the latest {name}.ckpt.mpack checkpoint when a final artifact is missing.
#SBATCH --job-name=tc-fmextract
#SBATCH --account=m5340_g
#SBATCH --qos=shared
#SBATCH --constraint=gpu
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail

module load conda
conda activate tc-nqs

REPO="${REPO:-$HOME/threed_TC/ThreeD_TC}"
cd "$REPO" || { echo "[submit] REPO not found: $REPO — set REPO=<clone path>"; exit 1; }

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "[submit] fm extraction: HX=${HX:-0.2}  LS=${LS:-<unset>}  sector=${SECTOR:-electric}"
bash nersc/extract_fm.sh
