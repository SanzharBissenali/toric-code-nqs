#!/bin/bash
# Submit the magnetic O_FM + Rényi-S₂ extraction for the hx-driven line (all fixed-hz
# dirs, one L) in a single GPU job. Thin sbatch wrapper over extract_fm_s2_hxline.sh;
# every knob (HZS, L, R, ASPECT, EVAL_SAMPLES, EVAL_CHAINS, PLANES, SKIP_EXISTING)
# passes straight through via --export=ALL.
#
#   L=4 sbatch nersc/submit_extract_hxline.sh                 # all 10 hz at L=4 (~2-3 h)
#   L=5 HZS="0.0 0.5 1.0" sbatch nersc/submit_extract_hxline.sh
#   squeue --me
#
# L=4 is compile-dominated (2 fresh JAX processes per hz × 10 hz); 4 h walltime is
# generous. SKIP_EXISTING=1 (driver default) makes a resubmit resume where it stopped.
#SBATCH --job-name=tc-hxline-fms2
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
cd "$REPO" || { echo "[submit] REPO not found: $REPO"; exit 1; }

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1

echo "[submit] hxline fm+s2: L=${L:-4}  HZS='${HZS:-<all>}'  SKIP_EXISTING=${SKIP_EXISTING:-1}"
bash nersc/extract_fm_s2_hxline.sh
