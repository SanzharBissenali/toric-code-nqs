#!/bin/bash
# ParaToric QMC reference run on a GPU-shared allocation. The sampler is CPU-only
# C++, but this project has GPU hours only (m5340 CPU allocation = 0.0), so we
# request 1 GPU on `shared` QOS — that buys 1/4 node = 16 physical EPYC cores,
# a 1:1 match for the driver's default 4 chains x 4 blocks — and let the GPU idle.
# Charges m5340_g at ~1 GPU-h per wall hour. Requires build_paratoric_perlmutter.sh
# to have run once (self-contained .so; no LD_LIBRARY_PATH needed here).
#
#   HX=0.6 HZ=0.15 NBS_MULT=4 sbatch nersc/submit_qmc_paratoric.sh      # production
#   VALIDATE=1 sbatch nersc/submit_qmc_paratoric.sh                     # anchor ladder
#
#SBATCH --job-name=tc-qmc
#SBATCH --account=m5340_g
#SBATCH --qos=shared
#SBATCH --constraint=gpu
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail

module load conda
conda activate tc-nqs
REPO="${REPO:-$HOME/toric-code-nqs}"
cd "$REPO"

L="${L:-4}"
HX="${HX:-0.2}"
HZ="${HZ:-0.1}"
BETA="${BETA:-12}"
CHAINS="${CHAINS:-4}"
BLOCKS="${BLOCKS:-4}"                # chains x blocks = 16 independent blocks
SAMPLES="${SAMPLES:-20000}"          # stored samples per chain
NBS_MULT="${NBS_MULT:-4}"            # x4 decorrelation = the audited precision recipe
SEED0="${SEED0:-$SLURM_JOB_ID}"      # fresh per run (audit rule); recorded in the JSON
VALIDATE="${VALIDATE:-0}"
BASIS="${BASIS:-x}"                  # z + FM=1 for the electric Z-string O_FM
FM="${FM:-0}"

if [ "$VALIDATE" = "1" ]; then
  # exact-anchor ladder (mandatory before trusting any new build/settings)
  srun -n 1 python -u analysis/paratoric_driver.py --validate \
    --beta "$BETA" --chains "$CHAINS" --samples "$SAMPLES"
else
  BTAG=""; [ "$BASIS" != "x" ] && BTAG="_b${BASIS}"      # never clobber x-basis files
  OUT="${OUT:-$PSCRATCH/tc_nqs/qmc/qmc_hx${HX}_hz${HZ}/paratoric_L${L}${BTAG}_beta${BETA}_x${NBS_MULT}_seed${SEED0}.json}"
  echo "[qmc] L=$L (hx=$HX, hz=$HZ) beta=$BETA basis=$BASIS fm=$FM  ${CHAINS}x${BLOCKS} blocks, nbs_mult=$NBS_MULT, seed0=$SEED0"
  srun -n 1 python -u analysis/paratoric_driver.py \
    --L "$L" --hx "$HX" --hz "$HZ" --beta "$BETA" \
    --chains "$CHAINS" --blocks "$BLOCKS" --samples "$SAMPLES" \
    --nbs_mult "$NBS_MULT" --seed0 "$SEED0" --basis "$BASIS" \
    $( [ "$FM" = "1" ] && echo --fm ) --out "$OUT"
fi
