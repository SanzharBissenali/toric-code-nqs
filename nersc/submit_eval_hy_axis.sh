#!/bin/bash
# S2 (+O_FM) snapshot replay for the hy-axis L=4 campaign: runs
# analysis/scripts/eval_snapshots.py --topological over the cold-lane runs,
# scoring every {name}.step*.mpack (50,100,...,500 — step 500 IS the
# end-of-training state) with tc3d.validation.topological_observables.
# S2 is sector-agnostic; fm_sector is pinned (auto = hx>=hz is ill-defined
# at hx=hz=0) so the O_FM byproduct is one coherent sector per pass.
#
# Chunk with GLOB (the topological eval costs wall-minutes per snapshot):
#   GLOB='*_hy0.[1-4]_*_k3.json' sbatch nersc/submit_eval_hy_axis.sh
#   GLOB='*_hz0.0_n2*_k3.json'   sbatch ...   # the hy=0.0 point (no _hy tag)
#
#SBATCH --job-name=tc-hyax-s2
#SBATCH --account=m5340_g
#SBATCH --qos=shared
#SBATCH --constraint=gpu
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=05:00:00
#SBATCH --output=%x-%j.out
set -u
module load conda
conda activate tc-nqs

REPO="${REPO:-$HOME/toric-code-nqs}"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
# extract-side wrappers do NOT inherit the training wrappers' JAX cache default
# -- export it or every chunk pays the ~20 min cold compile.
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$PSCRATCH/tc_nqs/jax_cache}"
mkdir -p "$JAX_COMPILATION_CACHE_DIR"

DIR="${DIR:-$PSCRATCH/tc_nqs/hy_axis/cold/L4}"
GLOB="${GLOB:-*_k3.json}"          # run JSONs end _k3.json; replay outputs don't
ROUNDS="${ROUNDS:-8}"              # match the campaign's --final_eval_rounds
SECTOR="${SECTOR:-electric}"
SUFFIX="${SUFFIX:-.snapeval_${SECTOR}.json}"
cd "$REPO"

echo "== import gate: verify tc3d resolves to \$REPO =="
python -c "
import tc3d, os
p = os.path.abspath(tc3d.__file__)
repo = os.path.abspath('$REPO')
print('tc3d:', p)
assert p.startswith(repo), f'tc3d NOT from {repo} -- PYTHONPATH shadow failed'
print('OK: tc3d resolves to this checkout')
"

srun -n 1 python -u analysis/scripts/eval_snapshots.py \
  --dir "$DIR" --glob "$GLOB" --rounds "$ROUNDS" \
  --topological --fm_sector "$SECTOR" --out_suffix "$SUFFIX"
echo "[eval] done: DIR=$DIR GLOB=$GLOB SECTOR=$SECTOR"
