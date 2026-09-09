#!/bin/bash
#SBATCH --job-name=tc-speedprof
#SBATCH --account=m5340_g
#SBATCH --qos=debug
#SBATCH --constraint=gpu
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out

# Speed-research profile job (branch p3d/speed-research; debug queue only): runs
# bench_hy_speed.py once PER VARIANT (impl:compute_dtype:qgt_solver:n_iter) as
# separate processes so each variant's GPU peak memory is clean, with the
# --microbench decomposition of the `grad` stage. Never touches the production
# clone ($HOME/toric-code-nqs); imports are gated to $REPO.
#
#   L=5 HY=0.4 sbatch nersc/speed_profile_job.sh
#   L=6 HY=0.4 CHUNK=2048 VARIANTS="dense:float32:kernel:3 dense:float64:kernel:3" sbatch ...
set -u
module load conda
conda activate tc-nqs
REPO="${REPO:-$HOME/toric-code-nqs-speed}"
export PYTHONPATH="$REPO"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$PSCRATCH/tc_nqs/jax_cache}"
export TC3D_PAULI_CACHE_DIR="${TC3D_PAULI_CACHE_DIR:-$PSCRATCH/tc_nqs/pauli_cache}"
mkdir -p "$JAX_COMPILATION_CACHE_DIR"
cd "$REPO"
echo "[job] tc3d commit $(git rev-parse --short HEAD) ($(git log -1 --format=%ci))"
python -c "
import tc3d, os
p = os.path.realpath(tc3d.__file__); repo = os.path.realpath('$REPO')
assert p.startswith(repo), f'tc3d NOT from {repo} -- PYTHONPATH shadow failed'
print('import gate OK:', p)
"
L="${L:?set L}"; HY="${HY:-0.4}"; HX="${HX:-0.2}"; HZ="${HZ:-0.26}"
CHUNK="${CHUNK:-2048}"; N_SAMPLES="${N_SAMPLES:-8192}"; N_CHAINS="${N_CHAINS:-1024}"
VARIANTS="${VARIANTS:-conv:float64:cholesky:3 dense:float64:cholesky:4 conv:float32:cholesky:4 dense:float32:cholesky:4}"
OUT="${OUT:-$PSCRATCH/tc_nqs/speed_bench}"; mkdir -p "$OUT"
for v in $VARIANTS; do                       # impl:compute_dtype:solver:n_iter[:chunk]
  IFS=: read -r IMPL DT SOLVER NIT VCHUNK <<< "$v"
  VCHUNK="${VCHUNK:-$CHUNK}"
  TAG="L${L}_hy${HY}_hx${HX}_hz${HZ}_${IMPL}_${DT}_${SOLVER}_c${VCHUNK}_n${N_SAMPLES}"
  echo "== $TAG  $(date +%H:%M:%S) =="
  srun -n 1 python -u analysis/scripts/bench_hy_speed.py \
    --L "$L" --hy "$HY" --hx "$HX" --hz "$HZ" --qgt dense --qgt_solver "$SOLVER" \
    --compute_dtype "$DT" --inv_impl "$IMPL" --chunk_size "$VCHUNK" \
    --n_samples "$N_SAMPLES" --n_chains "$N_CHAINS" --n_iter "$NIT" --microbench \
    --out "$OUT/$TAG.json"
done
echo "== DONE $(date +%H:%M:%S) =="
