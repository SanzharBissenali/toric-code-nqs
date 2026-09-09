#!/bin/bash
#SBATCH --job-name=tc-speedequiv
#SBATCH --account=m5340_g
#SBATCH --qos=debug
#SBATCH --constraint=gpu
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out

# Speed-research equivalence job (debug queue only; branch p3d/speed-research):
# same-seed tc3d.train runs of the lever variants at the L=4 complex production
# point (hx=0.2 hz=0.26 hy=0.4), plus (CHECK=1) the identical-samples gate
# analysis/scripts/check_equivalence.py. Results -> $OUT (default
# $PSCRATCH/tc_nqs/speed_equiv); compare with analysis/scripts/compare_equiv.py.
#
#   CHECK=1 VARIANTS="conv:float64:cholesky" sbatch nersc/speed_equiv_job.sh
#   VARIANTS="dense:float32:cholesky dense:float32:kernel dense:float32:cholesky:4096" sbatch ...
set -u
module load conda
conda activate tc-nqs
REPO="${REPO:-$HOME/toric-code-nqs-speed}"
export PYTHONPATH="$REPO"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$PSCRATCH/tc_nqs/jax_cache}"
export TC3D_PAULI_CACHE_DIR="${TC3D_PAULI_CACHE_DIR:-$PSCRATCH/tc_nqs/pauli_cache}"
cd "$REPO"
python -c "
import tc3d, os
p = os.path.abspath(tc3d.__file__); repo = os.path.abspath('$REPO')
assert p.startswith(repo), f'tc3d NOT from {repo} -- PYTHONPATH shadow failed'
print('import gate OK:', p)
"
L="${L:-4}"; HX="${HX:-0.2}"; HZ="${HZ:-0.26}"; HY="${HY:-0.4}"
N_ITER="${N_ITER:-60}"; DT="${DT:-0.02}"; DIAG_SHIFT="${DIAG_SHIFT:-1e-3}"; SEED="${SEED:-0}"
N_SAMPLES="${N_SAMPLES:-8192}"; N_CHAINS="${N_CHAINS:-1024}"; CHUNK="${CHUNK:-2048}"
OUT="${OUT:-$PSCRATCH/tc_nqs/speed_equiv}"; mkdir -p "$OUT"
CHECK="${CHECK:-0}"
VARIANTS="${VARIANTS:-conv:float64:cholesky}"        # impl:compute_dtype:solver[:n_samples]

if [ "$CHECK" = "1" ]; then
  echo "== check_equivalence L=$L  $(date +%H:%M:%S) =="
  srun -n 1 python -u analysis/scripts/check_equivalence.py --L "$L" --hx "$HX" --hz "$HZ" \
    --hy "$HY" --n_samples "$N_SAMPLES" --n_chains "$N_CHAINS" --chunk_size "$CHUNK" \
    --diag_shift "$DIAG_SHIFT" --dt "$DT" --warm_steps 5 --seed "$SEED" \
    --out "$OUT/check_L${L}_hy${HY}.json"
fi
for v in $VARIANTS; do
  IFS=: read -r IMPL DTYPE SOLVER NS <<< "$v"
  NS="${NS:-$N_SAMPLES}"
  NAME="equiv_L${L}_${IMPL}_${DTYPE}_${SOLVER}_n${NS}"
  echo "== $NAME  $(date +%H:%M:%S) =="
  srun -n 1 python -u -m tc3d.train --L "$L" --bc OBC --dual_basis --arch ToricCNN_gridinv \
    --noninv_hidden 4 8 --inv_hidden 8 8 --kernel_size $((L - 1)) \
    --hx "$HX" --hz "$HZ" --hy "$HY" --n_iter "$N_ITER" --dt "$DT" --diag_shift "$DIAG_SHIFT" \
    --qgt dense --qgt_solver "$SOLVER" --compute_dtype "$DTYPE" --inv_impl "$IMPL" \
    --n_samples "$NS" --n_chains "$N_CHAINS" --n_sweeps 48 --chunk_size "$CHUNK" --seed "$SEED" \
    --no_wandb --no_topological --checkpoint_every 0 --out_dir "$OUT" --name "$NAME"
done
echo "== DONE $(date +%H:%M:%S) =="
