#!/bin/bash
#SBATCH --job-name=tc-hyspeed
#SBATCH --account=m5340_g
#SBATCH --qos=debug
#SBATCH --constraint=gpu
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out

# PROTOTYPE debug-queue benchmark (NOT production): one (L, qgt, qgt_solver)
# variant per job, timing both a healthy (hy=0.3) and the pathological
# (hy=1.4) regime by warm-starting from the real hy-axis-L4 campaign
# checkpoints (results/-equivalent .mpack on scratch) instead of training --
# these are READ-ONLY loads of another job's finished checkpoints, never a
# write into that campaign's directories. Speed-investigation branch only;
# never touches the live campaign clone ($HOME/toric-code-nqs).
set -u
module load conda
conda activate tc-nqs

REPO="${REPO:-$HOME/toric-code-nqs-speed}"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$PSCRATCH/tc_nqs/jax_cache}"
mkdir -p "$JAX_COMPILATION_CACHE_DIR"

echo "== import gate: verify tc3d resolves to \$REPO =="
python -c "
import tc3d, os
p = os.path.abspath(tc3d.__file__)
repo = os.path.abspath('$REPO')
assert p.startswith(repo), f'tc3d NOT from {repo} -- PYTHONPATH shadow failed'
print('OK:', p)
"

L="${L:?set L=4/5/6}"
QGT="${QGT:-dense}"                    # dense / srt / onthefly / auto
QGT_SOLVER="${QGT_SOLVER:-}"            # "" (NetKet default CG) / cgN / cholesky / solve
QGT_SOLVERS="${QGT_SOLVERS:-}"          # space-sep list -> --qgt_solvers (A/B in ONE
                                        # process, amortizing build_state's H-assembly
                                        # cost across variants; needed at L>=6 where a
                                        # single cold-start build_state can exceed the
                                        # whole 30-min cap on its own). Overrides QGT_SOLVER.
N_ITER="${N_ITER:-6}"
HYS="${HYS:-0.3 1.4}"
CKPT_DIR="${CKPT_DIR:-$PSCRATCH/tc_nqs/hy_axis/cold/L$L}"
OUT="${OUT:-$PSCRATCH/tc_nqs/hy_speed_bench}"
mkdir -p "$OUT"
cd "$REPO"

TAG="${QGT}${QGT_SOLVER:+_${QGT_SOLVER}}"
[ -n "$QGT_SOLVERS" ] && TAG="${QGT}_AB_$(echo "$QGT_SOLVERS" | tr ' ' '-')"
KM1=$((L-1))
for HY in $HYS; do
  BASE="$CKPT_DIR/gridinv_dual_L${L}_OBC_hx0.0_hz0.0_hy${HY}_n2x4_nh4-8_inv8-8_k${KM1}"
  INIT_ARGS=()
  if [ -f "$BASE.mpack" ]; then
    INIT_ARGS=(--init_from "$BASE")
    echo "[bench] warm start from $BASE.mpack"
  else
    echo "[bench] no checkpoint at $BASE.mpack -- cold start (less representative)"
  fi
  OUTJSON="$OUT/L${L}_hy${HY}_${TAG}.json"
  if [ -f "$OUTJSON" ]; then
    echo "== skip L=$L hy=$HY $TAG (result exists) =="
    continue
  fi
  QARGS=(--qgt "$QGT")
  if [ -n "$QGT_SOLVERS" ]; then
    QARGS+=(--qgt_solvers $QGT_SOLVERS)
  elif [ -n "$QGT_SOLVER" ]; then
    QARGS+=(--qgt_solver "$QGT_SOLVER")
  fi
  echo "== L=$L hy=$HY qgt=$QGT solver=${QGT_SOLVERS:-${QGT_SOLVER:-cg-default}}  $(date +%H:%M:%S) =="
  srun -n 1 python -u analysis/scripts/bench_hy_speed.py \
    --L "$L" --hy "$HY" "${QARGS[@]}" --n_iter "$N_ITER" \
    "${INIT_ARGS[@]}" --out "$OUTJSON"
done
echo "== DONE  $(date +%H:%M:%S) =="
