#!/bin/bash
# DMRG-vs-NQS benchmark, bosonic 3D toric code, L=3 OBC (N=54), weak-field wave 1:
# 9 points, ours |h| <= 0.2 (hx = -0.05 -0.1 -0.2, hz = 0.05 0.1 0.2, hy = 0.05 0.1 0.2).
# Reference: results/dmrg_L3_obc/dmrg_L3_obc_reference.json (tc3d conventions; sweep.py
# --ref_json streams dE_ref per point). Production recipe: notes/transition_mapping_recipes.md §0.
#
#   bash nersc/launch_dmrg_L3_btc.sh                 # submit 3 axis array jobs (shared QOS)
#   DRY_RUN=1 bash nersc/launch_dmrg_L3_btc.sh       # print env + sbatch lines, submit nothing
#   MODE=smoke bash nersc/launch_dmrg_L3_btc.sh      # ONE gpu_debug job: 25 steps hx=-0.1 (dual)
#                                                    #   then 25 steps hy=0.1 (complex), timed
# Knobs: WALLTIME (default 01:30:00; set from the smoke timing), AUTO_RESUBMIT (default 1),
# OUT_BASE (default $PSCRATCH/tc_nqs/dmrg_L3/btc), REF_JSON, REPO.
#
#SBATCH --job-name=dmrgL3-smoke
#SBATCH --account=m5340_g
#SBATCH --qos=debug
#SBATCH --constraint=gpu
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
set -euo pipefail

MODE="${MODE:-prod}"
REF_JSON="${REF_JSON:-results/dmrg_L3_obc/dmrg_L3_obc_reference.json}"
OUT_BASE="${OUT_BASE:-${PSCRATCH:-/nonexistent}/tc_nqs/dmrg_L3/btc}"
WALLTIME="${WALLTIME:-01:30:00}"
NAME_TEMPLATE='btc_dual_L3_OBC_hx{hx}_hz{hz}'   # sweep.py appends _hy{hy} when hy != 0

# ---------------------------------------------------------------- smoke job body
if [ "$MODE" = "smoke_run" ]; then
  module load conda; conda activate tc-nqs
  REPO="${REPO:-$HOME/toric-code-nqs}"; cd "$REPO"
  export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"   # env's tc3d is editable against another clone
  export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$PSCRATCH/tc_nqs/jax_cache}"
  mkdir -p "$JAX_COMPILATION_CACHE_DIR"
  OUT="$OUT_BASE/smoke"; mkdir -p "$OUT"
  python -c "import tc3d, os; p = os.path.abspath(tc3d.__file__); assert p.startswith(os.path.abspath('$REPO')), p; print('tc3d:', p)"
  for PT in "hx -0.1" "hy 0.1"; do
    set -- $PT; AX="$1"; V="$2"
    HX=0.0; HY=0.0; HZ=0.0
    case "$AX" in hx) HX=$V;; hy) HY=$V;; esac
    REF_E=$(python -c "
import json; r=[x for x in json.load(open('$REF_JSON'))['rows'] if x['model']=='bosonic'
  and all(abs(x['ours'][k]-v)<1e-9 for k,v in (('hx',$HX),('hy',$HY),('hz',$HZ)))]
print(r[0]['ours']['E'])")
    NAME="smoke_btc_dual_L3_OBC_${AX}${V}"
    echo "== smoke block: $NAME  ref_E=$REF_E =="
    T0=$(date +%s)
    srun -n 1 python -u -m tc3d.train \
      --L 3 --bc OBC --model bosonic --arch ToricCNN_gridinv --dual_basis \
      --hx "$HX" --hy "$HY" --hz "$HZ" --ref_E "$REF_E" \
      --noninv_channels 4 --n_noninv 2 --noninv_hidden 4 8 --inv_hidden 8 8 --kernel_size 2 \
      --dt 0.02 --lr_min 0.002 --diag_shift 1e-3 --qgt dense \
      --compute_dtype float32 --inv_impl dense \
      --n_iter 25 --n_samples 8192 --n_chains 1024 --n_sweeps 48 --chunk_size 2048 \
      --final_eval_rounds 8 --no_topological --no_wandb \
      --out_dir "$OUT" --name "$NAME"
    T1=$(date +%s)
    echo "== smoke block $NAME wall: $((T1 - T0)) s (incl. compile + final eval) =="
    python - "$OUT/$NAME.curve.json" <<'PY'
import json, sys
t = [x["total"] for x in json.load(open(sys.argv[1]))["timing"]]
print(f"per-step total [s]: first={t[0]:.2f}  steady mean(steps 5+)={sum(t[5:])/max(len(t[5:]),1):.3f}  n={len(t)}")
print(f"=> 500 steps ~ {500*sum(t[5:])/max(len(t[5:]),1)/60:.1f} min/point (+compile, +final eval)")
PY
  done
  exit 0
fi

# ---------------------------------------------------------------- launcher
cd "$(dirname "$0")/.."
[ "$MODE" = "smoke" ] || [ "$MODE" = "prod" ] || { echo "MODE must be prod or smoke"; exit 1; }

run() {  # print the line; submit unless DRY_RUN=1
  printf '%q ' "$@"; echo
  [ "${DRY_RUN:-0}" = "1" ] || "$@"
}

if [ "$MODE" = "smoke" ]; then
  run env MODE=smoke_run REPO="$PWD" OUT_BASE="$OUT_BASE" REF_JSON="$REF_JSON" sbatch nersc/launch_dmrg_L3_btc.sh
  exit 0
fi

# production recipe (notes/transition_mapping_recipes.md §0 + agent-audited knobs)
RECIPE=(L=3 BC=OBC DUAL=1 NONINV_HIDDEN="4 8" INV="8 8" KERNEL=2 DT=0.02 LR_MIN=0.002
        DIAG_SHIFT=1e-3 N_ITER=500 N_SAMPLES=8192 N_CHAINS=1024 N_SWEEPS=48 CHUNK=2048
        QGT=dense TOPO=0 FINAL_EVAL_ROUNDS=8 INV_IMPL=dense COMPUTE_DTYPE=float32
        SNAPSHOT_EVERY=50 CHUNK_POINTS=3 AUTO_RESUBMIT="${AUTO_RESUBMIT:-1}" WALLTIME="$WALLTIME"
        NAME_TEMPLATE="$NAME_TEMPLATE" EXTRA_ARGS="--ref_json $REF_JSON")

#  axis  fixed-field env      field values
for SPEC in "x|HZ=0.0|-0.05 -0.1 -0.2|hx" "z|HX=0.0|0.05 0.1 0.2|hz" "y|HX=0.0 HZ=0.0|0.05 0.1 0.2|hy"; do
  IFS='|' read -r TAG FIXED VALS SWEEP <<<"$SPEC"
  run env "${RECIPE[@]}" SWEEP="$SWEEP" $FIXED FIELD_VALUES="$VALS" OUT_DIR="$OUT_BASE/$TAG" \
    sbatch --array=0-0 --time="$WALLTIME" --job-name="dmrgL3_btc_$TAG" nersc/submit_nqs_batch.sh
done
