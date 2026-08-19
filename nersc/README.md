# Running 3D toric-code NQS jobs on NERSC (Perlmutter)

The cluster side of the two experiment tracks: dual-basis NQS training/tuning
(GPU) and, post-processing aside, everything that produces the JSONs consumed by
`analysis/`. The pre-restructure ED/2D workflow (run_ed, submit_ed_sweep,
sweep_params, test_job, the `tc-ed` CPU env) lives at git tag `2d-final`.

## 0. The machine

- Login: `ssh <user>@perlmutter.nersc.gov` (sshproxy cert, 24 h, MFA re-mint).
- GPU nodes: 4× A100 40 GB each; jobs charge account `m5340_g` (GPU) / `m5340` (CPU).
- Filesystems: run and write under `$PSCRATCH` (purged after ~8 weeks idle);
  `$HOME` is small and slow. Archive keepers to `/global/cfs/cdirs/<proj>`.

## 1. One-time setup

```bash
git clone https://github.com/SanzharBissenali/toric-code-nqs.git ~/toric-code-nqs
cd ~/toric-code-nqs
bash nersc/setup_conda_gpu.sh      # builds conda env tc-nqs (jax[cuda12] + netket)
                                   # and pip-installs tc3d editable (importable anywhere)
wandb login                        # once, on a login node (for later `wandb sync`)
```

Every submit wrapper defaults to `REPO=$HOME/toric-code-nqs` and can be
repointed at submit time with `REPO=/other/clone sbatch …`.

## 2. Smoke test (gpu_debug)

One short dual-basis run through the full submit → train → checkpoint → finalize
path. The first-ever compile of a given (arch, L, dual) graph is ~20 min on a
cold cache — the wrappers all set `JAX_COMPILATION_CACHE_DIR`
(default `$PSCRATCH/tc_nqs/jax_cache`), so it is paid once ever.

```bash
L=4 BC=OBC HX=0.2 HZ=0.2 DUAL=1 N_ITER=40 \
  sbatch -q debug -t 00:30:00 nersc/submit_nqs_gridinv.sh
```

Outputs land in `$PSCRATCH/tc_nqs/gridinv/{name}.json/.mpack` (+ `.curve.json`
checkpoints). If the 30-min debug window dies inside the S2 finalize compile,
just re-`sbatch` — `--resume` skips training and finishes from the checkpoint.

## 3. Single production run (checkpoint + resume)

`nersc/submit_nqs_gridinv.sh` is one long job whose every knob is an env var:

```bash
# one config; override any knob (defaults in the script header)
L=4 BC=OBC DUAL=1 DT=0.01 DIAG_SHIFT=1e-3 N_NONINV=2 NONINV=4 INV="4 4" KERNEL=4 \
    N_ITER=400 sbatch nersc/submit_nqs_gridinv.sh

# unattended multi-slot run: auto-resubmits ~3 min before each wall limit;
# WALLTIME persists across the chain (a bare `sbatch --time=…` would not)
L=6 N_ITER=800 AUTO_RESUBMIT=1 WALLTIME=05:00:00 sbatch nersc/submit_nqs_gridinv.sh

# benchmark-gap streaming against a QMC reference (see results/qmc_*):
L=4 DUAL=1 REF_ARGS='--ref_E -174.5957 --ref_sig 0.0147' # (pass via train flags)
```

How the timeout-safety works (all in `tc3d/train.py`):

- `--checkpoint_every N` (default 10) atomically writes `{name}.ckpt.mpack`
  (weights **+ sampler RNG state**) and `{name}.curve.json` (completed step count
  + the full energy/error/delta curve) to `$PSCRATCH` every N steps. **Tail
  `{name}.curve.json` to watch progress live**, even with wandb offline.
- `--resume` (always passed by the script) reloads that checkpoint, continues the
  cosine-LR schedule from the right step, and appends to the curve. It is a no-op
  on the first run, so **re-`sbatch`-ing the same command always Just Continues**.
- `AUTO_RESUBMIT=1` traps Slurm's pre-timeout `USR1` signal and resubmits the job
  (bounded by `MAX_RESUBMITS`, default 8); the `$PSCRATCH` checkpoint is the
  hand-off. Leave it off to resubmit by hand.

**wandb from compute nodes:** compute nodes can't reach wandb.ai, so the script
passes `--wandb_offline` (logs to `$OUT_DIR/wandb/`). Afterwards, from a
**login** node: `bash nersc/sync_wandb.sh` (or `wandb sync $OUT_DIR/wandb/offline-run-*`).
A requeued job reuses a deterministic run id, so all chunks merge into **one**
wandb run on sync. `NO_WANDB=1` to rely on the JSON curve alone.

**Sanity anchor:** submit with `HX=0 HZ=0` and the energy must converge to the
exact unperturbed `E0` (PBC `-4L³`, OBC `-(L³+3(L-1)²L)`); at finite field a
converged run must sit *below* that anchor. `analysis/scripts/exact_benchmarks.py` has
the field-series references.

`submit_nqs_batch.sh` runs N field points sequentially in ONE process (the JAX
compile is paid once per job, not per point) — the workhorse for sweep chunks.
`submit_nqs_geocnn.sh` is the symmetry-unaware control at matched cost.

## 4. Multi-L phase sweeps (arrays)

Fix one field, sweep the other, one NQS per (L, field point); the transition is
located downstream from `O_FM` / `S2` extraction. Array size MUST equal the
number of points.

```bash
# hz sweep at fixed hx (electric cut) -> $PSCRATCH/tc_nqs/phase_hx0.2/L<L>/
L=4 sbatch --array=0-15 nersc/submit_nqs_hz_sweep.sh
# hx sweep at fixed hz (magnetic cut) -> $PSCRATCH/tc_nqs/phase_hz0.3/L<L>/
L=5 HZ=0.3 sbatch --array=0-6 nersc/submit_nqs_hx_sweep.sh
```

- Per-L defaults (kernel, diag_shift, walltime, n_iter) are baked into the
  scripts as `case` blocks — **kernel L→L−1 capped at 4; diag_shift 5e-3 at
  L≥6 (1e-2 over-regularized); N_ITER 150 (L≤6) / 175 (L=7); chunk 2048 for
  L=3..7**. `nersc/CAMPAIGN.md` is the canonical spec — keep them in sync.
- `run_phase_campaign.sh` submits the whole (hx, L) grid; idempotent (re-run to
  top up unfinished points).
- **QA gate before extraction:** `analysis/scripts/check_convergence.py --tree` (or
  `nersc/check_hxsweep.sh`) — finished-above-bound / DIVERGED / BAD-ESTIMATOR
  are the red flags; in-flight `descending` is informational.

## 5. Extraction (GPU) → local analysis

```bash
salloc -N 1 -C gpu --gpus 1 -q interactive -A m5340_g -t 00:30:00
module load conda && conda activate tc-nqs
HX=0.2 LS="4" bash nersc/extract_fm.sh    # one L per call; -> fm_L4_hx0.2_bulk.json
# batch versions: submit_extract_fm.sh / extract_fm_s2.sh / extract_s2.sh /
# extract_membrane_s2.sh / extract_energy.sh (login-node aggregation)
```

Pull each **placement** into its own local `results/` dir (mixing placements
double-counts an L), then run the analysis notebooks locally —
`analysis/scripts/plot_phase_diagram.py` and the notebooks are NetKet-free.

```bash
rsync -avz '<user>@perlmutter.nersc.gov:/pscratch/sd/s/<u>/tc_nqs/phase_hx0.2/fm_L*_bulk.json' \
    results/phase_hx0.2_bulk/
```

## 6. Monitoring & control

```bash
squeue --me                 # my queued/running jobs (NERSC also has `sqs`)
sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS,Submit,Start
scancel <jobid>             # kill one;  scancel --me  kills all mine
scontrol show job <jobid>   # full detail while pending/running
```

## 7. Gotchas

- **No default architecture** — every job MUST set `--constraint=cpu` (or `gpu`).
- Run and write under **`$PSCRATCH`**, not `$HOME` (quota + IO).
- `$PSCRATCH` is **purged** after ~8 weeks of no access — archive keepers.
- GPU account is `m5340_g`; CPU is `m5340`. Don't mix them up.
- Cold XLA compiles are the dominant smoke-test cost; never unset
  `JAX_COMPILATION_CACHE_DIR`. The inline S2 finalize adds its own one-off
  ~10-min compile (skippable with `--no_topological`).
- Set `OMP_NUM_THREADS` / `MKL_NUM_THREADS` to `$SLURM_CPUS_PER_TASK` for any
  CPU-side scipy work.

## 8. TODO

- [ ] Extract the fermionic numba sweep from `colab/fermionic_TC_colab.ipynb`
      into a batch script (parallel matvec → wants a full CPU node).
- [ ] Archive completed sweep JSONs from `$PSCRATCH` to `/global/cfs/cdirs/<proj>`.

## Quick reference

```bash
# smoke / production / sweep
L=4 DUAL=1 N_ITER=40 sbatch -q debug -t 00:30:00 nersc/submit_nqs_gridinv.sh
L=6 AUTO_RESUBMIT=1 WALLTIME=05:00:00 sbatch nersc/submit_nqs_gridinv.sh
L=4 sbatch --array=0-15 nersc/submit_nqs_hz_sweep.sh
# watch / cancel
squeue --me ; scancel <id>
# publish offline W&B runs (login node)
bash nersc/sync_wandb.sh
```
