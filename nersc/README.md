# Running the 3D bosonic toric-code NQS campaign on NERSC (Perlmutter)

Reproduction guide for the published work: the **phase3d** neural-quantum-state
(NQS) phase-diagram campaign, its **QMC** (ParaToric/PMRQMC) benchmarks, and the
NQS hyperparameter-tuning methodology behind both. Fermionic-TC material
(`launch_fermionic_ladder.sh`, `ladder_status.sh`, `colab/fermionic_TC_colab.ipynb`,
and fermionic knobs inside the shared wrappers) is a separate, still-active
track and out of scope for this document.

## 0. The machine

- Login: `ssh <user>@perlmutter.nersc.gov` (sshproxy cert, 24 h, MFA re-mint).
- GPU nodes: 4× A100 40 GB each; jobs charge account `m5340_g` (GPU) /
  `m5340` (CPU, W&B/QMC housekeeping only).
- Filesystems: run and write under `$PSCRATCH` (purged after ~8 weeks idle);
  `$HOME` is small and slow. Archive keepers to `/global/cfs/cdirs/<proj>`.

## 1. Environment

```bash
git clone https://github.com/SanzharBissenali/toric-code-nqs.git ~/toric-code-nqs
cd ~/toric-code-nqs
bash nersc/setup_conda_gpu.sh      # builds conda env tc-nqs (jax[cuda12] + netket)
                                   # and pip-installs tc3d editable (importable anywhere)
wandb login                        # once, on a login node (for later `wandb sync`)
```

Every submit wrapper defaults to `REPO=$HOME/toric-code-nqs` and can be
repointed at submit time with `REPO=/other/clone sbatch …`.

**Speed levers** (`--compute_dtype float32` + `--inv_impl dense`, ~4-8×
faster, behavior-preserving — see `tc3d.train --help`): the production
wrappers (`submit_nqs_gridinv.sh`, `submit_nqs_batch.sh`) read them from
`$PSCRATCH/tc_nqs/phase3d/defaults.env` at a job's *first* start, filling
only knobs left empty at submit time (so already-queued jobs adopt new
settings without losing queue age; a requeue never changes knobs mid-run).
Copy the committed template there once:

```bash
mkdir -p "$PSCRATCH/tc_nqs/phase3d"
cp nersc/defaults.env.example "$PSCRATCH/tc_nqs/phase3d/defaults.env"
```

## 2. Smoke test (gpu_debug)

```bash
L=4 BC=OBC HX=0.2 HZ=0.2 DUAL=1 N_ITER=40 \
  sbatch -q debug -t 00:30:00 nersc/submit_nqs_gridinv.sh
```

The first-ever compile of a given (arch, L, dual) graph is ~20 min on a cold
cache — the wrappers all set `JAX_COMPILATION_CACHE_DIR` (default
`$PSCRATCH/tc_nqs/jax_cache`), so it is paid once ever. Outputs land in
`$PSCRATCH/tc_nqs/gridinv/{name}.json/.mpack` (+ `.curve.json` checkpoints).
If the 30-min debug window dies inside the S2 finalize compile, just
re-`sbatch` — `--resume` skips training and finishes from the checkpoint.

## 3. The two NQS wrappers

Every knob is an environment variable (defaults live in the script header;
`train.py --help` / `sweep.py --help` document the underlying flags).

### `submit_nqs_gridinv.sh` — one point, one long job

```bash
L=4 BC=OBC DUAL=1 DT=0.01 DIAG_SHIFT=1e-3 N_NONINV=2 NONINV=4 INV="4 4" KERNEL=4 \
    N_ITER=400 sbatch nersc/submit_nqs_gridinv.sh

# unattended multi-slot run: auto-resubmits ~3 min before each wall limit;
# WALLTIME persists across the chain (a bare `sbatch --time=…` would not)
L=6 N_ITER=800 AUTO_RESUBMIT=1 WALLTIME=05:00:00 sbatch nersc/submit_nqs_gridinv.sh

# sign-full (h_y != 0): --dual_basis auto-selects complex weights; cold start
# is the default
L=4 BC=PBC DUAL=1 HX=0.0 HZ=0.0 HY=0.3 KERNEL=3 sbatch nersc/submit_nqs_gridinv.sh
```

Key knobs: `L BC HX HZ HY MODEL ARCH DT LR_MIN DIAG_SHIFT NONINV N_NONINV
NONINV_HIDDEN INV KERNEL N_ITER N_SAMPLES N_CHAINS N_SWEEPS QGT QGT_SOLVER
CHUNK EXACT_E0 REF_E REF_SIG SEED WANDB_PROJECT EXTRA_ARGS` plus the two
speed levers `COMPUTE_DTYPE INV_IMPL`.

### `submit_nqs_batch.sh` — N field points per job (array + chunk)

The workhorse for sweeps and the phase3d chain jobs: pays the ~10 min
JAX/XLA compile once per array task and reuses it across `CHUNK_POINTS`
field values trained sequentially in one process (`tc3d.sweep`).

```bash
# hz cut, L=4, 16 points, 4/job -> 4 chunks (array 0-3):
L=4 HX=0.2 HZ_N=16 CHUNK_POINTS=4 sbatch --array=0-3 --time=04:00:00 nersc/submit_nqs_batch.sh
# hx cut, L=4, 15 points, 4/job:
SWEEP=hx L=4 HZ=0.0 HX_N=15 CHUNK_POINTS=4 sbatch --array=0-3 --time=04:00:00 nersc/submit_nqs_batch.sh
# hy cut at fixed (hx,hz) (phase3d y-cuts): SWEEP=hy, GMIN/GMAX default 0.6/1.5
SWEEP=hy L=4 HX=0.0 HZ=0.0 HY_N=10 CHUNK_POINTS=4 sbatch --array=0-2 nersc/submit_nqs_batch.sh
```

Key knobs: `SWEEP={hz,hx,hy}` selects the swept field; `{HZ,HX,HY}_{MIN,MAX,N}`
define a uniform grid, or pass an explicit non-uniform `FIELD_VALUES="..."`
(campaign grids like Phase B's coarse+fine windows); `CHUNK_POINTS` sets
points/job (`--array` size must be `ceil(N_points / CHUNK_POINTS)`);
`NAME_TEMPLATE` controls output naming; `INIT_FROM=<BASE path, no extension>`
warm-starts from an external checkpoint (chain link jobs); `SNAPSHOT_EVERY`
keeps `{name}.step{N}.mpack` snapshots for post-hoc replay
(`analysis/scripts/eval_snapshots.py`); same speed levers/late-binding
`defaults.env` mechanism as `submit_nqs_gridinv.sh`. Per-L architecture
defaults (`KERNEL`, `DIAG_SHIFT`, `N_ITER`) are baked in as `case` blocks —
`nersc/CAMPAIGN.md` is the canonical spec, kept in sync by hand.

**Sanity anchor:** submit with `HX=0 HZ=0` and the energy must converge to
the exact unperturbed `E0` (PBC `-4L³`, OBC `-(L³+3(L-1)²L)`); at finite
field a converged run must sit *below* that anchor.
`analysis/scripts/exact_benchmarks.py` has the field-series references.

**wandb from compute nodes:** compute nodes can't reach wandb.ai, so runs
pass `--wandb_offline` (logs to `$OUT_DIR/wandb/`). From a **login** node:
`bash nersc/sync_wandb.sh` (or `wandb sync $OUT_DIR/wandb/offline-run-*`). A
requeued job reuses a deterministic run id, so all chunks merge into **one**
wandb run on sync.

## 4. The phase3d campaign

The published phase diagram's NQS side. Everything is idempotent and
state-driven — re-running any step just tops up what's still missing.

1. **Plan.** `analysis/scripts/phase3d_grid.py` is the pure-Python, NetKet-free
   deterministic grid (hy planes × L × cuts) and the sole source of "what
   point comes next" logic (priority order, dedup against manifests, the
   physics refusals). Inspect it standalone:
   ```bash
   python analysis/scripts/phase3d_grid.py --dry --hy 0.0 --L 4
   ```
2. **Launch.** `nersc/launch_phase3d.sh` turns `phase3d_grid.py plan --bash`
   output into `sbatch` calls (one hy **plane** per invocation) and appends a
   row per submitted point to a manifest TSV under
   `$PSCRATCH/tc_nqs/phase3d/manifests/`:
   ```bash
   HY=0.0 bash nersc/launch_phase3d.sh              # top up everything launchable
   HY=0.2 LS=5 bash nersc/launch_phase3d.sh          # LS is a display filter only
   DRYRUN=1 HY=0.0 bash nersc/launch_phase3d.sh      # print only, no sbatch/squeue
   ```
   For hand-made plan rows (e.g. extra points a review asked for), pass
   `PLAN_FILE=<tsv>` — but the launcher stamps the shell `HY` env var into
   every manifest row it writes, **not** a value read from the file, so
   `PLAN_FILE` and `HY` must agree (in particular `HY=y` for y-cut rows; a
   mismatch here mislabeled the y-cut probes as plane `1.0` once — see
   `analysis/scripts/phase3d_reseed.py`'s docstring).
3. **Watch.** `nersc/watch_phase3d.sh` cross-references a manifest against
   `sacct` + per-run logs + final JSONs and writes `watch_state.json`
   (state, diverged, E0-vs-bound, last step) — the live-state column the
   viewer/`phase3d_status.py` read.
4. **Automate.** `nersc/phase3d_cron_driver.sh` re-runs step 2 for one plane
   then step 3 over the union of *all* planes' manifests; install one
   scrontab line per plane from the template `nersc/phase3d_scrontab.txt`
   (fill in `<USER>`/`<USER_INITIAL>`/`<REPO>`/`<ACCOUNT>` — see §11). The
   template lists every plane the campaign ran (hy = 0.0, 0.2, 0.4, 0.6,
   0.8, 1.0, plus the `y` pseudo-plane sweeping hy itself at fixed hx,hz);
   by campaign end only the `y` driver was left running (its hourly tick
   also refreshes `watch_state.json` for every plane).
5. **Pull.** `analysis/scripts/pull_phase3d.sh` rsyncs finals + manifests +
   `watch_state.json` into `results/phase3d/` (committed) and splits
   per-step `.curve.json` learning curves into gitignored `data/tc_nqs/phase3d/`
   (CLAUDE.md's "commit summaries only" policy):
   ```bash
   bash analysis/scripts/pull_phase3d.sh              # every plane
   HY=0.0 bash analysis/scripts/pull_phase3d.sh        # one plane only
   ```
6. **Reseed.** `analysis/scripts/phase3d_reseed.py` is the best-of-N anchor
   reseed for a chain branch stuck in a bad local optimum (e.g. the
   y-polarized bimodality found 2026-09-23): train `N` seeded trials for the
   same anchor, `select` the lowest-energy healthy one against a physics
   gate, `--apply` to park the old branch and splice the winner in under the
   original name so the rest of the chain resumes untouched.

## 5. Provenance launchers (earlier cuts feeding the same diagram)

- `nersc/launch_phaseB.sh` / `launch_phaseB_rerun.sh` — the QMC-validation
  cuts (`results/phaseB*`), frozen spec in `nersc/CAMPAIGN.md` §Phase B:
  anchor (hx,hz)=(0.2,0.1) ("the tune-rect corner"), up cut sweeps hz at
  hx=0.2, right cut sweeps hx at hz=0.1, L∈{4,5,6}, every point cold-start.
  `_rerun` redoes the QMC-disagreement window with the cross-validated
  dt/diag_shift fixes (see its header).
- `nersc/launch_hy_cuts_L4.sh` — sign-full (h_y≠0) L=4 version of the same
  two cuts at hy∈{0.2,0.4} (`results/hy_cuts_L4/`).
- `nersc/launch_hy_axis_L4.sh` — the pure-hy line hx=hz=0, hy 0.0–1.5
  (cluster output `$PSCRATCH/tc_nqs/hy_axis/`), gated on the L=2 OBC Stage-0
  certification (§8).
- **tune-rect** (`results/tune_rect/`, the architecture/hyperparameter
  search that produced the winner config every launcher above uses) was run
  as ad-hoc `submit_nqs_gridinv.sh` calls, not a dedicated launcher script.
  The canonical winner settings are in `nersc/CAMPAIGN.md` ("Fixed across
  all L") and `notes/transition_mapping_recipes.md` §0: dual-basis
  `ToricCNN_gridinv`, `NONINV_HIDDEN="4 8"`, `INV="8 8"`, kernel = L−1,
  `DT=0.02`/`LR_MIN=0.002`, dense QGT.

## 6. QMC (ParaToric primary, PMRQMC cross-check)

Build order matters — the stdio patch's hunk offsets assume the membrane
patch applied first:

```bash
# macOS (Homebrew LLVM + boost, libc++ -- never mix with gcc):
bash external/build_paratoric_local.sh
# Perlmutter login node (gcc-15 via micromamba, statically linked .so):
bash nersc/build_paratoric_perlmutter.sh
```

Both scripts: clone ParaToric, apply the upstream `<print>` include fix,
then apply `external/paratoric_membrane.patch` (our
`fredenhagen_marcu_membrane` observable) followed by
`external/paratoric_stdio_taulog.patch` (routes the tau>0.1·N warning
through `fprintf(stderr)` — formatting it via Boost.Log segfaults batch
workers when the cluster build's `-static-libstdc++` meets conda's dynamic
Boost; **never let the ParaToric .so emit Boost.Log/iostreams**), and verify
the import touches the real extension (`paratoric.extended_toric_code.get_sample`
— `paratoric/__init__.py` swallows load failures silently).

Run a reference point:

```bash
HX=0.6 HZ=0.15 NBS_MULT=4 sbatch nersc/submit_qmc_paratoric.sh      # production
VALIDATE=1 sbatch nersc/submit_qmc_paratoric.sh                     # exact-anchor ladder
```

**Always pass the `--validate` ladder before trusting new numbers** (new
build, new settings, or a new L/field regime) — under-decorrelated runs
finish cleanly and return biased energies with confident error bars.
`VALIDATE=1` runs the energy ladder; chain `VALIDATE_FM=1` /
`VALIDATE_FM_MEMBRANE=1` / `VALIDATE_FM_MEMBRANE_R1=1` to also validate the
Z-string and X-membrane observable families. Production driver requirements:
`N_BETWEEN` ∝ β (≈120 updates/edge), `NBS_MULT=8` near h_c in the x-basis,
fresh `SEED0` per run, β≥24 for <1e-13 thermal bias — see
`analysis/scripts/paratoric_driver.py` and CLAUDE.md's QMC section.

**PMRQMC** is the independent cross-check, run via Colab (no Perlmutter
build): `colab/qmc_benchmarks_colab.ipynb` cell 6 generates `H.txt` from the
same stabilizer geometry as `analysis/scripts/export_pmrqmc.py`
(ground-state-isomorphic to `ThreeD_ToricCodeGeometry` at L=2 OBC, verified
to 1e-9) and runs PMRQMC on it at L=4.

## 7. Evaluation wrappers

- `nersc/submit_eval_ckpt.sh` — re-scores saved `train.py` checkpoints at a
  larger sample budget for honest error bars (`analysis/scripts/eval_ckpt.py`):
  `A_v/B_p/M_x/M_z` always, `+O_FM/S2` with `TOPO=1`, `+` the
  ParaToric-matched electric FM with `FM_PARATORIC=1`.
- `nersc/submit_eval_hy_axis.sh` — despite the name, a **generic**
  `analysis/scripts/eval_snapshots.py --topological` wrapper: replays every
  `{name}.step*.mpack` snapshot of a run through
  `tc3d.validation.topological_observables` for S2 (+O_FM). Used by the
  hy-axis campaign directly and by `launch_phase3d.sh`'s `POST_S2=1` hook
  (chain jobs' end-of-training S2).

## 8. L=2 OBC certification

`nersc/submit_hy_axis_l2_cert.sh` is the Stage-0 gate before any h_y≠0 L=4
production point: trains L=2 OBC dual-basis points cold and scores each
against a dense-ED referee vector
(`analysis/scripts/ed_referee_hy.py` → `results/hy_l2_certification/gs_*.npz`)
via `analysis/scripts/hy_cert_fidelity.py`. `HYS` sweeps the pure-hy axis by
default (`HX=HZ=0.0`); set `HX`/`HZ` for an off-axis point (this absorbed
the one-off `submit_hy_l2_cert.sh`, since removed) — each point still needs
a matching `gs_L2_OBC_hx<HX>_hy<HY>_hz<HZ>_dual.npz`. Note: those filenames
encode (hx,hy,hz) for lookup only — every banked
`results/hy_l2_certification/*.json` records its own hx/hy/hz/seed/diag_shift
in its own config, so naming drift in the bank never mislabels the physics.

## 9. Monitoring & control

```bash
squeue --me                 # my queued/running jobs (NERSC also has `sqs`)
sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS,Submit,Start
scancel <jobid>             # kill one;  scancel --me  kills all mine
scontrol show job <jobid>   # full detail while pending/running
bash nersc/phase3d_sample_jobs.sh 3   # spot-check N random running phase3d chains
```

## 10. Gotchas

- **No default architecture** — every job MUST set `--constraint=cpu` (or `gpu`).
- Run and write under **`$PSCRATCH`**, not `$HOME` (quota + IO); it is
  **purged** after ~8 weeks of no access — archive keepers.
- GPU account is `m5340_g`; CPU is `m5340`. Don't mix them up.
- Cold XLA compiles are the dominant smoke-test cost; never unset
  `JAX_COMPILATION_CACHE_DIR`. The inline S2 finalize adds its own one-off
  ~10-min compile (skippable with `--no_topological`).
- Set `OMP_NUM_THREADS` / `MKL_NUM_THREADS` to `$SLURM_CPUS_PER_TASK` for any
  CPU-side scipy work.

## 11. Site-specific settings (change these for a new user/allocation)

- **Account.** `#SBATCH --account=...` headers are baked into each wrapper
  (`m5340_g` GPU / `m5340` CPU) but every `#SBATCH` directive is overridable
  at submit time — use `sbatch -A <your_account> ...`, no file edits needed.
  The one exception is `nersc/phase3d_scrontab.txt`: `#SCRON` directives are
  **static** (not shell-evaluated), so its `<ACCOUNT>` placeholder must be
  edited into the file itself before `scrontab -e`.
- **`$PSCRATCH` layout.** Wrappers default to subdirectories of
  `$PSCRATCH/tc_nqs/` (`gridinv/`, `phase3d/`, `phaseB/`, `hy_cuts/`,
  `hy_axis/`, `qmc/`, `jax_cache/`); override per-run with `OUT_DIR=`/`OUT=`/
  `BASE_OUT=` (each script's header says which) if you want a different tree.
- **Conda env name.** `setup_conda_gpu.sh` creates `tc-nqs`; every wrapper's
  `conda activate tc-nqs` line must match if you rename it.
- **W&B entity/project.** `nersc/sync_wandb.sh` and `nersc/wb_regroup.py`
  read `WANDB_ENTITY` (falls back to the project's Caltech entity) and
  `PROJECT`/`WANDB_PROJECT`; `wandb login` once on a login node first.
- **Repo path.** Every wrapper reads `REPO` (default `$HOME/toric-code-nqs`);
  `nersc/phase3d_scrontab.txt`'s `<REPO>`/`<USER>`/`<USER_INITIAL>`
  placeholders are the static-directive exception above.
