# Running 3D toric-code jobs on NERSC (Perlmutter)

Working notes for submitting the ED / NQS sweeps to NERSC. The L=2 PBC 3D
toric-code ED (`N = 3·L³ = 24` qubits → 2²⁴ ≈ 16.7M states, ~2.7 GB Lanczos
workspace) OOMs the 8 GB dev laptop but is trivial on a Perlmutter CPU node.
Docs: https://docs.nersc.gov

## 0. The machine

**Perlmutter**, scheduled with **Slurm**. Two node types map onto our two codes:

| Workload | Code stack | Node | Specs |
|---|---|---|---|
| ED sweeps (`eigsh`, matrix-free) | numpy / scipy / numba | **CPU** | 2× AMD EPYC 7763, **128 cores, 512 GB** |
| NQS / variational | jax / jaxlib / netket / flax | **GPU** | 1× EPYC + **4× A100** (40/80 GB), 256 GB host |

Three Slurm commands: `sbatch` (submit a script), `salloc` (interactive node,
delivered **< 6 min** under `qos=interactive`), `srun` (launch on the node).

## 1. One-time setup

**Find your account/repo** (the `mXXXX` charged for compute; GPU jobs use the
`mXXXX_g` suffix):
```bash
sacctmgr -nP show assoc user=$USER format=account   # or check iris.nersc.gov
```
Put it in the `--account=` line of every script in this folder.

**Filesystems** — know which is which:
| Path | Use | Notes |
|---|---|---|
| `$HOME` (`/global/homes/...`) | code, configs | backed up, small quota, **not** for job IO |
| `$PSCRATCH` (`/pscratch/sd/...`) | **run jobs + write output here** | big, fast Lustre; purged after ~8 weeks idle |
| `/global/common/software/<proj>` | conda envs | read-optimised for compute nodes |

**Clone the repo** (into `$HOME`) and **build the CPU env** (once):
```bash
git clone <repo-url> $HOME/Approximate-Symmetries-TC
cd $HOME/Approximate-Symmetries-TC
export PROJ=mXXXX           # optional: put env in project software space
bash nersc/setup_conda_cpu.sh
```

## 2. Phase 1 — smoke test (gauge the mechanics + timing)

A throwaway job on `debug` QOS (fast scheduling, ≤30 min, ≤2 nodes). Uses the
base `python` module, so it does **not** need the conda env. Confirms the whole
login→queue→node→output loop and prints the queue wait.
```bash
# edit --account first
sbatch nersc/test_job.sh
squeue --me            # watch state: PD (pending) -> R (running) -> gone
cat tc-smoketest-*.out
```
The output reports `Submit` vs `Start` times — that's your real queue latency.
`debug` usually starts within a couple of minutes.

## 3. Phase 2 — interactive node (poke around live)

```bash
salloc --nodes 1 --qos interactive --time 01:00:00 --constraint cpu --account mXXXX
# ...drops you onto a compute node within ~6 min...
module load conda && conda activate tc-ed
cd $HOME/Approximate-Symmetries-TC/nersc
HX=0.3 HZ=0.5 L=2 python run_ed.py     # one ED point, live
```
Good for first runs, debugging, and timing a single point before committing a
batch sweep. Exit with `exit` to release the node.

## 4. Phase 3 — production sweep (batch array)

`submit_ed_sweep.sh` runs an h_z sweep as a Slurm **job array** on the `shared`
QOS — `shared` bills only the cores/memory you request (32 cores / 32 GB here),
not a whole 512 GB node, which suits the small single-process ED job.
```bash
# edit --account and the REPO= path first
sbatch nersc/submit_ed_sweep.sh        # default: 11 points, h_z = 0.0 .. 1.0
ls $PSCRATCH/tc_ed/hx0.3/              # one JSON per h_z
```
Each array task → `run_ed.py` → `tc3d/tests/colab_exact_diag.py::run`,
writing `ed_L2_hx0.3_hz<value>.json`. `run_ed.py` is a thin env-var wrapper
(`HX/HY/HZ/L/J/K/OUT`) so the same script can sweep any field.

## 4.5 Phase 4 — NQS hyperparameter sweep (GPU)

The variational (NetKet) path runs on the **GPU** nodes. One-time env build on a
**login** node, then a GPU job array — one config per task, each logging the
delta figure of merit to wandb.

```bash
# 1. one-time GPU env (jax[cuda12] + netket) + wandb auth (login node)
bash nersc/setup_conda_gpu.sh
module load conda && conda activate tc-nqs && wandb login

# 2. edit the grid in scripts/sweep_params.py, then size the array to it
N=$(python scripts/sweep_params.py --count)            # e.g. 36

# 3. submit; HZ_PRESET picks the hardcoded ED point (hard|mid|easy)
HZ_PRESET=mid sbatch --array=0-$((N-1)) nersc/submit_nqs_sweep.sh
```

Each task runs `tc3d/train.py` on **1 A100** with `--n_chains 1024`
(auto-detected) and `--hz_preset`, which sets both `h_z` and `E_exact` so
`delta = |E - E_exact|/|E_exact|` is printed per step and logged live to wandb.
All tasks share the wandb group `tc-nqs-sweep-<preset>` for side-by-side
comparison. Output JSON/weights go to `$PSCRATCH/tc_nqs/<preset>/`.

`scripts/sweep_params.py` is the single source of truth for the grid (learning
rate, diag_shift, pre-Wilson capacity, post-Wilson width); array index → combo,
exactly like the ED sweep maps index → h_z. If compute nodes can't reach wandb,
`export WANDB_MODE=offline` in the submit script and `wandb sync` later.

## 4.6 Phase 5 — large-L gridinv run (checkpoint + resume)

Phase 4 is an **L=2 grid array** that validates hyperparameters against ED. To
push the **grid-conv architecture (`ToricCNN_gridinv`) to larger L**, where one
run can outlast a single queue slot, use `nersc/submit_nqs_gridinv.sh`: a single
long job whose every knob is an env var, with on-disk checkpointing so a
timeout/pre-emption loses nothing.

```bash
# one config; override any knob via env vars (defaults in the script header)
L=4 BC=OBC DT=0.01 DIAG_SHIFT=1e-3 N_NONINV=2 NONINV=4 INV="4 4" KERNEL=4 \
    N_ITER=400 sbatch nersc/submit_nqs_gridinv.sh

# unattended multi-slot run: auto-resubmits ~3 min before each wall limit
L=6 N_ITER=800 AUTO_RESUBMIT=1 sbatch nersc/submit_nqs_gridinv.sh
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

**wandb from compute nodes:** Perlmutter compute nodes usually can't reach
wandb.ai, so the script passes `--wandb_offline` (logs to `$OUT_DIR/wandb/`).
After the run, from a **login** node: `wandb sync $OUT_DIR/wandb/offline-run-*`.
A requeued job reuses a deterministic run id, so all chunks merge into **one**
wandb run on sync. Set `WANDB_OFFLINE=0` if your project is reachable, or
`NO_WANDB=1` to rely on the JSON curve alone.

**Sanity anchor:** submit with `HX=0 HZ=0` and the energy must converge to the
exact unperturbed `E0` (PBC `-4L³`, OBC `-(L³+3(L-1)²L)`; see
`notes/progress_log.md`). This is the only exact reference at L>2.

## 4.7 Phase 6 — hz phase-diagram sweep (multi-L)

Map the topological→trivial boundary by fixing `hx` and sweeping `hz`, one NQS
per (L, hz), then locating the transition as the **peak of `dO_FM/dhz`** (the
Fredenhagen–Marcu order parameter's derivative) overlaid across L. Heavy compute
stays on the cluster; only tiny curve files come local.

```bash
# 1. Sweep hz at fixed hx, per L. Array size MUST equal HZ_N (default 16).
#    diag_shift auto-raises to 1e-2 at L>=6; chunk 2048 covers L=3..6.
L=3 sbatch --array=0-15 nersc/submit_nqs_hz_sweep.sh     # validate first (cheap)
L=4 sbatch --array=0-15 nersc/submit_nqs_hz_sweep.sh
L=5 sbatch --array=0-15 nersc/submit_nqs_hz_sweep.sh
L=6 sbatch --array=0-15 nersc/submit_nqs_hz_sweep.sh     # ~1 slot/point
#    knobs: HX (0.2), HZ_MIN (0.0), HZ_MAX (0.9), HZ_N (16), N_ITER (400), ...
#    outputs -> $PSCRATCH/tc_nqs/phase_hx0.2/L<L>/  (one {name}.json+.mpack per hz)

# 2. Compute O_FM per L on a GPU node (the ONLY on-cluster analysis step):
salloc -N 1 -C gpu --gpus 1 -q interactive -A m5340_g -t 00:30:00
module load conda && conda activate tc-nqs
HX=0.2 LS="3 4 5 6" bash nersc/extract_fm.sh             # -> phase_hx0.2/fm_L<L>_hx0.2.json

# 3. Pull the tiny curve files local, then plot the multi-L overlay + FSS:
rsync -avz <host>:$PSCRATCH/tc_nqs/phase_hx0.2/fm_L*.json ./results/phase_hx0.2/
python analysis/plot_phase_diagram.py --dir results/phase_hx0.2 --fss --out phase_hx0.2.png
```

- `submit_nqs_hz_sweep.sh` maps `SLURM_ARRAY_TASK_ID → hz` (like `submit_ed_sweep.sh`)
  and runs the validated gridinv train call (like `submit_nqs_gridinv.sh`);
  `--resume` is always on, so re-submitting the array continues unfinished points.
- `python -m tc3d.fm --dir … --L … --hx …` wraps the existing `fm_sweep` +
  `fit_transition` and dumps `{field, O, Oe, mz, h_c, h_c_fd, fit_curve, …}`.
- `analysis/plot_phase_diagram.py` is **NetKet-free** (reads only the extracted
  JSONs) — overlays `O_FM(hz)` + `dO_FM/dhz` for every L and (with `--fss`)
  extrapolates `h_c(L)` vs `1/L`.
- **Sanity per run** (reference-free, no ED past L=2): final `E0` must sit below
  `E0(0) = -(L³+3(L-1)²L)` (OBC); the L=3 sweep must show `O_FM` drop 1→0 with a
  clear derivative peak, and the `⟨σz⟩` susceptibility peak near the same `h_c`.

## 5. Monitoring & control

```bash
squeue --me                 # my queued/running jobs (NERSC also has `sqs`)
sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS,Submit,Start   # after the fact
scancel <jobid>             # kill one;  scancel --me  kills all mine
scontrol show job <jobid>   # full detail while pending/running
```

## 6. Gotchas

- **No default architecture** — every job MUST set `--constraint=cpu` (or `gpu`).
- Run and write under **`$PSCRATCH`**, not `$HOME` (quota + IO).
- `$PSCRATCH` is **purged** after ~8 weeks of no access — copy keepers to the
  community FS (`/global/cfs/cdirs/<proj>`) or pull them home.
- Set `OMP_NUM_THREADS` / `MKL_NUM_THREADS` to `$SLURM_CPUS_PER_TASK` so scipy's
  BLAS doesn't oversubscribe (the sweep script does this).
- GPU account is `mXXXX_g`; CPU is `mXXXX`. Don't mix them up.
- The scipy reference ED is largely BLAS/serial in the matvec — extra cores help
  modestly. The **numba** matrix-free path (parallel matvec) is the heavy one and
  currently lives in `colab/fermionic_TC_colab.ipynb`; extracting it to a script
  is the next step (see below).

## 7. Next steps / TODO

- [ ] **Extract the fermionic numba sweep** (`sweep_phase_diagram_3d_fermionic`)
      from `colab/fermionic_TC_colab.ipynb` into a `run_fermionic_sweep.py` so the
      Fredenhagen–Marcu / Wilson-loop sweep runs as a batch job (parallel matvec
      → wants the full 128-core node).
- [x] **GPU / NetKet env** for the NQS path — `nersc/setup_conda_gpu.sh` (env
      `tc-nqs`: `jax[cuda12]` + `netket`) and `nersc/submit_nqs_sweep.sh` (GPU job
      array, `--constraint=gpu --gpus=1 --account m5340_g`). See Phase 4 above.
- [ ] Decide output home: keep JSON on `$PSCRATCH` during runs, archive to
      `/global/cfs/cdirs/<proj>` once a sweep is complete.

## Quick reference

```bash
# submit / watch / cancel
sbatch nersc/test_job.sh ; squeue --me ; scancel <id>
# interactive CPU node
salloc -N 1 --qos interactive -t 01:00:00 -C cpu -A mXXXX
# one ED point
HX=0.3 HZ=0.5 L=2 python nersc/run_ed.py
```
