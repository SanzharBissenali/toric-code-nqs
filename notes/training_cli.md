# Training CLI reference (`tc3d/train.py`)

Every architecture + training hyperparameter is a CLI flag. **Omit any flag** and
it falls back to `TRAIN_DEFAULTS` / `builders.DEFAULTS` (the `argparse.SUPPRESS`
design), so you only list the knobs you're actually sweeping.

**2026-09 publication cleanup:** the ansätze `ToricCNN`, `ToricCNN_full`,
`VanillaCNN`, `VanillaWilsonCNN` and the flags that only served them (`--hidden`,
`--vanilla_depth`, `--noninv_random`, `--radius_plaq`), plus the dead `--hz_preset`
preset table, were removed from `tc3d/train.py` (see `ARCHIVE.md`). `--arch` now
takes only `ToricCNN_gridinv` (+ its dual-basis variant, selected via
`--dual_basis`) and `GeoCNN`. The table below is trimmed to match; it still lags
newer flags (`--dual_basis`, `--ref_E/--ref_sig`, the divergence-guard knobs,
speed levers) — `python -m tc3d.train --help` is always authoritative.

## Full flag set

| Group | Flag | Notes |
|---|---|---|
| System | `--L` (required), `--bc` | `PBC` \| `OBC` |
| | `--model` | `bosonic` \| `fermionic` |
| Hamiltonian | `--hx --hy --hz --J` | fields + coupling |
| | `--exact_E0` | manual `E_exact` at a custom field point (delta figure of merit) |
| Architecture | `--arch` | `ToricCNN_gridinv` \| `GeoCNN` (`--dual_basis` selects the Hadamard/dual variant of `ToricCNN_gridinv`) |
| | `--noninv_channels` | `ToricCNN_gridinv`: noninv channels (or use `--noninv_hidden` for per-layer widths) |
| | `--n_noninv` | `ToricCNN_gridinv`: # noninv layers |
| | `--inv_hidden` | post-Wilson hidden widths, e.g. `--inv_hidden 4 4` (final 1-ch appended) |
| | `--kernel_size` | `ToricCNN_gridinv` invariant grid-conv kernel (default auto = L) |
| | `--cnn_hidden` | `GeoCNN`: edge-conv channel widths (no Wilson), final 1-ch appended |
| Training | `--n_iter` | # VMC/SR steps |
| | `--dt` | (initial) learning rate |
| | `--lr_min` | cosine-decay lr → this over `n_iter`; set `== dt` for constant lr |
| | `--diag_shift` | SR regularization |
| | `--qgt` | `dense` \| `onthefly` \| `auto` (use `dense` on GPU) |
| | `--seed` | |
| Sampling | `--n_samples` | total MC samples |
| | `--n_chains` | # Metropolis chains (GPU auto-bumps if unset) |
| | `--n_sweeps` | sweeps between recorded samples (default `2N` = 48 at L=2) |
| | `--n_discard` | discarded warmup samples per chain |
| | `--chunk_size` | grad chunking (memory) |
| Output | `--name --out_dir` | auto name = `{model}_{arch}_L{L}_hx{hx}_hz{hz}` |
| | `--wandb_project --wandb_entity --wandb_group` | W&B routing |
| | `--no_wandb` | disable W&B |
| | `--wandb_offline` | log W&B to a local dir (`WANDB_MODE=offline`); `wandb sync` later — for compute nodes with no network |
| Checkpoint | `--checkpoint_every N` | atomically write weights + energy curve every N steps (default 10; 0 disables) — timeout-safe |
| | `--resume` | continue from `{out_dir}/{name}.ckpt.mpack` + `.curve.json` if present (resumes LR schedule + curve); re-run the same command to keep going |

## Colab cell template (every knob as a variable)

```python
# ---- system ----
L            = 2
BC           = "PBC"          # PBC | OBC
MODEL        = "bosonic"      # bosonic | fermionic

# ---- Hamiltonian ----
HX, HY, HZ   = 0.2, 0.0, 0.2
J            = 1.0
EXACT_E0     = None

# ---- architecture ----
ARCH         = "ToricCNN_gridinv"   # ToricCNN_gridinv | GeoCNN  (--dual_basis for the dual variant)
NONINV_CH    = 4
N_NONINV     = 2
INV_HIDDEN   = [4, 4]
KERNEL_SIZE  = 3

# ---- training ----
N_ITER       = 200
DT           = 7e-3
LR_MIN       = 7e-4           # == DT for constant lr
DIAG_SHIFT   = 5e-3
QGT          = "dense"
SEED         = 0

# ---- sampling ----
N_SAMPLES    = 4096
N_CHAINS     = 16
N_SWEEPS     = 96             # default 2N
N_DISCARD    = 8

# ---- output / wandb ----
NAME         = "run1"
OUT_DIR      = "outputs"
WANDB        = True
WANDB_GROUP  = "capacity_sweep"

# ---- assemble flags ----
flags  = f"--L {L} --bc {BC} --model {MODEL} --hx {HX} --hy {HY} --hz {HZ} --J {J}"
flags += f" --arch {ARCH} --kernel_size {KERNEL_SIZE}"
flags += f" --noninv_channels {NONINV_CH} --n_noninv {N_NONINV} --inv_hidden {' '.join(map(str, INV_HIDDEN))}"
flags += f" --n_iter {N_ITER} --dt {DT} --lr_min {LR_MIN} --diag_shift {DIAG_SHIFT} --qgt {QGT} --seed {SEED}"
flags += f" --n_samples {N_SAMPLES} --n_chains {N_CHAINS} --n_sweeps {N_SWEEPS} --n_discard {N_DISCARD}"
flags += f" --out_dir {OUT_DIR} --name {NAME}"
if EXACT_E0 is not None: flags += f" --exact_E0 {EXACT_E0}"
flags += f" --wandb_group {WANDB_GROUP}" if WANDB else " --no_wandb"

!python -u -m tc3d.train {flags}
```

## Gotchas

- `--inv_hidden 4 4` → invariant block `[4, 4, 1]`; `--inv_hidden` empty → `[1]`
  (the trailing width-1 readout is always appended).
- `ToricCNN_gridinv`: `--kernel_size` is the invariant grid-conv kernel; omit it for
  the default **auto = L** (full span, the topological-coverage choice). PBC uses
  CIRCULAR padding, OBC zero (`SAME`) padding + a masked readout — supports both BC.
- Keep `n_samples / n_chains` ≳ a few hundred per chain so `R_hat` / `tau_corr`
  are meaningful (e.g. 8192 samples / 512 chains = only 16 per chain — too few).
- For **slowing symmetry breaking**, `diag_shift` goes **up** (more conservative
  SR steps), not down. The 2D paper's tiny `5e-5` only worked because 2D sampling
  never stalled.
- **Cluster / timeout-safety:** the run writes `{name}.ckpt.mpack` (weights +
  sampler RNG) and `{name}.curve.json` (step count + full curve) every
  `--checkpoint_every` steps; `--resume` reloads them and continues. A requeued
  job reuses a deterministic wandb id, so offline chunks merge into one run on
  `wandb sync`. Tail `{name}.curve.json` to monitor a run with no network. The
  NERSC wrapper is `nersc/submit_nqs_gridinv.sh` (env-var knobs + `AUTO_RESUBMIT`
  for multi-slot runs).

---

**Doc gap (2026-08-04):** flags added after this doc was written are not listed —
notably `--dual_basis`, `--ref_E/--ref_sig`, `--no_topological`, `--fm_sector`,
`--init_from`, and the divergence-guard knobs. `python -m tc3d.train --help` is authoritative.
