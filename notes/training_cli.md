# Training CLI reference (`tc3d/train.py`)

Every architecture + training hyperparameter is a CLI flag. **Omit any flag** and
it falls back to `TRAIN_DEFAULTS` / `builders.DEFAULTS` (the `argparse.SUPPRESS`
design), so you only list the knobs you're actually sweeping.

## Full flag set

| Group | Flag | Notes |
|---|---|---|
| System | `--L` (required), `--bc` | `PBC` \| `OBC` |
| | `--model` | `bosonic` \| `fermionic` |
| Hamiltonian | `--hx --hy --hz --J` | fields + coupling |
| | `--hz_preset` | `hard`\|`mid`\|`easy` — sets `hz` AND exact `E0` (delta FOM) |
| | `--exact_E0` | manual `E_exact` at a custom `hz` (alt to preset) |
| Architecture | `--arch` | `ToricCNN` \| `ToricCNN_full` \| `ToricCNN_gridinv` \| `GeoCNN` \| `VanillaCNN` \| `VanillaWilsonCNN` |
| | `--hidden` | `ToricCNN`: invariant hidden width |
| | `--noninv_channels` | `ToricCNN_full`/`ToricCNN_gridinv`/`VanillaWilsonCNN`: noninv channels |
| | `--n_noninv` | `ToricCNN_full`/`ToricCNN_gridinv`/`VanillaWilsonCNN`: # noninv layers |
| | `--inv_hidden` | post-Wilson hidden widths, e.g. `--inv_hidden 4 4` (final 1-ch appended) |
| | `--kernel_size` | `VanillaCNN`/`VanillaWilsonCNN` conv kernel; `ToricCNN_gridinv` invariant grid-conv kernel (default auto = L) |
| | `--cnn_hidden` | `GeoCNN`: edge-conv channel widths (no Wilson), final 1-ch appended |
| | `--vanilla_depth` | `VanillaCNN` only: # hidden conv layers |
| | `--noninv_random` | `VanillaWilsonCNN`: random-init noninv (default = identity warm start) |
| Fermionic sign frame | `--sign_frame` | `none`\|`anaC`\|`table`\|`cup`\|`linear`\|`vote`\|`pt2` — formulation B: train a POSITIVE real trunk on `H~ = S H S` instead of signing log psi; `anaC` = analytic h=0 token-quadratic sign, `table` = a `--sign_table` lookup, and `cup`/`linear`/`vote`/`pt2` = the per-configuration decoder heads of `tc3d/sign_decoders.py` (exact h=0 cup-product sign, plus the three off-support recovery decoders of `analysis/scripts/sign_fidelity_ftc.py` evaluated row-by-row — no 2^N table, so they run at any L). Excludes `--phase_head*`/`--dual_basis`; host-side cost caps `anaC` at N_p≤64 and `table` at N≤24, while `cup` scales to the `CupSign` limit L≤5; `linear`/`vote`/`pt2` are OBC-only (at PBC a coset label has no unique decomposition into lit line classes) |
| | `--sign_k_cap` | `K` (default 8): lit-line-**class** cap for `--sign_frame vote`/`pt2`. A row whose lit line classes exceed it (`vote`: per connected component, `pt2`: in total) falls back to the `linear` head and is counted in the head's `pop_stats()['n_fallback']` |
| | `--sign_max_terms` | `M` (default 200000): recovery-**count** cap for `--sign_frame vote`/`pt2` — max minimal recoveries contracted per row (class product per component for `vote`, in total for `pt2`, plus the second-order candidate count). This is the cap that bites at L≥5 OBC (one coset asks for 7.5e5 recoveries/row at L=5, 1.1e8 at L=6); over-cap rows fall back to `linear`. Lower it to trade decoder accuracy for host-side speed |
| | `--sign_table` | `PATH.npy`: ±1 sign over all 2^N configs for `--sign_frame table` (bit i = qubit i, bit 1 = spin down, `tc3d.exact_diag` order) |
| | `--dtype` | `float64`\|`complex`: explicit ansatz+H dtype override. Fermionic default (no `--dtype`): complex if `hy≠0` OR (`model fermionic` and `sign_frame none`); real if `sign_frame≠none` and `hy=0` (S already carries the sign, trunk can be positive). `sign_frame≠none` + `hy≠0` + `--dtype float64` is refused outright (a real trunk can't hold the residual complex phase once h_y breaks stoquasticity past what S absorbs) — omit `--dtype` or pass `--dtype complex`. `sign_frame≠none` + `--dual_basis` is always refused (fermionic decoration isn't self-dual under H) |
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
HZ_PRESET    = None           # None | "hard"|"mid"|"easy"
EXACT_E0     = None

# ---- architecture ----
ARCH         = "ToricCNN_full"   # ToricCNN | ToricCNN_full | ToricCNN_gridinv | GeoCNN | VanillaCNN | VanillaWilsonCNN
NONINV_CH    = 4
N_NONINV     = 2
INV_HIDDEN   = [4, 4]
HIDDEN       = 8
KERNEL_SIZE  = 3
VANILLA_DEPTH= 2
NONINV_RANDOM= False

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
flags += f" --arch {ARCH} --hidden {HIDDEN} --kernel_size {KERNEL_SIZE} --vanilla_depth {VANILLA_DEPTH}"
flags += f" --noninv_channels {NONINV_CH} --n_noninv {N_NONINV} --inv_hidden {' '.join(map(str, INV_HIDDEN))}"
flags += f" --n_iter {N_ITER} --dt {DT} --lr_min {LR_MIN} --diag_shift {DIAG_SHIFT} --qgt {QGT} --seed {SEED}"
flags += f" --n_samples {N_SAMPLES} --n_chains {N_CHAINS} --n_sweeps {N_SWEEPS} --n_discard {N_DISCARD}"
flags += f" --out_dir {OUT_DIR} --name {NAME}"
if HZ_PRESET:    flags += f" --hz_preset {HZ_PRESET}"
if EXACT_E0 is not None: flags += f" --exact_E0 {EXACT_E0}"
if NONINV_RANDOM: flags += " --noninv_random"
flags += f" --wandb_group {WANDB_GROUP}" if WANDB else " --no_wandb"

!python -u -m tc3d.train {flags}
```

## Gotchas

- Don't mix `--hz` and `--hz_preset` — the preset overrides `hz` and also sets
  `exact_E0`.
- `--inv_hidden 4 4` → invariant block `[4, 4, 1]`; `--inv_hidden` empty → `[1]`.
  Holds for both `ToricCNN_full` (geometry-exact invariant convs) and
  `ToricCNN_gridinv` (standard grid `nn.Conv3D` invariant block) — only the conv
  *type* differs; the trailing width-1 readout is appended either way.
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

## Fermionic launcher run-name convention

`nersc/submit_fermionic_{hx_ladder,plane}.sh` name every run
`gridinv_fermionic_L2_OBC_hx{hx}_hz{hz}_k2_{arm}` (the ladder fixes `hz=0`); at
`h_y≠0` the plane launcher inserts an `_hy{hy}` tag before `_k2`:
`gridinv_fermionic_L2_OBC_hx{hx}_hz{hz}_hy{hy}_k2_{arm}`. `arm`/`tier` tags:
`plain` (GeoCNN, no head), `asymm` (gridinv, no head, sign-blind), `anaC_k0`/`anaC_k6`
(frozen in-network head, flux_penalty 0/6), `pt2sf`/`votesf`/`anaCsf` (sign-framed,
`--sign_frame table`/`table`/`anaC` with the pt2/vote/analytic lookup), `pt2sfc`
(`pt2sf` + `--dtype complex`). `analysis/scripts/{hx_ladder_summary,plane_summary}.py`
parse these names back into rows — keep the convention in sync if a launcher's
naming changes.

---

**Doc gap (2026-08-04):** flags added after this doc was written are not listed —
notably `--dual_basis`, `--ref_E/--ref_sig`, `--no_topological`, `--fm_sector`,
`--init_from`, and the divergence-guard knobs. `python -m tc3d.train --help` is authoritative.
