# tc3d — approximately-symmetric NQS for the 3D toric code

Neural-quantum-state study of the 3D **bosonic** toric code under uniform fields,
$H = -J\sum_v A_v - J\sum_p B_p - h_x\sum_i\sigma^x_i - h_y\sum_i\sigma^y_i - h_z\sum_i\sigma^z_i$
(including the sign-problem-full $h_y \neq 0$ sector, complex weights), mapping the
topological→trivial transitions with an **approximately-symmetric CNN** ansatz (a
geometry-exact Wilson-product change of coordinates + identity-initialised
non-invariant block, in the dual/Hadamard basis), validated against **QMC**
(ParaToric / PMRQMC) and analytic field series. The architecture generalises the 2D
construction of [Kufel et al., PRL 135, 056702 (2025)](https://arxiv.org/abs/2405.17541).

**This README documents the published bosonic-3D-toric-code track.** Two related
efforts share the tree but are out of scope here: the **fermionic** toric code
(separate, still-active track — see `notes/handoff_fermionic_tc.md`, `colab/fermionic_TC_colab.ipynb`,
the fermionic rows of `analysis/README.md`) and the **2D** surface code this project
grew out of, frozen in history at tag **`2d-final`** (not in this tree).

**Orientation documents, in reading order:**

1. `BLOG.md` — the living experiment log (what was run, what it showed, what's next).
2. `notes/transition_mapping_recipes.md` — the executable playbook for mapping
   transition cuts: §A second-order, §B first-order (warm chains / hysteresis),
   §C sign-full internal trust ladder. Distilled from the Phase-B NQS-vs-QMC
   reconciliation campaign; every threshold in it was validated against
   β-converged QMC at L=4–6.
3. `CLAUDE.md` — working rules for agent sessions (cluster charter, gotchas).

## North star

Reproduce the NQS-vs-QMC benchmark figures in `analysis/figs/` (all observables
+ learning curves), then the full ($h_x$, $h_y$, $h_z$) phase diagram: per-cut
transition points with finite-size scaling (`analysis/README.md`'s "Transition
locations + FSS" row), extended into the sign-full regime ($h_y \neq 0$) where
QMC cannot referee (`notes/transition_mapping_recipes.md` §C's QMC-free trust
ladder). The **phase3d** campaign (`nersc/README.md` §4) is the current mapping
effort; its outputs land in `results/phase3d/` once banked (see the data map
below). Earlier phase-diagram sweeps that predate the tuned architecture and the
recipes playbook are superseded by it; their data and one-off notebooks live in
`_archive/` (local, gitignored — recover pre-cleanup material from git history
instead, see `ARCHIVE.md`).

## Environment

```bash
pip install -e ".[analysis]"   # bare `pip install -e .` skips matplotlib/jupyter/nbstripout
```

`pyproject.toml` pins the exact JAX/NetKet stack every production run used:
**jax 0.5.2 / jaxlib 0.5.1 / netket 3.16.1.post1 / flax 0.10.4 / optax 0.2.5**
(newer NetKet releases, e.g. 3.22.x, fail to import on Python ≥3.13); numpy/scipy/numba/wandb
are bounded to the range spanning the NERSC production env and the local analysis
venv. `requirements.txt` is the **exact freeze of the NERSC Perlmutter conda env**
that trained every banked run — use it if `pip install -e .` resolves something
that misbehaves. On the cluster, `nersc/setup_conda_gpu.sh` builds that conda env
(`tc-nqs`, jax[cuda12] + netket) and pip-installs `tc3d` editable; see
`nersc/README.md` §1.

## Quickstart — the two experiment tracks

**1. NQS training / sweeps (sign-problem-free regime, and sign-full at h_y≠0)**

```bash
python -m tc3d.train --L 4 --bc OBC --hx 0.2 --hz 0.2 --dual_basis \
    --arch ToricCNN_gridinv --noninv_hidden 4 8 --inv_hidden 8 8 --kernel_size 3 \
    --n_iter 300 --ref_E -174.5957 --ref_sig 0.0147   # signed per-step gap vs the QMC reference
```

`--ref_E/--ref_sig` stream the benchmark gap; QMC reference values live in
`results/qmc_hx*_hz*/`. `python -m tc3d.sweep` batches several field points per
process (amortises the JAX compile). `--dual_basis` also carries the $h_y \neq 0$
sign-full path (complex weights auto-derived; `--force_complex` for complex-at-h_y=0
controls). Full flag reference: `python -m tc3d.train --help` (also
`notes/training_cli.md`, itself lagging `--help` — see its header note).

Production runs go through the `nersc/submit_*.sh` wrappers, never this laptop
(see `CLAUDE.md`'s working rules) — `nersc/README.md` is the full reproduction
guide for the **phase3d** phase-diagram campaign (the published result: plan →
launch → watch → automate → pull, §4) and its provenance launchers (Phase B,
hy-cuts, hy-axis, §5). Production ansatz flags for every launch:
`DUAL=1 NONINV_HIDDEN="4 8" INV="8 8" KERNEL=$((L-1))` (wrapper defaults differ) —
the tuning methodology behind this choice is in `notes/transition_mapping_recipes.md`
§0 and `analysis/notebooks/tune_rect_summary.ipynb` (architecture/hyperparameter
search, `analysis/README.md`'s tuning-methodology row).

**2. QMC validation (energy + stabilizers + magnetizations + order parameters)**

```bash
# one-time: clone the QMC codes into gitignored external/ (never committed)
git clone --recursive https://github.com/palmbart/ParaToric.git external/ParaToric
git clone https://github.com/LevBarash/PMRQMC.git external/PMRQMC
bash external/build_paratoric_local.sh                      # local macOS build (brew llvm boost hdf5 ninja)
python analysis/scripts/paratoric_driver.py --validate              # exact-anchor ladder — run BEFORE trusting numbers
python analysis/scripts/paratoric_driver.py --L 4 --hx 0.2 --hz 0.2 --beta 24 --nbs_mult 4 \
    --out results/qmc_hx0.2_hz0.2/run.json                  # nbs_mult>=4 for production (8 near a crossing)
python analysis/scripts/export_pmrqmc.py --verify                   # PMRQMC cross-check (+ colab/qmc_benchmarks_colab.ipynb)
```

Analytic anchors and low/high-field series (the zero-fit accuracy certificate):
`analysis/scripts/exact_benchmarks.py` (42 self-checks; run it directly). Near any
first-order crossing, β=12 x-basis references are thermally biased — β≥24 with
×8 decorrelation is mandatory there, and loaders take the highest-β subset only.
(Deep inside a phase, combining a no-drift β ladder into one reference is
legitimate — e.g. the (0.2, 0.2) anchor file.)

## Repo map

### `tc3d/` — the package (every module is live)

| Group | Modules |
|---|---|
| Entry points | `train.py` (single run; checkpoint/resume, `--dual_basis`, `--init_from` warm starts, h_y complex path), `sweep.py` (batched field points), `fm.py` (Fredenhagen–Marcu order parameters, electric loop / magnetic membrane), `renyi.py` (S₂ locator) |
| Model construction | `builders.py` (config → geometry+H+ansatz+sampler+vstate; shared `run_loop` with divergence guard), `geometry.py`, `hamiltonian.py` (bosonic + dual-basis + fermionic), `networks.py` (`ToricCNN_gridinv` + complex/dual variant for h_y, `GeoCNN` baseline, geometry-exact stencils), `sampler.py` (cluster-update MCMC rules) |
| Fermionic | `fermionic_decoration.py` (decorated stabilizers, GF(2) sign machinery, dressed strings) |
| Validation / infra | `validation.py` (end-of-training observables: energy/Vscore, stabilizers, magnetizations, O_FM, S2 — written into the run JSON/W&B, replayed on checkpoints by the `analysis/scripts/eval_*.py` tools), `exact_diag.py` (matrix-free Numba ED; QMC `--verify`), `io.py` (checkpoints), `config.py` (device probe), `wandb_logger.py` |

### `analysis/` — post-processing, QMC drivers, benchmark figures

Layout: `analysis/scripts/` (all .py tools, run from repo root), `analysis/notebooks/`
(all .ipynb, cwd = `analysis/notebooks/`, reach data via `../../results`),
`analysis/figs/` (the committed benchmark PNGs). **Full paper-output → script/notebook
→ input map, canonical-implementation table, and campaign-ops scripts:
`analysis/README.md`** — covers the QMC pipeline, transition-location + FSS
extraction (`transition_fit.py` + `transition_fss.ipynb`), the benchmark figures
(`phaseB_figs.py`), tuning methodology (`tune_rect_summary.ipynb`), exact anchors
(`exact_benchmarks.py`), h_y≠0 validation, and the fermionic sign-head track
(`prefit_phase_head.py`, `stencil_phase_head.py`, `ed_electric_line.py`).

### Figure directories (two, on purpose)

- **`analysis/figs/`** — git-tracked, curated benchmark figures. This is the
  reproducibility target; every PNG maps to a generator notebook (table below).
- **`figures/`** (repo root) — gitignored scratch target for paper-bound
  renders; `paper/current-version.tex` reads it via `\graphicspath`.
- Notebook `plt.savefig` lines stay **commented out** (and `SAVE_FIGS` gates
  stay `False`) — figures are promoted manually, never auto-saved. The one
  scripted exception: `python analysis/scripts/phaseB_figs.py` regenerates the eight
  committed `phaseB_*` PNGs bit-exactly (same venv).

| Figures in `analysis/figs/` | Generator |
|---|---|
| `phaseB_h_z_sweep_*` (energy, stabilizers, ⟨σ_z⟩, Z-string) | `phaseB_figs.py` |
| `phaseB_h_x_sweep_*` (energy, stabilizers, ⟨σ_x⟩, X-membrane) | `phaseB_figs.py` |
| `tune_rect_*` (scaling, learning curves, rel. errors) + `single_point_0.2_0.1_learning_curve` | `tune_rect_summary.ipynb` |
| `fermionic_h0_prefit_ladder`, `fermionic_h0_L3_ghost` | `fermionic_h0_prefit_ladder.ipynb` |
| `fermionic_arch_ladder`, `fermionic_ladder_E_L2` | `fermionic_arch_ladder.ipynb` |

### `nersc/` — Perlmutter wrappers (all env-var driven, resume-safe)

**Full reproduction guide: `nersc/README.md`.**

| Group | Scripts |
|---|---|
| NQS launch | `submit_nqs_gridinv.sh` (single run / warm-chain link), `submit_nqs_batch.sh` (batched sweep) |
| phase3d campaign | `launch_phase3d.sh` → `watch_phase3d.sh` → `phase3d_cron_driver.sh` (+ `phase3d_scrontab.txt` template, `phase3d_sample_jobs.sh`), planned by `analysis/scripts/phase3d_grid.py`, pulled by `analysis/scripts/pull_phase3d.sh` |
| Provenance launchers | `launch_phaseB{,_rerun}.sh` (`results/phaseB*`), `launch_hy_cuts_L4.sh`, `launch_hy_axis_L4.sh` |
| QMC | `submit_qmc_paratoric.sh`, `build_paratoric_perlmutter.sh` |
| Evaluation | `submit_eval_ckpt.sh`, `submit_eval_hy_axis.sh`, `submit_hy_axis_l2_cert.sh` (L=2 OBC h_y certification gate) |
| Speed levers | `speed_equiv_job.sh` (production float32/dense equivalence certificate), `defaults.env.example` |
| Fermionic | `launch_fermionic_ladder.sh`, `ladder_status.sh` |
| Monitoring / sync | `sync_wandb.sh`, `wb_regroup.py` |
| Setup / docs | `setup_conda_gpu.sh`, `README.md` (how-to), `CAMPAIGN.md` (frozen Phase-B config spec) |

### `results/` — data map (small derived JSONs only; checkpoints gitignored)

| Family | Contents |
|---|---|
| `qmc_hx*_hz*/` | QMC reference anchors: electric arc (h_x=0.2, h_z swept), magnetic arc (h_z=0.1, h_x swept), tuning points. Loaders take the **highest-β subset only** — never mix β. |
| `phaseB/`, `phaseB_rerun/`, `phaseB_ablation{A..D}/` | The Phase-B NQS-vs-QMC reconciliation campaign: cold sweeps, warm chains, hysteresis branches, ablations. |
| `tune_rect/` | Architecture-tuning campaign. Winner: dual-basis, non-inv 4→8, inv (8,8), 15-tap kernel k=L−1. |
| `hy_cuts_L4/` | Sign-full ($h_y \neq 0$) L=4 electric/magnetic cuts at $h_y \in \{0.2, 0.4\}$. |
| `hy_rect_L4/` | Sign-full tuning/benchmark A-B at $h_y = 0.2$ (primal vs. dual basis). |
| `hy_l2_certification/` | L=2 OBC dual-basis certification vs. the dense-ED referee (`analysis/scripts/ed_referee_hy.py`) — the Stage-0 gate before any $h_y \neq 0$, $L \geq 4$ production point. |
| `transitions/` | Banked per-cut locator + FSS records (written by `analysis/notebooks/transition_fss.ipynb` / `analysis/scripts/firstorder_fit.py`), read by `analysis/notebooks/phase_diagram_manual.ipynb`. |
| `speed_bench/`, `speed_equiv/` | Per-step speed-lever measurements (`notes/speed_levers.md`) and the production float32/dense-QGT equivalence gate. |
| `fermionic_ladder/`, `fermionic_h0/`, `fermionic_eline/` | Fermionic track: architecture ladder, h=0 sign-structure anchors + `ed_L2_electric.json` (the electric-line ED reference), electric-line NQS runs. |
| `qmc_hx0.88_hz0.0/`, `threed_bosonic.json` | Standalone anchors (membrane point-cube reference; provenance for constants in `train.py`). |
| `phase3d/`, `hy_axis_L4/` | **Banked at campaign end** (not yet present in a fresh checkout) — the published phase diagram's NQS finals (`nersc/README.md` §4, pulled by `analysis/scripts/pull_phase3d.sh`) and the pure-$h_y$-axis L=4 campaign. See `analysis/README.md`'s "(not yet banked)" notes for what depends on them meanwhile. |

### Everything else

| Path | Role |
|---|---|
| `tests/` | Standalone scripts, run via `tests/run_all.sh [python] [test_x.py ...]` (sets `PYTHONPATH=<repo root>`; one PASS/FAIL/SKIPPED line per file). |
| `colab/` | Self-contained notebooks: `qmc_benchmarks_colab.ipynb`, `fermionic_TC_colab.ipynb` (unique un-ported numba ED sweep). |
| `notes/` | Docs: the recipes playbook, `nqs_architecture.md`, `training_cli.md` / `training_gotchas.md`, `speed_levers.md`, the phase3d campaign design/plan/handoff notes, fermionic design notes + LaTeX write-ups, `log_and_plan.md` (frozen historical record). |
| `paper/` | Manuscript skeleton (`current-version.tex` + `refs.bib`; PDF gitignored). Figures pending the phase3d campaign banking. |
| `_archive/` | Local, gitignored archive of superseded material (old-architecture sweep data, retired one-off notebooks, provenance logs). Convention: archive non-regenerable material before `git rm`; plain-delete regenerable build artifacts. Repo-history removals from the 2026-09 publication cleanup are tracked in `ARCHIVE.md` instead (retrievable from any clone via git, no local archive needed). |
| `external/`, `data/`, `wandb/`, `slurm_logs/` | Non-committed working dirs: QMC clones (+3 force-added build/patch files), local cluster mirror, W&B cache, Slurm logs (`slurm_logs/` is self-ignored via its own tracked `.gitignore` so the dir survives a clone). |

## Conventions

- Notebook outputs are stripped on commit by nbstripout (`.gitattributes`);
  after cloning run `nbstripout --install` inside the venv.
- `pyproject.toml` pins the exact JAX/NetKet stack (see Environment above);
  `requirements.txt` is the exact NERSC production freeze if an install misbehaves.
- **Never run NQS training/sweeps, or 3D exact diagonalization at L≥2 PBC (2²⁴
  states, ~2.7 GB), on a laptop.** Use Perlmutter `gpu_debug` QOS instead; verify
  locally with the `tests/` proxies and `analysis/scripts/exact_benchmarks.py`
  anchors (L=2 OBC ED, N=12, is the one cheap exception — see `CLAUDE.md`).
- Raw checkpoints (`*.mpack`) and W&B dirs are never committed; only small
  derived JSONs enter `results/`.
- Error convention in comparisons: pull = (NQS−QMC)/σ_comb with NQS bars ×3
  (labelled); always also scan raw pulls for sign-coherent runs of points.
