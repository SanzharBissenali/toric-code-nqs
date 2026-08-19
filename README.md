# tc3d — approximately-symmetric NQS for the 3D toric code

Neural-quantum-state study of the 3D (bosonic & fermionic) toric code under
uniform fields, $H = -J\sum_v A_v - J\sum_p B_p - h_x\sum_i\sigma^x_i - h_z\sum_i\sigma^z_i$,
mapping the topological→trivial transitions with an **approximately-symmetric CNN**
ansatz (a geometry-exact Wilson-product change of coordinates + identity-initialised
non-invariant block), validated against **QMC** (ParaToric / PMRQMC) and analytic
field series. The architecture generalises the 2D construction of
[Kufel et al., PRL 135, 056702 (2025)](https://arxiv.org/abs/2405.17541).
The inherited 2D implementation is preserved in history at the tag **`2d-final`**.

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
+ learning curves), extract transition points with finite-size scaling, then
extend the same pipeline to the sign-full regime ($h_y \neq 0$) and the
fermionic toric code — with QMC as referee wherever the model is sign-free.

The earlier phase-diagram sweeps (pre-optimised architecture, pre-recipes
protocol) are **superseded**: those cuts will be re-run through the current
pipeline before the phase boundary is trusted. Their data and one-off notebooks
live in `_archive/` (local, gitignored).

## Quickstart — the two experiment tracks

**1. NQS training / sweeps (sign-problem-free regime)**

```bash
pip install -e ".[analysis]"   # bare `pip install -e .` skips matplotlib/jupyter/nbstripout
python -m tc3d.train --L 4 --bc OBC --hx 0.2 --hz 0.2 --dual_basis \
    --arch ToricCNN_gridinv --noninv_hidden 4 8 --inv_hidden 8 8 --kernel_size 3 \
    --n_iter 300 --ref_E -174.5957 --ref_sig 0.0147   # signed per-step gap vs the QMC reference
```

`--ref_E/--ref_sig` stream the benchmark gap; QMC reference values live in
`results/qmc_hx*_hz*/`. `python -m tc3d.sweep` batches several field points per
process (amortises the JAX compile). Production runs go through the
`nersc/submit_*.sh` wrappers (see `nersc/README.md`, `nersc/CAMPAIGN.md`).
Production ansatz flags for every launch:
`DUAL=1 NONINV_HIDDEN="4 8" INV="8 8" KERNEL=$((L-1))` (wrapper defaults differ).

**2. QMC validation (energy + stabilizers + magnetizations + order parameters)**

```bash
# one-time: clone the QMC codes into gitignored external/ (never committed)
git clone --recursive https://github.com/palmbart/ParaToric.git external/ParaToric
git clone https://github.com/LevBarash/PMRQMC.git external/PMRQMC
bash external/build_paratoric_local.sh                      # local macOS build (brew llvm boost hdf5 ninja)
python analysis/paratoric_driver.py --validate              # exact-anchor ladder — run BEFORE trusting numbers
python analysis/paratoric_driver.py --L 4 --hx 0.2 --hz 0.2 --beta 24 --nbs_mult 4 \
    --out results/qmc_hx0.2_hz0.2/run.json                  # nbs_mult>=4 for production (8 near a crossing)
python analysis/export_pmrqmc.py --verify                   # PMRQMC cross-check (+ colab/qmc_benchmarks_colab.ipynb)
```

Analytic anchors and low/high-field series (the zero-fit accuracy certificate):
`analysis/exact_benchmarks.py` (42 self-checks; run it directly). Near any
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
| Validation / infra | `validation.py` (L=2 ED harness), `exact_diag.py` (matrix-free Numba ED; QMC `--verify`), `io.py` (checkpoints), `config.py` (device probe), `wandb_logger.py` |

### `analysis/` — post-processing, QMC drivers, benchmark figures

| Group | Files |
|---|---|
| QMC pipeline | `paratoric_driver.py` (validation ladder + production), `export_pmrqmc.py`, `exact_benchmarks.py` (analytic series), `qmc_arcs_observables.ipynb` (pure-QMC arc sanity to L=12) |
| QA / evaluation | `check_convergence.py` (pre-extraction gate), `eval_ckpt.py` (re-eval at larger samples), `eval_snapshots.py` (snapshot replay → convergence-vs-step), `bank_point.py` (bank a plateaued sweep point), `test_grad_guard.py` (divergence-guard regression) |
| Benchmark figures | `phaseB_figs.py` — **the generator of the 8 committed `phaseB_*` PNGs** (β-honest QMC refs + per-point best-state substitution table; bit-exact in the repo venv). `phaseB_summary.ipynb` (pre-reconciliation campaign notebook, kept for the record; its `SAVE_FIGS` gate stays False), `tune_rect_summary.ipynb` (architecture tuning + learning curves → 6 figs), `fermionic_arch_ladder.ipynb` + `fermionic_h0_prefit_ladder.ipynb` (→ 4 figs) |
| FSS / extraction machinery | `plot_phase_diagram.py` (scriptable sigmoid fit + `--fss` over `tc3d.fm` JSONs), `tuning_table.py`, `ablation_report*.py` (reusable pull-table pattern). The retired notebook templates (O_FM/S₂ fits, PDG errors, exponent sweeps) live in `_archive/analysis_archive/{vertical_line_hz,xz_cut}.ipynb` — start the redo-campaign extraction notebooks from them. |
| Fermionic sign-head track | `prefit_phase_head.py`, `stencil_phase_head.py`, `ed_electric_line.py`, `decoder_sign_prototype.py` (WIP — pending commit by the fermionic session) |
| Paper assets | `arch_figure.py` (Wilson-CNN architecture diagram) |

### Figure directories (two, on purpose)

- **`analysis/figs/`** — git-tracked, curated benchmark figures. This is the
  reproducibility target; every PNG maps to a generator notebook (table below).
- **`figures/`** (repo root) — gitignored scratch target for paper-bound
  renders; `paper/current-version.tex` reads it via `\graphicspath`.
- Notebook `plt.savefig` lines stay **commented out** (and `SAVE_FIGS` gates
  stay `False`) — figures are promoted manually, never auto-saved. The one
  scripted exception: `python analysis/phaseB_figs.py` regenerates the eight
  committed `phaseB_*` PNGs bit-exactly (same venv).

| Figures in `analysis/figs/` | Generator |
|---|---|
| `phaseB_h_z_sweep_*` (energy, stabilizers, ⟨σ_z⟩, Z-string) | `phaseB_figs.py` |
| `phaseB_h_x_sweep_*` (energy, stabilizers, ⟨σ_x⟩, X-membrane) | `phaseB_figs.py` |
| `tune_rect_*` (scaling, learning curves, rel. errors) + `single_point_0.2_0.1_learning_curve` | `tune_rect_summary.ipynb` |
| `fermionic_h0_prefit_ladder`, `fermionic_h0_L3_ghost` | `fermionic_h0_prefit_ladder.ipynb` |
| `fermionic_arch_ladder`, `fermionic_ladder_E_L2` | `fermionic_arch_ladder.ipynb` |

### `nersc/` — Perlmutter wrappers (all env-var driven, resume-safe)

| Group | Scripts |
|---|---|
| NQS launch | `submit_nqs_gridinv.sh` (single run / warm-chain link), `submit_nqs_batch.sh` (batched sweep), `submit_nqs_{hz,hx}_sweep.sh` (arrays), `submit_nqs_geocnn.sh` (symmetry-unaware baseline), `run_phase_campaign.sh` (grid driver), `launch_phaseB{,_rerun}.sh` (executable provenance of `results/phaseB*`) |
| QMC | `submit_qmc_paratoric.sh`, `build_paratoric_perlmutter.sh` |
| Extraction | `extract_fm.sh`, `extract_s2.sh`, `extract_fm_s2.sh` (electric), `extract_membrane_s2.sh` (magnetic), `extract_energy.sh` (energy-kink diagnostic), `submit_extract_fm{,_s2}.sh`, `run_extract_campaign.sh` |
| QA / monitoring | `check_hxsweep.sh` (gate before extraction), `submit_eval_ckpt.sh`, `sync_wandb.sh`, `ladder_status.sh` |
| Fermionic | `launch_fermionic_ladder.sh` |
| Setup / docs | `setup_conda_gpu.sh`, `README.md` (how-to), `CAMPAIGN.md` (frozen config provenance of the July campaign — superseded for new runs by the recipes doc) |

### `results/` — data map (small derived JSONs only; checkpoints gitignored)

| Family | Contents |
|---|---|
| `qmc_hx*_hz*/` | QMC reference anchors: electric arc (h_x=0.2, h_z swept), magnetic arc (h_z=0.1, h_x swept), tuning points. Loaders take the **highest-β subset only** — never mix β. |
| `phaseB/`, `phaseB_rerun/`, `phaseB_ablation{A..D}/` | The Phase-B NQS-vs-QMC campaign (current pipeline): cold sweeps, warm chains, hysteresis branches, ablations. |
| `tune_rect/` | Architecture-tuning campaign. Winner: dual-basis, non-inv 4→8, inv (8,8), 15-tap kernel k=L−1. |
| `fermionic_ladder/`, `fermionic_h0/`, `fermionic_eline/` | Fermionic track: architecture ladder, h=0 sign-structure anchors + `ed_L2_electric.json` (the electric-line ED reference), electric-line NQS runs. |
| `qmc_hx0.88_hz0.0/`, `threed_bosonic.json` | Standalone anchors (membrane point-cube reference; provenance for constants in `train.py`). |

### Everything else

| Path | Role |
|---|---|
| `tests/` | 11 standalone checks (`cd tests && ../.venv/bin/python test_geometry.py`, run each directly). **Cluster-only, never local:** `test_exact_diag.py` (ED reference generator, ~2.7 GB) and `test_hamiltonian.py` (3× 2²⁴-row `to_sparse()`). |
| `colab/` | Self-contained notebooks: `dual_basis_colab.ipynb` (L=4 tuning/AB), `qmc_benchmarks_colab.ipynb`, `fermionic_TC_colab.ipynb` (unique un-ported numba ED sweep). |
| `notes/` | Docs: the recipes playbook, `nqs_architecture.md`, `training_cli.md` / `training_gotchas.md`, fermionic design notes + LaTeX write-ups, `log_and_plan.md` (frozen historical record). |
| `paper/` | Manuscript skeleton (`current-version.tex` + `refs.bib`; PDF gitignored). Figures pending the re-run campaign. |
| `_archive/` | Local, gitignored archive of superseded material (old-architecture sweep data, retired one-off notebooks, provenance logs). Convention: archive non-regenerable material before `git rm`; plain-delete regenerable build artifacts. |
| `external/`, `data/`, `wandb/`, `slurm_logs/` | Non-committed working dirs: QMC clones (+3 force-added build/patch files), local cluster mirror, W&B cache, Slurm logs (`slurm_logs/` is self-ignored via its own tracked `.gitignore` so the dir survives a clone). |

## Conventions

- Notebook outputs are stripped on commit by nbstripout (`.gitattributes`);
  after cloning run `nbstripout --install` inside the venv.
- `pyproject.toml` carries unpinned deps for `pip install -e`; `requirements.txt`
  pins the known-good stack (jax 0.5.2, netket 3.16.1) if an install misbehaves.
- Never run 3D exact diagonalization locally at L≥2 PBC (2²⁴ states, ~2.7 GB);
  use the `tests/` proxies and `analysis/exact_benchmarks.py` anchors.
- Raw checkpoints (`*.mpack`) and W&B dirs are never committed; only small
  derived JSONs enter `results/`.
- Error convention in comparisons: pull = (NQS−QMC)/σ_comb with NQS bars ×3
  (labelled); always also scan raw pulls for sign-coherent runs of points.
