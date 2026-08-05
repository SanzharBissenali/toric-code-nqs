# tc3d — approximately-symmetric NQS for the 3D toric code

Neural-quantum-state study of the 3D (bosonic & fermionic) toric code under
uniform fields, $H = -J\sum_v A_v - J\sum_p B_p - h_x\sum_i\sigma^x_i - h_z\sum_i\sigma^z_i$,
mapping the topological→trivial transitions with an **approximately-symmetric CNN**
ansatz (a geometry-exact Wilson-product change of coordinates + identity-initialised
non-invariant block), validated against **QMC** (ParaToric / PMRQMC) and analytic
field series.

The architecture generalises the 2D construction of
[Kufel et al., PRL 135, 056702 (2025)](https://arxiv.org/abs/2405.17541).
The inherited 2D implementation (surface code + factored-attention transformer)
is preserved in history at the tag **`2d-final`**.

## The two experiment tracks

**1. NQS training / hyperparameter tuning (sign-problem-free regime)**

```bash
pip install -e ".[analysis]"   # bare `pip install -e .` skips matplotlib/jupyter/nbstripout
python -m tc3d.train --L 4 --bc OBC --hx 0.2 --hz 0.2 --dual_basis \
    --arch ToricCNN_gridinv --n_iter 300 \
    --ref_E -174.5957 --ref_sig 0.0147   # signed gap vs the combined QMC reference
```

`--ref_E/--ref_sig` print + log the per-step signed benchmark gap; QMC reference
values live in `results/qmc_hx*_hz*/`. `tc3d/sweep.py` batches several field
points per process (amortises the JAX compile). Production runs go through the
`nersc/submit_*.sh` wrappers (see `nersc/README.md`, `nersc/CAMPAIGN.md`).

**2. QMC validation (energies + stabilizers + magnetization)**

```bash
# one-time: clone the QMC codes into gitignored external/ (never committed)
git clone --recursive https://github.com/palmbart/ParaToric.git external/ParaToric
git clone https://github.com/LevBarash/PMRQMC.git external/PMRQMC
bash external/build_paratoric_local.sh                      # local macOS build (brew LLVM+boost)
python analysis/paratoric_driver.py --validate              # exact-anchor ladder — run BEFORE trusting numbers
python analysis/paratoric_driver.py --L 4 --hx 0.2 --hz 0.2 --beta 24 --out results/qmc_hx0.2_hz0.2/run.json
python analysis/export_pmrqmc.py --verify                   # PMRQMC cross-check (+ colab/qmc_benchmarks_colab.ipynb)
```

Analytic anchors and low/high-field series (the zero-fit accuracy certificate):
`analysis/exact_benchmarks.py` (42 self-checks; run it directly).

## Layout

| Path | Role |
|---|---|
| `tc3d/` | The package: geometry, Hamiltonian, networks (ansätze), sampler, builders, `train`/`sweep` entry points, FM + Rényi extraction, validation, checkpoint I/O. |
| `tests/` | Standalone test files (`python tests/test_geometry.py`, …). Exception: `test_exact_diag.py` is an ED reference *generator* (L=2 PBC Lanczos, ~2.7 GB) — cluster/Colab only. |
| `analysis/` | Post-processing over `results/` JSONs (pure numpy/scipy/matplotlib) + QMC drivers + analytic benchmarks. |
| `nersc/` | Slurm submit wrappers, campaign drivers, extraction jobs for Perlmutter. |
| `colab/` | Self-contained notebooks: dual-basis tuning, QMC benchmarks, fermionic ED sweep. |
| `results/` | Small derived artifacts only (curve/fit/reference JSONs); raw checkpoints are never committed. |
| `paper/` | Manuscript (`current-version.tex` + `refs.bib`; PDF gitignored). |
| `notes/` | Living docs: `log_and_plan.md` (campaign log), `nqs_architecture.md`, `handoff_fermionic_tc.md`, training CLI/gotchas. |

## Notes

- Notebook outputs are stripped on commit by nbstripout (`.gitattributes`);
  after cloning run `nbstripout --install` inside your venv.
- Never run 3D exact diagonalization locally at L≥2 PBC (2²⁴ states); use the
  cheap proxies in `tests/` and the analytic anchors instead.
