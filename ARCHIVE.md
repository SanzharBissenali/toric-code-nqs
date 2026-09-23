# ARCHIVE — publication cleanup (branch `chore/publication-cleanup`)

What was removed or renamed on this branch relative to tag **`pre-publication-cleanup`**,
why (inferred from `git log pre-publication-cleanup..HEAD`), and how to get it back.
Nothing here is gone from history — the tag is a permanent snapshot.

**Retrieve a removed file's content:**
```
git show pre-publication-cleanup:<path>
```
**Restore it into the working tree:**
```
git checkout pre-publication-cleanup -- <path>
```
**See the full pre-cleanup source of a file that was only trimmed** (e.g.
`tc3d/networks.py` before the dead ansätze were dropped):
```
git show pre-publication-cleanup:tc3d/networks.py
```

## Files removed

### `analysis/scripts/` and `analysis/notebooks/`

| Path | What it was | Why removed |
|---|---|---|
| `analysis/scripts/plot_phase_diagram.py` | Scriptable sigmoid-fit + `--fss` phase-diagram plotter over `tc3d.fm`'s output JSONs | Superseded by `analysis/scripts/transition_fit.py` (sigmoid/Richards/fd-peak locators + FSS) and `analysis/notebooks/transition_fss.ipynb` |
| `analysis/scripts/bench_hy_speed.py` | Per-step speed-lever microbenchmark (`--microbench`; µs/config, TFLOP/s HLO census) | Its callers (`nersc/speed_profile_job.sh`, `nersc/submit_hy_speed_bench.sh`) were removed first; superseded by `nersc/speed_equiv_job.sh`, the production float32/dense equivalence certificate. Numbers it produced are archived in `notes/speed_levers.md` |
| `analysis/scripts/summarize_speed_bench.py` | Table generator over `bench_hy_speed.py`'s raw JSONs | Speed levers are now merged into production defaults; no longer needed as a live tool. Numbers kept in `notes/speed_levers.md`; raw JSONs stay in `results/speed_bench/` |
| `analysis/notebooks/phase3d_progress.ipynb` | Interactive progress viewer for the phase3d campaign | Ops duplicate of `python analysis/scripts/phase3d_status.py --out STATUS.md` |

### `colab/`

| Path | What it was | Why removed |
|---|---|---|
| `colab/dual_basis_colab.ipynb` | L=4 dual-basis tuning/A-B notebook | Superseded L=4 tuning notebook; no callers, campaign moved to the cluster wrappers |

### `.claude/`

| Path | What it was | Why removed |
|---|---|---|
| `.claude/workflows/deep-research-nqs-error-bars.js` | An unused Claude Code workflow script | Dead — no references anywhere in the tree |

### `nersc/` — retired per-point sweep family and one-offs

| Path | What it was | Why removed |
|---|---|---|
| `nersc/submit_nqs_hz_sweep.sh` | Per-point array launcher, sweeps h_z (`phase_hz{HZ}/L*`) | Superseded by the batched `nersc/submit_nqs_batch.sh` workhorse used by the phase3d campaign |
| `nersc/submit_nqs_hx_sweep.sh` | Per-point array launcher, sweeps h_x (`phase_hx{HX}/L*`) | Same — superseded by `submit_nqs_batch.sh` |
| `nersc/run_phase_campaign.sh` | Original (h_x, L) grid driver: 24 array jobs, one per (h_x × L), 13 h_z tasks each | Superseded by `analysis/scripts/phase3d_grid.py` + `nersc/launch_phase3d.sh` (the phase3d campaign planner) |
| `nersc/check_hxsweep.sh` | QA gate before extraction, for the retired per-point sweep family | Family retired; phase3d/Phase-B have their own convergence gates (`analysis/scripts/check_convergence.py`, `phase3d_status.add_health`) |
| `nersc/extract_energy.sh` | Energy-kink diagnostic extractor, retired sweep family | Family retired |
| `nersc/run_extract_campaign.sh` | Extraction driver, retired sweep family | Family retired |
| `nersc/submit_extract_fm.sh` | Extraction wrapper (electric FM), retired sweep family | Family retired |
| `nersc/submit_extract_fm_s2.sh` | Extraction wrapper (electric FM + S2), retired sweep family | Family retired |
| `nersc/submit_nqs_geocnn.sh` | One-off symmetry-unaware `GeoCNN` architecture-comparison control | Campaign long closed |
| `nersc/speed_profile_job.sh` | Speed-lever profiling job (drove `bench_hy_speed.py`) | Superseded by `nersc/speed_equiv_job.sh` |
| `nersc/submit_hy_speed_bench.sh` | Speed-lever benchmark submit wrapper | Superseded by `nersc/speed_equiv_job.sh` |
| `nersc/submit_hy_l2_cert.sh` | One-off single-point (h_x=0.2, h_y=0.2, h_z=0.1) L=2 OBC certification script | Folded into `nersc/submit_hy_axis_l2_cert.sh` as `HX`/`HZ` env knobs (default 0, unchanged pure-h_y-axis behavior) |

### `tests/` — cluster-only, orphaned

| Path | What it was | Why removed |
|---|---|---|
| `tests/test_exact_diag.py` | L=2 PBC Lanczos ED reference generator (~2.7 GB workspace) | Fed the dead L=2 PBC reference-comparison harness in `validation.py` (also removed, see below); cannot run on a dev box and nothing consumes its output any more |
| `tests/test_hamiltonian.py` | 3× `to_sparse()` on the 2²⁴-row Hamiltonian (~75 min, ~1.7 GB each) | Same — cluster-only and orphaned once its consumer was gone |

## Renamed (not removed)

| From | To | Note |
|---|---|---|
| `analysis/scripts/test_grad_guard.py` | `tests/test_grad_guard.py` | Divergence-guard regression test moved into the standard `tests/` location; runs via `tests/run_all.sh` like every other test |

## Code symbols removed from `tc3d/` (files themselves survive, trimmed)

| Module | Symbols removed | Why |
|---|---|---|
| `tc3d/networks.py` | Ansätze `ToricCNN`, `ToricCNN_full`, `VanillaCNN`, `VanillaWilsonCNN` + their private support (`CNN_invariant_3D`, `compute_edges_3D`) and builder branches | No branch, script, notebook, wrapper or banked `results/` config used them — every banked config is `ToricCNN_gridinv`(`_dual`) or `GeoCNN`. `DEFAULTS["arch"]` (was `ToricCNN_full`) is now `ToricCNN_gridinv` |
| `tc3d/train.py` | CLI flags `--hidden`, `--vanilla_depth`, `--noninv_random`, `--radius_plaq` | Served only the removed ansätze above |
| `tc3d/train.py` | `--hz_preset` / `HZ_PRESETS` table, dead `is_gpu`, an unused import | L=2 PBC preset table with no caller on any branch or wrapper; `--exact_E0` remains the way to set a custom energy target |
| `tc3d/hamiltonian.py` | `Jy_v`, `Jy_p`, `Jbond` Hamiltonian terms | No config, CLI flag, script or branch ever set them; the bosonic `build_hamiltonian` always assembles from the cached Pauli-string parts now. `create_hamiltonian` keeps `bonds` as an ignored optional arg for existing external callers (e.g. `ed_referee_hy.py`) |
| `tc3d/validation.py` | `load_reference`, `find_reference`, `_REF_KEYS`, `_dev`, `nqs_metrics`, `train_one`, `run_validation` + `builders` re-exports | Dead L=2 PBC reference-comparison harness, no caller on any branch (current importers use only `nqs_observables`, `pooled_final_observables`, `topological_observables`, `build_eval_operators`) |
| `tc3d/geometry.py` | `construct_Wilson_generators`, `find_generators`, `select_subset`, `qubit_select` (called a nonexistent `_mapping2Dto1D`), `select_bulk`, the `vertex_bulk_hetero`/`vertex_edge_hetero` split | Unused 2D-era helpers, no caller on any branch. `bonds`/`Nbonds` were kept — the fermionic Hamiltonian consumes `geo.bonds` |
| `tc3d/sampler.py` | `create_custom_sampler` | Unused (bulk-only star flips, deprecated `n_sweeps` kwarg); production sampler is `builders.build_sampler` |
| `tc3d/fm.py` | `magnetic_membrane_edges` (superseded flat "Option A" sheet), `verify_fm_geometry`, `verify_fm_charge_flux`, `plot_fm_sweep` | No caller anywhere; the production magnetic operator is `magnetic_cube_edges`. The `placement="boundary"` path was kept |
| `tc3d/renyi.py` | `verify_s2_geometry`, `s2_stabilizer_exact`, `_gf2_rank` | No caller anywhere; the S2 anchors `S2_EXACT_HZ0`/`S2_EXACT_HZINF` were kept |

## Data added / promoted to tracked (not a removal, noted for completeness)

`analysis/scripts/transition_fit.py`, the notebooks `transition_fss.ipynb`,
`cut_fss_explorer.ipynb`, `phase_diagram_manual.ipynb`, `hy_axis_L4_S2.ipynb`, and
`results/transitions/*.json` (25 banked locator records) were brought in from an
untracked working checkout during this cleanup — see `BLOG.md`'s 2026-09-24 entry.

## Fermionic and 2D material

Out of scope for this cleanup by design (user decision) — untouched. Fermionic docs,
scripts and notebooks remain exactly as they were; see `README.md`'s "separate tracks"
note. The 2D implementation was never in this tree; it lives at git tag `2d-final`.
