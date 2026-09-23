# analysis/ — from campaign results to the paper's numbers

Pure post-processing over `results/` JSONs (plus the QMC drivers and a few cluster-side
re-evaluation tools). Scripts live in `scripts/` and run from the repo root; notebooks live
in `notebooks/` and run with cwd = `analysis/notebooks/` (those that use a script add
`analysis/scripts` to `sys.path` themselves). Committed figures are in `figs/`; notebook
`savefig` lines stay commented out, so re-running a notebook never overwrites them.

Inputs marked **(not yet banked)** are still untracked while the phase-diagram campaign
runs: `results/phase3d/` and `results/hy_axis_L4/` are committed at campaign end. Until
then those notebooks only run in a checkout that has the pulled data
(`scripts/pull_phase3d.sh`). `data/tc_nqs/` is the gitignored raw mirror (per-step
curves, the pre-optimization campaign's `fm_L*.json`); nothing below needs it except
where noted.

## Paper outputs → code → inputs

| Output | Script / notebook | Inputs | Run |
|---|---|---|---|
| **Final phase diagram** (hand-vetted points: h_y=0 FSS line, L=4 planes at h_y=0/0.2/0.4, 3D pocket) | `notebooks/phase_diagram_manual.ipynb` | none: values are copied in by hand from `cut_fss_explorer` / `transition_fss` / `STATUS.md` | Run-All |
| L=4 map across all h_y planes (per-cut curves, 2D planes, 3D boundary, h_c vs h_y) | `notebooks/phase3d_L4_planes.ipynb` | `results/phase3d/summary.json` **(not yet banked)**, `results/transitions/hy*_{hx0.2_sweep-hz,hz0.1_sweep-hx}.json`, `results/hy_axis_L4/` **(not yet banked)** | regenerate the summary first: `python analysis/scripts/phase3d_status.py --export-summary --root results/phase3d --out results/phase3d/summary.json` |
| Interactive campaign viewer (drill-down + phase-diagram view) | `viewer/phase3d_viewer.html` (template), `viewer/phase3d_extras.json` (off-campaign locators), `viewer/cut_status.json` (hand-kept per-cut status); built by `scripts/phase3d_viewer_build.py` | `results/phase3d/` **(not yet banked)**, `data/tc_nqs/phase3d/` curves | `phase3d_status.py --export-viewer HY --root results/phase3d --curves-root data/tc_nqs/phase3d --out viewer_hyHY.json` per plane (and `y`), then `phase3d_viewer_build.py OUT.html viewer_hy*.json` |
| **Transition locations + FSS** (per-L locators → h_c(∞), banked records) | `scripts/transition_fit.py` (library) + `notebooks/transition_fss.ipynb` (cut registry driver) | `results/phaseB*/`, `results/hy_cuts_L4/`, `results/phase3d/` **(not yet banked; the default `CUT`)**, optional `data/tc_nqs/phase_h*/` (old lane) | pick `CUT`, Run-All; set `WRITE_JSON = True` to rebank `results/transitions/<tag>.json` (§8 = self-tests) |
| Hand-driven cut analysis (electric sigmoid FSS; magnetic up/dn branches + energy crossing) | `notebooks/cut_fss_explorer.ipynb` | `results/phase3d/` **(not yet banked)** | edit the cut knobs, Run-All |
| Banked locator records | `results/transitions/*.json` (25 records: prod, `@phase3d`, `@old` lanes) | written by `transition_fss.ipynb` / `firstorder_fit.py --out` | — |
| First-order / trivial→trivial locators (energy branch crossing, M_x / stabilizer jumps, O_FM on the winner curve) | `scripts/firstorder_fit.py` | per-run finals with `_up`/`_dn` chain names | `python analysis/scripts/firstorder_fit.py --runs DIR ... --sweep hx --fixed hz=0.1 hy=0.0 [--out results/transitions/<tag>.json]`; tests: `tests/test_firstorder_fit.py` |
| h_y dependence of the electric/magnetic cuts at L=4 (O_FM, Richards h_z,c(h_y), S₂, hysteresis) | `notebooks/hy_cuts_L4_transitions.ipynb` | `results/hy_cuts_L4/`, `results/phaseB/up/L4`, `results/phaseB_rerun/up/L4` | Run-All |
| Pure-h_y axis (S₂ collapse, V-score) | `notebooks/hy_axis_L4_S2.ipynb` | `results/hy_axis_L4/cold/L4` **(not yet banked)** | Run-All |
| **QMC-vs-NQS benchmark figures** (the 8 `figs/phaseB_*.png`) | `scripts/phaseB_figs.py` (provenance of those PNGs) | `results/phaseB*/`, `results/qmc_hx*_hz*/` | `python analysis/scripts/phaseB_figs.py [--cut right] [--no-save]` |
| QMC-vs-NQS comparison tables/plots (interactive) | `notebooks/phaseB_summary.ipynb` | same as above | Run-All (`SAVE_FIGS` stays False) |
| QMC-only observables along both arcs, L=4..12 | `notebooks/qmc_arcs_observables.ipynb` | `results/qmc_hx0.2_hz*/`, `results/qmc_hx*_hz0.1/` | Run-All |
| QMC references themselves | `scripts/paratoric_driver.py` (ParaToric, primary; `--validate` ladder first), `scripts/export_pmrqmc.py` (PMRQMC H.txt, cross-check) | `external/` builds (`external/build_paratoric_local.sh`) | see each `--help` |
| **NQS hyperparameter-tuning methodology** (sign-free lane) | `notebooks/tune_rect_summary.ipynb` + `scripts/tuning_table.py` | `results/tune_rect/`, `results/qmc_*/` | `python analysis/scripts/tuning_table.py --runs 'results/tune_rect/*/*.json' --out_md ... --out_json ...`, then Run-All |
| Sign-full (h_y≠0) tuning/benchmark lane | `notebooks/hy_rect_summary.ipynb` | `results/hy_rect_L4/`, `results/tune_rect/` | Run-All |
| Training-length / depth ablations (Phase B) | `scripts/ablation_report.py`, `scripts/ablation_report_c.py` | `results/phaseB/`, `results/phaseB_ablation{A,B,C}/`, `results/qmc_*/` (`phaseB_ablationD/` has no reader) | `python analysis/scripts/ablation_report.py` |
| Speed-lever equivalence gates (production config unchanged by the levers) | `scripts/check_equivalence.py` (cluster), `scripts/compare_equiv.py` | `results/speed_equiv/` | see `--help` |
| Architecture figure | `scripts/arch_figure.py` | geometry only | `python analysis/scripts/arch_figure.py` (writes repo-root `figures/`) |
| **Exact / analytic anchors** (low/high-field series, dualities, h=0 energies, exact h_c) | `scripts/exact_benchmarks.py` | none | `python analysis/scripts/exact_benchmarks.py` (self-checks) |
| **h_y≠0 validation** (QMC-free): dense L=2 OBC ED referee; NQS fidelity vs the ED ground state | `scripts/ed_referee_hy.py`, `scripts/hy_cert_fidelity.py` | writes/reads `results/hy_l2_certification/` | `python analysis/scripts/ed_referee_hy.py --hx 0 --hy 0.4 --hz 0 --dual --out ...`; the fidelity scorer needs the run's `.mpack` |

Cluster-side re-evaluation tools (NetKet/JAX, run on a GPU): `scripts/eval_ckpt.py`
(re-evaluate checkpoints with more samples), `scripts/eval_snapshots.py` (replay
`--snapshot_every` snapshots → `*.snapshots.json`), `scripts/bank_point.py` (bank a
plateaued sweep point). `scripts/check_convergence.py` is the QA gate for the
pre-optimization campaign trees (`phase_hx*/L*`).

## Canonical implementations

Where the same thing is computed in several places, this is the one to trust and extend:

| Functionality | Canonical | Other copies (kept, not canonical) |
|---|---|---|
| Sigmoid / Richards / fd-peak locators, per-L marker policy, FSS | `transition_fit.fit_logistic`, `fit_richards`, `locate_all`, `combine_default`, `fss_fit` | `hy_cuts_L4_transitions.ipynb` §3b local Richards fit (same inflections to ~1e-6, different error bars); `tc3d.fm.fit_transition` (in-package legacy sigmoid) |
| Energy branch crossing (up vs dn) | `firstorder_fit.energy_crossing` (bracket + error model) | `transition_fit.branch_crossing` (bare interpolation, used in `cut_fss_explorer.ipynb`) |
| Per-run health flags | `phase3d_status.add_health` (E0 vs the h=0 bound, hot Vscore; `diverged` from the final) | `phase3d_tick_checks.py` (ops tick), `check_convergence.py` (old campaign trees) |
| Exact anchors | `exact_benchmarks.REFERENCE` (+ `counts`/`Counts` for the h=0 energy) | constants repeated in `transition_fit.EXACT`, `phase3d_L4_planes.ipynb`, `phase_diagram_manual.ipynb`, `phase3d_status.bound` |
| Run loading (lowest-energy non-diverged run wins per point) | `transition_fit.load_runs` / `load_runs_branched` | `firstorder_fit.load_branches` (keeps diverged runs flagged), notebook-local loaders in `hy_cuts_L4_transitions.ipynb` |

## Campaign operations (phase3d; the campaign is still running)

Not paper outputs; they plan, pull and monitor the campaign that produces `results/phase3d/`.

| Script | Role |
|---|---|
| `scripts/phase3d_grid.py` | Deterministic grid + state-driven planner consumed by `nersc/launch_phase3d.sh` (self-tests run on every call; `--dry` to inspect). |
| `scripts/phase3d_status.py` | STATUS.md / summary.json / viewer-JSON generator; `--selftest`. |
| `scripts/pull_phase3d.sh` | rsync finals → `results/phase3d/`, per-step curves → `data/tc_nqs/phase3d/`. |
| `scripts/phase3d_reseed.py` | Best-of-N anchor reseed for stuck y-polarized chain branches. |
| `scripts/phase3d_tt_diag_probe.py` | One-off trivial↔trivial probe plan (emits launcher PLAN_FILE lines). |
| `scripts/phase3d_tick_checks.py` | Numeric health checks for the periodic monitoring tick. |
| `scripts/phase3d_local_tick.sh` | The tick itself (pull → STATUS/summary/viewer → checks); hard-codes the operator's local paths. |
| `scripts/phase3d_viewer_build.py` | Embeds per-plane viewer JSON into `viewer/phase3d_viewer.html`. |

## Fermionic toric code (separate track)

`notebooks/fermionic_arch_ladder.ipynb`, `notebooks/fermionic_h0_prefit_ladder.ipynb`,
`scripts/ed_electric_line.py`, `scripts/prefit_phase_head.py`,
`scripts/stencil_phase_head.py` and the `figs/fermionic_*.png` figures. The three scripts
execute on import (argparse at module level; `ed_electric_line.py` runs an L=2 PBC ED,
cluster only), so run them only as scripts.
