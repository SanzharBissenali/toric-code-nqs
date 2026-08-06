# Research blog — toric-code-nqs

Living log of research progress. Newest entries first. Supersedes
`notes/log_and_plan.md`, which is frozen as the historical design record of the
architecture-extension era (old-layout paths there resolve at tag `2d-final`).

## Current goal — NQS hyperparameter tuning in the dual basis

The active work is **track 1**: tune the dual-basis NQS
(`python -m tc3d.train --dual_basis` — Hadamard-rotated inputs + star-token
`ToricCNN_gridinv`) in the sign-problem-free regime, across system sizes.

- **Accuracy yardstick:** `--ref_E/--ref_sig` streams the signed per-step gap
  against the QMC benchmark. Canonical point so far: L=4 OBC,
  (h_x=0.2, h_z=0.2), combined ParaToric reference **E = −174.5957(147)**
  (`results/qmc_hx0.2_hz0.2/`; PMRQMC cross-check at h_z=0.1).
- **Status:** tune-rect campaign COMPLETE (2026-08-06, below) — canonical
  architecture locked: **dual · nh(4→8) → inv(8,8) · 15-tap · kernel=L−1 ·
  dt=0.02→0.002 · ds=1e-3 (3e-3 at strong field for L=5)**; validated vs QMC
  at 4 points × L∈{4,5,6}. Summary: `analysis/tune_rect_summary.ipynb`.
- **Track 2 (support):** QMC validation via `analysis/paratoric_driver.py`
  (ParaToric primary; `--validate` ladder mandatory) +
  `analysis/export_pmrqmc.py --verify` (PMRQMC cross-check). New benchmark
  points get generated here as tuning moves through the phase diagram.

---

## 2026-08-06 — tune-rect campaign: dual-basis architecture locked, scaled to L=6

Full campaign in ~24 h (75+ GPU jobs, ~35 GPU-h): hyperparameter search for the
dual-basis `ToricCNN_gridinv` on the rectangle hx {0.2,0.6} × hz {0.1,0.15}, judged
against dedicated ParaToric references on all observables, then the winner scaled to
L=5/6. Everything in `analysis/tune_rect_summary.ipynb` +
`results/tune_rect/tuning_table_L4_all.{md,json}`; 59+ runs on W&B (`tune_rect_L4`).

- **QMC reference grid built**: 3 new L=4 corners + L=5/6 at all corners (14 runs,
  every sum rule ≤0.06σ, β-drift z=+1.18 at the hardest corner). GPU-shared trick:
  ParaToric on 1-GPU shared QOS (16 cores), ~1–3 min per L=4 point.
- **Dual beats primal at equal budget**: winner nh(4→8)→inv(8,8) k3 @ dt=0.02/ds=1e-3
  hits −173.4099(48) at (0.2,0.1) vs QMC −173.4493(179) with Vscore 2.9e-4 (primal:
  7.2e-3 at +2.5σ, 5,319 params vs 5,345). Enablers: invariant width (2,2,2)→(8,8)
  cut the gap 4× (capacity, as predicted 2026-07-29); hot schedule dt=0.02 converges
  in 150 steps (0.01 still descending; 0.04 spikes). Damping ladder monotone: 1e-3 best.
- **r0.9 NN-only stencil fails at strong field** (+0.21…0.28 vs +0.09…0.13 at hx=0.6;
  one divergence; ds=3e-3 retry stays bad → capacity, not fragility). The dropped
  d=1.0 taps are the plaquette-adjacent correlations that matter once ⟨B_p⟩≈0.85.
  User's r0.9 tie-breaker overridden on this evidence (robustness-first rule).
- **Scaling (kernel=L−1 → params 5,345/10,377/18,673, i.e. ≈35 per spin at every L)**:
  L=6 statistically consistent with QMC
  at both weak-field points (+1.3σ / **+0.7σ**, rel 1.0e-4 / 2.9e-5), degrading
  smoothly to +4.8σ (6.1e-4) at (0.6,0.15). L=6 beats L=5 everywhere — kernel span +
  iters co-scale against fixed width. **L=5 needed ds=3e-3 at hx=0.6** (both diverged
  at 1e-3); L=6 (k=5) stable at 1e-3.
- Ops: timing smokes before scaling (L=5 15.6 s/step, L=6 59.8, nodes ±40%);
  AUTO_RESUBMIT chains carried all four L=6 runs across the 5 h cap; Phase-B name
  collision taught: training knobs belong in the checkpoint name (SEED/NAME tags);
  inline O_FM returned nothing at L=6 (R=3) — post-hoc via `tc3d.fm` if needed.

## 2026-08-05 — post-promotion hardening; cluster back online

- **Adversarial audit of the promotion** (3-agent sweep, `99b0a8f`): fixed the
  real defects it surfaced — `WALLTIME` env knob now survives AUTO_RESUBMIT
  requeues (a bare `sbatch --time` silently reset on every chain link); JAX
  cache export hoisted above the requeue trap; `setup_conda_gpu.sh` now
  `pip install -e .`; `run_phase_campaign.sh` cds to repo root and exports
  `REPO`. Tree hygiene (stale `model/ simulation/ utils/` husks removed,
  `*.egg-info` untracked) and a docs-truthfulness pass (`nersc/README.md`
  rewritten to match the actual scripts; README/CLAUDE.md corrections).
- **JAX compilation cache unified** (`212c091`): all five training submit
  wrappers default `JAX_COMPILATION_CACHE_DIR=$PSCRATCH/tc_nqs/jax_cache`.
  Measured: cold dual L=4 compile ~20 min → cached seconds. Never disable it.
- **NERSC back from maintenance a day early.** Login + `gpu_ss11` healthy;
  `$PSCRATCH/tc_nqs` survived intact (the pre-maintenance local mirror wasn't
  needed); cluster clone `~/toric-code-nqs` at parity with `main`.

## 2026-08-04 — repo promotion: ThreeD_TC → toric-code-nqs

The project moved from the old working tree to this repo, **keeping full git
history** (in-place restructure, never a fresh-history copy):

- **New canonical remote:** `github.com/SanzharBissenali/toric-code-nqs`;
  cluster clone path repointed to `~/toric-code-nqs` (`48f7778` — 15 nersc
  wrappers, Colab clone URL).
- **Package promotion** (`9a318ab`): `Three_TC/` became the flat installable
  **`tc3d/`** package (`pip install -e ".[analysis]"`); tests run against the
  installed package with no path shims.
- **2D legacy frozen at tag `2d-final`:** the 2D surface-code implementation,
  factored-attention transformer, and all dead/ED-era code were dropped from
  the tree. What survived the cut is exactly what serves the two tracks:
  geometry/Hamiltonian/networks, `builders.py`, cluster-update sampler,
  `train.py`/`sweep.py`, `fm.py`/`renyi.py` extraction, QMC drivers,
  analytic benchmarks (42 self-checks), NERSC wrappers.
- **Dual-basis branch merged:** `--dual_basis` (Hadamard + star tokens) is now
  a first-class `train.py` flag rather than a feature branch.

Everything before this point lives in `notes/log_and_plan.md` (frozen) and the
git history — highlights: 2D→3D architecture extension and L=2 arch comparison
(symmetry-aware ~100× lower energy error than plain CNN), the 2026-07 phase
campaigns (`phase_hx*`/`phase_hz*` sweeps, FM + S2 extraction, x–z magnetic
line), and the QMC benchmark pipeline build-out.
