# Session kickoff prompt — 3D toric-code NQS program (tc3d)

> **How to use:** paste this whole file at the start of a Claude Code session, then fill in
> the **SESSION FOCUS** block at the bottom. Everything above it is stable orientation;
> the block is the only part you edit per session. A ready-made FOCUS for the
> hyperparameter campaign is included — replace it when the campaign moves on.

---

You are resuming work on a neural-quantum-state (NQS) study of **3D bosonic and fermionic
toric codes** under uniform fields. The repo was promoted 2026-08-04: single installable
package **`tc3d/`**, canonical remote **github.com/SanzharBissenali/toric-code-nqs**, 2D
legacy frozen at tag `2d-final`. Everything in the tree serves **two first-class tracks**:

1. **NQS hyperparameter tuning** — `python -m tc3d.train --dual_basis` (+ `tc3d.sweep`)
   in the sign-problem-free regime across L; `--ref_E/--ref_sig` streams the signed
   per-step gap against a benchmark (QMC) energy.
2. **QMC validation** — ParaToric (primary) + PMRQMC (cross-check) via
   `analysis/paratoric_driver.py` / `analysis/export_pmrqmc.py`; energies, stabilizers,
   magnetization. References live in `results/qmc_hx0.2_hz{0.1,0.2}/`.

Build a working mental model of the areas below before acting. **Be context-efficient:**
read `CLAUDE.md` (auto-loaded) and `notes/log_and_plan.md` yourself; for a broad focus,
dispatch an Explore subagent over the ★ files and take back a ≤1-page map. Do **not**
read the whole repo.

## 0. Read first
- ★ `CLAUDE.md` — layout, working rules, QMC pipeline traps, cluster charter + job gate.
- ★ `notes/log_and_plan.md` — living campaign log (decisions, per-L configs, status).

## 1. Architecture & training
- ★ `tc3d/networks.py` — ansätze (`ToricCNN_gridinv` = production; `ToricCNN_full`,
  `GeoCNN`, Vanilla* baselines; `KernelManager3D` geometry-exact kernel).
- ★ `tc3d/builders.py` — config → geometry+H+ansatz+sampler+vstate; arch registry;
  shared `run_loop`. Everything imports it.
- `tc3d/{geometry,hamiltonian,sampler,fermionic_decoration}.py`;
  entry points `tc3d/{train,sweep}.py` (checkpoint/resume-safe).
- ★ `notes/nqs_architecture.md` (authoritative write-up), `notes/training_cli.md`,
  `notes/training_gotchas.md` (read both before touching hyperparameters).

## 2. Benchmarks & extraction
- ★ `analysis/exact_benchmarks.py` — analytic low/high-field series + h=0 anchors
  (zero-fit accuracy certificate; 39 self-checks, run it directly).
- ★ `analysis/paratoric_driver.py` — QMC driver (`--validate` ladder is MANDATORY before
  trusting new QMC numbers); `analysis/export_pmrqmc.py --verify` cross-check.
- `tc3d/fm.py` / `tc3d/renyi.py` — FM order parameter + S2 extraction (north star);
  `analysis/plot_phase_diagram.py` is imported by fm.py (load-bearing).
- `analysis/check_convergence.py` + `nersc/check_hxsweep.sh` — QA gate before extraction.

## 3. NERSC operations
- ★ `nersc/CAMPAIGN.md` (canonical FSS config spec), `nersc/README.md` (how-to).
- ★ `nersc/submit_nqs_gridinv.sh` — single run, every knob an env var, resume-safe,
  AUTO_RESUBMIT chains; `nersc/submit_nqs_batch.sh` — N field points per process
  (amortizes compile); `submit_nqs_{hz,hx}_sweep.sh` — arrays.
- `JAX_COMPILATION_CACHE_DIR` defaults to `$PSCRATCH/tc_nqs/jax_cache` in all wrappers
  (2026-08-04): cold compile ≈ 20 min for dual L=4, cached ≈ seconds. Never disable it.
- W&B: jobs log **offline** with deterministic run-ids (resubmit chunks merge);
  publish from a login node with `nersc/sync_wandb.sh` / `wandb sync`.

## Operational essentials (bind every session)
- **Never run 3D ED/sweeps locally** (L=2 PBC = 2²⁴ states OOMs the 8 GB Mac; L=2 OBC ok).
- **Cluster:** `ssh perlmutter` (sshproxy cert `~/.ssh/nersc`, 24 h, human re-mints via
  `sshproxy -u sanzharb`; stale socket → `rm -f ~/.ssh/cm-*`). Repo `~/toric-code-nqs`;
  account `m5340_g`; env `tc-nqs`; data `$PSCRATCH/tc_nqs/`.
- **Job gate:** `gpu_debug` smokes autonomously; **production/campaign runs need explicit
  per-request approval**. ≤5 h walltime + AUTO_RESUBMIT chains. Only `scancel` own jobs.
- **Git:** work on a feature branch; never commit to `main` autonomously.
- **Tests:** `cd tests && ../.venv/bin/python test_geometry.py` (etc.). `test_exact_diag.py`
  is a cluster-only ED reference generator — never run it locally.
- nbstripout strips notebook outputs on commit; fresh clones need `nbstripout --install`.

## Deliverable of this orientation
A short internal map (few lines per area, no file dumps); confirm you can locate the
ansatz definitions, the QMC benchmark path, the submit→train→extract pipeline, and the
campaign status in the log. Then address SESSION FOCUS. Ask before any non-gpu_debug job.

---

## SESSION FOCUS

**Find the best architecture + training hyperparameters for the dual-basis
approximately-symmetric NQS (`ToricCNN_gridinv --dual_basis`) in the sign-problem-free
regime, judged against QMC.**

Continuation context (2026-08-04): the promoted repo is fully validated — local smokes
reproduced pre-restructure results bit-identically; the Perlmutter chain
submit→train (2.8 s/step A100, L=4 dual)→checkpoint→resume→observables→final JSON passed
(jobs 56333004→56336504); the XLA cache at `$PSCRATCH/tc_nqs/jax_cache` is warm for the
L=4 dual graph. Prior tuning knowledge: memory notes `dual-basis-vertex-tokens`,
`l7-architecture-tuning`, `nqs-sr-qgt-dense`, `noninv-kernel-size-saturates`; L=2 OBC
dual vs primal A/B is in `notes/log_and_plan.md` (2026-07-29).

Suggested shape (adapt, don't follow blindly):
1. **Anchor point** (hx=0.2, hz=0.2), L=4 OBC — QMC target E = −174.5957(147)
   (`results/qmc_hx0.2_hz0.2/paratoric_L4_combined.json`); score runs by the
   `--ref_E -174.5957 --ref_sig 0.0147` per-step gap, final Vscore, and the
   `exact_benchmarks` series as a second opinion. (0.2, 0.1) refs exist for L=4–7.
2. **Search grid** over env-var knobs of `submit_nqs_gridinv.sh`: `KERNEL` (2,3,4…
   saturation expected near L−1), `N_NONINV`×`NONINV` (depth×width of the pre-Wilson
   block), `INV` (invariant stack), `DIAG_SHIFT` (1e-3…1e-2), `DT`, `N_SAMPLES`
   (watch n_samples < n_params under-determination), `DUAL=1` vs `DUAL=0` A/B.
3. **Workflow:** a few gpu_debug pilots to size step-time per config (autonomous),
   then present the campaign matrix + GPU-hour estimate and **ask approval** to launch
   batched arrays (`submit_nqs_batch.sh` amortizes compile across points).
4. Scale the winner: L=4 → L=5 (→L=6) at fixed config; check the Vscore-vs-L capacity
   trend (log_and_plan 2026-07-04 section is the precedent) before any bigger push.
