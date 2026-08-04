# Session kickoff prompt — 3D toric-code phase-diagram program

> **How to use:** paste this whole file at the start of a Claude Code session, then fill in
> the **SESSION FOCUS** block at the bottom. Everything above it is stable orientation;
> the block is the only part you edit per session.

---

You are resuming work on a neural-quantum-state (NQS) study of **3D bosonic and fermionic
toric codes** under X/Y/Z fields. The north star is **mapping topological→trivial
phase-transition diagrams** using an **approximately-symmetric CNN** ansatz, with production
runs on **NERSC Perlmutter**. The 2D surface-code code is an inherited reference and is kept
but is not the focus.

Before doing the task in SESSION FOCUS, build a working mental model of the four areas below.
**Be context-efficient:** read `CLAUDE.md` (auto-loaded) and `notes/log_and_plan.md` yourself,
then — if the focus is broad — dispatch a `general-purpose` subagent to read the ★ files and
return a ≤1-page map rather than dumping files into the main context. For a narrow focus, just
read the ★ files for the relevant area(s). Do **not** read the whole repo.

## 0. Read first (orientation)
- ★ `CLAUDE.md` — layout map, working rules, cluster charter + the current `gpu_debug`-only job gate.
- ★ `notes/log_and_plan.md` — living campaign log: current sweeps, per-L config decisions, status.
- `notes/progress_log.md` — milestone checkpoints + the only exact energy anchors past L=2
  (E₀ = −4L³ PBC / −(L³+3(L−1)²L) OBC) for sanity-checking runs.

## 1. Architecture (approximately-symmetric CNN)
- ★ `tc3d/networks.py` — the ansätze: `KernelManager3D` (geometry-exact equivariant
  kernel), `ToricCNN` (fully symmetric), `ToricCNN_full` (approx-symm, identity-init non-invariant
  block; the complex/Y-field variant), `ToricCNN_gridinv`, `GeoCNN`, Vanilla*.
- ★ `tc3d/builders.py` — the hub: config dict → geometry+Hamiltonian+ansatz+sampler+vstate,
  the arch registry, bosonic/fermionic dispatch, and the shared `run_loop`. Everything imports it.
- `tc3d/geometry.py` — 3D lattice (`vertex_all`/`plaq_all`/bonds, PBC+OBC).
- `tc3d/hamiltonian.py` — NetKet H with hx/hy/hz + J (field enters as Pauli-string weights).
- `tc3d/fermionic_decoration.py` — decorated plaquette B̃_p for the fermionic model.
- ★ `notes/nqs_architecture.md` — authoritative arch write-up; `notes/handoff_fermionic_tc.md`
  — the fermionic model + dressed Wilson-loop / Fredenhagen–Marcu order parameter derivation.

## 2. Order-parameter extraction (topological order → phase boundary)
- ★ `tc3d/fm.py` — Fredenhagen–Marcu (BFFM) detection from trained checkpoints: electric
  (hz, R=1 Wilson loop) and magnetic (hx, membrane) sectors, `fit_transition`, plotting.
- ★ `tc3d/renyi.py` — S2-Rényi central-patch transition locator (independent cross-check of fm.py).
- `tc3d/validation.py` — `nqs_observables` (E, Vscore, stabilizer/magnetization deviations);
  `tc3d/fidelity.py` — L=2 ED-fidelity guardrail (needed for the sign-full Y-field regime).
- ★ `analysis/plot_phase_diagram.py` — multi-L FM-curve fitting (logistic + finite-size scaling);
  **imported by `fm.py`**, so it is load-bearing, not just a plotting script.
- `notes/pipeline.md` (§6b = FM detection methodology); `notes/distinguishing_transition_order.md`
  (energy-kink dE/dh = −N⟨m⟩ diagnostic for 1st- vs 2nd-order).

## 3. NERSC job submission
- ★ `nersc/CAMPAIGN.md` — canonical FSS config spec (fixed arch/optimizer/sampler + per-L
  kernel/diag_shift/n_iter/walltime/E0 bounds). `nersc/README.md` — the operational how-to.
- ★ `nersc/submit_nqs_gridinv.sh` — single-run wrapper (every hyperparameter an env var; resume-safe).
- `nersc/submit_nqs_batch.sh` — batches N field points per process (amortizes the ~10-min JAX compile).
- `nersc/submit_nqs_{hz,hx}_sweep.sh` — per-cut arrays; `nersc/run_{phase,extract}_campaign.sh` — drivers.
- `nersc/extract_{fm,fm_s2,s2,membrane_s2,energy}.sh` — read-side extraction jobs.
- `analysis/check_convergence.py` + `nersc/check_hxsweep.sh` — the QA gate run BEFORE extraction.
- Actual entry points: `tc3d/train.py` (single run), `tc3d/sweep.py` (batched points).

## 4. Phase-diagram assembly & status
- ★ `analysis/vertical_line_hz.ipynb` — canonical hz-transition (O_FM R=1 + S2, tanh/Richards, FSS).
- `analysis/horizontal_line_hx.ipynb` — hx-transition (membrane sector); `analysis/phase_diagram.ipynb`
  — the assembled (hz,hx) boundary.
- `analysis/transition_order.ipynb` + `analysis/hysteresis.ipynb` — order-of-transition study.
- All analysis notebooks are pure post-processing (numpy/scipy/matplotlib) over pulled JSONs in
  `results/` — safe to run locally; they never call NetKet.

## Operational essentials (bind for every session)
- **Never run 3D ED/sweeps locally** (L=2 PBC = 2²⁴ states → OOMs this Mac). Verify with cheap
  proxies; run real jobs on NERSC.
- **Cluster:** `ssh perlmutter` (cert at `~/.ssh/nersc`; if it fails, cert likely expired — ask the
  user to re-run `sshproxy -u sanzharb`; also try `rm -f ~/.ssh/cm-*` for a stale socket). Repo on
  NERSC: `~/threed_TC/ThreeD_TC`; account `m5340_g`; env `tc-nqs`; data `$PSCRATCH/tc_nqs/`.
- **Job gate:** only `gpu_debug` smokes autonomously; **production/campaign runs need explicit
  approval.** Only `scancel` your own jobs. Commit to a feature branch, never `main`.
- **Tests:** `cd tc3d/tests && ../../.venv/bin/python test_geometry.py` (also test_fm/renyi_units/…).
- Notebook outputs are stripped by nbstripout on commit — a fresh clone needs `nbstripout --install`.

## Deliverable of this orientation
Produce a short internal map (a few lines per area — do NOT paste file contents), confirm you can
locate: the ansatz definitions, the FM+S2 extraction path, the submit→extract→plot pipeline, and
the current campaign status. Then address the SESSION FOCUS. Ask before any non-`gpu_debug` job.

---

## SESSION FOCUS


